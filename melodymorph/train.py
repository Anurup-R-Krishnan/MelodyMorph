"""Training loop: AdamW + cosine schedule with warmup, AMP on CUDA, best-val
checkpointing. Everything a checkpoint needs to regenerate from (weights, model
config, vocab) is bundled in a single .pt file.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .corpus import build_corpus, load_corpus
from .dataset import MelodyWindowDataset, build_token_stream, split_melodies
from .model import MelodyTransformer, ModelConfig
from .tokenizer import MelodyTokenizer

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # data
    data_cache: str = "data/melodies.jsonl"
    corpus_limit: int | None = None
    val_fraction: float = 0.1
    transpose_range: int = 5
    block_size: int = 256
    stride: int = 128
    # model
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    # optimisation
    batch_size: int = 64
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = False  # Turing GPUs (e.g. GTX 16xx) have no tensor cores;
                            # fp16 autocast is a net slowdown there, not a speedup
    # bookkeeping
    seed: int = 1337
    checkpoint_path: str = "checkpoints/best.pt"
    run_dir: str = "runs"
    resume_from: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**raw)


def _cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = _device()
    log.info("training on device: %s", device)

    try:
        melodies = load_corpus(cfg.data_cache)
    except FileNotFoundError:
        melodies = build_corpus(cfg.data_cache, limit=cfg.corpus_limit)

    tokenizer = MelodyTokenizer()
    train_melodies, val_melodies = split_melodies(melodies, cfg.val_fraction, cfg.seed)
    log.info("melodies: %d train / %d val", len(train_melodies), len(val_melodies))

    train_stream = build_token_stream(train_melodies, tokenizer, cfg.transpose_range)
    val_stream = build_token_stream(val_melodies, tokenizer, transpose_range=0)
    log.info("tokens: %d train / %d val", len(train_stream), len(val_stream))

    train_ds = MelodyWindowDataset(train_stream, cfg.block_size, cfg.stride)
    val_ds = MelodyWindowDataset(val_stream, cfg.block_size, cfg.block_size)

    effective_batch = min(cfg.batch_size, max(1, len(train_ds)))
    drop_last = len(train_ds) >= effective_batch * 2
    train_loader = DataLoader(train_ds, batch_size=effective_batch, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, max(1, len(val_ds))), shuffle=False)

    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_model=cfg.d_model,
        block_size=cfg.block_size,
        dropout=cfg.dropout,
    )
    model = MelodyTransformer(model_cfg).to(device)
    log.info("model parameters: %d", model.num_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_enabled = cfg.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    total_steps = cfg.epochs * max(len(train_loader), 1)
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))

    history = {"train_loss": [], "val_loss": [], "val_perplexity": []}
    best_val = float("inf")
    start_epoch = 0
    step = 0
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.resume_from:
        log.info("resuming training from checkpoint: %s", cfg.resume_from)
        try:
            ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=True)
        except Exception:
            ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                log.warning("could not load optimizer state: %s", e)
        if "scaler_state" in ckpt and amp_enabled:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception as e:
                log.warning("could not load scaler state: %s", e)
        start_epoch = ckpt.get("epoch", 0)
        step = ckpt.get("step", 0)
        best_val = ckpt.get("val_loss", float("inf"))
        hist_file = run_dir / "history.json"
        if hist_file.exists():
            try:
                history = json.loads(hist_file.read_text())
            except Exception:
                pass
        log.info("resumed at epoch %d, step %d, best_val=%.4f", start_epoch, step, best_val)

    if start_epoch >= cfg.epochs:
        log.warning(
            "resumed checkpoint epoch (%d) >= configured epochs (%d); nothing to train",
            start_epoch, cfg.epochs,
        )

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            lr = _cosine_warmup_lr(step, total_steps, warmup_steps, cfg.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=amp_enabled):
                _, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        train_loss = epoch_loss / max(n_batches, 1)
        val_loss = _evaluate_loss(model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 20))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_perplexity"].append(val_ppl)

        log.info(
            "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_ppl=%.2f  (%.1fs)",
            epoch + 1, cfg.epochs, train_loss, val_loss, val_ppl, time.time() - t0,
        )

        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(
                cfg, model_cfg, model, tokenizer, cfg.checkpoint_path,
                optimizer=optimizer, scaler=scaler, epoch=epoch + 1, step=step, val_loss=val_loss,
            )

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    _plot_loss_curve(history, run_dir / "loss_curve.png")

    return {"best_val_loss": best_val, "history": history}


def _evaluate_loss(model: MelodyTransformer, loader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total += loss.item()
            n += 1
    return total / max(n, 1)


def _save_checkpoint(
    cfg: TrainConfig, model_cfg: ModelConfig, model: MelodyTransformer,
    tokenizer: MelodyTokenizer, path: str, optimizer=None, scaler=None,
    epoch: int = 0, step: int = 0, val_loss: float = 0.0,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model_cfg.to_dict(),
        "vocab": tokenizer.itos,
        "train_config": asdict(cfg),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, device: str | None = None) -> tuple[MelodyTransformer, MelodyTokenizer]:
    device = device or _device()
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location=device, weights_only=False)

    tokenizer = MelodyTokenizer()
    if tokenizer.itos != ckpt["vocab"]:
        raise ValueError("checkpoint vocabulary does not match the current tokenizer")

    model_cfg = ModelConfig(**ckpt["model_config"])
    model = MelodyTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


def _plot_loss_curve(history: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["train_loss"], label="train loss")
    ax.plot(history["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("MelodyMorph training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
