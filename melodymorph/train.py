"""Training loop: AdamW + cosine schedule with warmup, optional AMP, best-val
checkpointing with early stopping.

``checkpoint_path`` holds the best weights for inference (model config + vocab,
no optimizer state). ``<checkpoint>.last.pt`` holds the latest epoch with
optimizer state, for ``--resume``.
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

from .corpus import build_corpus, load_records, split_records
from .dataset import MelodyWindowDataset, augment
from .model import MelodyTransformer, ModelConfig
from .tokenizer import MelodyTokenizer

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    # data
    data_cache: str = "data/melodies.jsonl"
    corpus_limit: int | None = None
    val_fraction: float = 0.1
    transpose_range: int = 2
    block_size: int = 256
    stride: int = 128
    # model
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    # optimisation
    batch_size: int = 64
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    patience: int | None = 3  # stop after this many epochs without val improvement
    use_amp: bool = False  # Turing GPUs (e.g. GTX 16xx) have no tensor cores;
                            # fp16 autocast is a net slowdown there, not a speedup
    # bookkeeping
    seed: int = 1337
    checkpoint_path: str = "checkpoints/best.pt"
    run_dir: str = "runs"
    resume_from: str | None = None
    device: str | None = None

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(**raw)


def _cosine_warmup_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(progress, 1.0)))


def _resolve_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def last_checkpoint_path(checkpoint_path: str) -> str:
    p = Path(checkpoint_path)
    return str(p.with_name(p.stem + ".last.pt"))


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    log.info("training on device: %s", device)

    try:
        records = load_records(cfg.data_cache)
    except FileNotFoundError:
        records = build_corpus(cfg.data_cache, limit=cfg.corpus_limit)

    tokenizer = MelodyTokenizer()
    train_recs, val_recs = split_records(records, cfg.val_fraction)
    train_melodies = [r.notes for r in train_recs]
    val_melodies = [r.notes for r in val_recs]
    train_bars, val_bars = [r.bar for r in train_recs], [r.bar for r in val_recs]
    log.info("triple-metre melodies: %d train / %d val",
             train_bars.count(12), val_bars.count(12))
    log.info("melodies: %d train / %d val", len(train_melodies), len(val_melodies))

    aug, aug_bars = augment(train_melodies, cfg.transpose_range, train_bars)
    train_ds = MelodyWindowDataset(aug, tokenizer, cfg.block_size, cfg.stride, bars=aug_bars)
    val_ds = MelodyWindowDataset(val_melodies, tokenizer, cfg.block_size, cfg.stride, bars=val_bars)
    log.info("windows: %d train / %d val", len(train_ds), len(val_ds))

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
        pad_id=tokenizer.pad_id,
    )
    model = MelodyTransformer(model_cfg).to(device)
    log.info("model parameters: %d", model.num_parameters())

    optimizer = torch.optim.AdamW(model.param_groups(cfg.weight_decay), lr=cfg.lr)
    amp_enabled = cfg.use_amp and device.startswith("cuda")
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
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" not in ckpt:
            log.warning("%s has no optimizer state (resume from %s instead)",
                        cfg.resume_from, last_checkpoint_path(cfg.checkpoint_path))
        else:
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
        best_val = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        hist_file = run_dir / "history.json"
        if hist_file.exists():
            # keep only the epochs this checkpoint has actually seen
            saved = json.loads(hist_file.read_text())
            history = {k: saved.get(k, [])[:start_epoch] for k in history}
        if total_steps != ckpt.get("total_steps", total_steps):
            log.warning("epoch count changed: the LR schedule is stretched from step %d on", step)
        log.info("resumed at epoch %d, step %d, best_val=%.4f", start_epoch, step, best_val)

    if start_epoch >= cfg.epochs:
        log.warning(
            "resumed checkpoint epoch (%d) >= configured epochs (%d); nothing to train",
            start_epoch, cfg.epochs,
        )

    last_ckpt_path = last_checkpoint_path(cfg.checkpoint_path)
    stale_epochs = 0

    try:
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
                with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
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
            val_loss = _evaluate_loss(model, val_loader, device, amp_enabled=amp_enabled)
            val_ppl = math.exp(min(val_loss, 20))

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_perplexity"].append(val_ppl)

            log.info(
                "epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_ppl=%.2f  (%.1fs)",
                epoch + 1, cfg.epochs, train_loss, val_loss, val_ppl, time.time() - t0,
            )

            # Always save latest checkpoint for seamless resumption
            _save_checkpoint(
                cfg, model_cfg, model, tokenizer, last_ckpt_path,
                optimizer=optimizer, scaler=scaler, epoch=epoch + 1, step=step,
                val_loss=val_loss, best_val=min(best_val, val_loss), total_steps=total_steps,
            )
            if val_loss < best_val:
                best_val, stale_epochs = val_loss, 0
                _save_checkpoint(
                    cfg, model_cfg, model, tokenizer, cfg.checkpoint_path,
                    epoch=epoch + 1, step=step, val_loss=val_loss, best_val=best_val,
                )
            else:
                stale_epochs += 1

            (run_dir / "history.json").write_text(json.dumps(history, indent=2))
            _plot_loss_curve(history, run_dir / "loss_curve.png")
            if cfg.patience is not None and stale_epochs >= cfg.patience:
                log.info("early stop: no val improvement for %d epochs", stale_epochs)
                break

    except KeyboardInterrupt:
        log.warning("training interrupted by user (Ctrl+C); saving state...")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        _plot_loss_curve(history, run_dir / "loss_curve.png")

    return {"best_val_loss": best_val, "history": history}


def _evaluate_loss(
    model: MelodyTransformer, loader: DataLoader, device: str, amp_enabled: bool = False,
) -> float:
    model.eval()
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.amp.autocast(device_type=device_type, enabled=amp_enabled):
                _, loss = model(xb, yb)
            # weight by scored tokens, not batches (padding and partial batches vary)
            n_tok = int((yb != model.cfg.pad_id).sum())
            total += loss.item() * n_tok
            n += n_tok
    return total / max(n, 1)


def _save_checkpoint(
    cfg: TrainConfig, model_cfg: ModelConfig, model: MelodyTransformer,
    tokenizer: MelodyTokenizer, path: str, optimizer=None, scaler=None,
    epoch: int = 0, step: int = 0, val_loss: float = 0.0, best_val: float = float("inf"),
    total_steps: int | None = None,
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
        "best_val_loss": best_val,
        "total_steps": total_steps,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None and scaler.is_enabled():
        payload["scaler_state"] = scaler.state_dict()
    torch.save(payload, path)


def load_checkpoint_payload(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_checkpoint(path: str, device: str | None = None) -> tuple[MelodyTransformer, MelodyTokenizer]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"checkpoint file {path!r} not found. Run `melodymorph train` first to produce a checkpoint."
        )
    device = device or _resolve_device()
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # checkpoints from before the time-signature tokens use the 73-token prefix
    tokenizer = MelodyTokenizer(meter_tokens=len(ckpt["vocab"]) > len(MelodyTokenizer(False)))
    if tokenizer.itos != ckpt["vocab"]:
        raise ValueError("checkpoint vocabulary does not match the current tokenizer")

    model_cfg = ModelConfig(**ckpt["model_config"])
    model = MelodyTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


def _plot_loss_curve(history: dict, path: Path) -> None:
    if not history.get("train_loss"):
        return
    import matplotlib.ticker as ticker
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 4))
    ax = fig.add_subplot()
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax.plot(epochs, history["train_loss"], label="train loss", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="val loss", marker="s", markersize=3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("MelodyMorph training curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)

