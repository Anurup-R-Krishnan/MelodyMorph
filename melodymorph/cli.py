"""Command-line entry point.

    melodymorph prepare-data
    melodymorph train --config configs/base.yaml
    melodymorph generate --seed-text "C4/q E4/q G4/h" --mode continuation -k 4 --out out/
    melodymorph evaluate --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("melodymorph")


def _cmd_prepare_data(args: argparse.Namespace) -> None:
    from .corpus import build_corpus, corpus_stats

    melodies = build_corpus(cache=args.cache, limit=args.limit, force=args.force)
    stats = corpus_stats(melodies)
    log.info("corpus ready: %s", stats)


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig.from_yaml(args.config)
    if getattr(args, "resume", None):
        cfg.resume_from = args.resume
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    if getattr(args, "lr", None) is not None:
        cfg.lr = args.lr
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "device", None) is not None:
        cfg.device = args.device

    result = train(cfg)
    log.info("training complete, best val loss = %.4f", result["best_val_loss"])


def _resolve_seed(args: argparse.Namespace):
    from .midi_io import PRESET_SEEDS, parse_note_string, read_midi

    if args.seed_midi:
        return read_midi(args.seed_midi)
    if args.seed_text:
        return parse_note_string(args.seed_text)
    if args.seed_preset:
        if args.seed_preset not in PRESET_SEEDS:
            available = ", ".join(PRESET_SEEDS)
            raise SystemExit(f"unknown preset {args.seed_preset!r}. available: {available}")
        return parse_note_string(PRESET_SEEDS[args.seed_preset])
    raise SystemExit("one of --seed-midi / --seed-text / --seed-preset is required")


def _cmd_generate(args: argparse.Namespace) -> None:
    from .generate import generate_continuations, generate_variations
    from .midi_io import write_midi
    from .train import load_checkpoint
    from .viz import save_piano_roll

    seed = _resolve_seed(args)
    model, tokenizer = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device.type

    fn = generate_continuations if args.mode == "continuation" else generate_variations
    candidates = fn(
        model, tokenizer, seed,
        n_bars=args.bars, k=args.k, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repetition_penalty,
        device=device,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("generated %d candidate(s)", len(candidates))

    for i, cand in enumerate(candidates):
        stem = out_dir / f"{args.mode}_{i + 1}"
        write_midi(cand.melody, stem.with_suffix(".mid"), tempo_bpm=args.tempo)
        save_piano_roll(
            cand.melody, str(stem.with_suffix(".png")),
            seed_len=cand.seed_len_steps,
            title=f"{args.mode} #{i + 1}" + (f" (motif={cand.motif_score:.2f})" if cand.motif_score else ""),
        )
        log.info("wrote %s.mid / .png", stem)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from .corpus import load_corpus
    from .dataset import split_melodies
    from .evaluate import (
        evaluate_generations,
        markov_baseline_perplexity,
        pitch_markov_baseline_perplexity,
        perplexity_and_accuracy,
    )
    from .train import load_checkpoint

    model, tokenizer = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device.type

    melodies = load_corpus(args.cache)
    train_melodies, val_melodies = split_melodies(melodies, seed=1337)

    metrics = perplexity_and_accuracy(model, tokenizer, val_melodies, device)
    log.info(
        "Transformer  val_loss=%.4f  val_ppl=%.2f  next_token_acc=%.3f  pitch_acc=%.3f",
        metrics["loss"], metrics["perplexity"], metrics["next_token_accuracy"], metrics["pitch_accuracy"],
    )

    baseline_ppl = markov_baseline_perplexity(tokenizer, train_melodies, val_melodies, order=args.markov_order)
    log.info("order-%d Token Markov baseline   val_ppl=%.2f", args.markov_order, baseline_ppl)

    pitch_baseline_ppl = pitch_markov_baseline_perplexity(train_melodies, val_melodies, order=2)
    log.info("order-2 Pitch Markov baseline   val_ppl=%.2f", pitch_baseline_ppl)

    if metrics["perplexity"] < baseline_ppl:
        log.info("Transformer beats token Markov baseline (%.2f < %.2f)", metrics["perplexity"], baseline_ppl)
    else:
        log.warning("Transformer did NOT beat token Markov baseline (%.2f >= %.2f) -- consider more training",
                    metrics["perplexity"], baseline_ppl)

    if args.sample_generations:
        from .generate import generate_continuations
        sample_seeds = val_melodies[:8]
        generated = []
        for s in sample_seeds:
            cands = generate_continuations(model, tokenizer, s, n_bars=2, k=1, device=device)
            generated.extend(c.melody for c in cands)
        if generated:
            gen_metrics = evaluate_generations(generated, train_melodies)
            log.info(
                "Sample generation metrics: in_scale=%.1f%%  repetition=%.1f%%  distinctness=%.2f  JS_div=%.4f  rhythm_entropy=%.2f",
                gen_metrics.in_scale_ratio * 100,
                gen_metrics.repetition_rate * 100,
                gen_metrics.pairwise_distinctness,
                gen_metrics.pitch_class_js_divergence,
                gen_metrics.rhythm_entropy,
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="melodymorph", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("prepare-data", help="extract and cache the training corpus")
    pd.add_argument("--cache", default="data/melodies.jsonl")
    pd.add_argument("--limit", type=int, default=None)
    pd.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    pd.set_defaults(func=_cmd_prepare_data)

    tr = sub.add_parser("train", help="train the Transformer")
    tr.add_argument("--config", required=True)
    tr.add_argument("--resume", help="path to checkpoint to resume training from")
    tr.add_argument("--epochs", type=int, help="override number of training epochs")
    tr.add_argument("--lr", type=float, help="override learning rate")
    tr.add_argument("--batch-size", type=int, help="override batch size")
    tr.add_argument("--device", help="override compute device ('cuda', 'cpu', 'mps')")
    tr.set_defaults(func=_cmd_train)

    gen = sub.add_parser("generate", help="generate continuations/variations from a seed")
    gen.add_argument("--checkpoint", default="checkpoints/best.pt")
    seed_group = gen.add_mutually_exclusive_group(required=False)
    seed_group.add_argument("--seed-midi")
    seed_group.add_argument("--seed-text", help='e.g. "C4/q E4/q G4/h A4/e G4/e"')
    seed_group.add_argument("--seed-preset")
    gen.add_argument("--mode", choices=["continuation", "variation"], default="continuation")
    gen.add_argument("-k", type=int, default=4, help="number of candidates")
    gen.add_argument("--bars", type=int, default=4, help="bars to generate")
    gen.add_argument("--temperature", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--repetition-penalty", type=float, default=1.15, help="penalty against token repetition")
    gen.add_argument("--tempo", type=int, default=100)
    gen.add_argument("--out", default="out")
    gen.set_defaults(func=_cmd_generate)

    ev = sub.add_parser("evaluate", help="perplexity + baseline comparison on held-out data")
    ev.add_argument("--checkpoint", default="checkpoints/best.pt")
    ev.add_argument("--cache", default="data/melodies.jsonl")
    ev.add_argument("--markov-order", type=int, default=3)
    ev.add_argument("--sample-generations", action="store_true", help="evaluate musical quality metrics on generated samples")
    ev.set_defaults(func=_cmd_evaluate)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main(sys.argv[1:])
