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

log = logging.getLogger("melodymorph")


def _cmd_prepare_data(args: argparse.Namespace) -> None:
    from .corpus import build_corpus, corpus_stats

    records = build_corpus(cache=args.cache, limit=args.limit, force=args.force,
                           align_meter=not args.legacy_extraction)
    log.info("corpus ready: %s", corpus_stats([r.notes for r in records]))


def _parse_override(cfg, item: str) -> None:
    import yaml

    key, _, raw = item.partition("=")
    if not hasattr(cfg, key):
        raise SystemExit(f"unknown config key {key!r}")
    setattr(cfg, key, yaml.safe_load(raw))


def _cmd_train(args: argparse.Namespace) -> None:
    from .train import TrainConfig, train

    cfg = TrainConfig.from_yaml(args.config)
    for flag in ("epochs", "lr", "batch_size", "device"):
        if getattr(args, flag) is not None:
            setattr(cfg, flag, getattr(args, flag))
    if args.resume:
        cfg.resume_from = args.resume
    if args.amp:
        cfg.use_amp = True
    for item in args.set or []:
        _parse_override(cfg, item)

    result = train(cfg)
    log.info("training complete, best val loss = %.4f", result["best_val_loss"])


def _resolve_seed(args: argparse.Namespace):
    """Returns ``(seed, bar_length)``. A MIDI seed's metre comes from the file."""
    from .corpus import bar_steps
    from .midi_io import PRESET_SEEDS, midi_bar, midi_time_signature, parse_note_string, read_midi

    if args.seed_midi:
        return read_midi(args.seed_midi), midi_bar(midi_time_signature(args.seed_midi)) or 16
    bar = bar_steps(args.meter)
    if bar is None:
        raise SystemExit(f"unsupported --meter {args.meter!r}")
    if args.seed_text:
        return parse_note_string(args.seed_text), bar
    if args.seed_preset:
        if args.seed_preset not in PRESET_SEEDS:
            available = ", ".join(PRESET_SEEDS)
            raise SystemExit(f"unknown preset {args.seed_preset!r}. available: {available}")
        return parse_note_string(PRESET_SEEDS[args.seed_preset]), bar
    raise SystemExit("one of --seed-midi / --seed-text / --seed-preset is required")


def _cmd_generate(args: argparse.Namespace) -> None:
    from .generate import generate_continuations, generate_variations
    from .midi_io import write_midi
    from .train import load_checkpoint
    from .viz import save_piano_roll

    seed, bar = _resolve_seed(args)
    model, tokenizer = load_checkpoint(args.checkpoint)
    common = dict(k=args.k, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                  key_normalize=not args.no_key_normalize, bar=bar)
    if args.mode == "continuation":
        candidates = generate_continuations(model, tokenizer, seed, n_bars=args.bars,
                                            repetition_penalty=args.repetition_penalty, **common)
    else:
        candidates = generate_variations(model, tokenizer, seed, **common)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("generated %d candidate(s)", len(candidates))
    for i, cand in enumerate(candidates):
        stem = out_dir / f"{args.mode}_{i + 1}"
        write_midi(cand.melody, stem.with_suffix(".mid"), tempo_bpm=args.tempo, bar=bar)
        score = f" (motif={cand.motif_score:.2f})" if cand.motif_score is not None else ""
        save_piano_roll(cand.melody, str(stem.with_suffix(".png")),
                        seed_len=cand.seed_len_steps, title=f"{args.mode} #{i + 1}{score}",
                        bar=bar)
        log.info("wrote %s.mid / .png", stem)


def _cmd_evaluate(args: argparse.Namespace) -> None:
    import json

    import torch

    from .corpus import load_records, split_records
    from .evaluate import (
        evaluate_continuations,
        evaluate_variations,
        family_nll,
        kneser_ney_baseline,
        make_seeds,
    )
    from .train import load_checkpoint, load_checkpoint_payload

    train_cfg = load_checkpoint_payload(args.checkpoint).get("train_config", {})
    cache = args.cache or train_cfg.get("data_cache", "data/melodies.jsonl")
    model, tokenizer = load_checkpoint(args.checkpoint)
    train_recs, val_recs = split_records(load_records(cache), train_cfg.get("val_fraction", 0.1))
    if not tokenizer.meter_tokens:  # pre-TS-token checkpoint: duple tunes only
        train_recs = [r for r in train_recs if r.bar == 16]
        val_recs = [r for r in val_recs if r.bar == 16]
    train_m, val_m = [r.notes for r in train_recs], [r.notes for r in val_recs]
    train_b, val_b = [r.bar for r in train_recs], [r.bar for r in val_recs]
    report: dict = {"checkpoint": args.checkpoint, "data": cache,
                    "n_train": len(train_m), "n_val": len(val_m)}

    report["transformer"] = family_nll(model, tokenizer, val_m, bars=val_b)
    t = report["transformer"]
    log.info("Transformer  pitch_ppl=%.3f  note_nll=%.4f  token_ppl=%.3f  bits/note=%.3f",
             t["pitch_ppl"], t["note_nll"], t["token_ppl"], t["all_bits_per_note"])
    if args.kn_orders:
        report["kneser_ney"] = kneser_ney_baseline(tokenizer, train_m, val_m, tuple(args.kn_orders),
                                                   train_bars=train_b, val_bars=val_b)
        for order, r in report["kneser_ney"].items():
            log.info("KN order-%d   pitch_ppl=%.3f  note_nll=%.4f  token_ppl=%.3f",
                     order, r["pitch_ppl"], r["note_nll"], r["token_ppl"])

    if args.generation:
        torch.manual_seed(0)
        seeds, seed_bars = make_seeds(val_m, limit=args.n_seeds, bars=val_b)
        report["continuation"] = evaluate_continuations(
            model, tokenizer, seeds, train_m, seed_bars=seed_bars).to_dict()
        report["variation"] = evaluate_variations(
            model, tokenizer, seeds, train_m, seed_bars=seed_bars).to_dict()
        for mode in ("continuation", "variation"):
            log.info("%s: %s", mode, {k: round(v, 3) if isinstance(v, float) else v
                                      for k, v in report[mode].items()})
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))
        log.info("wrote %s", args.report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="melodymorph", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("prepare-data", help="extract and cache the training corpus")
    pd.add_argument("--cache", default="data/melodies.jsonl")
    pd.add_argument("--limit", type=int, default=None)
    pd.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    pd.add_argument("--legacy-extraction", action="store_true",
                    help="old extraction (all metres, pickups shifted to step 0); for ablations")
    pd.set_defaults(func=_cmd_prepare_data)

    tr = sub.add_parser("train", help="train the Transformer")
    tr.add_argument("--config", required=True)
    tr.add_argument("--resume", help="checkpoint to resume from (use <checkpoint>.last.pt)")
    tr.add_argument("--epochs", type=int, help="override number of training epochs")
    tr.add_argument("--lr", type=float, help="override learning rate")
    tr.add_argument("--batch-size", type=int, help="override batch size")
    tr.add_argument("--device", help="override compute device ('cuda', 'cpu', 'mps')")
    tr.add_argument("--amp", action="store_true", help="fp16 autocast (GPUs with tensor cores)")
    tr.add_argument("--set", action="append", metavar="KEY=VALUE",
                    help="override any config key, e.g. --set data_cache=data/x.jsonl")
    tr.set_defaults(func=_cmd_train)

    gen = sub.add_parser("generate", help="generate continuations/variations from a seed")
    gen.add_argument("--checkpoint", default="checkpoints/best.pt")
    seed_group = gen.add_mutually_exclusive_group(required=False)
    seed_group.add_argument("--seed-midi")
    seed_group.add_argument("--seed-text", help='e.g. "C4/q E4/q G4/h A4/e G4/e"')
    seed_group.add_argument("--seed-preset")
    gen.add_argument("--meter", default="4/4",
                     help="time signature of a text/preset seed, e.g. 3/4 or 6/8 (MIDI: from the file)")
    gen.add_argument("--mode", choices=["continuation", "variation"], default="continuation")
    gen.add_argument("-k", type=int, default=4, help="number of candidates")
    gen.add_argument("--bars", type=int, default=4, help="bars to generate (continuation)")
    gen.add_argument("--temperature", type=float, default=0.95)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--repetition-penalty", type=float, default=1.0,
                     help="penalty on recently used pitches (continuation)")
    gen.add_argument("--no-key-normalize", action="store_true",
                     help="don't transpose the seed into the training key")
    gen.add_argument("--tempo", type=int, default=100)
    gen.add_argument("--out", default="out")
    gen.set_defaults(func=_cmd_generate)

    ev = sub.add_parser("evaluate", help="held-out likelihood, n-gram baseline, generation metrics")
    ev.add_argument("--checkpoint", default="checkpoints/best.pt")
    ev.add_argument("--cache", help="corpus (default: the one the checkpoint was trained on)")
    ev.add_argument("--kn-orders", type=int, nargs="*", default=[3, 6, 9],
                    help="Kneser-Ney baseline orders (none to skip)")
    ev.add_argument("--generation", action="store_true", help="also score sampled generations")
    ev.add_argument("--n-seeds", type=int, default=48)
    ev.add_argument("--report", help="write the full report as JSON")
    ev.set_defaults(func=_cmd_evaluate)

    return p


def main(argv: list[str] | None = None) -> None:
    # configured here, not at import time, so importing the package never
    # hijacks the host application's logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main(sys.argv[1:])
