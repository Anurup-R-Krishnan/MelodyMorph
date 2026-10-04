"""Score checkpoints side by side on the current corpus' validation split.

    python -m scripts.compare checkpoints/best.pt checkpoints/new.pt

Caveat for checkpoints trained on an older corpus: their training split differs,
so some of these validation tunes may have been in their training data. That
biases the comparison *in favour of* the old checkpoint.
"""

from __future__ import annotations

import sys

import torch

from melodymorph.corpus import load_records, split_records
from melodymorph.evaluate import (
    evaluate_continuations,
    evaluate_variations,
    family_nll,
    kneser_ney_baseline,
    make_seeds,
)
from melodymorph.tokenizer import MelodyTokenizer
from melodymorph.train import load_checkpoint
from scripts._io import write_json

CORPUS = "data/melodies.jsonl"
REPORT = "reports/compare.json"


def main(argv: list[str] | None = None) -> None:
    ckpts = sys.argv[1:] if argv is None else argv
    if not ckpts:
        raise SystemExit("usage: python -m scripts.compare <checkpoint> [<checkpoint> ...]")

    train, val = split_records(load_records(CORPUS), 0.1)
    # Pre-TS-token checkpoints only read duple tunes, so every model is scored on the
    # duple validation tunes (and the TS models additionally on the triple ones).
    val_d = [r for r in val if r.bar == 16]
    train_m = [r.notes for r in train]
    val_m, val_b = [r.notes for r in val_d], [r.bar for r in val_d]
    val_t = [r.notes for r in val if r.bar == 12]
    seeds, seed_bars = make_seeds(val_m, limit=48, bars=val_b)

    report = {"kneser_ney": kneser_ney_baseline(
        MelodyTokenizer(), train_m, val_m, (6, 9),
        train_bars=[r.bar for r in train], val_bars=val_b)}

    for ck in ckpts:
        model, tok = load_checkpoint(ck)
        torch.manual_seed(0)
        report[ck] = {
            "likelihood": family_nll(model, tok, val_m, bars=val_b),
            "continuation": evaluate_continuations(
                model, tok, seeds, train_m, seed_bars=seed_bars).to_dict(),
            "variation": evaluate_variations(
                model, tok, seeds, train_m, seed_bars=seed_bars).to_dict(),
        }
        if tok.meter_tokens and val_t:
            report[ck]["likelihood_triple"] = family_nll(model, tok, val_t, bars=[12] * len(val_t))

    rows = [(f"KN-{n}", r) for n, r in report["kneser_ney"].items()]
    rows += [(ck, report[ck]["likelihood"]) for ck in ckpts]
    rows += [(f"{ck} (triple tunes)", report[ck]["likelihood_triple"])
             for ck in ckpts if "likelihood_triple" in report[ck]]

    print(f"duple validation tunes: {len(val_m)}, triple: {len(val_t)}")
    print(f"{'model':28s}{'pitch_ppl':>11s}{'note_nll':>10s}{'dur_nll':>9s}{'bits/note':>11s}")
    for name, r in rows:
        print(f"{name:28s}{r['pitch_ppl']:>11.3f}{r['note_nll']:>10.4f}{r['dur_nll']:>9.4f}"
              f"{r['all_bits_per_note']:>11.3f}")

    for mode in ("continuation", "variation"):
        print(f"\n{mode}")
        keys = [k for k, v in report[ckpts[0]][mode].items() if isinstance(v, float) and v == v]
        print(f"{'':28s}" + "".join(f"{k[:12]:>13s}" for k in keys))
        for ck in ckpts:
            print(f"{ck:28s}" + "".join(f"{report[ck][mode][k]:>13.3f}" for k in keys))

    write_json(REPORT, report)
    print("\nwrote", REPORT)


if __name__ == "__main__":
    main()
