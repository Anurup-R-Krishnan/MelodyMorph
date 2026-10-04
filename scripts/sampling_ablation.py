"""Decide the sampling defaults with numbers, on held-out seeds.

1. repetition penalty (pitch-only) 1.0 / 1.1 / 1.2 vs the old all-token 1.15
2. key normalisation on/off, with the seeds moved to random keys
3. variation mode: old (continue from half the seed) vs new (rhythm-locked re-voicing)

    python -m scripts.sampling_ablation [<checkpoint>] [<out.json>]
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable

import torch

from melodymorph.corpus import load_records, split_records
from melodymorph.evaluate import (
    evaluate_continuations,
    evaluate_variations,
    in_scale_ratio,
    jensen_shannon,
    make_seeds,
    pairwise_distinctness,
    pitch_class_histogram,
    random_key_seeds,
    repetition_rate,
    rhythm_entropy,
)
from melodymorph.keys import estimate_key
from melodymorph.sample import motif_similarity
from melodymorph.tokenizer import melody_duration
from melodymorph.train import load_checkpoint, load_checkpoint_payload
from scripts._io import write_json
from scripts.legacy import old_generate

DEFAULT_CKPT = "checkpoints/best.pt"
DEFAULT_OUT = "logs/sampling_ablation.json"


def _mean(xs: Iterable) -> float:
    """Mean that reports NaN for an empty run instead of dividing by zero."""
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    ckpt = argv[0] if argv else DEFAULT_CKPT
    out_path = argv[1] if len(argv) > 1 else DEFAULT_OUT

    cfg = load_checkpoint_payload(ckpt)["train_config"]
    model, tok = load_checkpoint(ckpt)
    train, val = split_records(load_records(cfg["data_cache"]), cfg["val_fraction"])
    train_m, val_m = [r.notes for r in train], [r.notes for r in val]
    seeds, _ = make_seeds(val_m, limit=64)  # duple seeds only
    report: dict = {"checkpoint": ckpt, "n_seeds": len(seeds)}

    def show(name, r):
        report[name] = r.to_dict() if hasattr(r, "to_dict") else r
        d = report[name]
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}, flush=True)

    # old sampler, all-token penalty, for reference (old continuation code path)
    def old_continuations(seed_list, rp):
        gens, completed, in_key, per_seed = [], [], [], []
        t0 = time.time()
        for s in seed_list:
            tonic, mode = estimate_key(s)
            cands = old_generate.generate_continuations(
                model, tok, s, n_bars=4, k=4, repetition_penalty=rp)
            regions = [[n for n in c.melody if n.onset >= c.seed_len_steps] for c in cands]
            target = melody_duration(s) + 64
            completed += [melody_duration(c.melody) >= target - 8 for c in cands]
            in_key += [in_scale_ratio(r, tonic, mode) for r in regions]
            per_seed.append(pairwise_distinctness(regions))
            gens += regions
        return {"completion_rate": _mean(completed), "in_key": _mean(in_key),
                "distinctness": _mean(per_seed),
                "rhythm_entropy": _mean(map(rhythm_entropy, gens)),
                "repetition_rate": _mean(map(repetition_rate, gens)),
                "pitch_class_js": jensen_shannon(pitch_class_histogram(gens),
                                                 pitch_class_histogram(train_m)),
                "seconds": time.time() - t0}

    def old_variations(seed_list):
        novel, scores, per_seed, in_key, copies = [], [], [], [], []
        t0 = time.time()
        for s in seed_list:
            tonic, mode = estimate_key(s)
            cands = old_generate.generate_variations(model, tok, s, n_bars=4, k=4)
            melodies = [c.melody for c in cands]
            seed_notes = {(n.onset, n.pitch) for n in s}
            for m in melodies:
                head = [n for n in m if n.onset < melody_duration(s)]
                novel.append(1 - sum((n.onset, n.pitch) in seed_notes for n in head)
                             / max(len(head), 1))
                scores.append(motif_similarity(s, head))
                in_key.append(in_scale_ratio(m, tonic, mode))
                copies.append([n.pitch for n in head[: len(s) // 2]]
                              == [n.pitch for n in s[: len(s) // 2]])
            per_seed.append(pairwise_distinctness(melodies))
        return {"novel_note_rate_in_seed_span": _mean(novel),
                "opens_with_verbatim_seed_half": _mean(copies),
                "motif_score": _mean(scores), "distinctness": _mean(per_seed),
                "in_key": _mean(in_key), "seconds": time.time() - t0}

    def new_variation_novelty(seed_list):
        from melodymorph.generate import generate_variations

        novel, copies = [], []
        for s in seed_list:
            seed_notes = {(n.onset, n.pitch) for n in s}
            for c in generate_variations(model, tok, s, k=4):
                novel.append(1 - sum((n.onset, n.pitch) in seed_notes for n in c.melody)
                             / max(len(c.melody), 1))
                copies.append([n.pitch for n in c.melody[: len(s) // 2]]
                              == [n.pitch for n in s[: len(s) // 2]])
        return {"novel_note_rate_in_seed_span": _mean(novel),
                "opens_with_verbatim_seed_half": _mean(copies)}

    # 1. repetition penalty (pitch-only, new sampler)
    for rp in (1.0, 1.1, 1.2):
        torch.manual_seed(0)
        show(f"cont_rp{rp}",
             evaluate_continuations(model, tok, seeds, train_m, repetition_penalty=rp))

    for rp in (1.0, 1.15):
        torch.manual_seed(0)
        show(f"old_cont_rp{rp}", old_continuations(seeds, rp))

    # 2. key normalisation, seeds in random keys
    rk = random_key_seeds(seeds)
    for norm in (False, True):
        torch.manual_seed(0)
        show(f"randkey_norm_{norm}",
             evaluate_continuations(model, tok, rk, train_m, key_normalize=norm))

    # 3. variation: old vs new
    torch.manual_seed(0)
    show("old_variation", old_variations(seeds))
    torch.manual_seed(0)
    show("new_variation", evaluate_variations(model, tok, seeds, train_m))
    torch.manual_seed(0)
    show("new_variation_novelty", new_variation_novelty(seeds))
    for temp in (0.8, 1.2):
        torch.manual_seed(0)
        show(f"new_variation_t{temp}",
             evaluate_variations(model, tok, seeds, train_m, temperature=temp))

    write_json(out_path, report)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
