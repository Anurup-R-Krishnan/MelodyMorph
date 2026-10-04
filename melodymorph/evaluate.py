"""Evaluation that measures what it claims to.

* **Likelihood per token family.** Overall perplexity is dominated by POS tokens,
  which the grammar makes almost free. The headline is *pitch* perplexity; the
  per-note NLL (pitch + duration) is comparable across tokenizations of the same
  notes, which is what the metre-alignment ablation needs.
* **Kneser-Ney n-gram baseline** scored the same way, at several orders (an
  order-3 token model only sees one note of context).
* Both models score each validation tune on its own, from its BOS.
* **Generation metrics on the generated notes only**, from short seeds, with
  several candidates per seed so diversity means diversity.
"""

from __future__ import annotations

import csv
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import build_token_stream
from .keys import estimate_key
from .model import MelodyTransformer
from .sample import levenshtein
from .tokenizer import Melody, MelodyTokenizer, melody_duration, transpose

FAMILIES = ("pitch", "dur", "pos", "bar", "eos")


def _families(tokenizer: MelodyTokenizer) -> dict[str, set[int]]:
    return {
        "pitch": set(tokenizer.pitch_ids),
        "dur": set(tokenizer.dur_ids),
        "pos": set(tokenizer.pos_ids),
        "bar": {tokenizer.bar_id},
        "eos": {tokenizer.eos_id},
    }


def _summarise(nll_by_family: dict[str, list[float]]) -> dict:
    out = {}
    for fam in FAMILIES:
        vals = nll_by_family.get(fam, [])
        out[f"{fam}_nll"] = sum(vals) / max(len(vals), 1)
    n_notes = len(nll_by_family.get("pitch", []))
    total = sum(sum(v) for v in nll_by_family.values())
    n_tokens = sum(len(v) for v in nll_by_family.values())
    out["token_nll"] = total / max(n_tokens, 1)
    out["token_ppl"] = math.exp(out["token_nll"])
    out["pitch_ppl"] = math.exp(out["pitch_nll"])
    pitch_dur = sum(nll_by_family.get("pitch", [])) + sum(nll_by_family.get("dur", []))
    out["note_nll"] = pitch_dur / max(n_notes, 1)
    out["all_bits_per_note"] = total / max(n_notes, 1) / math.log(2)
    out["n_notes"] = n_notes
    return out


# --------------------------------------------------------------------------
# Transformer likelihood
# --------------------------------------------------------------------------
@torch.no_grad()
def family_nll(
    model: MelodyTransformer, tokenizer: MelodyTokenizer, melodies: list[Melody],
    batch_size: int = 32, bars: list[int] | None = None,
) -> dict:
    """NLL per token family over held-out melodies.

    Each melody is scored on its own, in the same windows the model trains on:
    starting at its BOS, every target scored exactly once (see
    :func:`melodymorph.dataset.melody_windows`). No token is ever conditioned on
    a different tune.
    """
    from .dataset import MelodyWindowDataset

    model.eval()
    device = next(model.parameters()).device
    ds = MelodyWindowDataset(melodies, tokenizer, model.cfg.block_size, bars=bars)
    fam_of = {t: f for f, ids in _families(tokenizer).items() for t in ids}
    nll_by_family: dict[str, list[float]] = defaultdict(list)
    for i in range(0, len(ds), batch_size):
        x, y = ds.x[i : i + batch_size].to(device), ds.y[i : i + batch_size].to(device)
        logits, _ = model(x)
        nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
        for t, v in zip(y.flatten().tolist(), nll.flatten().tolist()):
            fam = fam_of.get(t)  # PAD (unscored) targets have no family
            if fam is not None:
                nll_by_family[fam].append(v)
    return _summarise(nll_by_family)


# --------------------------------------------------------------------------
# interpolated Kneser-Ney n-gram baseline over the same token vocabulary
# --------------------------------------------------------------------------
class KneserNey:
    def __init__(self, order: int = 6, discount: float = 0.75):
        self.order = order
        self.d = discount

    def fit(self, stream: list[int], vocab_size: int) -> KneserNey:
        self.V = vocab_size
        n = self.order
        # counts[k][context] -> Counter(next); k = context length.
        # Highest order uses raw counts, lower orders continuation counts.
        self.counts: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]
        for i in range(len(stream)):
            w = stream[i]
            ctx = tuple(stream[max(0, i - n + 1) : i])
            if len(ctx) == n - 1:
                self.counts[n - 1][ctx][w] += 1
        for k in range(n - 1, 0, -1):
            for ctx, nxt in self.counts[k].items():
                for w in nxt:
                    self.counts[k - 1][ctx[1:]][w] += 1  # distinct left extensions
        self.totals = [{c: sum(v.values()) for c, v in level.items()} for level in self.counts]
        return self

    def prob(self, ctx: tuple, w: int) -> float:
        p = 1.0 / self.V
        for k in range(0, self.order):
            if k > len(ctx):
                break
            c = ctx[len(ctx) - k :] if k else ()
            nxt = self.counts[k].get(c)
            if not nxt:
                continue
            total = self.totals[k][c]
            p = max(nxt.get(w, 0) - self.d, 0) / total + self.d * len(nxt) / total * p
        return p

    def family_nll(self, sequences: list[list[int]], tokenizer: MelodyTokenizer) -> dict:
        """Score each sequence on its own (context never crosses into another tune)."""
        fam_of = {t: f for f, ids in _families(tokenizer).items() for t in ids}
        nll_by_family: dict[str, list[float]] = defaultdict(list)
        n = self.order
        for seq in sequences:
            for i in range(1, len(seq)):
                fam = fam_of.get(seq[i])
                if fam is not None:
                    ctx = tuple(seq[max(0, i - n + 1) : i])
                    nll_by_family[fam].append(-math.log(self.prob(ctx, seq[i])))
        return _summarise(nll_by_family)


def kneser_ney_baseline(
    tokenizer: MelodyTokenizer, train_melodies: list[Melody], val_melodies: list[Melody],
    orders: tuple[int, ...] = (3, 6, 9), transpose_range: int = 0,
    train_bars: list[int] | None = None, val_bars: list[int] | None = None,
) -> dict[int, dict]:
    train_stream = build_token_stream(train_melodies, tokenizer, transpose_range, bars=train_bars)
    val_bars = val_bars or [16] * len(val_melodies)
    val_seqs = [tokenizer.encode(m, bar=b) for m, b in zip(val_melodies, val_bars)]
    return {
        n: KneserNey(n).fit(train_stream, tokenizer.vocab_size).family_nll(val_seqs, tokenizer)
        for n in orders
    }


# --------------------------------------------------------------------------
# musical-quality metrics
# --------------------------------------------------------------------------
_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}


def in_scale_ratio(melody: Melody, tonic_pc: int = 0, mode: str = "major") -> float:
    """Fraction of notes in the (natural) scale of the given key."""
    if not melody:
        return 0.0
    major_tonic = tonic_pc if mode == "major" else (tonic_pc + 3) % 12
    return sum((n.pitch - major_tonic) % 12 in _MAJOR_SCALE for n in melody) / len(melody)


def pitch_class_histogram(melodies: list[Melody]) -> list[float]:
    counts = [0] * 12
    for melody in melodies:
        for n in melody:
            counts[n.pitch % 12] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def jensen_shannon(p: list[float], q: list[float]) -> float:
    """Jensen-Shannon divergence in [0, 1] (base-2 logarithm)."""
    def kl(a, b):
        return sum(ai * math.log2(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)

    m = [max((pi + qi) / 2, 1e-12) for pi, qi in zip(p, q)]
    return max(0.0, min(1.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def repetition_rate(melody: Melody, n: int = 4) -> float:
    """Fraction of pitch n-grams that repeat elsewhere in the melody."""
    pitches = [note.pitch for note in melody]
    if len(pitches) < n + 1:
        return 0.0
    grams = Counter(tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1))
    return sum(c for c in grams.values() if c > 1) / sum(grams.values())


def pairwise_distinctness(melodies: list[Melody]) -> float:
    """Mean normalised pitch edit distance between all pairs (1.0 for < 2 melodies)."""
    if len(melodies) < 2:
        return 1.0
    scores = []
    for i in range(len(melodies)):
        for j in range(i + 1, len(melodies)):
            a = [n.pitch for n in melodies[i]]
            b = [n.pitch for n in melodies[j]]
            scores.append(levenshtein(a, b) / max(len(a), len(b), 1))
    return sum(scores) / len(scores)


def rhythm_entropy(melody: Melody) -> float:
    """Shannon entropy (bits) of note durations."""
    if not melody:
        return 0.0
    counts = Counter(n.dur for n in melody)
    total = len(melody)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


@dataclass
class GenerationReport:
    n_seeds: int
    n_candidates: int
    completion_rate: float       # candidates that reached the requested length
    in_key: float                # generated notes in the seed's key
    pitch_class_js: float        # generated vs reference pitch-class distribution
    repetition_rate: float
    distinctness: float          # between candidates of the same seed
    rhythm_entropy: float
    notes_per_bar: float
    motif_score: float | None = None
    in_band_rate: float | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def make_seeds(
    melodies: list[Melody], n_notes: int = 10, limit: int = 48, bars: list[int] | None = None,
) -> tuple[list[Melody], list[int]]:
    """Seeds of ``n_notes`` notes (the task's 5-15 range), cut from melody openings
    and kept bar-phase so a pickup stays a pickup; returned with their bar lengths."""
    bars = bars or [16] * len(melodies)
    pairs = [(m[:n_notes], b) for m, b in zip(melodies, bars) if len(m) >= n_notes + 4][:limit]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def evaluate_continuations(
    model, tokenizer, seeds: list[Melody], reference: list[Melody],
    n_bars: int = 4, k: int = 4, seed_bars: list[int] | None = None, **sampling,
) -> GenerationReport:
    from .generate import generate_continuations

    t0 = time.time()
    gens, per_seed, completed, in_key, bars = [], [], [], [], []
    for seed, bar in zip(seeds, seed_bars or [16] * len(seeds)):
        tonic, mode = estimate_key(seed)
        cands = generate_continuations(model, tokenizer, seed, n_bars=n_bars, k=k, bar=bar, **sampling)
        regions = [[n for n in c.melody if n.onset >= c.seed_len_steps] for c in cands]
        regions = [r for r in regions if r]
        per_seed.append(pairwise_distinctness(regions))
        completed += [c.completed for c in cands]
        for r in regions:
            in_key.append(in_scale_ratio(r, tonic, mode))
            bars.append(len(r) / n_bars)
        gens += regions
    return GenerationReport(
        n_seeds=len(seeds), n_candidates=len(gens),
        completion_rate=_mean(completed), in_key=_mean(in_key),
        pitch_class_js=jensen_shannon(pitch_class_histogram(gens), pitch_class_histogram(reference)),
        repetition_rate=_mean(repetition_rate(g) for g in gens),
        distinctness=_mean(per_seed), rhythm_entropy=_mean(rhythm_entropy(g) for g in gens),
        notes_per_bar=_mean(bars), seconds=time.time() - t0,
    )


def evaluate_variations(
    model, tokenizer, seeds: list[Melody], reference: list[Melody], k: int = 4,
    band: tuple[float, float] = (0.35, 0.85), seed_bars: list[int] | None = None, **sampling,
) -> GenerationReport:
    from .generate import generate_variations

    t0 = time.time()
    gens, per_seed, in_key, scores, bars = [], [], [], [], []
    for seed, bar in zip(seeds, seed_bars or [16] * len(seeds)):
        tonic, mode = estimate_key(seed)
        cands = generate_variations(model, tokenizer, seed, k=k, similarity_band=band, bar=bar,
                                    **sampling)
        per_seed.append(pairwise_distinctness([c.melody for c in cands]))
        n_bars = max(melody_duration(seed) / bar, 1)
        for c in cands:
            gens.append(c.melody)
            in_key.append(in_scale_ratio(c.melody, tonic, mode))
            scores.append(c.motif_score)
            bars.append(len(c.melody) / n_bars)
    lo, hi = band
    return GenerationReport(
        n_seeds=len(seeds), n_candidates=len(gens), completion_rate=1.0,
        in_key=_mean(in_key),
        pitch_class_js=jensen_shannon(pitch_class_histogram(gens), pitch_class_histogram(reference)),
        repetition_rate=_mean(repetition_rate(g) for g in gens),
        distinctness=_mean(per_seed), rhythm_entropy=_mean(rhythm_entropy(g) for g in gens),
        notes_per_bar=_mean(bars), motif_score=_mean(scores),
        in_band_rate=_mean(lo <= s <= hi for s in scores), seconds=time.time() - t0,
    )


def random_key_seeds(seeds: list[Melody], seed: int = 0) -> list[Melody]:
    """The same seeds, each moved to a random key (within the model's range)."""
    import random

    from .tokenizer import in_range

    rng = random.Random(seed)
    out = []
    for s in seeds:
        shifts = [d for d in range(-6, 7) if d and in_range(transpose(s, d))]
        out.append(transpose(s, rng.choice(shifts)) if shifts else s)
    return out


# --------------------------------------------------------------------------
# human evaluation logging
# --------------------------------------------------------------------------
RATING_FIELDS = [
    "timestamp", "checkpoint", "mode", "take", "seed", "melody", "params",
    "musicality_1_5", "motif_preservation_1_5", "notes",
]


def log_human_rating(path: str | Path, row: dict) -> None:
    """Append one rating with enough context (seed, output, sampling params,
    checkpoint) to analyse it later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RATING_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **row})
