"""Quantitative evaluation: perplexity, musical-quality metrics, and an n-gram
Markov baseline so the Transformer's numbers have something concrete to beat.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .dataset import MelodyWindowDataset, build_token_stream, split_melodies
from .model import MelodyTransformer
from .tokenizer import MelodyTokenizer, Melody


# --------------------------------------------------------------------------
# perplexity / accuracy on held-out data
# --------------------------------------------------------------------------
@torch.no_grad()
def perplexity_and_accuracy(
    model: MelodyTransformer, tokenizer: MelodyTokenizer,
    melodies: list[Melody], device: str | None = None, block_size: int | None = None,
) -> dict:
    device = device or str(next(model.parameters()).device)
    block_size = block_size or model.cfg.block_size
    stream = build_token_stream(melodies, tokenizer, transpose_range=0)
    ds = MelodyWindowDataset(stream, block_size, stride=block_size)

    total_loss, total_correct, total_tokens, n_batches = 0.0, 0, 0, 0
    model.eval()
    for i in range(len(ds)):
        xb, yb = ds[i]
        xb, yb = xb.unsqueeze(0).to(device), yb.unsqueeze(0).to(device)
        logits, loss = model(xb, yb)
        total_loss += loss.item()
        n_batches += 1

        mask = yb != tokenizer.pad_id
        preds = logits.argmax(-1)
        total_correct += int(((preds == yb) & mask).sum().item())
        total_tokens += int(mask.sum().item())

    avg_loss = total_loss / max(n_batches, 1)
    return {
        "loss": avg_loss,
        "perplexity": math.exp(min(avg_loss, 20)),
        "next_token_accuracy": total_correct / max(total_tokens, 1),
    }


# --------------------------------------------------------------------------
# order-N Markov baseline over the same token vocabulary
# --------------------------------------------------------------------------
class MarkovBaseline:
    """A simple order-N token n-gram model, evaluated the same way as the Transformer
    so the two perplexities are directly comparable."""

    def __init__(self, order: int = 3):
        self.order = order
        self.counts: dict[tuple, Counter] = defaultdict(Counter)

    def fit(self, stream: list[int]) -> None:
        for i in range(len(stream) - self.order):
            ctx = tuple(stream[i : i + self.order])
            nxt = stream[i + self.order]
            self.counts[ctx][nxt] += 1

    def perplexity(self, stream: list[int], vocab_size: int) -> float:
        log_prob_sum, n = 0.0, 0
        for i in range(len(stream) - self.order):
            ctx = tuple(stream[i : i + self.order])
            nxt = stream[i + self.order]
            counter = self.counts.get(ctx)
            total = sum(counter.values()) if counter else 0
            # add-one smoothing keeps this well-defined for unseen contexts
            prob = ((counter.get(nxt, 0) if counter else 0) + 1) / (total + vocab_size)
            log_prob_sum += -math.log(prob)
            n += 1
        return math.exp(log_prob_sum / max(n, 1))


def markov_baseline_perplexity(
    tokenizer: MelodyTokenizer, train_melodies: list[Melody], val_melodies: list[Melody],
    order: int = 3,
) -> float:
    train_stream = build_token_stream(train_melodies, tokenizer)
    val_stream = build_token_stream(val_melodies, tokenizer)
    baseline = MarkovBaseline(order)
    baseline.fit(train_stream)
    return baseline.perplexity(val_stream, tokenizer.vocab_size)


# --------------------------------------------------------------------------
# musical-quality metrics over a set of generated candidates
# --------------------------------------------------------------------------
_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}  # relative to an arbitrary tonic


def in_scale_ratio(melody: Melody, tonic_pc: int = 0) -> float:
    if not melody:
        return 0.0
    hits = sum(1 for n in melody if (n.pitch - tonic_pc) % 12 in _MAJOR_SCALE)
    return hits / len(melody)


def pitch_class_histogram(melody: Melody) -> list[float]:
    counts = [0] * 12
    for n in melody:
        counts[n.pitch % 12] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


def jensen_shannon(p: list[float], q: list[float]) -> float:
    def kl(a, b):
        return sum(ai * math.log(ai / bi) for ai, bi in zip(a, b) if ai > 0 and bi > 0)

    m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
    m = [x if x > 0 else 1e-12 for x in m]
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def repetition_rate(melody: Melody, n: int = 4) -> float:
    """Fraction of pitch n-grams that repeat elsewhere in the melody."""
    pitches = [note.pitch for note in melody]
    if len(pitches) < n + 1:
        return 0.0
    grams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(grams)


def pairwise_distinctness(melodies: list[Melody]) -> float:
    """Mean fraction of differing pitches between all candidate pairs -- how
    distinct the k generated options actually are from each other."""
    if len(melodies) < 2:
        return 1.0
    scores = []
    for i in range(len(melodies)):
        for j in range(i + 1, len(melodies)):
            a = [n.pitch for n in melodies[i]]
            b = [n.pitch for n in melodies[j]]
            length = max(len(a), len(b), 1)
            overlap = sum(1 for x, y in zip(a, b) if x == y)
            scores.append(1 - overlap / length)
    return sum(scores) / len(scores)


def corpus_pitch_class_histogram(melodies: list[Melody]) -> list[float]:
    counts = [0] * 12
    for melody in melodies:
        for n in melody:
            counts[n.pitch % 12] += 1
    total = sum(counts) or 1
    return [c / total for c in counts]


@dataclass
class GenerationMetrics:
    in_scale_ratio: float
    pitch_class_js_divergence: float
    repetition_rate: float
    pairwise_distinctness: float
    mean_notes: float


def evaluate_generations(
    candidates: list[Melody], reference_corpus: list[Melody], tonic_pc: int = 0,
) -> GenerationMetrics:
    ref_hist = corpus_pitch_class_histogram(reference_corpus)
    gen_hist = corpus_pitch_class_histogram(candidates)

    return GenerationMetrics(
        in_scale_ratio=sum(in_scale_ratio(m, tonic_pc) for m in candidates) / max(len(candidates), 1),
        pitch_class_js_divergence=jensen_shannon(gen_hist, ref_hist),
        repetition_rate=sum(repetition_rate(m) for m in candidates) / max(len(candidates), 1),
        pairwise_distinctness=pairwise_distinctness(candidates),
        mean_notes=sum(len(m) for m in candidates) / max(len(candidates), 1),
    )


# --------------------------------------------------------------------------
# human evaluation logging
# --------------------------------------------------------------------------
def log_human_rating(
    path: str | Path, candidate_id: str, musicality: int, motif_preservation: int,
    notes: str = "",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["candidate_id", "musicality_1_5", "motif_preservation_1_5", "notes"])
        writer.writerow([candidate_id, musicality, motif_preservation, notes])
