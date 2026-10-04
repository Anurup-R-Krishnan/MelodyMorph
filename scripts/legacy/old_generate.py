"""User-facing generation: continuation and variation modes.

Both modes wrap the low-level :func:`melodymorph.sample.generate` sampler and add
what the proposal calls for: several distinct candidates per request, and (for
variation) a motif-preservation filter so results stay recognisably related to
the seed instead of wandering off or degenerating into a copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from melodymorph.model import MelodyTransformer
from scripts.legacy.old_sample import generate, motif_similarity, region_within
from melodymorph.tokenizer import STEPS_PER_BAR, Melody, MelodyTokenizer, melody_duration


@dataclass
class Candidate:
    melody: Melody
    seed_len_steps: int
    motif_score: float | None = None


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Drop candidates that are near-duplicates of an earlier one (same pitch
    sequence in the generated region), so the user gets genuinely distinct
    options rather than k copies of the model's favourite continuation."""
    seen: set[tuple[int, ...]] = set()
    unique: list[Candidate] = []
    for cand in candidates:
        gen_region = region_within(cand.melody, cand.seed_len_steps, 10**9)
        key = tuple(n.pitch for n in gen_region)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def generate_continuations(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    seed: Melody,
    n_bars: int = 4,
    k: int = 4,
    temperature: float = 0.95,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    device: str | None = None,
    max_attempts: int | None = None,
) -> list[Candidate]:
    """Extend ``seed`` forward by ``n_bars`` bars, collecting ``k`` distinct candidates."""
    device = device or str(next(model.parameters()).device)
    seed_len = melody_duration(seed)
    prompt = tokenizer.encode(seed, add_special=True)[:-1]  # drop EOS, keep BOS
    target_steps = seed_len + n_bars * STEPS_PER_BAR
    max_new = int((target_steps - seed_len) * 1.5) + 32  # slack for BAR/markers

    attempts = max_attempts or k * 5
    unique: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(attempts):
        if len(unique) >= k:
            break
        ids = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            device=device,
        )
        melody = tokenizer.decode(ids)
        melody = [n for n in melody if n.onset < target_steps]
        if melody_duration(melody) <= seed_len:
            continue  # model produced nothing new; skip

        gen_region = region_within(melody, seed_len, 10**9)
        key = tuple(n.pitch for n in gen_region)
        if key in seen:
            continue
        seen.add(key)
        unique.append(Candidate(melody=melody, seed_len_steps=seed_len))

    return unique[:k]


def generate_variations(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    seed: Melody,
    n_bars: int = 4,
    k: int = 4,
    temperature: float = 1.05,
    top_k: int = 0,
    top_p: float = 0.96,
    repetition_penalty: float = 1.15,
    device: str | None = None,
    similarity_band: tuple[float, float] = (0.30, 0.85),
    max_attempts: int = 32,
) -> list[Candidate]:
    """Generate variations that explore melodic re-imaginings related to ``seed``.

    Instead of discarding the seed, the model is conditioned on the motif's harmonic
    and melodic anchor (first 2-3 notes or half the motif), and full-length interval
    contour similarity is evaluated against the complete seed so genuine thematic
    relationship is preserved.
    """
    device = device or str(next(model.parameters()).device)
    seed_len = melody_duration(seed)
    # Use the thematic opening anchor (half the seed or up to 3 notes) rather than a single note
    anchor_notes = seed[:max(2, len(seed) // 2)] if len(seed) >= 2 else seed
    anchor = tokenizer.encode(anchor_notes, add_special=True)[:-1] if anchor_notes else [tokenizer.bos_id]

    target_steps = seed_len + n_bars * STEPS_PER_BAR
    max_new = int(target_steps * 1.5) + 32
    lo, hi = similarity_band

    unique: list[Candidate] = []
    seen: set[tuple[int, ...]] = set()

    for _ in range(max_attempts):
        if len(unique) >= k:
            break
        ids = generate(
            model, tokenizer, anchor,
            max_new_tokens=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            device=device,
        )
        melody = tokenizer.decode(ids)
        melody = [n for n in melody if n.onset < target_steps]
        if melody_duration(melody) < max(STEPS_PER_BAR, seed_len // 2):
            continue

        # Evaluate similarity over the entire seed length, not just bar 1
        gen_head = region_within(melody, 0, max(seed_len, STEPS_PER_BAR))
        score = motif_similarity(seed, gen_head)

        key = tuple(n.pitch for n in melody)
        if key in seen:
            continue

        if lo <= score <= hi:
            seen.add(key)
            unique.append(Candidate(melody=melody, seed_len_steps=0, motif_score=score))

    if len(unique) < k:
        # If strict band yielded fewer than k, backfill with closest candidates
        for _ in range(max_attempts // 2):
            if len(unique) >= k:
                break
            ids = generate(
                model, tokenizer, anchor,
                max_new_tokens=max_new, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                device=device,
            )
            melody = tokenizer.decode(ids)
            melody = [n for n in melody if n.onset < target_steps]
            if not melody:
                continue
            key = tuple(n.pitch for n in melody)
            if key not in seen:
                seen.add(key)
                gen_head = region_within(melody, 0, max(seed_len, STEPS_PER_BAR))
    target_score = (lo + hi) / 2.0
    unique.sort(key=lambda c: abs((c.motif_score if c.motif_score is not None else 0.5) - target_score))
    return unique[:k]

