"""User-facing generation: continuation and variation modes.

Both modes wrap the low-level :func:`melodymorph.sample.generate` sampler and add
what the proposal calls for: several distinct candidates per request, and (for
variation) a motif-preservation filter so results stay recognisably related to
the seed instead of wandering off or degenerating into a copy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import MelodyTransformer
from .sample import generate, motif_similarity, region_within
from .tokenizer import STEPS_PER_BAR, Melody, MelodyTokenizer, melody_duration


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
    device: str | None = None,
    max_attempts: int | None = None,
) -> list[Candidate]:
    """Extend ``seed`` forward by ``n_bars`` bars, ``k`` distinct times."""
    device = device or str(next(model.parameters()).device)
    seed_len = melody_duration(seed)
    prompt = tokenizer.encode(seed, add_special=True)[:-1]  # drop EOS, keep BOS
    target_steps = seed_len + n_bars * STEPS_PER_BAR
    max_new = int((target_steps - seed_len) * 1.5) + 24  # generous slack for BAR/markers

    attempts = max_attempts or k * 3
    candidates: list[Candidate] = []
    for _ in range(attempts):
        if len(candidates) >= k:
            break
        ids = generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, device=device,
        )
        melody = tokenizer.decode(ids)
        melody = [n for n in melody if n.onset < target_steps]
        if melody_duration(melody) <= seed_len:
            continue  # model produced nothing new; skip
        candidates.append(Candidate(melody=melody, seed_len_steps=seed_len))

    return _dedupe(candidates)[:k] or candidates[:k]


def generate_variations(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    seed: Melody,
    n_bars: int = 4,
    k: int = 4,
    temperature: float = 1.15,
    top_k: int = 0,
    top_p: float = 0.97,
    device: str | None = None,
    similarity_band: tuple[float, float] = (0.35, 0.9),
    max_attempts: int = 24,
) -> list[Candidate]:
    """Regenerate the seed's own first bar (at higher temperature) then continue,
    keeping only candidates whose opening bar's interval contour is close-but-not-
    identical to the seed's -- the motif should be recognisable, not copy-pasted.
    """
    device = device or str(next(model.parameters()).device)
    seed_len = melody_duration(seed)
    first_bar_seed = region_within(seed, 0, STEPS_PER_BAR)
    # prompt the model with only the very first note, so it must regenerate the
    # rest of the motif itself rather than parroting the given seed back
    anchor = tokenizer.encode(seed[:1], add_special=True)[:-1] if seed else [tokenizer.bos_id]

    target_steps = seed_len + n_bars * STEPS_PER_BAR
    max_new = int(target_steps * 1.5) + 24
    lo, hi = similarity_band

    scored: list[Candidate] = []
    for _ in range(max_attempts):
        if len(scored) >= k:
            break
        ids = generate(
            model, tokenizer, anchor,
            max_new_tokens=max_new, temperature=temperature,
            top_k=top_k, top_p=top_p, device=device,
        )
        melody = tokenizer.decode(ids)
        melody = [n for n in melody if n.onset < target_steps]
        if melody_duration(melody) < STEPS_PER_BAR:
            continue

        first_bar_gen = region_within(melody, 0, STEPS_PER_BAR)
        score = motif_similarity(first_bar_seed, first_bar_gen)
        if lo <= score <= hi:
            scored.append(Candidate(melody=melody, seed_len_steps=seed_len, motif_score=score))

    if not scored:
        # relax the band rather than returning nothing
        scored = generate_variations(
            model, tokenizer, seed, n_bars=n_bars, k=k,
            temperature=temperature, top_k=top_k, top_p=top_p, device=device,
            similarity_band=(0.0, 1.0), max_attempts=max_attempts,
        ) if similarity_band != (0.0, 1.0) else scored

    return _dedupe(scored)[:k] or scored[:k]
