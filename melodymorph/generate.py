"""User-facing generation: continuation and variation modes.

Both modes transpose the seed into the training key (C major / A minor), sample
a batch of candidates in one pass, keep distinct ones, and transpose back.

* **continuation** extends the seed by ``n_bars`` bars.
* **variation** restates the seed with its rhythm held fixed and new pitches:
  the model sees the whole seed as context, then the seed's BAR/POS/DUR tokens
  are teacher-forced while every PITCH is sampled. A contour-similarity band
  keeps results recognisably related without being copies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .keys import shift_to_training_key
from .model import MelodyTransformer
from .sample import generate_batch, motif_similarity, region_within
from .tokenizer import STEPS_PER_BAR, Melody, MelodyTokenizer, Note, melody_duration, transpose

log = logging.getLogger(__name__)

# The task is 5-15 note motifs; anything far longer (a whole uploaded MIDI file)
# would overflow the context and make "variation" a re-voicing of a whole piece.
MAX_SEED_NOTES = 32


@dataclass
class Candidate:
    melody: Melody
    seed_len_steps: int
    motif_score: float | None = None
    completed: bool = True
    bar: int = STEPS_PER_BAR


def _bar_ceil(steps: int, bar: int = STEPS_PER_BAR) -> int:
    return -(-steps // bar) * bar


def trim_seed(
    seed: Melody, max_notes: int = MAX_SEED_NOTES, keep: str = "last", bar: int = STEPS_PER_BAR,
) -> Melody:
    """Cap a seed at ``max_notes`` notes, shifting by whole bars so every note
    keeps its position in the bar. ``keep="last"`` (continuation: what matters is
    where the seed ends) or ``"first"`` (variation: the motif is the opening)."""
    if not seed:
        raise ValueError("seed contains no notes")
    if len(seed) <= max_notes:
        return seed
    log.warning("seed has %d notes; using the %s %d", len(seed), keep, max_notes)
    notes = seed[-max_notes:] if keep == "last" else seed[:max_notes]
    shift = (notes[0].onset // bar) * bar
    return [Note(n.onset - shift, n.pitch, n.dur) for n in notes]


def generate_continuations(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    seed: Melody,
    n_bars: int = 4,
    k: int = 4,
    temperature: float = 0.95,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    key_normalize: bool = True,
    oversample: int = 3,
    bar: int = STEPS_PER_BAR,
) -> list[Candidate]:
    """Extend ``seed`` by ``n_bars`` bars of ``bar`` steps (16: duple, 12: triple),
    returning up to ``k`` distinct candidates."""
    if bar != STEPS_PER_BAR and not tokenizer.meter_tokens:
        raise ValueError("this checkpoint only supports duple (4/4) metre")
    seed = trim_seed(seed, keep="last", bar=bar)
    shift = shift_to_training_key(seed, bar) if key_normalize else 0
    seed_t = transpose(seed, shift)
    seed_len = melody_duration(seed_t)
    stop = _bar_ceil(seed_len + n_bars * bar, bar)
    prompt = tokenizer.encode(seed_t, add_special=True, bar=bar)[:-1]  # drop EOS, keep BOS

    rows = generate_batch(
        model, tokenizer, prompt, n=k * oversample,
        max_new_tokens=(stop - seed_len) * 3 + 8, temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, stop_step=stop,
    )
    seen: set[tuple] = set()
    out: list[Candidate] = []
    # completed rows first, so a row cut off by the token budget is a last resort
    for ids, completed in sorted(rows, key=lambda r: not r[1]):
        melody = tokenizer.decode(ids, bar=bar)
        gen = region_within(melody, seed_len, stop)
        key = tuple((n.onset, n.pitch, n.dur) for n in gen)
        if not gen or key in seen:
            continue
        seen.add(key)
        melody = [n for n in melody if n.onset < stop]
        out.append(Candidate(transpose(melody, -shift), seed_len, completed=completed, bar=bar))
        if len(out) == k:
            break
    return out


def generate_variations(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    seed: Melody,
    k: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    similarity_band: tuple[float, float] = (0.35, 0.85),
    key_normalize: bool = True,
    oversample: int = 6,
    bar: int = STEPS_PER_BAR,
) -> list[Candidate]:
    """Rhythm-locked re-voicings of ``seed``, ranked in-band first."""
    if bar != STEPS_PER_BAR and not tokenizer.meter_tokens:
        raise ValueError("this checkpoint only supports duple (4/4) metre")
    seed = trim_seed(seed, keep="first", bar=bar)
    shift = shift_to_training_key(seed, bar) if key_normalize else 0
    seed_t = transpose(seed, shift)
    offset = _bar_ceil(melody_duration(seed_t), bar)  # restate from the next bar line
    restated = [Note(n.onset + offset, n.pitch, n.dur) for n in seed_t]

    prompt = tokenizer.encode(seed_t, add_special=True, bar=bar)[:-1]
    full = tokenizer.encode(seed_t + restated, add_special=True, bar=bar)[:-1]
    template = [None if tokenizer.is_pitch(t) else t for t in full[len(prompt):]]

    rows = generate_batch(
        model, tokenizer, prompt, n=k * oversample, temperature=temperature,
        top_k=top_k, top_p=top_p, template=template,
    )
    lo, hi = similarity_band
    seed_pitches = [n.pitch for n in seed_t]
    seen: set[tuple] = set()
    cands: list[Candidate] = []
    for ids, _ in rows:
        var = [Note(n.onset - offset, n.pitch, n.dur)
               for n in region_within(tokenizer.decode(ids, bar=bar), offset, 10**9)]
        pitches = tuple(n.pitch for n in var)
        if list(pitches) == seed_pitches or pitches in seen:
            continue  # a verbatim copy is not a variation
        seen.add(pitches)
        cands.append(Candidate(transpose(var, -shift), 0, motif_similarity(seed_t, var), bar=bar))

    mid = (lo + hi) / 2
    cands.sort(key=lambda c: (not lo <= c.motif_score <= hi, abs(c.motif_score - mid)))
    return cands[:k]
