"""Harmony: which chord goes under each part of a melody, and which key it is in.

Used by the arranger (to write the band) and by key normalisation (to move a seed
into the key the Transformer was trained in). A profile-only key estimate confuses
a key with its relative minor on short fragments, so the key is found *jointly*
with the chords: every key is asked to harmonise the melody, and the one whose
best chord path scores highest wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from .keys import key_correlations
from .tokenizer import STEPS_PER_BAR, Melody, melody_duration

_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
TONES = {"maj": (0, 4, 7), "min": (0, 3, 7), "dom7": (0, 4, 7, 10), "dim": (0, 3, 6)}
_SUFFIX = {"maj": "", "min": "m", "dom7": "7", "dim": "dim"}
# keys whose tonic is conventionally spelled with flats
_FLAT_MAJOR = {5, 10, 3, 8, 1}
_FLAT_MINOR = {2, 7, 0, 5, 10, 3}


@dataclass(frozen=True)
class Chord:
    """A chord over ``[start, start + dur)`` in 16th steps."""

    start: int
    dur: int
    root: int  # pitch class 0-11
    quality: str  # maj | min | dom7 | dim
    flats: bool = False

    @property
    def pcs(self) -> tuple[int, ...]:
        return tuple((self.root + t) % 12 for t in TONES[self.quality])

    @property
    def name(self) -> str:
        names = _FLAT if self.flats else _SHARP
        return names[self.root] + _SUFFIX[self.quality]


# --------------------------------------------------------------------------
# chord inference
# --------------------------------------------------------------------------
# (root offset from tonic, quality, prior): how at home each chord is in the key
_MAJOR_POOL = [
    (0, "maj", 1.0),
    (5, "maj", 0.9),
    (7, "maj", 0.9),
    (9, "min", 0.85),
    (2, "min", 0.6),
    (4, "min", 0.45),
    (7, "dom7", 0.55),
]
_MINOR_POOL = [
    (0, "min", 1.0),
    (5, "min", 0.9),
    (7, "maj", 0.8),
    (8, "maj", 0.85),
    (10, "maj", 0.7),
    (3, "maj", 0.65),
    (7, "min", 0.4),
    (7, "dom7", 0.5),
]
# favoured root movements, as (from offset, to offset) -- the grammar of pop harmony
_MOVES = {
    (7, 0): 0.8,
    (5, 7): 0.7,
    (5, 0): 0.5,
    (0, 5): 0.5,
    (9, 5): 0.6,
    (0, 7): 0.4,
    (2, 7): 0.7,
    (7, 9): 0.4,
    (9, 2): 0.4,
    (4, 9): 0.5,
    (0, 9): 0.5,
    (0, 8): 0.5,
    (8, 10): 0.5,
    (10, 0): 0.5,
    (8, 7): 0.5,
    (3, 8): 0.5,
    (10, 3): 0.4,
    (0, 10): 0.4,
    (8, 5): 0.4,
}
_STAY = 0.45


def _slot_score(chord_pcs: tuple[int, ...], melody: Melody, lo: int, hi: int) -> float:
    """How well a chord sits under the melody notes sounding in ``[lo, hi)``."""
    total = 0.0
    for n in melody:
        overlap = min(n.onset + n.dur, hi) - max(n.onset, lo)
        if overlap <= 0:
            continue
        weight = overlap * (1.6 if lo <= n.onset < hi and n.onset % 4 == 0 else 1.0)
        pc = n.pitch % 12
        if pc in chord_pcs:
            total += 2.0 * weight
        elif any((pc - t) % 12 in (1, 11) for t in chord_pcs):
            total -= 1.2 * weight  # a held half-step clash against the chord
        else:
            total -= 0.25 * weight  # an ordinary passing / colour tone
    return total / (hi - lo)


def _viterbi(melody: Melody, bar: int, tonic: int, mode: str) -> tuple[float, list[Chord]]:
    """Best chord path in one key: ``(score per slot, chords)``."""
    flats = (tonic in _FLAT_MAJOR) if mode == "major" else (tonic in _FLAT_MINOR)
    pool = [
        ((tonic + off) % 12, q, prior, off)
        for off, q, prior in (_MAJOR_POOL if mode == "major" else _MINOR_POOL)
    ]
    steps = -(-max(melody_duration(melody), bar) // bar) * bar
    slot = bar // 2 if bar == STEPS_PER_BAR else bar
    n_slots = steps // slot
    emit = [
        [
            _slot_score(tuple((r + t) % 12 for t in TONES[q]), melody, s * slot, (s + 1) * slot)
            + 0.8 * prior
            for r, q, prior, _ in pool
        ]
        for s in range(n_slots)
    ]

    def trans(a: int, b: int) -> float:
        if pool[a][:2] == pool[b][:2]:
            return _STAY
        return _MOVES.get((pool[a][3], pool[b][3]), 0.0)

    k = len(pool)
    best = [[0.0] * k for _ in range(n_slots)]
    back = [[0] * k for _ in range(n_slots)]
    for j in range(k):
        best[0][j] = emit[0][j] + (1.0 if pool[j][3] == 0 else 0.0)
    for s in range(1, n_slots):
        for j in range(k):
            cands = [best[s - 1][i] + trans(i, j) for i in range(k)]
            i = max(range(k), key=cands.__getitem__)
            best[s][j] = cands[i] + emit[s][j]
            back[s][j] = i
    last = [best[-1][j] + (2.0 if pool[j][3] == 0 and pool[j][1] != "dom7" else 0.0) for j in range(k)]
    j = max(range(k), key=last.__getitem__)
    total = last[j]
    path = [j]
    for s in range(n_slots - 1, 0, -1):
        j = back[s][j]
        path.append(j)
    path.reverse()

    chords: list[Chord] = []
    for s, j in enumerate(path):
        root, quality = pool[j][0], pool[j][1]
        if chords and chords[-1].root == root and chords[-1].quality == quality:
            prev = chords[-1]
            chords[-1] = Chord(prev.start, prev.dur + slot, root, quality, flats)
        else:
            chords.append(Chord(s * slot, slot, root, quality, flats))
    return total / n_slots, chords


# how much the pitch-class profile (Krumhansl) counts next to the chord fit when choosing a key
_PROFILE_WEIGHT = 0.6


def infer_key_and_chords(
    melody: Melody, bar: int = STEPS_PER_BAR, key: tuple[int, str] | None = None
) -> tuple[tuple[int, str], list[Chord]]:
    """The key and chords that best explain the melody.

    A profile-only key estimate confuses a key with its relative minor on short
    fragments (a C-major arpeggio reads as E minor). So instead every one of the
    24 keys is asked to harmonise the melody, and the winner is the key whose
    best chord path scores highest -- with the pitch-class profile as a tiebreak.
    """
    if not melody:
        return key or (0, "major"), []
    if key is not None:
        return key, _viterbi(melody, bar, *key)[1]
    profile = key_correlations(melody)
    best: tuple[float, tuple[int, str], list[Chord]] | None = None
    for cand, corr in profile.items():
        score, chords = _viterbi(melody, bar, *cand)
        total = score + _PROFILE_WEIGHT * corr
        if best is None or total > best[0]:
            best = (total, cand, chords)
    assert best is not None
    return best[1], best[2]


def infer_chords(melody: Melody, bar: int = STEPS_PER_BAR, key: tuple[int, str] | None = None) -> list[Chord]:
    """Choose a chord for every half bar (every bar in 3/4) by Viterbi search.

    Each chord is scored by how many melody notes it contains, weighted towards
    strong beats, plus how natural the move from the previous chord is. The
    path is nudged to start on the tonic and to cadence at the end. With no
    ``key`` given, the key is found jointly with the chords.
    """
    return infer_key_and_chords(melody, bar, key)[1]
