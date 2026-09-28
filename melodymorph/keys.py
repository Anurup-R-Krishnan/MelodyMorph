"""Key estimation for seeds (Krumhansl-Schmuckler, duration-weighted).

The training corpus is transposed to C major / A minor, so a seed in any other
key is out of distribution. Generation transposes the seed into the training
key first and transposes the results back.
"""

from __future__ import annotations

from .tokenizer import Melody, in_range, transpose

# Krumhansl-Kessler probe-tone profiles, index 0 = tonic
_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _corr(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def estimate_key(melody: Melody) -> tuple[int, str]:
    """Return ``(tonic_pitch_class, "major" | "minor")``."""
    hist = [0.0] * 12
    for n in melody:
        hist[n.pitch % 12] += n.dur
    best = (float("-inf"), 0, "major")
    for tonic in range(12):
        rotated = hist[tonic:] + hist[:tonic]
        for mode, profile in (("major", _MAJOR), ("minor", _MINOR)):
            score = _corr(rotated, profile)
            if score > best[0]:
                best = (score, tonic, mode)
    return best[1], best[2]


def shift_to_training_key(melody: Melody) -> int:
    """Semitone shift that moves ``melody`` to C major / A minor and keeps it
    inside the model's pitch range (0 if no shift keeps it in range)."""
    tonic, mode = estimate_key(melody)
    target = 0 if mode == "major" else 9
    base = (target - tonic) % 12
    if base > 6:
        base -= 12
    for shift in sorted((base, base - 12, base + 12), key=abs):
        if in_range(transpose(melody, shift)):
            return shift
    return 0
