"""Melody-aligned token windows with transposition augmentation.

Each training window starts at a melody's ``BOS`` (or, for a melody longer than
the context, at a later offset of that same melody) and is padded with ``PAD``
after its ``EOS``. Windows never mix two tunes, which matches inference: the
model is always prompted with ``BOS`` + one seed at position 0, never with the
tail of an unrelated tune in its context.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .tokenizer import STEPS_PER_BAR, Melody, MelodyTokenizer, in_range, transpose


def transpose_in_range(melody: Melody, semitones: int) -> Melody | None:
    """Transpose a whole melody, or return None if it would leave the range."""
    shifted = transpose(melody, semitones)
    return shifted if in_range(shifted) else None


def augment(
    melodies: list[Melody], transpose_range: int = 0, bars: list[int] | None = None,
) -> tuple[list[Melody], list[int]]:
    """Every melody at every in-range shift in ``[-transpose_range, transpose_range]``,
    with each copy's bar length."""
    bars = bars or [STEPS_PER_BAR] * len(melodies)
    out: list[Melody] = []
    out_bars: list[int] = []
    for shift in range(-transpose_range, transpose_range + 1):
        for melody, bar in zip(melodies, bars):
            shifted = transpose_in_range(melody, shift)
            if shifted is not None:
                out.append(shifted)
                out_bars.append(bar)
    return out, out_bars


def build_token_stream(
    melodies: list[Melody], tokenizer: MelodyTokenizer, transpose_range: int = 0, seed: int = 1337,
    bars: list[int] | None = None,
) -> list[int]:
    """All melodies (shuffled, optionally augmented) as one flat stream.

    Used by the n-gram baseline, which has no context window to respect.
    """
    items = list(zip(*augment(melodies, transpose_range, bars)))
    random.Random(seed).shuffle(items)
    return [t for m, bar in items for t in tokenizer.encode(m, bar=bar)]


def melody_windows(ids: list[int], block_size: int, stride: int) -> list[tuple[int, list[int]]]:
    """Windows of up to ``block_size + 1`` tokens over one encoded melody.

    Returns ``(score_from, window)`` pairs: the first window scores every target;
    later windows overlap the previous one and score only targets it did not,
    so each token is predicted exactly once (with at least
    ``block_size - stride`` tokens of context after the first window).
    """
    span = block_size + 1
    if len(ids) <= span:
        return [(0, ids)]
    out = [(0, ids[:span])]
    scored_to = span - 1  # targets ids[1:span] are scored
    start = stride
    while scored_to < len(ids) - 1:
        start = min(start, len(ids) - span)
        window = ids[start : start + span]
        out.append((scored_to - start, window))
        scored_to = start + span - 1
        start += stride
    return out


class MelodyWindowDataset(Dataset):
    """One or more padded windows per melody; targets shifted by one.

    Targets that are padding, or that an earlier overlapping window already
    covered, are set to ``PAD`` and ignored by the loss.
    """

    def __init__(
        self, melodies: list[Melody], tokenizer: MelodyTokenizer, block_size: int,
        stride: int | None = None, bars: list[int] | None = None,
    ):
        if not melodies:
            raise ValueError("no melodies to build windows from")
        self.block_size = block_size
        pad = tokenizer.pad_id
        stride = stride or block_size // 2
        xs, ys = [], []
        bars = bars or [STEPS_PER_BAR] * len(melodies)
        for melody, bar in zip(melodies, bars):
            for score_from, w in melody_windows(tokenizer.encode(melody, bar=bar), block_size, stride):
                x = w[:-1] + [pad] * (block_size + 1 - len(w))
                y = w[1:] + [pad] * (block_size + 1 - len(w))
                y = [pad] * score_from + y[score_from:]
                xs.append(x)
                ys.append(y)
        self.x = torch.tensor(xs, dtype=torch.long)
        self.y = torch.tensor(ys, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]
