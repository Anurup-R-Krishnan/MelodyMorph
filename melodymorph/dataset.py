"""Windowed token dataset with transposition augmentation."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from .tokenizer import MAX_PITCH, MIN_PITCH, Melody, MelodyTokenizer, Note


def transpose(melody: Melody, semitones: int) -> Melody | None:
    """Transpose a whole melody, or return None if it would leave the range."""
    if semitones == 0:
        return melody
    lo = min(n.pitch for n in melody) + semitones
    hi = max(n.pitch for n in melody) + semitones
    if lo < MIN_PITCH or hi > MAX_PITCH:
        return None
    return [Note(n.onset, n.pitch + semitones, n.dur) for n in melody]


def split_melodies(
    melodies: list[Melody], val_fraction: float = 0.1, seed: int = 1337
) -> tuple[list[Melody], list[Melody]]:
    """Split at the *melody* level so validation windows cannot leak into train."""
    shuffled = list(melodies)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def build_token_stream(
    melodies: list[Melody],
    tokenizer: MelodyTokenizer,
    transpose_range: int = 0,
) -> list[int]:
    """Encode melodies (optionally with transposed copies) into one flat stream."""
    stream: list[int] = []
    shifts = range(-transpose_range, transpose_range + 1) if transpose_range else [0]
    for melody in melodies:
        for shift in shifts:
            shifted = transpose(melody, shift)
            if shifted is not None:
                stream.extend(tokenizer.encode(shifted))
    return stream


class MelodyWindowDataset(Dataset):
    """Fixed-length windows over a flat token stream, targets shifted by one."""

    def __init__(self, stream: list[int], block_size: int, stride: int | None = None):
        if len(stream) < block_size + 1:
            raise ValueError(
                f"token stream ({len(stream)}) shorter than block_size + 1 ({block_size + 1})"
            )
        self.data = torch.tensor(stream, dtype=torch.long)
        self.block_size = block_size
        self.stride = stride or block_size // 2
        self.starts = list(range(0, len(self.data) - block_size - 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[i]
        chunk = self.data[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]
