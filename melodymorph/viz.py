"""Piano-roll rendering for melodies."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # safe in headless/CLI contexts; Streamlit re-selects its own backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .tokenizer import STEPS_PER_BAR, Melody, melody_duration

_NOTE_NAMES = ["C", "", "D", "", "E", "F", "", "G", "", "A", "", "B"]


def _pitch_label(pitch: int) -> str:
    name = _NOTE_NAMES[pitch % 12]
    octave = pitch // 12 - 1
    return f"{name}{octave}" if name else ""


def plot_piano_roll(
    melody: Melody,
    seed_len: int = 0,
    title: str = "",
    ax=None,
):
    """Draw a piano roll. Notes with onset < seed_len are shaded as the seed.

    ``seed_len`` is in 16th-note steps. Returns the matplotlib Figure (or the
    Axes' figure, if ``ax`` was supplied) so callers can save or embed it.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if not melody:
        ax.set_title(title or "Empty melody")
        return fig

    pitches = [n.pitch for n in melody]
    lo, hi = min(pitches) - 2, max(pitches) + 2
    total_steps = melody_duration(melody)

    for note in melody:
        is_seed = note.onset < seed_len
        color = "#4C72B0" if is_seed else "#DD8452"
        ax.add_patch(
            Rectangle(
                (note.onset, note.pitch - 0.4),
                width=max(note.dur, 0.8),
                height=0.8,
                facecolor=color,
                edgecolor="black",
                linewidth=0.4,
            )
        )

    for bar_step in range(0, total_steps + 1, STEPS_PER_BAR):
        ax.axvline(bar_step, color="grey", linewidth=0.5, alpha=0.5)

    if seed_len:
        ax.axvline(seed_len, color="black", linewidth=1.5, linestyle="--")

    ax.set_xlim(0, max(total_steps, STEPS_PER_BAR))
    ax.set_ylim(lo, hi)
    ax.set_yticks(range(lo, hi + 1))
    ax.set_yticklabels([_pitch_label(p) for p in range(lo, hi + 1)], fontsize=7)
    ax.set_xlabel("16th-note step")
    ax.set_title(title)
    if own_fig:
        fig.tight_layout()
    return fig


def save_piano_roll(melody: Melody, path: str, seed_len: int = 0, title: str = "") -> str:
    fig = plot_piano_roll(melody, seed_len=seed_len, title=title)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
