"""Piano-roll rendering for melodies."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # safe in headless/CLI contexts; Streamlit re-selects its own backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .tokenizer import STEPS_PER_BAR, Melody, melody_duration

_ALL_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_label(pitch: int) -> str:
    name = _ALL_PITCH_NAMES[pitch % 12]
    octave = pitch // 12 - 1
    # Label natural notes with name+octave, sharps with small sharp sign
    return f"{name}{octave}"


def plot_piano_roll(
    melody: Melody,
    seed_len: int = 0,
    title: str = "",
    theme: str = "light",
    ax=None,
):
    """Draw a piano roll. Notes with onset < seed_len are shaded as the seed.

    ``seed_len`` is in 16th-note steps. Returns the matplotlib Figure.
    Supports native ``theme="light"`` and ``theme="dark"``.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    is_dark = theme == "dark"
    bg_color = "#0C0C0E" if is_dark else "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    seed_color = "#D8C79A" if is_dark else "#4C72B0"
    gen_color = "#7EE8B8" if is_dark else "#DD8452"
    edge_color = "#2A2415" if is_dark else "#222222"
    grid_color = (1, 1, 1, 0.12) if is_dark else "grey"
    divider_color = "#D8C79A" if is_dark else "black"
    text_color = "#85858C" if is_dark else "#333333"

    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.15) if is_dark else "#CCCCCC")
    ax.tick_params(colors=text_color, labelsize=8)
    ax.xaxis.label.set_color(text_color)
    ax.xaxis.label.set_fontsize(9)

    if not melody:
        ax.set_title(title or "Empty melody", color=text_color)
        ax.set_xlim(0, STEPS_PER_BAR)
        ax.set_ylim(55, 75)
        return fig

    pitches = [n.pitch for n in melody]
    lo, hi = min(pitches) - 1, max(pitches) + 1
    total_steps = melody_duration(melody)
    max_steps = max(total_steps + 2, STEPS_PER_BAR)

    # Subtle beat lines (every 4 steps / quarter note)
    for beat_step in range(0, max_steps + 1, 4):
        if beat_step % STEPS_PER_BAR != 0:
            ax.axvline(beat_step, color=grid_color, linewidth=0.3, alpha=0.3, linestyle=":")

    # Measure lines (every 16 steps)
    for bar_step in range(0, max_steps + 1, STEPS_PER_BAR):
        ax.axvline(bar_step, color=grid_color, linewidth=0.7, alpha=0.7)

    for note in melody:
        is_seed = (note.onset < seed_len) if seed_len > 0 else False
        color = seed_color if is_seed else gen_color
        # Slightly shrink width so consecutive notes don't visually fuse
        visual_width = max(note.dur - 0.15, 0.6)
        ax.add_patch(
            Rectangle(
                (note.onset, note.pitch - 0.38),
                width=visual_width,
                height=0.76,
                facecolor=color,
                edgecolor=edge_color,
                linewidth=0.5,
                alpha=0.95,
            )
        )

    if seed_len > 0 and seed_len < total_steps:
        ax.axvline(seed_len, color=divider_color, linewidth=1.5, linestyle="--", alpha=0.9)

    ax.set_xlim(0, max_steps)
    ax.set_ylim(lo - 0.5, hi + 0.5)

    # Adaptive Y-ticks based on register span to prevent label collisions
    span = hi - lo
    if span <= 20:
        y_ticks = list(range(lo, hi + 1))
    elif span <= 36:
        # Naturals only
        y_ticks = [p for p in range(lo, hi + 1) if "#" not in _pitch_label(p)]
    else:
        # Octaves and fifths
        y_ticks = [p for p in range(lo, hi + 1) if _pitch_label(p).startswith("C") or _pitch_label(p).startswith("G")]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([_pitch_label(p) for p in y_ticks], fontsize=7)

    # Measure-based X-ticks (e.g. m.1, m.2, m.3)
    bar_ticks = list(range(0, max_steps, STEPS_PER_BAR))
    ax.set_xticks(bar_ticks)
    ax.set_xticklabels([f"m.{b // STEPS_PER_BAR + 1}" for b in bar_ticks], fontsize=8)
    ax.set_xlabel("measure (4/4 time)")
    if title:
        ax.set_title(title, color=text_color, fontsize=11)
    if own_fig:
        fig.tight_layout(pad=1.0)
    return fig


def save_piano_roll(melody: Melody, path: str, seed_len: int = 0, title: str = "", theme: str = "light") -> str:
    fig = plot_piano_roll(melody, seed_len=seed_len, title=title, theme=theme)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_contour_strip(
    melody: Melody,
    seed_len: int = 0,
    theme: str = "dark",
    figsize: tuple[float, float] = (4.5, 1.6),
    ax=None,
):
    """Pitch-contour strip: melodic trajectory at a glance.

    Notes with onset < seed_len are shaded as the seed. Supports theme="dark" and theme="light".
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    is_dark = theme == "dark"
    bg_color = "#0C0C0E" if is_dark else "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    notes = sorted(melody, key=lambda n: n.onset)
    seed_color = "#D8C79A" if is_dark else "#4C72B0"
    gen_color = "#7EE8B8" if is_dark else "#DD8452"
    spine_color = (1, 1, 1, 0.10) if is_dark else "#CCCCCC"

    if notes:
        xs = [n.onset for n in notes]
        ys = [n.pitch for n in notes]
        if seed_len > 0:
            ax.axvspan(0, seed_len, color=seed_color, alpha=0.10 if is_dark else 0.20)
        ax.step(xs, ys, where="post", color=gen_color, linewidth=1.4)
        ax.plot(xs, ys, "o", color=gen_color, markersize=3)
        ax.set_xlim(0, max(max(xs) + 4, STEPS_PER_BAR))
        y_min, y_max = min(ys), max(ys)
        ax.set_ylim(y_min - 2, y_max + 2)
    else:
        ax.set_xlim(0, STEPS_PER_BAR)
        ax.set_ylim(55, 75)

    for spine in ax.spines.values():
        spine.set_color(spine_color)
    ax.set_xticks([])
    ax.set_yticks([])
    if own_fig:
        fig.tight_layout(pad=0.4)
    return fig


contour_strip = plot_contour_strip


def save_contour_strip(
    melody: Melody,
    path: str,
    seed_len: int = 0,
    theme: str = "dark",
    figsize: tuple[float, float] = (4.5, 1.6),
) -> str:
    fig = plot_contour_strip(melody, seed_len=seed_len, theme=theme, figsize=figsize)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path

