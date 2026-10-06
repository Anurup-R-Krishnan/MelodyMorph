"""Generative artwork for the MelodyMorph interface.

Everything here is drawn from the melody itself, so the artwork is evidence, not
decoration:

* :func:`arc_poster_svg` -- the *arc poster*. Every note becomes a fragment of a
  ring: the ring is the pitch (low notes sit inside, high notes outside), the
  angle is the onset, the sweep is the duration. Repeated pitches share a ring.
* :func:`roll_figure_html` -- a piano roll on a visible construction grid (the
  16th-step grid the tokenizer quantises to), with the bar, beat and step lines
  revealable. It is plain HTML/CSS because Streamlit sanitises inline ``<svg>``.
* :func:`contour_svg` -- the interval contour, the fingerprint the motif filter
  compares (one bar per interval, up or down).
* :func:`logo_svg` -- the mark: the same arc grammar, drawn once.

The SVG builders return self-contained documents (their own ``<style>``), meant
to be shipped as an image element via :func:`svg_img`; the roll is DOM so a page script
can drive its playhead. No Streamlit dependency, so it is unit-testable.
"""

from __future__ import annotations

import base64
import math
from html import escape

from .tokenizer import (
    MAX_PITCH,
    MIN_PITCH,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
    Melody,
    melody_duration,
    pitch_name,
)

# The palette lives here once; app/static/theme.css mirrors it as custom properties.
FIELD = "#F2512B"  # vermilion poster ground
INK = "#12100E"
PAPER = "#F5F4EF"
ULTRA = "#1F3BE8"  # playhead and grid only

_BLACK_KEYS = {1, 3, 6, 8, 10}
_NO_MOTION = "@media (prefers-reduced-motion:reduce){.mm-draw{animation:none;stroke-dashoffset:0}.mm-spin{animation:none}}"


def _point(cx: float, cy: float, r: float, deg: float) -> tuple[float, float]:
    rad = math.radians(deg)
    return cx + r * math.cos(rad), cy + r * math.sin(rad)


def _arc(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    """SVG path for a clockwise arc of radius ``r`` from ``a0`` to ``a1`` degrees."""
    x0, y0 = _point(cx, cy, r, a0)
    x1, y1 = _point(cx, cy, r, a1)
    large = 1 if (a1 - a0) > 180 else 0
    return f"M{x0:.2f} {y0:.2f}A{r:.2f} {r:.2f} 0 {large} 1 {x1:.2f} {y1:.2f}"


def _pitch_span(melody: Melody, min_rows: int = 12) -> tuple[int, int]:
    """Inclusive pitch range to draw, widened to at least ``min_rows`` semitones."""
    lo = min(n.pitch for n in melody)
    hi = max(n.pitch for n in melody)
    short = min_rows - (hi - lo + 1)
    if short > 0:
        lo -= short // 2
        hi += short - short // 2
    return max(lo, MIN_PITCH - 12), min(hi, MAX_PITCH + 12)


def _describe(melody: Melody) -> str:
    if not melody:
        return "empty melody"
    lo = min(n.pitch for n in melody)
    hi = max(n.pitch for n in melody)
    return f"{len(melody)} notes from {pitch_name(lo)} to {pitch_name(hi)}"


def svg_img(svg: str, alt: str, cls: str = "") -> str:
    """Ship an SVG document as an image element (an inline svg is stripped by
    Streamlit's sanitiser; CSS animation inside an SVG image still runs)."""
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    cls_attr = f' class="{cls}"' if cls else ""
    return f'<img{cls_attr} src="{uri}" alt="{escape(alt, quote=True)}" draggable="false">'


# --------------------------------------------------------------------------
# the mark
# --------------------------------------------------------------------------
def logo_svg(size: int = 48, fg: str = INK, bg: str | None = None) -> str:
    """The MelodyMorph mark: four open rings that climb like a phrase, around a
    held dot. ``bg`` adds a square ground (favicon use)."""
    c = 32.0
    # (radius, start angle, sweep): each ring opens at a different place, so the
    # four read as a phrase climbing outwards rather than a target
    rings = [(7.5, -90, 270), (14.5, -160, 235), (21.5, -215, 205), (28.0, -262, 170)]
    paths = "".join(f'<path d="{_arc(c, c, r, a0, a0 + sweep)}"/>' for r, a0, sweep in rings)
    ground = f'<rect width="64" height="64" fill="{bg}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{size}" height="{size}" '
        f'role="img" aria-label="MelodyMorph">{ground}'
        f'<g fill="none" stroke="{fg}" stroke-width="5.2">{paths}</g>'
        f'<circle cx="{c}" cy="{c}" r="3.6" fill="{fg}"/></svg>'
    )


# --------------------------------------------------------------------------
# the arc poster
# --------------------------------------------------------------------------
def arc_poster_svg(
    melody: Melody,
    *,
    seed_len: int = 0,
    bar: int = STEPS_PER_BAR,
    width: int = 1200,
    height: int = 600,
    animate: bool = True,
    center: tuple[float, float] = (0.74, 0.56),
    radius: float = 0.78,
) -> str:
    """One melody as a poster of concentric ring fragments.

    Notes before ``seed_len`` (the seed) are drawn in paper white, generated notes
    in ink. The artwork bleeds off the frame on purpose, and the fragments draw
    themselves clockwise, in note order, when the image loads.
    """
    cx, cy = width * center[0], height * center[1]
    r_out, r_in = height * radius, height * 0.09
    css = (
        f".a{{fill:none;stroke-linecap:butt}}.a.s{{stroke:{PAPER}}}.a.g{{stroke:{INK}}}"
        f".k{{fill:none;stroke:{INK};stroke-opacity:.34;stroke-width:1}}"
    )
    if animate:
        css += (
            ".mm-draw{stroke-dasharray:var(--l);stroke-dashoffset:var(--l);"
            "animation:draw .9s cubic-bezier(.16,1,.3,1) forwards;animation-delay:calc(var(--i)*38ms + .15s)}"
            ".mm-spin{animation:spin 160s linear infinite}"
            "@keyframes draw{to{stroke-dashoffset:0}}@keyframes spin{to{transform:rotate(360deg)}}"
            + _NO_MOTION
        )
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" preserveAspectRatio="xMidYMid slice"><style>{css}</style>'
    ]

    if not melody:
        for i in range(1, 9):
            r = r_in + (r_out - r_in) * i / 8
            parts.append(f'<circle class="k" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"/>')
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_in * 0.55:.1f}" fill="{INK}"/></svg>')
        return "".join(parts)

    lo, hi = _pitch_span(melody)
    rows = hi - lo + 1
    pitch_step = (r_out - r_in) / rows
    total = max(melody_duration(melody), bar)
    start, sweep = -212.0, 300.0

    def ring(pitch: int) -> float:
        return r_in + pitch_step * (pitch - lo + 0.5)

    # construction: one hairline ring per C, and a radial tick per barline
    construction = [
        f'<circle class="k" cx="{cx:.1f}" cy="{cy:.1f}" r="{ring(p):.1f}"/>'
        for p in range(lo, hi + 1)
        if p % 12 == 0
    ]
    for b in range(0, total // bar + 1):
        ang = start + sweep * (b * bar) / total
        if ang > start + sweep + 0.01:
            break
        x0, y0 = _point(cx, cy, r_in * 0.6, ang)
        x1, y1 = _point(cx, cy, r_out + pitch_step, ang)
        construction.append(f'<line class="k" x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>')
    spin = f' style="transform-origin:{cx:.1f}px {cy:.1f}px" class="mm-spin"' if animate else ""
    parts.append(f"<g{spin}>{''.join(construction)}</g>")

    stroke = max(pitch_step * 0.86, 2.0)
    parts.append(f'<g stroke-width="{stroke:.2f}">')
    for i, n in enumerate(sorted(melody, key=lambda m: (m.onset, m.pitch))):
        a0 = start + sweep * n.onset / total
        a1 = max(start + sweep * (n.onset + n.dur) / total - 0.5, a0 + 0.8)
        kind = "s" if n.onset < seed_len else "g"
        r = ring(n.pitch)
        length = r * math.radians(a1 - a0)
        anim = f' style="--i:{i};--l:{length:.1f}"' if animate else ""
        parts.append(
            f'<path class="a {kind}{" mm-draw" if animate else ""}" d="{_arc(cx, cy, r, a0, a1)}"{anim}/>'
        )
    parts.append("</g>")
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_in * 0.55:.1f}" fill="{INK}"/></svg>')
    return "".join(parts)


def arc_poster_img(melody: Melody, alt_prefix: str = "Arc artwork", cls: str = "", **kwargs) -> str:
    """The arc poster as an image element, with alt text describing the melody."""
    return svg_img(arc_poster_svg(melody, **kwargs), f"{alt_prefix}: {_describe(melody)}", cls)


# --------------------------------------------------------------------------
# the roll
# --------------------------------------------------------------------------
def roll_geometry(melody: Melody, bar: int = STEPS_PER_BAR, min_steps: int | None = None) -> dict:
    """Pitch range and bar-aligned length of a roll."""
    lo, hi = _pitch_span(melody) if melody else (60, 71)
    steps = max(melody_duration(melody) if melody else 0, min_steps or 0, bar)
    steps = -(-steps // bar) * bar
    return {"lo": lo, "hi": hi, "rows": hi - lo + 1, "steps": steps, "bar": bar}


def roll_html(
    melody: Melody,
    *,
    seed_len: int = 0,
    bar: int = STEPS_PER_BAR,
    min_steps: int | None = None,
    animate: bool = True,
    chords: list[tuple[int, int, str]] | None = None,
) -> str:
    """The roll itself: pitch labels, bar numbers, a grid that CSS can reveal,
    and one element per note positioned in percent (so it scales with its card).
    Seed notes are vermilion, generated notes ink."""
    g = roll_geometry(melody, bar, min_steps)
    lo, hi, rows, steps = g["lo"], g["hi"], g["rows"], g["steps"]
    row_pct = 100.0 / rows

    labels = "".join(
        f'<span style="top:{(hi - p) * row_pct:.3f}%;height:{row_pct:.3f}%">{pitch_name(p)}</span>'
        for p in range(lo, hi + 1)
        if p % 12 == 0
    )
    bars = "".join(
        f'<span style="left:{b * bar / steps * 100:.3f}%">{b + 1}</span>' for b in range(steps // bar)
    )
    keys = "".join(
        f'<i class="bk" style="top:{(hi - p) * row_pct:.3f}%;height:{row_pct:.3f}%"></i>'
        for p in range(lo, hi + 1)
        if p % 12 in _BLACK_KEYS
    ) + "".join(
        f'<i class="cl" style="top:{(hi - p + 1) * row_pct:.3f}%"></i>'
        for p in range(lo, hi + 1)
        if p % 12 == 0
    )
    notes = []
    for i, n in enumerate(sorted(melody, key=lambda m: (m.onset, m.pitch))):
        kind = "s" if n.onset < seed_len else "g"
        grow = " mm-grow" if animate else ""
        notes.append(
            f'<i class="n {kind}{grow}" style="--i:{i};left:{n.onset / steps * 100:.3f}%;'
            f'top:{(hi - n.pitch) * row_pct:.3f}%;width:{n.dur / steps * 100:.3f}%;height:{row_pct:.3f}%" '
            f'data-on="{n.onset}" data-du="{n.dur}" title="{pitch_name(n.pitch)}"></i>'
        )
    chord_row = ""
    if chords:
        cells = "".join(
            f'<span style="left:{c0 / steps * 100:.3f}%;width:{cd / steps * 100:.3f}%">{escape(name)}</span>'
            for c0, cd, name in chords
        )
        chord_row = f'<div class="rl-chordcorner"></div><div class="rl-chords">{cells}</div>'
    return (
        f'<div class="rl{" has-chords" if chords else ""}" style="--steps:{steps};--bar:{bar};--rows:{rows}" '
        f'role="img" aria-label="Piano roll: {escape(_describe(melody), quote=True)}">'
        f'<div class="rl-corner"></div><div class="rl-bars">{bars}</div>{chord_row}'
        f'<div class="rl-labels">{labels}</div>'
        f'<div class="rl-grid" data-steps="{steps}">{keys}{"".join(notes)}<b class="ph"></b></div></div>'
    )


def mixer_html(stems: list[dict]) -> str:
    """One row per instrument: mute, solo and a volume fader. The rows are driven by
    the delegated listener in ``app/static/player.js``; with no JS they simply sit still."""
    rows = []
    for st in stems:
        key, label = escape(st["key"], quote=True), escape(st["label"])
        vol = int(round(st.get("volume", 1.0) * 100))
        rows.append(
            f'<div class="mx" data-stem="{key}"><span class="mx-n">{label}</span>'
            f'<button type="button" class="mx-m" aria-pressed="false" aria-label="Mute {label}">M</button>'
            f'<button type="button" class="mx-s" aria-pressed="false" aria-label="Solo {label}">S</button>'
            f'<input type="range" class="mx-v" min="0" max="100" value="{vol}" aria-label="{label} volume"></div>'
        )
    return f'<div class="mm-mix" role="group" aria-label="Mixer">{"".join(rows)}</div>'


def roll_figure_html(
    melody: Melody,
    *,
    uid: str,
    seed_len: int = 0,
    bar: int = STEPS_PER_BAR,
    tempo: int = 100,
    audio_b64: str | None = None,
    stems: list[dict] | None = None,
    chords: list[tuple[int, int, str]] | None = None,
    min_steps: int | None = None,
    animate: bool = True,
) -> str:
    """The roll with its controls: a play button, the grid reveal and a clock.

    Audio is either one WAV (``audio_b64``, the built-in synth) or a band of MP3
    ``stems`` (dicts with ``key``, ``label``, ``b64`` and a default ``volume``),
    which get a mixer under the roll. ``chords`` adds a chord lane above the grid.

    The grid reveal is a pure-CSS checkbox, so it needs no rerun. Playback, the
    moving playhead and the mixer are driven by ``app/static/player.js``.
    """
    sps = 60.0 / tempo / STEPS_PER_BEAT
    if stems:
        audio = "".join(
            f'<audio preload="auto" data-stem="{escape(st["key"], quote=True)}" '
            f'src="data:audio/mpeg;base64,{st["b64"]}"></audio>'
            for st in stems
        )
    elif audio_b64:
        audio = f'<audio preload="none" src="data:audio/wav;base64,{audio_b64}"></audio>'
    else:
        audio = ""
    play = (
        '<button type="button" class="mm-play" aria-label="Play"><i class="ic"></i></button>' if audio else ""
    )
    mixer = mixer_html(stems) if stems and len(stems) > 1 else ""
    return (
        f'<figure class="mm-roll" data-sps="{sps:.5f}">'
        f'<input type="checkbox" id="grid-{uid}" class="mm-gridbox">'
        f'<div class="mm-rollbar">{play}'
        f'<label for="grid-{uid}" class="mm-gridlabel"><span class="mm-switch" aria-hidden="true"></span>'
        f'show construction grid</label><span class="mm-clock">0:00</span></div>'
        f"{roll_html(melody, seed_len=seed_len, bar=bar, min_steps=min_steps, animate=animate, chords=chords)}"
        f"{mixer}{audio}</figure>"
    )


# --------------------------------------------------------------------------
# the contour fingerprint
# --------------------------------------------------------------------------
def contour_svg(melody: Melody, *, width: int = 320, height: int = 56) -> str:
    """One bar per melodic interval, up above the midline and down below, scaled
    by size. This is the sequence the motif-similarity filter compares."""
    notes = sorted(melody, key=lambda n: n.onset)
    steps = [b.pitch - a.pitch for a, b in zip(notes, notes[1:])]
    mid = height / 2
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<line x1="0" y1="{mid}" x2="{width}" y2="{mid}" stroke="{INK}" stroke-opacity=".5"/>',
    ]
    if steps:
        biggest = max(max(abs(s) for s in steps), 1)
        slot = width / len(steps)
        bw = max(min(slot - 2, 14), 2)
        for i, s in enumerate(steps):
            hgt = max(abs(s) / biggest * (mid - 3), 2 if s == 0 else 3)
            x = i * slot + (slot - bw) / 2
            y = mid - hgt if s > 0 else mid
            fill = INK if s > 0 else FIELD if s < 0 else "#8A857D"
            stroke = f' stroke="{INK}" stroke-width="1.2"' if s < 0 else ""
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{hgt:.1f}" fill="{fill}"{stroke}/>'
            )
    out.append("</svg>")
    return "".join(out)


def contour_img(melody: Melody, **kwargs) -> str:
    n = max(len(melody) - 1, 0)
    return svg_img(contour_svg(melody, **kwargs), f"Interval contour of {n} steps", "mm-contour")
