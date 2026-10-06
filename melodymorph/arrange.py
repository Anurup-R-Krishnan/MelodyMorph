"""Turn a melody into a band: chords, bass, comping and drums.

The Transformer writes the tune; this module writes everything underneath it.
It is deliberately rule-based (no training): it infers a chord for each half bar
by dynamic programming over the diatonic chords of the estimated key, then plays
those chords through a *style* -- a small table of rhythmic patterns for the
comping instrument, an optional pad, the bass and the drums -- with smooth
voice-leading and a light human touch (phrase-shaped velocity, tiny timing
drift) so it does not sound like a grid.

Everything is expressed in MIDI ticks (480 per quarter, so 120 per 16th step)
and returned as an :class:`Arrangement`, which can be written as a multi-track
MIDI file or split into one single-track file per instrument (the stems the app
renders and mixes).
"""

from __future__ import annotations

import hashlib
import io
import random
from dataclasses import dataclass, field

import mido

from .harmony import TONES, Chord, infer_key_and_chords
from .tokenizer import STEPS_PER_BAR, Melody, melody_duration

PPQ = 480
TICKS_PER_STEP = PPQ // 4  # 120 ticks per 16th
DRUM_CHANNEL = 9

# General MIDI programs the app offers for the lead voice
LEAD_INSTRUMENTS: dict[str, int] = {
    "Violin": 40,
    "Flute": 73,
    "Clarinet": 71,
    "Piano": 0,
    "Electric piano": 4,
    "Nylon guitar": 24,
    "Clean guitar": 27,
    "Vibraphone": 11,
    "Cello": 42,
    "Alto sax": 65,
    "Trumpet": 56,
    "Harp": 46,
    "Marimba": 12,
    "Synth lead": 81,
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Placed:
    """A note in absolute ticks."""

    tick: int
    dur: int
    pitch: int
    vel: int


@dataclass
class Track:
    key: str  # lead | chords | pad | bass | drums
    label: str
    program: int
    channel: int
    notes: list[Placed] = field(default_factory=list)
    volume: float = 1.0  # default mixer level (0-1)

    @property
    def drum(self) -> bool:
        return self.channel == DRUM_CHANNEL


@dataclass
class Arrangement:
    tracks: list[Track]
    chords: list[Chord]
    tempo: int
    bar: int
    total_steps: int
    style: str
    key: tuple[int, str]

    @property
    def total_ticks(self) -> int:
        return self.total_steps * TICKS_PER_STEP

    def track(self, key: str) -> Track | None:
        return next((t for t in self.tracks if t.key == key), None)


def _chord_at(chords: list[Chord], step: int) -> Chord:
    for c in chords:
        if c.start <= step < c.start + c.dur:
            return c
    return chords[-1]


# --------------------------------------------------------------------------
# voicing
# --------------------------------------------------------------------------
def _candidates(chord: Chord, lo: int, hi: int) -> list[list[int]]:
    """Every close-position inversion of ``chord`` that fits in ``[lo, hi]``."""
    pcs = list(chord.pcs)
    out = []
    for k in range(len(pcs)):
        rot = pcs[k:] + pcs[:k]
        base = lo + ((rot[0] - lo) % 12)
        while base <= hi:
            notes, cur = [base], base
            for pc in rot[1:]:
                cur += (pc - cur) % 12 or 12
                notes.append(cur)
            if notes[-1] <= hi:
                out.append(notes)
            base += 12
    return out


def voice(chord: Chord, prev: list[int] | None, lo: int = 53, hi: int = 74) -> list[int]:
    """The inversion nearest the previous voicing (smooth voice-leading); the
    first chord is a root-position triad near middle C."""
    cands = _candidates(chord, lo, hi) or [[60 + (chord.root % 12) + t for t in TONES[chord.quality]]]
    if prev is None:
        return min(cands, key=lambda v: (v[0] % 12 != chord.root, abs(sum(v) / len(v) - 62)))
    return min(
        cands,
        key=lambda v: (
            sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev), strict=False))
            + 2 * abs(len(v) - len(prev))
        ),
    )


def _bass_pitch(chord: Chord, kind: str) -> int:
    root = 36 + ((chord.root - 36) % 12)  # E1-D#2... keeps the bass in 36..47
    if kind == "fifth":
        return root + 7 if root + 7 <= 47 else root - 5
    if kind == "octave":
        return root + 12
    return root


# --------------------------------------------------------------------------
# styles
# --------------------------------------------------------------------------
# one event per row. comp: (step, dur, selection, vel, strum ticks) -- selection is
# "all" or indices into the voicing. bass: (step, dur, kind, vel). drums: (step, dur, note, vel)
HAT, KICK, SNARE, CRASH, TAMB, MARACAS, TOM = 42, 36, 38, 49, 54, 70, 45


@dataclass(frozen=True)
class Pattern:
    comp: tuple = ()
    pad: tuple = ()
    bass: tuple = ()
    drums: tuple = ()


def _arp(steps, idx, vels, dur=4):
    return tuple((s, dur, (i,), v, 0) for s, i, v in zip(steps, idx, vels, strict=True))


_EIGHTHS16 = tuple(range(0, 16, 2))
_EIGHTHS12 = tuple(range(0, 12, 2))


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    blurb: str
    patterns: dict  # bar length -> Pattern
    comp_program: int
    comp_label: str
    pad_program: int | None
    bass_program: int
    bass_label: str
    lead_default: str
    suggested_bpm: int
    fill: bool = False  # snare fill on the last beat of every 4th bar

    @property
    def metres(self) -> tuple[int, ...]:
        return tuple(self.patterns)


STYLES: dict[str, Style] = {
    "ballad": Style(
        "ballad",
        "pop ballad",
        "Soft piano arpeggios, string pad, sustained bass, gentle backbeat.",
        {
            16: Pattern(
                comp=_arp(_EIGHTHS16, (0, 1, 2, 1, 2, 1, 2, 1), (66, 50, 58, 50, 60, 50, 58, 50)),
                pad=((0, 16, "all", 38, 0),),
                bass=((0, 8, "root", 82), (8, 8, "root", 72)),
                drums=tuple((s, 2, HAT, 34 + (8 if s % 4 == 0 else 0)) for s in _EIGHTHS16)
                + ((0, 2, KICK, 74), (10, 2, KICK, 60), (4, 2, SNARE, 50), (12, 2, SNARE, 58)),
            ),
            12: Pattern(
                comp=_arp(_EIGHTHS12, (0, 1, 2, 1, 2, 1), (66, 50, 58, 50, 60, 50)),
                pad=((0, 12, "all", 38, 0),),
                bass=((0, 12, "root", 80),),
                drums=((0, 2, KICK, 70), (4, 2, HAT, 40), (8, 2, HAT, 40)),
            ),
        },
        0,
        "Piano",
        48,
        33,
        "Finger bass",
        "Violin",
        72,
    ),
    "rock": Style(
        "rock",
        "pop rock",
        "Strummed steel guitar, driving eighth-note bass, full kit with fills.",
        {
            16: Pattern(
                comp=tuple(
                    (s, 2, "all", v, 6)
                    for s, v in zip(_EIGHTHS16, (92, 66, 84, 66, 90, 66, 84, 70), strict=True)
                ),
                bass=tuple(
                    (s, 2, k, v)
                    for s, k, v in zip(
                        _EIGHTHS16,
                        ("root", "root", "root", "octave", "root", "root", "root", "octave"),
                        (94, 72, 82, 76, 90, 72, 82, 78),
                        strict=True,
                    )
                ),
                drums=tuple((s, 2, HAT, 78 if s % 4 == 0 else 52) for s in _EIGHTHS16)
                + (
                    (0, 2, KICK, 100),
                    (8, 2, KICK, 96),
                    (10, 2, KICK, 80),
                    (4, 2, SNARE, 100),
                    (12, 2, SNARE, 104),
                ),
            )
        },
        25,
        "Steel guitar",
        None,
        34,
        "Pick bass",
        "Clean guitar",
        120,
        fill=True,
    ),
    "folk": Style(
        "folk",
        "folk strum",
        "Nylon-string guitar strum, walking root-fifth bass, tambourine.",
        {
            16: Pattern(
                comp=(
                    (0, 4, "all", 86, 10),
                    (4, 2, "all", 70, 10),
                    (6, 2, "all", 48, 8),
                    (10, 2, "all", 52, 8),
                    (12, 2, "all", 72, 10),
                    (14, 2, "all", 50, 8),
                ),
                bass=((0, 4, "root", 82), (4, 4, "fifth", 66), (8, 4, "root", 74), (12, 4, "fifth", 66)),
                drums=((4, 2, TAMB, 56), (12, 2, TAMB, 60))
                + tuple((s, 2, MARACAS, 30 + (8 if s % 4 == 0 else 0)) for s in _EIGHTHS16),
            )
        },
        24,
        "Nylon guitar",
        None,
        32,
        "Acoustic bass",
        "Flute",
        96,
    ),
    "waltz": Style(
        "waltz",
        "waltz",
        "Accordion oom-pah-pah over a bass downbeat, light brushed kit.",
        {
            12: Pattern(
                comp=((4, 4, "all", 66, 0), (8, 4, "all", 60, 0)),
                bass=((0, 4, "root", 90), (6, 2, "fifth", 60)),
                drums=((0, 2, KICK, 62), (4, 2, HAT, 44), (8, 2, HAT, 44)),
            )
        },
        21,
        "Accordion",
        None,
        32,
        "Acoustic bass",
        "Clarinet",
        100,
    ),
}


def styles_for(bar: int) -> list[Style]:
    """The styles that have a pattern for this bar length (4/4 or 3/4)."""
    return [s for s in STYLES.values() if bar in s.patterns]


# --------------------------------------------------------------------------
# arranging
# --------------------------------------------------------------------------
def _seed_for(melody: Melody, style: str, salt: int) -> int:
    digest = hashlib.sha256(
        repr([(n.onset, n.pitch, n.dur) for n in melody]).encode() + style.encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") ^ salt


def lead_track(melody: Melody, program: int, bar: int, rng: random.Random | None = None) -> Track:
    """The melody as a performed line: downbeats and long notes lean louder, high
    notes a touch brighter, and each note is slightly detached so repeats speak."""
    notes = []
    for n in melody:
        vel = 78
        vel += 12 if n.onset % bar == 0 else 6 if n.onset % 4 == 0 else 0
        vel += 8 if n.dur >= 8 else 0
        vel += max(-8, min(8, int((n.pitch - 66) * 0.5)))
        if rng is not None:
            vel += rng.randint(-4, 4)
        notes.append(
            Placed(
                n.onset * TICKS_PER_STEP, max(n.dur * TICKS_PER_STEP - 8, 24), n.pitch, max(40, min(118, vel))
            )
        )
    return Track("lead", "lead", program, 0, notes, 1.0)


def arrange(
    melody: Melody,
    *,
    style: str = "ballad",
    bar: int = STEPS_PER_BAR,
    tempo: int = 100,
    lead_program: int | None = None,
    humanize: bool = True,
    salt: int = 0,
) -> Arrangement:
    """Write a band under ``melody`` in the given style."""
    if not melody:
        raise ValueError("cannot arrange an empty melody")
    sty = STYLES[style]
    if bar not in sty.patterns:
        raise ValueError(f"style {style!r} has no pattern for {bar}-step bars")
    pat = sty.patterns[bar]
    key, chords = infer_key_and_chords(melody, bar)
    total = -(-max(melody_duration(melody), bar) // bar) * bar
    rng = random.Random(_seed_for(melody, style, salt)) if humanize else None

    def jitter(ticks: int) -> int:
        return rng.randint(-ticks, ticks) if rng else 0

    def vel_j(v: int, spread: int = 5) -> int:
        return max(1, min(127, v + (rng.randint(-spread, spread) if rng else 0)))

    lead = lead_track(
        melody, lead_program if lead_program is not None else LEAD_INSTRUMENTS[sty.lead_default], bar, rng
    )
    comp = Track("chords", sty.comp_label.lower(), sty.comp_program, 1, [], 0.8)
    pad = Track("pad", "strings", sty.pad_program or 48, 2, [], 0.5) if sty.pad_program is not None else None
    bass = Track("bass", sty.bass_label.lower(), sty.bass_program, 3, [], 0.9)
    drums = Track("drums", "drums", 0, DRUM_CHANNEL, [], 0.8)

    prev_voicing: list[int] | None = None
    n_bars = total // bar
    for b in range(n_bars):
        base = b * bar
        last = b == n_bars - 1
        if last:  # close on the final (cadence) chord, held, with bass and a crash
            ch = chords[-1]
            v = voice(ch, prev_voicing)
            comp.notes += [Placed(base * TICKS_PER_STEP, bar * TICKS_PER_STEP - 20, p, vel_j(70)) for p in v]
            if pad:
                pad.notes += [
                    Placed(base * TICKS_PER_STEP, bar * TICKS_PER_STEP - 20, p, vel_j(44)) for p in v
                ]
            bass.notes.append(
                Placed(base * TICKS_PER_STEP, bar * TICKS_PER_STEP - 20, _bass_pitch(ch, "root"), 84)
            )
            drums.notes += [
                Placed(base * TICKS_PER_STEP, 240, KICK, 96),
                Placed(base * TICKS_PER_STEP, 960, CRASH, 90),
            ]
            break
        for step, dur, sel, vel, strum in pat.comp:
            ch = _chord_at(chords, base + step)
            prev_voicing = voice(ch, prev_voicing)
            idx = range(len(prev_voicing)) if sel == "all" else [i % len(prev_voicing) for i in sel]
            for order, i in enumerate(idx):
                t = (base + step) * TICKS_PER_STEP + order * strum + jitter(6)
                comp.notes.append(Placed(max(t, 0), dur * TICKS_PER_STEP, prev_voicing[i], vel_j(vel)))
        if pad is not None:
            # sustained pad: one voicing per chord, cut where the harmony changes inside the bar
            vel = pat.pad[0][3] if pat.pad else 38
            for ch in chords:
                lo, hi = max(ch.start, base), min(ch.start + ch.dur, base + bar)
                if hi <= lo:
                    continue
                for p in voice(ch, prev_voicing):
                    pad.notes.append(
                        Placed(lo * TICKS_PER_STEP, (hi - lo) * TICKS_PER_STEP - 20, p, vel_j(vel, 3))
                    )
        for step, dur, kind, vel in pat.bass:
            ch = _chord_at(chords, base + step)
            bass.notes.append(
                Placed(
                    max((base + step) * TICKS_PER_STEP + jitter(5), 0),
                    dur * TICKS_PER_STEP - 12,
                    _bass_pitch(ch, kind),
                    vel_j(vel),
                )
            )
        fill = sty.fill and b % 4 == 3
        for step, dur, note, vel in pat.drums:
            if fill and step >= 12 and note in (HAT, SNARE):
                continue
            drums.notes.append(
                Placed(
                    max((base + step) * TICKS_PER_STEP + jitter(3), 0),
                    dur * TICKS_PER_STEP,
                    note,
                    vel_j(vel, 6),
                )
            )
        if fill:
            for i, s in enumerate((12, 13, 14, 15)):
                drums.notes.append(
                    Placed(
                        (base + s) * TICKS_PER_STEP, TICKS_PER_STEP, SNARE if i % 2 == 0 else TOM, 70 + i * 12
                    )
                )
        if b % 4 == 0 and style == "rock":
            drums.notes.append(Placed(base * TICKS_PER_STEP, 960, CRASH, 88))
    tracks = [lead, comp] + ([pad] if pad else []) + [bass, drums]
    for t in tracks:
        t.notes.sort(key=lambda p: (p.tick, p.pitch))
    return Arrangement(tracks, chords, tempo, bar, total, style, key)


def lead_only(melody: Melody, *, program: int, bar: int = STEPS_PER_BAR, tempo: int = 100) -> Arrangement:
    """A single-instrument arrangement (used to audition a seed on its own)."""
    total = -(-max(melody_duration(melody), bar) // bar) * bar
    return Arrangement(
        [lead_track(melody, program, bar)],
        [],
        tempo,
        bar,
        total,
        "solo",
        infer_key_and_chords(melody, bar)[0],
    )


# --------------------------------------------------------------------------
# MIDI out
# --------------------------------------------------------------------------
def _track_events(track: Track) -> list[tuple[int, int, mido.Message]]:
    """(tick, order, message): note-offs sort before note-ons at the same tick."""
    ev: list[tuple[int, int, mido.Message]] = [
        (0, 0, mido.Message("program_change", program=track.program, channel=track.channel, time=0))
    ]
    for n in track.notes:
        ev.append(
            (n.tick, 2, mido.Message("note_on", note=n.pitch, velocity=n.vel, channel=track.channel, time=0))
        )
        ev.append(
            (
                n.tick + n.dur,
                1,
                mido.Message("note_off", note=n.pitch, velocity=0, channel=track.channel, time=0),
            )
        )
    return sorted(ev, key=lambda e: (e[0], e[1]))


def _mido_track(track: Track, end_tick: int) -> mido.MidiTrack:
    mt = mido.MidiTrack()
    mt.append(mido.MetaMessage("track_name", name=track.label, time=0))
    now = 0
    for tick, _, msg in _track_events(track):
        mt.append(msg.copy(time=tick - now))
        now = tick
    mt.append(mido.MetaMessage("end_of_track", time=max(end_tick - now, 0)))
    return mt


def _header(arr: Arrangement) -> mido.MidiTrack:
    mt = mido.MidiTrack()
    mt.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(arr.tempo), time=0))
    num = 4 if arr.bar == STEPS_PER_BAR else 3
    mt.append(mido.MetaMessage("time_signature", numerator=num, denominator=4, time=0))
    return mt


def _end_tick(arr: Arrangement) -> int:
    last = max((n.tick + n.dur for t in arr.tracks for n in t.notes), default=0)
    return max(arr.total_ticks, last) + PPQ


def to_midi_bytes(arr: Arrangement, only: str | None = None) -> bytes:
    """The arrangement as a type-1 MIDI file; ``only`` keeps a single track (a
    stem) but every stem ends at the same tick, so rendered stems stay aligned."""
    mid = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    mid.tracks.append(_header(arr))
    end = _end_tick(arr)
    for t in arr.tracks:
        if only is None or t.key == only:
            mid.tracks.append(_mido_track(t, end))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def chord_chart(arr: Arrangement) -> list[tuple[int, int, str]]:
    """``(start, dur, name)`` per chord, for the UI's chord lane."""
    return [(c.start, c.dur, c.name) for c in arr.chords]
