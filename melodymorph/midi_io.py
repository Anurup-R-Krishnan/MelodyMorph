"""MIDI reading/writing and human-friendly seed parsing.

MIDI is handled with ``mido`` directly.  Our internal melody form is simple
enough that a direct implementation is clearer -- and much faster -- than routing
through a full score library.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mido

from .corpus import METRES
from .tokenizer import (
    MAX_DUR,
    MAX_PITCH,
    MIN_PITCH,
    STEPS_PER_BEAT,
    Melody,
    Note,
    fit_range,
    normalize,
    pitch_name,
)

log = logging.getLogger(__name__)

TICKS_PER_BEAT = 480
TICKS_PER_STEP = TICKS_PER_BEAT // STEPS_PER_BEAT  # 120

# duration letters -> 16th steps
_DUR_CODES = {
    "w": 16,   # whole
    "h": 8,    # half
    "q": 4,    # quarter
    "e": 2,    # eighth
    "s": 1,    # sixteenth
    "h.": 12,  # dotted half
    "q.": 6,   # dotted quarter
    "e.": 3,   # dotted eighth
}
_DUR_TO_CODE = {v: k for k, v in _DUR_CODES.items()}

_PITCH_CLASSES = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)(?:/([whqes]\.?|\d+))?$")

PRESET_SEEDS: dict[str, str] = {
    "Ode to Joy (opening)": "E4/q E4/q F4/q G4/q G4/q F4/q E4/q D4/q",
    "Rising arpeggio": "C4/e E4/e G4/e C5/e G4/q E4/q",
    "Folk phrase": "G4/q A4/e B4/e C5/q B4/e A4/e G4/h",
    "Minor motif": "A4/q C5/e B4/e A4/q E4/q",
    "Syncopated hook": "C4/e D4/e E4/q. G4/e A4/q G4/q",
}


def _dur_to_code(steps: int) -> str:
    """Duration code for a number of 16th steps: a letter when one exists, else
    the bare step count (e.g. ``5``), so every duration round-trips exactly."""
    return _DUR_TO_CODE.get(steps, str(steps))


def _code_to_dur(code: str | None, token: str) -> int:
    if not code:
        return 4
    code = code.lower()
    if code.isdigit():
        steps = int(code)
        if not 1 <= steps <= MAX_DUR:
            raise ValueError(f"duration {code} in {token!r} must be 1-{MAX_DUR} sixteenths")
        return steps
    if code not in _DUR_CODES:
        raise ValueError(
            f"invalid duration {code!r} in {token!r} -- expected w/h/q/e/s (optionally dotted) "
            "or a number of sixteenths"
        )
    return _DUR_CODES[code]


# --------------------------------------------------------------------------
# text seeds
# --------------------------------------------------------------------------
def parse_note_string(text: str) -> Melody:
    """Parse a seed like ``"C4/q E4/q G4/h"`` into a melody.

    Pitch is scientific notation (C4 = MIDI 60), the optional suffix after ``/``
    is the duration code (w/h/q/e/s, optionally dotted). A missing duration
    defaults to a quarter note. ``R`` or ``rest`` inserts a rest.
    """
    melody: Melody = []
    onset = 0

    for raw in text.replace(",", " ").split():
        token = raw.strip()
        if not token:
            continue

        low = token.lower()
        if low == "r" or low.startswith("r/") or low == "rest" or low.startswith("rest/"):
            _, _, dur_code = token.partition("/")
            onset += _code_to_dur(dur_code, token)
            continue

        match = _NOTE_RE.match(token)
        if not match:
            raise ValueError(
                f"cannot parse note {token!r} -- expected e.g. C4/q, F#3/e, Bb4"
            )
        letter, accidental, octave, dur_code = match.groups()
        pitch = _PITCH_CLASSES[letter.lower()] + (int(octave) + 1) * 12
        if accidental == "#":
            pitch += 1
        elif accidental == "b":
            pitch -= 1
        if not MIN_PITCH <= pitch <= MAX_PITCH:
            raise ValueError(
                f"note {token!r} (MIDI {pitch}) is outside the supported range "
                f"{MIN_PITCH}-{MAX_PITCH} (C3-C6)"
            )

        dur = _code_to_dur(dur_code, token)
        melody.append(Note(onset, pitch, min(dur, MAX_DUR)))
        onset += dur

    if not melody:
        raise ValueError("seed contains no notes")
    return melody


def melody_to_note_string(melody: Melody) -> str:
    """Inverse of :func:`parse_note_string`, including rests so round-tripping is lossless."""
    parts: list[str] = []
    current_step = 0

    for note in sorted(melody, key=lambda n: (n.onset, n.pitch)):
        if note.onset > current_step:
            # Emit rest for gap
            gap = note.onset - current_step
            while gap > 0:
                chunk = min(gap, 16)
                parts.append(f"R/{_dur_to_code(chunk)}")
                gap -= chunk
        parts.append(f"{pitch_name(note.pitch)}/{_dur_to_code(note.dur)}")
        current_step = note.end

    return " ".join(parts)


# --------------------------------------------------------------------------
# MIDI
# --------------------------------------------------------------------------
def write_midi(melody: Melody, path: str | Path, tempo_bpm: int = 100,
               program: int = 0, bar: int = 16) -> Path:
    """Write a melody to a single-track MIDI file (4/4, or 3/4 for 12-step bars)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))
    track.append(mido.MetaMessage("time_signature", numerator=bar // 4, denominator=4))
    track.append(mido.Message("program_change", program=program, time=0))

    # build absolute-time events, then convert to delta times
    events: list[tuple[int, int, int, int]] = []  # (tick, is_note_on, pitch, order)
    for note in sorted(melody, key=lambda n: n.onset):
        events.append((note.onset * TICKS_PER_STEP, 1, note.pitch, 1))
        events.append((note.end * TICKS_PER_STEP, 0, note.pitch, 0))
    # note_off before note_on at the same tick, so repeated pitches retrigger
    events.sort(key=lambda e: (e[0], e[3]))

    prev = 0
    for tick, is_on, pitch, _ in events:
        delta = tick - prev
        prev = tick
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=pitch,
                velocity=80 if is_on else 0,
                time=delta,
            )
        )

    # Required by SMF standard: terminate track explicitly
    track.append(mido.MetaMessage("end_of_track", time=0))

    midi.save(str(path))
    return path


def midi_time_signature(path: str | Path) -> str | None:
    """The first time signature in a MIDI file (``"3/4"``), or None if it has none."""
    for track in mido.MidiFile(str(path)).tracks:
        for msg in track:
            if msg.type == "time_signature":
                return f"{msg.numerator}/{msg.denominator}"
    return None


def midi_bar(time_signature: str | None) -> int | None:
    """Model bar length for a MIDI file's metre (no signature = 4/4; None = unsupported).

    Only metres that need no rescaling (4/4, 3/4, 6/8, ...) are accepted: MIDI
    ticks are read at face value, one quarter note = 4 steps.
    """
    scale, bar = METRES.get(time_signature or "4/4", (None, None))
    return bar if scale == 1.0 else None


def read_midi(path: str | Path) -> Melody:
    """Read the melody track out of a MIDI file on the 16th grid, prioritizing lead voices.

    Bars follow the file's time signature (see :func:`midi_bar`). A file in an
    unsupported metre (5/4, 7/8, 3/2, ...) is still read, as 4/4, with a warning.
    """
    ts = midi_time_signature(path)
    bar = midi_bar(ts)
    if bar is None:
        log.warning("%s is in %s, which the model does not support; reading it as 4/4", path, ts)
        bar = 16
    midi = mido.MidiFile(str(path))
    tpb = midi.ticks_per_beat or TICKS_PER_BEAT

    candidates_tracks: list[Melody] = []

    for track in midi.tracks:
        abs_tick = 0
        # Stack per pitch to handle overlapping legato and retriggers cleanly
        pending: dict[int, list[int]] = {}
        track_notes: Melody = []

        for msg in track:
            abs_tick += msg.time
            # Ignore percussion channel (channel 9 in 0-indexed, 10 in 1-indexed)
            if getattr(msg, "channel", None) == 9:
                continue

            if msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault(msg.note, []).append(abs_tick)
            elif msg.type in ("note_off", "note_on"):
                starts = pending.get(msg.note)
                if starts:
                    start = starts.pop(0)
                    onset = int(round(start / tpb * STEPS_PER_BEAT))
                    dur = max(1, int(round((abs_tick - start) / tpb * STEPS_PER_BEAT)))
                    track_notes.append(Note(onset, msg.note, min(dur, MAX_DUR)))

        if len(track_notes) >= 3:
            candidates_tracks.append(track_notes)

    if not candidates_tracks:
        # Fallback to any notes found in file
        all_notes: Melody = []
        for track in midi.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    all_notes.append(Note(int(round(abs_tick / tpb * STEPS_PER_BEAT)), msg.note, 4))
        if not all_notes:
            raise ValueError(f"no playable notes found in {path}")
        candidates_tracks.append(all_notes)

    # Select the track with the highest average pitch (standard soprano/melody heuristic)
    best_track = max(candidates_tracks, key=lambda tr: sum(n.pitch for n in tr) / len(tr))

    # Keep the top note at each onset
    by_onset: dict[int, Note] = {}
    for note in best_track:
        best = by_onset.get(note.onset)
        if best is None or note.pitch > best.pitch:
            by_onset[note.onset] = note

    # align_bars: MIDI time 0 is a barline, so a pickup keeps its metric position
    melody = normalize(sorted(by_onset.values(), key=lambda n: n.onset), align_bars=True, bar=bar)
    return fit_range(melody)
