"""MIDI reading/writing and human-friendly seed parsing.

MIDI is handled with ``mido`` directly.  Our internal melody form is simple
enough that a direct implementation is clearer -- and much faster -- than routing
through a full score library.
"""

from __future__ import annotations

import re
from pathlib import Path

import mido

from .tokenizer import (
    MAX_DUR,
    MAX_PITCH,
    MIN_PITCH,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
    Melody,
    Note,
    melody_duration,
    normalize,
)

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

_NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)(?:/([whqes]\.?))?$")

PRESET_SEEDS: dict[str, str] = {
    "Ode to Joy (opening)": "E4/q E4/q F4/q G4/q G4/q F4/q E4/q D4/q",
    "Rising arpeggio": "C4/e E4/e G4/e C5/e G4/q E4/q",
    "Folk phrase": "G4/q A4/e B4/e C5/q B4/e A4/e G4/h",
    "Minor motif": "A4/q C5/e B4/e A4/q E4/q",
    "Syncopated hook": "C4/e D4/e E4/q. G4/e A4/q G4/q",
}


def _dur_to_code(steps: int) -> str:
    """Best-effort duration code for a given number of 16th steps."""
    if steps in _DUR_TO_CODE:
        return _DUR_TO_CODE[steps]
    # Fallback to closest standard duration
    closest = min(_DUR_TO_CODE.keys(), key=lambda k: abs(k - steps))
    return _DUR_TO_CODE[closest]


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
            dur_str = dur_code.lower()
            if dur_str and dur_str not in _DUR_CODES:
                raise ValueError(
                    f"invalid rest duration {dur_code!r} in {token!r} -- expected w/h/q/e/s optionally dotted"
                )
            onset += _DUR_CODES.get(dur_str, 4)
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

        dur = _DUR_CODES.get((dur_code or "q").lower(), 4)
        melody.append(Note(onset, pitch, min(dur, MAX_DUR)))
        onset += dur

    if not melody:
        raise ValueError("seed contains no notes")
    return melody


def melody_to_note_string(melody: Melody) -> str:
    """Inverse of :func:`parse_note_string`, including rests so round-tripping is lossless."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
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
        name = names[note.pitch % 12] + str(note.pitch // 12 - 1)
        parts.append(f"{name}/{_dur_to_code(note.dur)}")
        current_step = note.end

    return " ".join(parts)


# --------------------------------------------------------------------------
# MIDI
# --------------------------------------------------------------------------
def write_midi(melody: Melody, path: str | Path, tempo_bpm: int = 100,
               program: int = 0) -> Path:
    """Write a melody to a single-track MIDI file conforming to SMF standard."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    midi = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm)))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
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


def read_midi(path: str | Path) -> Melody:
    """Read the melody track out of a MIDI file on the 16th grid, prioritizing lead voices."""
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

    melody = normalize(sorted(by_onset.values(), key=lambda n: n.onset))
    return _fit_range(melody)


def _fit_range(melody: Melody) -> Melody:
    """Octave-shift the melody as a whole into the supported pitch range."""
    if not melody:
        return melody
    lo = min(n.pitch for n in melody)
    hi = max(n.pitch for n in melody)
    shift = 0
    if hi - lo <= (MAX_PITCH - MIN_PITCH):
        while lo + shift < MIN_PITCH:
            shift += 12
        while hi + shift > MAX_PITCH:
            shift -= 12
    else:
        # Span exceeds total register; center the midpoint in the active range
        mid = (lo + hi) / 2.0
        target_mid = (MIN_PITCH + MAX_PITCH) / 2.0
        shift = int(round((target_mid - mid) / 12.0)) * 12

    if shift:
        melody = [Note(n.onset, n.pitch + shift, n.dur) for n in melody]
    return [n for n in melody if MIN_PITCH <= n.pitch <= MAX_PITCH]


def bars_of(melody: Melody) -> float:
    return melody_duration(melody) / STEPS_PER_BAR
