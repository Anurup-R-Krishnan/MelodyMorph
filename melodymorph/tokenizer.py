"""REMI-style event vocabulary for monophonic melodies.

A melody is a list of ``Note`` objects placed on a 16th-note grid.  It is encoded
as a flat sequence of events::

    BOS TS_16 BAR POS_0 PITCH_60 DUR_4 POS_4 PITCH_64 DUR_4 BAR POS_0 PITCH_67 DUR_8 EOS

Every note is the fixed triple ``POS_p PITCH_n DUR_d``.  Explicit ``BAR`` markers
and in-bar positions give the model the metrical structure directly, which is what
keeps generated rhythm from drifting.

The time-signature token says how long a bar is: ``TS_16`` (duple: 4/4, 2/4, ...)
or ``TS_12`` (triple: 3/4, 6/8, ...). The vocabulary is deliberately tiny
(75 tokens) so a small model converges quickly on a modest corpus. Checkpoints
from before the TS tokens existed use the first 73 tokens and duple bars only.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- grid constants -------------------------------------------------------
STEPS_PER_BEAT = 4          # 16th-note resolution
STEPS_PER_BAR = 16          # default (duple) bar; also the POS vocabulary size
BAR_LENGTHS = (16, 12)      # duple and triple bars, in 16th steps

MIN_PITCH = 48              # C3
MAX_PITCH = 84              # C6
MIN_DUR = 1                 # one 16th
MAX_DUR = 16                # one bar


@dataclass(frozen=True)
class Note:
    """A single melody note. ``onset`` and ``dur`` are in 16th-note steps."""

    onset: int
    pitch: int
    dur: int

    @property
    def end(self) -> int:
        return self.onset + self.dur


Melody = list[Note]


class MelodyTokenizer:
    """Bidirectional map between melodies and token id sequences."""

    def __init__(self, meter_tokens: bool = True) -> None:
        tokens: list[str] = ["PAD", "BOS", "EOS", "BAR"]
        tokens += [f"POS_{i}" for i in range(STEPS_PER_BAR)]
        tokens += [f"PITCH_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
        tokens += [f"DUR_{d}" for d in range(MIN_DUR, MAX_DUR + 1)]
        # appended last so token ids of older (73-token) checkpoints are unchanged
        self.meter_tokens = meter_tokens
        if meter_tokens:
            tokens += [f"TS_{b}" for b in BAR_LENGTHS]

        self.itos: list[str] = tokens
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(tokens)}

        self.pad_id = self.stoi["PAD"]
        self.bos_id = self.stoi["BOS"]
        self.eos_id = self.stoi["EOS"]
        self.bar_id = self.stoi["BAR"]

        self.pos_ids = [self.stoi[f"POS_{i}"] for i in range(STEPS_PER_BAR)]
        self.pitch_ids = [self.stoi[f"PITCH_{p}"] for p in range(MIN_PITCH, MAX_PITCH + 1)]
        self.dur_ids = [self.stoi[f"DUR_{d}"] for d in range(MIN_DUR, MAX_DUR + 1)]

        self._pos_set = set(self.pos_ids)
        self._pitch_set = set(self.pitch_ids)
        self._dur_set = set(self.dur_ids)

        self.ts_ids: dict[int, int] = {b: self.stoi[f"TS_{b}"] for b in BAR_LENGTHS} if meter_tokens else {}
        self._ts_bar: dict[int, int] = {idx: b for b, idx in self.ts_ids.items()}

        self._pos_map: dict[int, int] = {idx: i for i, idx in enumerate(self.pos_ids)}
        self._pitch_map: dict[int, int] = {idx: MIN_PITCH + i for i, idx in enumerate(self.pitch_ids)}
        self._dur_map: dict[int, int] = {idx: MIN_DUR + i for i, idx in enumerate(self.dur_ids)}

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.itos)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def token(self, idx: int) -> str:
        return self.itos[idx]

    def id(self, token: str) -> int:
        return self.stoi[token]

    # -- token family predicates ------------------------------------------
    def is_pos(self, idx: int) -> bool:
        return idx in self._pos_set

    def is_pitch(self, idx: int) -> bool:
        return idx in self._pitch_set

    def is_dur(self, idx: int) -> bool:
        return idx in self._dur_set

    def is_ts(self, idx: int) -> bool:
        return idx in self._ts_bar

    def bar_of_ts(self, idx: int) -> int:
        return self._ts_bar[idx]

    def pitch_of(self, idx: int) -> int:
        return self._pitch_map[idx]

    def dur_of(self, idx: int) -> int:
        return self._dur_map[idx]

    def pos_of(self, idx: int) -> int:
        return self._pos_map[idx]

    def pitch_token(self, pitch: int) -> int:
        return self.pitch_ids[pitch - MIN_PITCH]

    def dur_token(self, dur: int) -> int:
        return self.dur_ids[dur - MIN_DUR]

    def pos_token(self, pos: int) -> int:
        return self.pos_ids[pos]

    # -- encode / decode ---------------------------------------------------
    def encode(self, melody: Melody, add_special: bool = True, bar: int = STEPS_PER_BAR) -> list[int]:
        """Encode a melody to token ids, with bars of ``bar`` 16th steps.

        Notes are assumed sorted by onset; overlapping notes are trimmed so the
        stream stays strictly monophonic. Out-of-range pitches/durations are
        clamped rather than dropped, so encoding never silently loses a note.
        """
        if bar not in BAR_LENGTHS or (bar != STEPS_PER_BAR and not self.meter_tokens):
            raise ValueError(f"unsupported bar length {bar} for this tokenizer")
        ids: list[int] = [self.bos_id] if add_special else []
        if add_special and self.meter_tokens:
            ids.append(self.ts_ids[bar])
        bar_len = bar
        current_bar = -1

        for note in sorted(melody, key=lambda n: (n.onset, n.pitch)):
            pitch = max(MIN_PITCH, min(MAX_PITCH, note.pitch))
            dur = max(MIN_DUR, min(MAX_DUR, note.dur))
            bar, pos = divmod(note.onset, bar_len)

            # emit bar lines for every bar we crossed, including empty ones
            while current_bar < bar:
                ids.append(self.bar_id)
                current_bar += 1

            ids.append(self.pos_token(pos))
            ids.append(self.pitch_token(pitch))
            ids.append(self.dur_token(dur))

        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], bar: int = STEPS_PER_BAR) -> Melody:
        """Decode token ids back to a melody (a ``TS_*`` token overrides ``bar``).

        Defensive by design: sampled sequences can be ill-formed, so incomplete or
        out-of-order triples are skipped instead of raising.
        Correctly prevents bar 0 collisions when notes precede or follow the first BAR marker.
        """
        melody: Melody = []
        bar_len = bar
        bar = 0
        seen_bar = False
        pos: int | None = None
        pitch: int | None = None

        for idx in ids:
            if idx in self._ts_bar:
                bar_len = self._ts_bar[idx]
            elif idx == self.bar_id:
                if seen_bar or len(melody) > 0:
                    bar += 1
                else:
                    bar = 0
                seen_bar = True
                pos = pitch = None
            elif self.is_pos(idx):
                pos = self.pos_of(idx)
                pitch = None
            elif self.is_pitch(idx):
                pitch = self.pitch_of(idx) if pos is not None else None
            elif self.is_dur(idx):
                if pos is not None and pitch is not None:
                    onset = bar * bar_len + pos
                    melody.append(Note(onset, pitch, self.dur_of(idx)))
                pos = pitch = None
            elif idx == self.eos_id:
                break
            # PAD / BOS are simply ignored

        return melody


def normalize(melody: Melody, align_bars: bool = False, bar: int = STEPS_PER_BAR) -> Melody:
    """Shift a melody so it starts at step 0 (or bar boundary if align_bars) and trim any overlaps."""
    if not melody:
        return []
    # If there are simultaneous notes (e.g. from chord or polyphonic MIDI), keep the highest pitch
    by_onset: dict[int, Note] = {}
    for n in sorted(melody, key=lambda x: (x.onset, x.pitch)):
        by_onset[n.onset] = n  # highest pitch wins
    notes = sorted(by_onset.values(), key=lambda n: n.onset)

    if align_bars:
        offset = (notes[0].onset // bar) * bar
    else:
        offset = notes[0].onset

    out: Melody = []
    for i, note in enumerate(notes):
        dur = note.dur
        if i + 1 < len(notes):
            dur = min(dur, notes[i + 1].onset - note.onset)
        dur = max(MIN_DUR, min(dur, MAX_DUR))
        out.append(Note(note.onset - offset, note.pitch, dur))
    return out


def melody_duration(melody: Melody) -> int:
    """Total length of a melody in 16th steps."""
    return max((n.end for n in melody), default=0)


def transpose(melody: Melody, semitones: int) -> Melody:
    return [Note(n.onset, n.pitch + semitones, n.dur) for n in melody]


def in_range(melody: Melody) -> bool:
    return all(MIN_PITCH <= n.pitch <= MAX_PITCH for n in melody)


def fit_range(melody: Melody) -> Melody:
    """Octave-shift a melody as a whole into the model's pitch range.

    A melody whose span exceeds the range is centred, and the notes that still
    fall outside are dropped.
    """
    if not melody:
        return melody
    lo = min(n.pitch for n in melody)
    hi = max(n.pitch for n in melody)
    shift = 0
    if hi - lo <= MAX_PITCH - MIN_PITCH:
        while lo + shift < MIN_PITCH:
            shift += 12
        while hi + shift > MAX_PITCH:
            shift -= 12
    else:
        mid, target_mid = (lo + hi) / 2.0, (MIN_PITCH + MAX_PITCH) / 2.0
        shift = int(round((target_mid - mid) / 12.0)) * 12
    return [n for n in transpose(melody, shift) if MIN_PITCH <= n.pitch <= MAX_PITCH]


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(pitch: int) -> str:
    """Scientific pitch name, C4 = MIDI 60."""
    return f"{PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}"
