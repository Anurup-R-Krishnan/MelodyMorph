"""REMI-style event vocabulary for monophonic melodies.

A melody is a list of ``Note`` objects placed on a 16th-note grid.  It is encoded
as a flat sequence of events::

    BOS BAR POS_0 PITCH_60 DUR_4 POS_4 PITCH_64 DUR_4 BAR POS_0 PITCH_67 DUR_8 EOS

Every note is the fixed triple ``POS_p PITCH_n DUR_d``.  Explicit ``BAR`` markers
and in-bar positions give the model the metrical structure directly, which is what
keeps generated rhythm from drifting.  The vocabulary is deliberately tiny (~73
tokens) so a small model converges quickly on a modest corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- grid constants -------------------------------------------------------
STEPS_PER_BEAT = 4          # 16th-note resolution
BEATS_PER_BAR = 4           # everything is normalised to 4/4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR  # 16

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

    def __init__(self) -> None:
        tokens: list[str] = ["PAD", "BOS", "EOS", "BAR"]
        tokens += [f"POS_{i}" for i in range(STEPS_PER_BAR)]
        tokens += [f"PITCH_{p}" for p in range(MIN_PITCH, MAX_PITCH + 1)]
        tokens += [f"DUR_{d}" for d in range(MIN_DUR, MAX_DUR + 1)]

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

    def pitch_of(self, idx: int) -> int:
        return MIN_PITCH + self.pitch_ids.index(idx)

    def dur_of(self, idx: int) -> int:
        return MIN_DUR + self.dur_ids.index(idx)

    def pos_of(self, idx: int) -> int:
        return self.pos_ids.index(idx)

    def pitch_token(self, pitch: int) -> int:
        return self.pitch_ids[pitch - MIN_PITCH]

    def dur_token(self, dur: int) -> int:
        return self.dur_ids[dur - MIN_DUR]

    def pos_token(self, pos: int) -> int:
        return self.pos_ids[pos]

    # -- encode / decode ---------------------------------------------------
    def encode(self, melody: Melody, add_special: bool = True) -> list[int]:
        """Encode a melody to token ids.

        Notes are assumed sorted by onset; overlapping notes are trimmed so the
        stream stays strictly monophonic.  Out-of-range pitches/durations are
        clamped rather than dropped, so encoding never silently loses a note.
        """
        ids: list[int] = [self.bos_id] if add_special else []
        current_bar = -1

        for note in sorted(melody, key=lambda n: (n.onset, n.pitch)):
            pitch = max(MIN_PITCH, min(MAX_PITCH, note.pitch))
            dur = max(MIN_DUR, min(MAX_DUR, note.dur))
            bar, pos = divmod(note.onset, STEPS_PER_BAR)

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

    def decode(self, ids: list[int]) -> Melody:
        """Decode token ids back to a melody.

        Defensive by design: sampled sequences can be ill-formed, so incomplete or
        out-of-order triples are skipped instead of raising.
        """
        melody: Melody = []
        bar = -1
        pos: int | None = None
        pitch: int | None = None

        for idx in ids:
            if idx == self.bar_id:
                bar += 1
                pos = pitch = None
            elif self.is_pos(idx):
                pos = self.pos_of(idx)
                pitch = None
            elif self.is_pitch(idx):
                pitch = self.pitch_of(idx) if pos is not None else None
            elif self.is_dur(idx):
                if pos is not None and pitch is not None:
                    onset = max(bar, 0) * STEPS_PER_BAR + pos
                    melody.append(Note(onset, pitch, self.dur_of(idx)))
                pos = pitch = None
            elif idx == self.eos_id:
                break
            # PAD / BOS are simply ignored

        return melody


def normalize(melody: Melody) -> Melody:
    """Shift a melody so it starts at step 0 and trim any overlaps."""
    if not melody:
        return []
    notes = sorted(melody, key=lambda n: (n.onset, n.pitch))
    offset = notes[0].onset
    out: Melody = []
    for i, note in enumerate(notes):
        dur = note.dur
        if i + 1 < len(notes):
            dur = min(dur, notes[i + 1].onset - note.onset)
        if dur >= MIN_DUR:
            out.append(Note(note.onset - offset, note.pitch, min(dur, MAX_DUR)))
    return out


def melody_duration(melody: Melody) -> int:
    """Total length of a melody in 16th steps."""
    return max((n.end for n in melody), default=0)
