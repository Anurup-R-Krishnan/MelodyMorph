"""Build a corpus of normalised monophonic melodies from the music21 bundle.

music21 ships its corpus inside the pip package, so no external download is
needed.  We prefer the Essen folksong collection (thousands of genuinely
monophonic European folk tunes -- exactly the right shape for melody
continuation) and top up with the soprano lines of the Bach chorales.

Every melody is quantised to a 16th-note grid and transposed to C major / A minor
so the model learns relative pitch relationships instead of memorising keys.

The tokenizer's ``BAR``/``POS`` tokens only mean something if they line up with
real barlines, so extraction is metre-aware: only duple metres (which tile a
16-step bar) are kept, and a pickup is placed at the end of bar 0 so the first
full measure starts exactly on a ``BAR`` token.
"""

from __future__ import annotations

import json
import logging
import zlib
from dataclasses import dataclass
from pathlib import Path

from .tokenizer import (
    MAX_DUR,
    STEPS_PER_BAR,
    STEPS_PER_BEAT,
    Melody,
    Note,
    fit_range,
    in_range,
    normalize,
)

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/melodies.jsonl")
MIN_NOTES = 16

# metres whose bar length divides (or is a multiple of) the 16-step bar
DUPLE_METRES = {"2/4", "4/4", "2/2", "4/2"}


@dataclass
class MelodyRecord:
    id: str
    meter: str
    notes: Melody


def _quantize(offset_quarters: float) -> int:
    """Quarter-note offset -> 16th-note grid step."""
    return int(round(float(offset_quarters) * STEPS_PER_BEAT))


def _meters(part) -> list[str]:
    from music21 import meter

    return [ts.ratioString for ts in part.recurse().getElementsByClass(meter.TimeSignature)]


def _pickup_steps(part, bar_quarters: float) -> int:
    """Length of the anacrusis in 16th steps (0 if the first measure is full)."""
    from music21 import stream

    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        return 0
    first = float(measures[0].duration.quarterLength)
    if 0 < first < bar_quarters - 1e-6:
        return _quantize(first)
    return 0


def _part_to_melody(part, align_meter: bool = True) -> Melody | None:
    """Flatten one music21 part into monophonic notes on the 16th grid.

    With ``align_meter`` the part must be in a single duple metre, and onsets are
    shifted so the first downbeat lands on a bar boundary. Returns ``None`` for
    parts that cannot be aligned.
    """
    from music21 import chord, meter, note as m21note

    try:
        part = part.stripTies()
    except Exception:
        pass

    shift = 0
    if align_meter:
        metres = set(_meters(part))
        if len(metres) != 1 or not metres <= DUPLE_METRES:
            return None
        bar_quarters = meter.TimeSignature(metres.pop()).barDuration.quarterLength
        pickup = _pickup_steps(part, bar_quarters)
        shift = (STEPS_PER_BAR - pickup) % STEPS_PER_BAR

    notes: Melody = []
    for element in part.flatten().notes:
        if isinstance(element, chord.Chord):
            pitch = max(p.midi for p in element.pitches)  # keep the top voice
        elif isinstance(element, m21note.Note):
            pitch = element.pitch.midi
        else:
            continue
        dur = max(1, _quantize(element.quarterLength))
        notes.append(Note(_quantize(element.offset) + shift, pitch, min(dur, MAX_DUR)))

    return normalize(notes, align_bars=align_meter)


def _transpose_to_c(score):
    """Transpose a score so its tonic is C (major) or A (minor), within a tritone."""
    from music21 import interval, pitch as m21pitch

    try:
        key = score.analyze("key")
    except Exception:  # analysis can fail on degenerate scores
        return score

    target = "C4" if key.mode == "major" else "A4"
    try:
        tonic = key.tonic
        if tonic.octave is None:
            tonic.octave = 4
        semi = interval.Interval(tonic, m21pitch.Pitch(target)).semitones
        while semi > 6:
            semi -= 12
        while semi < -6:
            semi += 12
        return score.transpose(interval.Interval(semi))
    except Exception:
        return score


def _iter_essen():
    from music21 import corpus

    paths = [p for p in corpus.getCorePaths() if "essenFolksong" in str(p)]
    log.info("essenFolksong files found: %d", len(paths))
    for path in paths:
        try:
            opus = corpus.parse(path)
        except Exception as exc:
            log.debug("skipping %s: %s", path, exc)
            continue
        # ABC collections parse into an Opus of many small scores
        scores = list(opus.scores) if hasattr(opus, "scores") else [opus]
        for i, score in enumerate(scores):
            part = score.parts[0] if len(score.parts) else score
            yield f"essen:{Path(path).stem}:{i}", part


def _iter_chorales():
    from music21 import corpus

    try:
        iterator = corpus.chorales.Iterator(numberingSystem="bwv")
    except Exception as exc:
        log.warning("chorale iterator unavailable: %s", exc)
        return
    for i, score in enumerate(iterator):
        soprano = None
        for part in score.parts:
            name = (part.partName or "").lower()
            if "soprano" in name or "sopran" in name:
                soprano = part
                break
        if soprano is None and len(score.parts) > 0:
            soprano = score.parts[0]
        if soprano is not None:
            yield f"bach:{i}", soprano


def extract_records(
    limit: int | None = None, include_chorales: bool = True, align_meter: bool = True,
) -> list[MelodyRecord]:
    """Extract melodies from the music21 bundle."""
    records: list[MelodyRecord] = []
    rejected = 0
    sources = [_iter_essen()] + ([_iter_chorales()] if include_chorales else [])
    for source in sources:
        for rec_id, part in source:
            metres = _meters(part)
            melody = _part_to_melody(_transpose_to_c(part), align_meter=align_meter)
            melody = fit_range(melody) if melody else melody
            if not melody or len(melody) < MIN_NOTES or not in_range(melody):
                rejected += 1
                continue
            records.append(MelodyRecord(rec_id, metres[0] if metres else "?", melody))
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break
    log.info("extracted %d melodies (%d rejected)", len(records), rejected)
    return records


def build_corpus(
    cache: Path = DEFAULT_CACHE,
    limit: int | None = None,
    include_chorales: bool = True,
    force: bool = False,
    align_meter: bool = True,
) -> list[MelodyRecord]:
    """Extract melodies and cache them as JSONL (one record per line)."""
    cache = Path(cache)
    if cache.exists() and not force:
        log.info("using cached corpus at %s", cache)
        return load_records(cache)

    records = extract_records(limit, include_chorales, align_meter)
    if not records:
        raise RuntimeError(
            "No melodies could be extracted from the music21 corpus. "
            "Check that music21 is installed with its bundled corpus."
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        for r in records:
            notes = [[n.onset, n.pitch, n.dur] for n in r.notes]
            fh.write(json.dumps({"id": r.id, "meter": r.meter, "notes": notes}) + "\n")
    log.info("wrote %d melodies to %s", len(records), cache)
    return records


def load_records(cache: Path = DEFAULT_CACHE) -> list[MelodyRecord]:
    cache = Path(cache)
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found -- run `melodymorph prepare-data` first.")
    records: list[MelodyRecord] = []
    with cache.open() as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if isinstance(raw, list):  # legacy cache: bare note triples, no id
                raw = {"id": f"legacy:{i}", "meter": "?", "notes": raw}
            records.append(MelodyRecord(raw["id"], raw["meter"], [Note(*t) for t in raw["notes"]]))
    return records


def load_corpus(cache: Path = DEFAULT_CACHE) -> list[Melody]:
    return [r.notes for r in load_records(cache)]


def split_records(
    records: list[MelodyRecord], val_fraction: float = 0.1,
) -> tuple[list[MelodyRecord], list[MelodyRecord]]:
    """Split by a hash of the melody id: stable across corpus rebuilds and
    extraction variants, so the same tune is always on the same side."""
    cut = int(val_fraction * 10_000)
    train, val = [], []
    for r in records:
        (val if zlib.crc32(r.id.encode()) % 10_000 < cut else train).append(r)
    return train, val


def corpus_stats(melodies: list[Melody]) -> dict:
    from .tokenizer import melody_duration

    note_counts = [len(m) for m in melodies]
    return {
        "melodies": len(melodies),
        "total_notes": sum(note_counts),
        "mean_notes": sum(note_counts) / max(len(note_counts), 1),
        "min_notes": min(note_counts, default=0),
        "max_notes": max(note_counts, default=0),
        "mean_bars": sum(melody_duration(m) for m in melodies) / max(len(melodies), 1) / STEPS_PER_BAR,
    }
