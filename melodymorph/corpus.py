"""Build a corpus of normalised monophonic melodies from the music21 bundle.

music21 ships its corpus inside the pip package, so no external download is
needed.  We prefer the Essen folksong collection (thousands of genuinely
monophonic European folk tunes -- exactly the right shape for melody
continuation) and top up with the soprano lines of the Bach chorales.

Every melody is quantised to a 16th-note grid and transposed to C major / A minor
so the model learns relative pitch relationships instead of memorising keys.

The tokenizer's ``BAR``/``POS`` tokens only mean something if they line up with
real barlines, so extraction is metre-aware. Each supported metre maps to a
16-step (duple) or 12-step (triple) bar; a pickup is placed at the end of bar 0
so the first full measure starts exactly on a ``BAR`` token. Mixed or other
metres (5/4, 7/8, changing time signatures) are skipped.
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

# metre -> (time scale, bar length in 16th steps). 2/4 and 3/8 bars are packed
# two to a model bar. 4/2, 3/2 and 6/4 are the same music as 4/4 / 3/4 notated
# at double note values; rescaling keeps one rhythm vocabulary instead of a fast
# and a slow copy.
METRES: dict[str, tuple[float, int]] = {
    "4/4": (1.0, 16), "2/4": (1.0, 16), "2/2": (1.0, 16), "4/2": (0.5, 16),
    "3/4": (1.0, 12), "6/8": (1.0, 12), "3/8": (1.0, 12), "3/2": (0.5, 12), "6/4": (0.5, 12),
}
DUPLE_METRES = {m for m, (_, bar) in METRES.items() if bar == 16}


def bar_steps(metre: str | None) -> int | None:
    """Model bar length for a time signature (None: unsupported)."""
    return METRES[metre][1] if metre in METRES else None


@dataclass
class MelodyRecord:
    id: str
    meter: str
    notes: Melody

    @property
    def bar(self) -> int:
        """Model bar length (legacy-extraction records are treated as duple)."""
        return bar_steps(self.meter) or STEPS_PER_BAR


def _quantize(offset_quarters: float) -> int:
    """Quarter-note offset -> 16th-note grid step."""
    return int(round(float(offset_quarters) * STEPS_PER_BEAT))


def _meters(part) -> list[str]:
    from music21 import meter

    return [ts.ratioString for ts in part.recurse().getElementsByClass(meter.TimeSignature)]


def _pickup_quarters(part, bar_quarters: float) -> float:
    """Length of the anacrusis in quarter notes (0 if the first measure is full)."""
    from music21 import stream

    measures = list(part.getElementsByClass(stream.Measure))
    if not measures:
        return 0
    first = float(measures[0].duration.quarterLength)
    return first if 0 < first < bar_quarters - 1e-6 else 0.0


def _part_to_melody(part, align_meter: bool = True) -> Melody | None:
    """Flatten one music21 part into monophonic notes on the 16th grid.

    With ``align_meter`` the part must be in a single duple metre, and onsets are
    shifted so the first downbeat lands on a bar boundary. Returns ``None`` for
    parts that cannot be aligned.
    """
    from music21 import chord, meter
    from music21 import note as m21note

    try:
        part = part.stripTies()
    except Exception:
        pass

    elements = []
    for element in part.flatten().notes:
        if isinstance(element, chord.Chord):
            pitch = max(p.midi for p in element.pitches)  # keep the top voice
        elif isinstance(element, m21note.Note):
            pitch = element.pitch.midi
        else:
            continue
        elements.append((float(element.offset), float(element.quarterLength), pitch))

    shift, scale, bar = 0, 1.0, STEPS_PER_BAR
    if align_meter:
        metres = set(_meters(part))
        if len(metres) != 1 or not metres <= METRES.keys():
            return None
        metre = metres.pop()
        scale, bar = METRES[metre]
        # rescale only if every onset still lands on the 16th grid
        if scale != 1.0 and any((o * STEPS_PER_BEAT * scale) % 1 for o, _, _ in elements):
            scale = 1.0
        bar_quarters = meter.TimeSignature(metre).barDuration.quarterLength
        pickup = _quantize(_pickup_quarters(part, bar_quarters) * scale)
        shift = (bar - pickup) % bar

    notes: Melody = []
    for offset, length, pitch in elements:
        dur = max(1, _quantize(length * scale))
        notes.append(Note(_quantize(offset * scale) + shift, pitch, min(dur, MAX_DUR)))

    return normalize(notes, align_bars=align_meter, bar=bar)


def _transpose_to_c(score):
    """Transpose a score so its tonic is C (major) or A (minor), within a tritone."""
    from music21 import interval
    from music21 import pitch as m21pitch

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


def _to_record(rec_id: str, part, align_meter: bool) -> MelodyRecord | None:
    metres = _meters(part)
    # cheap metre check before the expensive key analysis + transposition
    if align_meter and (len(set(metres)) != 1 or metres[0] not in METRES):
        return None
    melody = _part_to_melody(_transpose_to_c(part), align_meter=align_meter)
    melody = fit_range(melody) if melody else melody
    if not melody or len(melody) < MIN_NOTES or not in_range(melody):
        return None
    return MelodyRecord(rec_id, metres[0] if metres else "?", melody)


def _essen_file(path: str, align_meter: bool) -> tuple[list[MelodyRecord], int]:
    """Worker: all melodies in one Essen ABC collection."""
    from music21 import corpus

    try:
        opus = corpus.parse(path)
    except Exception as exc:
        log.debug("skipping %s: %s", path, exc)
        return [], 0
    # ABC collections parse into an Opus of many small scores
    scores = list(opus.scores) if hasattr(opus, "scores") else [opus]
    records, rejected = [], 0
    for i, score in enumerate(scores):
        part = score.parts[0] if len(score.parts) else score
        rec = _to_record(f"essen:{Path(path).stem}:{i}", part, align_meter)
        if rec is None:
            rejected += 1
        else:
            records.append(rec)
    return records, rejected


def _chorales(align_meter: bool) -> tuple[list[MelodyRecord], int]:
    """Worker: soprano lines of the Bach chorales."""
    from music21 import corpus

    try:
        iterator = corpus.chorales.Iterator(numberingSystem="bwv")
    except Exception as exc:
        log.warning("chorale iterator unavailable: %s", exc)
        return [], 0
    records, rejected = [], 0
    for i, score in enumerate(iterator):
        soprano = None
        for part in score.parts:
            name = (part.partName or "").lower()
            if "soprano" in name or "sopran" in name:
                soprano = part
                break
        if soprano is None and len(score.parts) > 0:
            soprano = score.parts[0]
        rec = _to_record(f"bach:{i}", soprano, align_meter) if soprano is not None else None
        if rec is None:
            rejected += 1
        else:
            records.append(rec)
    return records, rejected


def extract_records(
    limit: int | None = None, include_chorales: bool = True, align_meter: bool = True,
    workers: int | None = None,
) -> list[MelodyRecord]:
    """Extract melodies from the music21 bundle, one worker process per file."""
    from concurrent.futures import ProcessPoolExecutor

    from music21 import corpus

    paths = sorted(str(p) for p in corpus.getCorePaths() if "essenFolksong" in str(p))
    log.info("essenFolksong files found: %d", len(paths))
    records: list[MelodyRecord] = []
    rejected = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_essen_file, p, align_meter) for p in paths]
        if include_chorales:
            jobs.append(pool.submit(_chorales, align_meter))
        for job in jobs:  # in submission order, so the output is deterministic
            recs, rej = job.result()
            records.extend(recs)
            rejected += rej
    records = dedupe(records)
    log.info("extracted %d melodies (%d rejected)", len(records), rejected)
    return records[:limit] if limit is not None else records


def _content_key(notes: Melody, n: int = 16) -> tuple:
    """Transposition-invariant fingerprint of a tune's opening (intervals + rhythm)."""
    head = notes[:n]
    return tuple((b.pitch - a.pitch, b.onset - a.onset) for a, b in zip(head, head[1:]))


def dedupe(records: list[MelodyRecord]) -> list[MelodyRecord]:
    """Drop exact duplicates (same notes after key normalisation). Essen and the
    chorales both contain the same tune more than once."""
    seen: set[tuple] = set()
    out = []
    for r in records:
        key = tuple((n.onset, n.pitch, n.dur) for n in r.notes)
        if key not in seen:
            seen.add(key)
            out.append(r)
    if len(out) < len(records):
        log.info("dropped %d duplicate melodies", len(records) - len(out))
    return out


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
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if isinstance(raw, list):
                raise ValueError(
                    f"{cache} is an old-format corpus (no ids, pre metre alignment). "
                    "Rebuild it with `melodymorph prepare-data --force`."
                )
            records.append(MelodyRecord(raw["id"], raw["meter"], [Note(*t) for t in raw["notes"]]))
    return records


def load_corpus(cache: Path = DEFAULT_CACHE) -> list[Melody]:
    return [r.notes for r in load_records(cache)]


def split_records(
    records: list[MelodyRecord], val_fraction: float = 0.1,
) -> tuple[list[MelodyRecord], list[MelodyRecord]]:
    """Split by a hash of each tune's opening (transposition-invariant).

    Near-duplicates -- the same tune notated twice, or variants sharing an opening
    -- always land on the same side, so validation never scores a tune the model
    trained on. The split is also stable across rebuilds and extraction variants.
    """
    cut = int(val_fraction * 10_000)
    train, val = [], []
    for r in records:
        h = zlib.crc32(repr(_content_key(r.notes)).encode()) % 10_000
        (val if h < cut else train).append(r)
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
