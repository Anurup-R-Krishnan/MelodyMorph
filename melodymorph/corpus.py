"""Build a corpus of normalised monophonic melodies from the music21 bundle.

music21 ships its corpus inside the pip package, so no external download is
needed.  We prefer the Essen folksong collection (thousands of genuinely
monophonic European folk tunes -- exactly the right shape for melody
continuation) and top up with the soprano lines of the Bach chorales.

Every melody is quantised to a 16th-note grid and transposed to C major / A minor
so the model learns relative pitch relationships instead of memorising keys.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .tokenizer import (
    MAX_DUR,
    MAX_PITCH,
    MIN_PITCH,
    STEPS_PER_BEAT,
    Melody,
    Note,
    normalize,
)

log = logging.getLogger(__name__)

DEFAULT_CACHE = Path("data/melodies.jsonl")
MIN_NOTES = 16


def _quantize(offset_quarters: float) -> int:
    """Quarter-note offset -> 16th-note grid step."""
    return int(round(float(offset_quarters) * STEPS_PER_BEAT))


def _part_to_melody(part) -> Melody:
    """Flatten one music21 part into monophonic notes on the 16th grid."""
    from music21 import chord, note as m21note

    notes: Melody = []
    for element in part.flatten().notes:
        if isinstance(element, chord.Chord):
            pitch = max(p.midi for p in element.pitches)  # keep the top voice
        elif isinstance(element, m21note.Note):
            pitch = element.pitch.midi
        else:
            continue

        onset = _quantize(element.offset)
        dur = _quantize(element.quarterLength)
        if dur <= 0:
            dur = 1
        notes.append(Note(onset, pitch, min(dur, MAX_DUR)))

    return normalize(notes)


def _transpose_to_c(score):
    """Transpose a score so its tonic is C (major) or A (minor)."""
    from music21 import interval, pitch as m21pitch

    try:
        key = score.analyze("key")
    except Exception:  # analysis can fail on degenerate scores
        return score

    target = "C" if key.mode == "major" else "A"
    try:
        step = interval.Interval(key.tonic, m21pitch.Pitch(target))
        return score.transpose(step)
    except Exception:
        return score


def _melody_ok(melody: Melody) -> bool:
    if len(melody) < MIN_NOTES:
        return False
    return all(MIN_PITCH <= n.pitch <= MAX_PITCH for n in melody)


def _fit_range(melody: Melody) -> Melody:
    """Octave-shift a melody as a whole to fit the model's pitch range."""
    if not melody:
        return melody
    lo = min(n.pitch for n in melody)
    hi = max(n.pitch for n in melody)
    shift = 0
    while lo + shift < MIN_PITCH and hi + shift + 12 <= MAX_PITCH:
        shift += 12
    while hi + shift > MAX_PITCH and lo + shift - 12 >= MIN_PITCH:
        shift -= 12
    if shift == 0:
        return melody
    return [Note(n.onset, n.pitch + shift, n.dur) for n in melody]


def _iter_essen(limit: int | None):
    from music21 import corpus

    paths = [p for p in corpus.getCorePaths() if "essenFolksong" in str(p)]
    log.info("essenFolksong files found: %d", len(paths))
    count = 0
    for path in paths:
        try:
            opus = corpus.parse(path)
        except Exception as exc:
            log.debug("skipping %s: %s", path, exc)
            continue
        # ABC collections parse into an Opus of many small scores
        scores = list(opus.scores) if hasattr(opus, "scores") else [opus]
        for score in scores:
            yield score
            count += 1
            if limit is not None and count >= limit:
                return


def _iter_chorales(limit: int | None):
    from music21 import corpus

    count = 0
    try:
        iterator = corpus.chorales.Iterator(numberingSystem="bwv")
    except Exception as exc:
        log.warning("chorale iterator unavailable: %s", exc)
        return
    for score in iterator:
        soprano = None
        for part in score.parts:
            name = (part.partName or "").lower()
            if "soprano" in name:
                soprano = part
                break
        if soprano is None and len(score.parts) > 0:
            soprano = score.parts[0]
        if soprano is None:
            continue
        yield soprano
        count += 1
        if limit is not None and count >= limit:
            return


def build_corpus(
    cache: Path = DEFAULT_CACHE,
    limit: int | None = None,
    include_chorales: bool = True,
    force: bool = False,
) -> list[Melody]:
    """Extract melodies from the music21 bundle and cache them as JSONL."""
    cache = Path(cache)
    if cache.exists() and not force:
        log.info("using cached corpus at %s", cache)
        return load_corpus(cache)

    melodies: list[Melody] = []

    for score in _iter_essen(limit):
        melody = _fit_range(_part_to_melody(_transpose_to_c(score)))
        if _melody_ok(melody):
            melodies.append(melody)

    log.info("melodies from essenFolksong: %d", len(melodies))

    if include_chorales and (limit is None or len(melodies) < limit):
        remaining = None if limit is None else limit - len(melodies)
        before = len(melodies)
        for part in _iter_chorales(remaining):
            melody = _fit_range(_part_to_melody(_transpose_to_c(part)))
            if _melody_ok(melody):
                melodies.append(melody)
        log.info("melodies from Bach chorale sopranos: %d", len(melodies) - before)

    if not melodies:
        raise RuntimeError(
            "No melodies could be extracted from the music21 corpus. "
            "Check that music21 is installed with its bundled corpus."
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        for melody in melodies:
            fh.write(json.dumps([[n.onset, n.pitch, n.dur] for n in melody]) + "\n")

    log.info("wrote %d melodies to %s", len(melodies), cache)
    return melodies


def load_corpus(cache: Path = DEFAULT_CACHE) -> list[Melody]:
    cache = Path(cache)
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} not found -- run `melodymorph prepare-data` first."
        )
    melodies: list[Melody] = []
    with cache.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                melodies.append([Note(*triple) for triple in json.loads(line)])
    return melodies


def corpus_stats(melodies: list[Melody]) -> dict:
    from .tokenizer import melody_duration

    note_counts = [len(m) for m in melodies]
    return {
        "melodies": len(melodies),
        "total_notes": sum(note_counts),
        "mean_notes": sum(note_counts) / max(len(note_counts), 1),
        "min_notes": min(note_counts, default=0),
        "max_notes": max(note_counts, default=0),
        "mean_bars": sum(melody_duration(m) for m in melodies)
        / max(len(melodies), 1)
        / 16,
    }
