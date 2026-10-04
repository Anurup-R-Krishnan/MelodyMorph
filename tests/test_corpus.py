from music21 import meter, note, stream

from melodymorph.corpus import MelodyRecord, _part_to_melody, dedupe, load_records, split_records
from melodymorph.tokenizer import Note


def _part(ts: str, pickup: list[str], bars: list[list[str]], beat: float = 1.0):
    """Build a part: an optional pickup measure, then full measures of equal notes."""
    part = stream.Part()
    measures = []
    if pickup:
        m0 = stream.Measure(number=0)
        m0.append(meter.TimeSignature(ts))
        for p in pickup:
            m0.append(note.Note(p, quarterLength=beat))
        measures.append(m0)
    for i, notes in enumerate(bars, 1):
        m = stream.Measure(number=i)
        if not measures:
            m.append(meter.TimeSignature(ts))
        for p in notes:
            m.append(note.Note(p, quarterLength=beat))
        measures.append(m)
    offset = 0.0
    for m in measures:
        part.insert(offset, m)
        offset += m.duration.quarterLength
    return part


def test_pickup_lands_before_the_first_barline():
    part = _part("4/4", ["G4"], [["C5", "D5", "E5", "F5"], ["G5", "A5", "B5", "C6"]])
    melody = _part_to_melody(part)
    assert melody[0].onset == 12      # one-beat anacrusis at the end of bar 0
    assert melody[1].onset == 16      # bar 1 downbeat on the barline
    assert melody[5].onset == 32


def test_no_pickup_starts_on_a_barline():
    melody = _part_to_melody(_part("4/4", [], [["C5", "D5", "E5", "F5"]]))
    assert melody[0].onset == 0


def test_two_four_downbeats_stay_on_eighth_bar_grid():
    part = _part("2/4", ["G4"], [["C5", "D5"], ["E5", "F5"]])
    melody = _part_to_melody(part)
    downbeats = [melody[1].onset, melody[3].onset]
    assert downbeats == [16, 24]


def test_four_two_is_rescaled_to_four_four():
    part = _part("4/2", [], [["C5", "D5", "E5", "F5"]], beat=2.0)
    melody = _part_to_melody(part)
    assert [n.dur for n in melody] == [4, 4, 4, 4]


def test_triple_metre_gets_twelve_step_bars():
    part = _part("3/4", ["G4"], [["C5", "D5", "E5"], ["F5", "G5", "A5"]])
    melody = _part_to_melody(part)
    assert [n.onset for n in melody[:5]] == [8, 12, 16, 20, 24]   # downbeats at 12 and 24


def test_unsupported_metre_is_rejected_when_aligning():
    part = _part("5/4", [], [["C5", "D5", "E5", "F5", "G5"]])
    assert _part_to_melody(part) is None
    assert _part_to_melody(part, align_meter=False) is not None


def _tune(i: int, transpose_by: int = 0, onset_shift: int = 0):
    import random

    rng = random.Random(i)
    notes, t = [], onset_shift
    for _ in range(20):
        d = rng.choice([2, 4, 8])
        notes.append(Note(t, 60 + rng.randrange(12) + transpose_by, d))
        t += d
    return notes


def test_split_is_stable_and_disjoint():
    recs = [MelodyRecord(f"essen:x:{i}", "4/4", _tune(i)) for i in range(2000)]
    train, val = split_records(recs, 0.1)
    assert {r.id for r in train}.isdisjoint({r.id for r in val})
    assert 150 < len(val) < 250
    _, val_rev = split_records(list(reversed(recs)), 0.1)
    assert {r.id for r in val} == {r.id for r in val_rev}


def test_split_keeps_near_duplicates_together():
    """A tune and its transposed / re-barred copy must not straddle the split."""
    recs = []
    for i in range(500):
        recs.append(MelodyRecord(f"a:{i}", "4/4", _tune(i)))
        recs.append(MelodyRecord(f"b:{i}", "4/4", _tune(i, transpose_by=2, onset_shift=4)))
    train, val = split_records(recs, 0.2)
    val_ids = {r.id.split(":")[1] for r in val}
    assert val_ids
    assert not val_ids & {r.id.split(":")[1] for r in train}


def test_dedupe_drops_exact_copies():
    recs = [MelodyRecord("a", "4/4", _tune(1)), MelodyRecord("b", "4/4", _tune(1)),
            MelodyRecord("c", "4/4", _tune(2))]
    assert [r.id for r in dedupe(recs)] == ["a", "c"]


def test_old_format_cache_is_refused(tmp_path):
    import pytest

    path = tmp_path / "old.jsonl"
    path.write_text("[[0, 60, 4], [4, 62, 4]]\n")
    with pytest.raises(ValueError, match="old-format"):
        load_records(path)
