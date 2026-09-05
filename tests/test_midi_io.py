import tempfile
from pathlib import Path

from melodymorph.midi_io import parse_note_string, read_midi, write_midi


def test_parse_note_string_basic():
    melody = parse_note_string("C4/q E4/q G4/h")
    assert [n.pitch for n in melody] == [60, 64, 67]
    assert [n.dur for n in melody] == [4, 4, 8]
    assert [n.onset for n in melody] == [0, 4, 8]


def test_parse_note_string_accidentals_and_rests():
    melody = parse_note_string("C#4/e R/e Bb3/q")
    assert melody[0].pitch == 61
    assert melody[1].pitch == 58  # Bb3, after skipping the rest
    assert melody[1].onset == 4   # C#4/e (2 steps) + rest R/e (2 steps)


def test_write_and_read_midi_roundtrip():
    melody = parse_note_string("C4/q E4/q G4/h A4/e G4/e")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "seed.mid"
        write_midi(melody, path)
        assert path.exists()
        recovered = read_midi(path)
    assert [n.pitch for n in recovered] == [n.pitch for n in melody]
