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


def test_write_midi_contains_end_of_track():
    import mido
    melody = parse_note_string("C4/q E4/q")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "track.mid"
        write_midi(melody, path)
        mf = mido.MidiFile(str(path))
        # Last message of the track must be end_of_track
        assert mf.tracks[0][-1].type == "end_of_track"


def test_lossless_rest_roundtrip():
    from melodymorph.midi_io import melody_to_note_string
    # Melody with rest between note 1 and note 2
    melody = parse_note_string("C4/q R/h E4/q")
    text = melody_to_note_string(melody)
    reparsed = parse_note_string(text)
    assert [(n.onset, n.pitch, n.dur) for n in reparsed] == [(n.onset, n.pitch, n.dur) for n in melody]


def test_invalid_rest_raises():
    import pytest
    with pytest.raises(ValueError, match="invalid duration"):
        parse_note_string("C4/q R/invalid E4/q")
    with pytest.raises(ValueError, match="cannot parse note"):
        parse_note_string("re4")


def test_read_midi_keeps_a_pickup_on_its_beat():
    from melodymorph.tokenizer import Note
    melody = [Note(12, 67, 4), Note(16, 72, 4), Note(20, 74, 4)]  # one-beat pickup
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pickup.mid"
        write_midi(melody, path)
        assert [n.onset for n in read_midi(path)] == [12, 16, 20]


def test_any_duration_roundtrips_through_text():
    from melodymorph.midi_io import melody_to_note_string
    from melodymorph.tokenizer import Note
    melody = [Note(0, 60, 5), Note(5, 62, 7), Note(19, 64, 13)]  # 5, 7, rest of 7, 13 steps
    assert parse_note_string(melody_to_note_string(melody)) == melody


def test_midi_time_signature_is_detected():
    import mido

    from melodymorph.midi_io import midi_bar, midi_time_signature
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "waltz.mid"
        write_midi(parse_note_string("C4/q E4/q G4/q"), path)
        assert midi_time_signature(path) == "4/4" and midi_bar("4/4") == 16
        mf = mido.MidiFile(str(path))
        for msg in mf.tracks[0]:
            if msg.type == "time_signature":
                msg.numerator = 3
        mf.save(str(path))
        assert midi_time_signature(path) == "3/4" and midi_bar("3/4") == 12
        assert midi_bar("5/4") is None and midi_bar("3/2") is None  # 3/2 would need rescaling


def test_three_four_midi_keeps_its_bars():
    from melodymorph.tokenizer import Note
    melody = [Note(8, 67, 4), Note(12, 72, 4), Note(16, 74, 4)]  # one-beat pickup in 3/4
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "waltz.mid"
        write_midi(melody, path, bar=12)
        assert [n.onset for n in read_midi(path)] == [8, 12, 16]
