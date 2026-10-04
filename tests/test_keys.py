from melodymorph.keys import estimate_key, shift_to_training_key
from melodymorph.midi_io import parse_note_string
from melodymorph.tokenizer import transpose


def test_estimates_major_and_minor_keys():
    assert estimate_key(parse_note_string("C4/q E4/q G4/q E4/q D4/q F4/q E4/q C4/h")) == (0, "major")
    assert estimate_key(parse_note_string("A4/q C5/q E5/q C5/q B4/q D5/q C5/q A4/h")) == (9, "minor")


def test_shift_moves_any_key_to_c_major():
    c_major = parse_note_string("C4/q E4/q G4/q E4/q D4/q F4/q E4/q C4/h")
    for d in range(-5, 7):
        seed = transpose(c_major, d)
        shift = shift_to_training_key(seed)
        assert (d + shift) % 12 == 0
        assert abs(shift) <= 12
