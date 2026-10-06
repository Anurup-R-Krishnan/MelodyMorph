import pytest

from melodymorph.harmony import infer_chords, infer_key_and_chords
from melodymorph.keys import estimate_key, shift_to_training_key
from melodymorph.midi_io import PRESET_SEEDS, parse_note_string
from melodymorph.tokenizer import transpose

# well-known tunes (and the app's presets) with their true keys: (tonic pitch class, mode, notes)
TUNES = {
    "Ode to Joy": (0, "major", "E4/q E4/q F4/q G4/q G4/q F4/q E4/q D4/q C4/q C4/q D4/q E4/q E4/q. D4/e D4/h"),
    "Twinkle": (0, "major", "C4/q C4/q G4/q G4/q A4/q A4/q G4/h F4/q F4/q E4/q E4/q D4/q D4/q C4/h"),
    "Frere Jacques": (0, "major", "C4/q D4/q E4/q C4/q C4/q D4/q E4/q C4/q E4/q F4/q G4/h E4/q F4/q G4/h"),
    "Happy Birthday": (0, "major", "G4/e G4/e A4/q G4/q C5/q B4/h G4/e G4/e A4/q G4/q D5/q C5/h"),
    "When the Saints": (
        0,
        "major",
        "C4/e E4/e F4/e G4/h C4/e E4/e F4/e G4/h C4/e E4/e F4/e G4/q E4/q C4/q E4/q D4/h",
    ),
    "Amazing Grace": (7, "major", "D4/q G4/h B4/e G4/e B4/h A4/q G4/h E4/q D4/h"),
    "Greensleeves": (
        9,
        "minor",
        "A3/q C4/h D4/q E4/q. F4/e E4/q D4/q B3/q. G3/e A3/q B3/q C4/q A3/q A3/e G#3/e A3/q B3/q G#3/q E3/h",
    ),
    "preset: Rising arpeggio": (0, "major", PRESET_SEEDS["Rising arpeggio"]),
    "preset: Folk phrase": (7, "major", PRESET_SEEDS["Folk phrase"]),
    "preset: Minor motif": (9, "minor", PRESET_SEEDS["Minor motif"]),
    "preset: Ode to Joy": (0, "major", PRESET_SEEDS["Ode to Joy (opening)"]),
    "preset: Syncopated hook": (0, "major", PRESET_SEEDS["Syncopated hook"]),
}


@pytest.mark.parametrize("name", TUNES)
def test_harmonic_key_finder_names_the_true_key(name):
    tonic, mode, notes = TUNES[name]
    assert infer_key_and_chords(parse_note_string(notes), 16)[0] == (tonic, mode)


def test_profile_alone_confuses_short_tunes_with_their_relative_minor():
    """Why the key is found jointly with the chords: the profile alone calls a
    plain C-major arpeggio E minor."""
    arpeggio = parse_note_string(PRESET_SEEDS["Rising arpeggio"])
    assert estimate_key(arpeggio) == (4, "minor")
    assert infer_key_and_chords(arpeggio, 16)[0] == (0, "major")


@pytest.mark.parametrize("name", ["Twinkle", "Happy Birthday", "preset: Ode to Joy", "Amazing Grace"])
def test_key_normalisation_brings_any_transposition_back_to_the_training_key(name):
    tonic, mode, notes = TUNES[name]
    tune = parse_note_string(notes)
    for d in (-5, -2, 3, 5):
        moved = transpose(tune, d)
        # the tune lands in the training key (C major / A minor), so its tonic ends up on C (or A)
        assert (tonic + d + shift_to_training_key(moved)) % 12 == (0 if mode == "major" else 9)


def test_chords_use_the_key_they_were_found_in():
    names = [c.name for c in infer_chords(parse_note_string(TUNES["Twinkle"][2]), 16)]
    assert names[0] == "C" and names[-1] == "C" and "F" in names and "G" in names
