import random

from melodymorph.tokenizer import MAX_PITCH, MIN_PITCH, MelodyTokenizer, Note, normalize


def test_vocab_size_matches_families():
    tok = MelodyTokenizer()
    # PAD BOS EOS BAR + 16 POS + 37 PITCH + 16 DUR
    assert tok.vocab_size == 4 + 16 + 37 + 16


def test_roundtrip_simple_melody():
    tok = MelodyTokenizer()
    melody = [Note(0, 60, 4), Note(4, 64, 4), Note(8, 67, 8)]
    ids = tok.encode(melody)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    decoded = tok.decode(ids)
    assert decoded == melody


def test_roundtrip_across_bars():
    tok = MelodyTokenizer()
    melody = [Note(0, 60, 4), Note(20, 62, 4), Note(40, 64, 16)]
    decoded = tok.decode(tok.encode(melody))
    assert decoded == melody


def test_decode_is_defensive_to_garbage():
    tok = MelodyTokenizer()
    rng = random.Random(0)
    for _ in range(20):
        garbage = [rng.randrange(tok.vocab_size) for _ in range(30)]
        melody = tok.decode(garbage)  # must not raise
        for note in melody:
            assert MIN_PITCH <= note.pitch <= MAX_PITCH
            assert note.dur >= 1


def test_clamps_out_of_range_pitch_and_duration():
    tok = MelodyTokenizer()
    melody = [Note(0, 10, 100)]  # way outside range
    decoded = tok.decode(tok.encode(melody))
    assert decoded[0].pitch == MIN_PITCH
    assert decoded[0].dur == 16


def test_normalize_shifts_to_zero_and_trims_overlap():
    melody = [Note(4, 60, 8), Note(8, 64, 4)]  # first note overlaps the second
    normed = normalize(melody)
    assert normed[0].onset == 0
    assert normed[0].dur == 4  # trimmed to not overlap the next note
