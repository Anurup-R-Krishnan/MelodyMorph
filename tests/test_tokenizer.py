import random

from melodymorph.tokenizer import MAX_PITCH, MIN_PITCH, MelodyTokenizer, Note, normalize


def test_vocab_size_matches_families():
    tok = MelodyTokenizer()
    # PAD BOS EOS BAR + 16 POS + 37 PITCH + 16 DUR + 2 TS
    assert tok.vocab_size == 4 + 16 + 37 + 16 + 2
    # the legacy vocabulary is an exact prefix, so old checkpoints keep their token ids
    legacy = MelodyTokenizer(meter_tokens=False)
    assert tok.itos[: legacy.vocab_size] == legacy.itos


def test_triple_bars_roundtrip_with_time_signature_token():
    tok = MelodyTokenizer()
    melody = [Note(8, 60, 4), Note(12, 64, 4), Note(24, 67, 12)]
    ids = tok.encode(melody, bar=12)
    assert ids[1] == tok.ts_ids[12]
    assert ids.count(tok.bar_id) == 3
    assert tok.decode(ids) == melody             # the TS token tells decode the bar length
    assert tok.decode(tok.encode(melody)) == melody


def test_legacy_tokenizer_refuses_triple_bars():
    import pytest
    with pytest.raises(ValueError, match="bar length"):
        MelodyTokenizer(meter_tokens=False).encode([Note(0, 60, 4)], bar=12)


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


def test_bar_zero_collision_prevention():
    tok = MelodyTokenizer()
    # Sequence with note before first BAR, followed by BAR, then second note
    tokens = [tok.pos_token(4), tok.pitch_token(60), tok.dur_token(4),
              tok.bar_id,
              tok.pos_token(4), tok.pitch_token(64), tok.dur_token(4)]
    melody = tok.decode(tokens)
    assert len(melody) == 2
    # Note 1 is at bar 0 step 4; Note 2 is at bar 1 step 4 (onset 20). They must NOT collide!
    assert melody[0].onset == 4
    assert melody[1].onset == 20


def test_normalize_chords_preserves_highest_voice():
    # Three simultaneous notes at onset 8
    notes = [Note(8, 55, 4), Note(8, 67, 4), Note(8, 60, 4)]
    normed = normalize(notes)
    assert len(normed) == 1
    assert normed[0].pitch == 67  # highest pitch wins


def test_fast_o1_lookup_consistency():
    tok = MelodyTokenizer()
    for p in range(MIN_PITCH, MAX_PITCH + 1):
        token_id = tok.pitch_token(p)
        assert tok.pitch_of(token_id) == p
    for d in range(1, 17):
        token_id = tok.dur_token(d)
        assert tok.dur_of(token_id) == d
