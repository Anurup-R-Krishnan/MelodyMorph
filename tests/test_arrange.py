import mido
import pytest

from melodymorph.arrange import (
    DRUM_CHANNEL,
    STYLES,
    TICKS_PER_STEP,
    arrange,
    lead_only,
    styles_for,
    to_midi_bytes,
    voice,
)
from melodymorph.harmony import infer_chords
from melodymorph.midi_io import parse_note_string

# four bars in C major whose notes clearly belong to C, G, Am, F; a C bar closes it
PROGRESSION = parse_note_string(
    "C4/q E4/q G4/q E4/q  B3/q D4/q G4/q D4/q  A3/q C4/q E4/q C4/q  F3/q A3/q C4/q A3/q  C4/w"
)


def _names(chords):
    return [c.name for c in chords]


def test_chords_follow_a_clear_progression():
    names = _names(infer_chords(PROGRESSION, 16))
    assert names[0] == "C" and names[-1] == "C"
    assert {"G", "Am", "F"} <= set(names)


def test_chords_tile_the_whole_melody_without_gaps():
    chords = infer_chords(PROGRESSION, 16)
    assert chords[0].start == 0
    for a, b in zip(chords, chords[1:], strict=False):
        assert a.start + a.dur == b.start
    assert chords[-1].start + chords[-1].dur == 80


def test_flat_keys_are_spelled_with_flats():
    f_major = parse_note_string("F4/q A4/q C5/q A4/q  Bb4/q D5/q F5/q D5/q  C5/q E5/q G5/q E5/q  F4/w")
    assert any("b" in n for n in _names(infer_chords(f_major, 16))) or "F" in _names(
        infer_chords(f_major, 16)
    )
    assert all("#" not in n for n in _names(infer_chords(f_major, 16)))


def test_minor_key_gets_a_minor_tonic_chord():
    a_minor = parse_note_string("A3/q C4/q E4/q C4/q  D4/q F4/q A4/q F4/q  E4/q G#4/q B4/q G#4/q  A3/w")
    chords = infer_chords(a_minor, 16)
    assert chords[0].name == "Am" and chords[-1].name == "Am"


def test_empty_melody_has_no_chords():
    assert infer_chords([], 16) == []
    with pytest.raises(ValueError):
        arrange([])


def test_voice_leading_is_smooth():
    chords = infer_chords(PROGRESSION, 16)
    prev, jumps = None, []
    for c in chords:
        v = voice(c, prev)
        if prev:
            jumps.append(sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev), strict=False)))
        prev = v
        assert all(53 <= p <= 74 for p in v)
    assert max(jumps) <= 9  # nothing leaps like root-position block chords would


@pytest.mark.parametrize("style", ["ballad", "rock", "folk"])
def test_band_has_every_part_in_a_sane_range(style):
    arr = arrange(PROGRESSION, style=style, bar=16)
    keys = [t.key for t in arr.tracks]
    assert keys[0] == "lead" and "bass" in keys and "drums" in keys and "chords" in keys
    for t in arr.tracks:
        assert t.notes, t.key
        assert (t.channel == DRUM_CHANNEL) == (t.key == "drums")
        for n in t.notes:
            assert 0 <= n.tick and n.dur > 0 and 1 <= n.vel <= 127
    assert all(36 <= n.pitch <= 59 for n in arr.track("bass").notes)
    end = arr.total_ticks
    assert all(n.tick < end for t in arr.tracks for n in t.notes)


def test_comp_notes_belong_to_the_chord_playing_at_that_moment():
    arr = arrange(PROGRESSION, style="folk", bar=16)
    by_chord = {c.start: c for c in arr.chords}
    for n in arr.track("chords").notes:
        step = n.tick // TICKS_PER_STEP
        ch = max((c for c in arr.chords if c.start <= step), key=lambda c: c.start)
        assert n.pitch % 12 in ch.pcs or abs(step - ch.start - ch.dur) <= 2, (step, by_chord)


def test_arranging_is_deterministic_and_salt_changes_the_feel():
    a = arrange(PROGRESSION, style="rock")
    b = arrange(PROGRESSION, style="rock")
    c = arrange(PROGRESSION, style="rock", salt=7)
    ticks = lambda arr: [(n.tick, n.vel) for n in arr.track("drums").notes]  # noqa: E731
    assert ticks(a) == ticks(b) and ticks(a) != ticks(c)
    assert to_midi_bytes(a) == to_midi_bytes(b)


def test_unhumanised_playing_sits_exactly_on_the_grid():
    arr = arrange(PROGRESSION, style="ballad", humanize=False)
    assert all(n.tick % TICKS_PER_STEP == 0 for n in arr.track("bass").notes)


def test_waltz_is_oom_pah_pah():
    waltz = parse_note_string("C4/q E4/q G4/q  C4/q E4/q G4/q  D4/q F4/q A4/q  C4/h.")
    arr = arrange(waltz, style="waltz", bar=12, humanize=False)
    body = arr.total_ticks - 12 * TICKS_PER_STEP  # the closing bar holds one chord on the downbeat
    bass_steps = {n.tick // TICKS_PER_STEP % 12 for n in arr.track("bass").notes if n.tick < body}
    comp_steps = {n.tick // TICKS_PER_STEP % 12 for n in arr.track("chords").notes if n.tick < body}
    assert 0 in bass_steps and 0 not in comp_steps and {4, 8} <= comp_steps
    closing = [n for n in arr.track("chords").notes if n.tick >= body]
    assert closing and all(n.tick == body for n in closing)


def test_styles_are_only_offered_for_metres_they_support():
    assert {s.key for s in styles_for(16)} == {"ballad", "rock", "folk"}
    assert {s.key for s in styles_for(12)} == {"ballad", "waltz"}
    with pytest.raises(ValueError):
        arrange(PROGRESSION, style="waltz", bar=16)
    assert set(STYLES) == {"ballad", "rock", "folk", "waltz"}


def test_midi_file_is_valid_and_carries_every_part():
    arr = arrange(PROGRESSION, style="rock", tempo=120)
    mid = mido.MidiFile(file=__import__("io").BytesIO(to_midi_bytes(arr)))
    assert mid.ticks_per_beat == 480
    assert len(mid.tracks) == 1 + len(arr.tracks)
    tempo = next(m.tempo for m in mid.tracks[0] if m.type == "set_tempo")
    assert round(mido.tempo2bpm(tempo)) == 120
    channels = {m.channel for t in mid.tracks for m in t if m.type == "note_on"}
    assert channels == {t.channel for t in arr.tracks}


def test_stems_end_at_the_same_tick_so_they_stay_aligned():
    arr = arrange(PROGRESSION, style="ballad")
    lengths = set()
    for t in arr.tracks:
        mid = mido.MidiFile(file=__import__("io").BytesIO(to_midi_bytes(arr, only=t.key)))
        assert len(mid.tracks) == 2
        lengths.add(sum(m.time for m in mid.tracks[1]))
    assert len(lengths) == 1


def test_lead_only_is_a_single_voice():
    arr = lead_only(PROGRESSION, program=73, tempo=90)
    assert [t.key for t in arr.tracks] == ["lead"] and arr.track("lead").program == 73
