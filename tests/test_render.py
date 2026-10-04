import matplotlib.pyplot as plt

from melodymorph.audio import melody_to_wave
from melodymorph.midi_io import parse_note_string
from melodymorph.viz import plot_contour_strip, plot_piano_roll, save_contour_strip


def test_audio_is_finite_bounded_and_timed():
    wave = melody_to_wave(parse_note_string("C4/q"), tempo_bpm=120)
    assert (abs(wave) <= 1.0).all()
    assert len(wave) > 0.5 * 44100  # a quarter at 120 bpm lasts 0.5 s


def test_piano_roll_has_measure_ticks():
    fig = plot_piano_roll(parse_note_string("C3/q G4/q C6/q"), theme="dark")
    assert any("m.1" in t.get_text() for t in fig.axes[0].get_xticklabels())
    plt.close(fig)


def test_contour_strip_renders_and_saves(tmp_path):
    plt.close(plot_contour_strip([], theme="dark"))
    out = tmp_path / "contour.png"
    save_contour_strip(parse_note_string("C4/q E4/q G4/h"), str(out), seed_len=4, theme="light")
    assert out.stat().st_size > 0
