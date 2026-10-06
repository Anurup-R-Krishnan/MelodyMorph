import io
import shutil
import wave

import numpy as np
import pytest

from melodymorph.arrange import arrange, lead_only
from melodymorph.midi_io import parse_note_string
from melodymorph.render import (
    TAIL,
    available,
    duration_seconds,
    find_soundfont,
    fit_wav,
    render_mix,
    render_stems,
)

MELODY = parse_note_string("C4/q E4/q G4/q E4/q  F4/q A4/q C5/q A4/q  G4/q B4/q D5/q B4/q  C5/w")


def _wav(seconds: float, rate: int = 44100, amp: int = 8000) -> bytes:
    n = int(seconds * rate)
    tone = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, n)) * amp).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.repeat(tone, 2).tobytes())
    return buf.getvalue()


def _read(wav: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav)) as r:
        return np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).reshape(-1, 2), r.getframerate()


def test_fit_wav_trims_a_long_tail_to_the_exact_length():
    frames, rate = _read(fit_wav(_wav(6.0), 2.5))
    assert len(frames) == round(2.5 * rate)


def test_fit_wav_pads_a_short_render_with_silence():
    frames, rate = _read(fit_wav(_wav(1.0), 2.0))
    assert len(frames) == 2 * rate
    assert not frames[-rate // 4 :].any()


def test_fit_wav_fades_out_so_a_cut_never_clicks():
    frames, rate = _read(fit_wav(_wav(3.0), 1.0, fade=0.4))
    assert abs(int(frames[-1, 0])) < 50 and abs(frames[: rate // 2]).max() > 4000


def test_duration_is_the_musical_length_plus_the_tail():
    arr = arrange(MELODY, style="ballad", tempo=120)
    beats = arr.total_ticks / 480
    assert duration_seconds(arr) == pytest.approx(beats * 0.5 + TAIL)


def test_no_soundfont_means_not_available(monkeypatch, tmp_path):
    monkeypatch.setattr("melodymorph.render.SOUNDFONT_DIRS", (tmp_path,))
    monkeypatch.delenv("MELODYMORPH_SOUNDFONT", raising=False)
    assert find_soundfont() is None and not available()


def test_env_var_selects_the_soundfont(monkeypatch, tmp_path):
    sf = tmp_path / "x.sf2"
    sf.write_bytes(b"RIFF")
    monkeypatch.setenv("MELODYMORPH_SOUNDFONT", str(sf))
    assert find_soundfont() == sf


needs_synth = pytest.mark.skipif(not available(), reason="needs fluidsynth, lame and a soundfont")


@needs_synth
def test_every_instrument_renders_to_audible_stems_of_identical_length():
    arr = arrange(MELODY, style="rock", tempo=120)
    stems = render_stems(arr)
    assert set(stems) == {t.key for t in arr.tracks}
    assert all(b[:3] == b"ID3" or b[0] == 0xFF for b in stems.values())
    assert len({len(b) // 1000 for b in stems.values()}) <= 3  # similar sizes: no 40 s cymbal tail


@needs_synth
@pytest.mark.skipif(not shutil.which("ffprobe"), reason="needs ffprobe")
def test_mixdown_is_a_real_mp3_of_the_right_length(tmp_path):
    import subprocess

    arr = arrange(MELODY, style="folk", tempo=96)
    path = tmp_path / "mix.mp3"
    path.write_bytes(render_mix(arr))
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(out.stdout) == pytest.approx(duration_seconds(arr), abs=0.3)


@needs_synth
def test_a_lone_lead_renders_too():
    stems = render_stems(lead_only(MELODY, program=73, tempo=100))
    assert list(stems) == ["lead"] and len(stems["lead"]) > 5000
