"""Render an :class:`~melodymorph.arrange.Arrangement` to real instrument audio.

The band is played by FluidSynth through a General MIDI soundfont and encoded to
MP3 with LAME. Each instrument is rendered as its own *stem* (they all end on the
same tick, so they line up), which is what lets the app offer a mixer with mute,
solo and per-instrument volume.

Both tools are optional: :func:`available` says whether rendering can happen, and
callers fall back to the built-in additive synth when it cannot. The soundfont is
not bundled (it is tens of megabytes); ``python -m scripts.get_soundfont`` fetches
one into ``assets/soundfonts/``.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .arrange import PPQ, Arrangement, to_midi_bytes

ROOT = Path(__file__).resolve().parent.parent
SOUNDFONT_DIRS = (
    ROOT / "assets" / "soundfonts",
    Path("/usr/share/soundfonts"),
    Path("/usr/share/sounds/sf2"),
)
SAMPLE_RATE = 44100
GAIN = (
    1.2  # FluidSynth's default (0.2) leaves a lone stem ~20 dB below full scale; stems peak near -10 dB here
)
TAIL = 1.2  # seconds of ring-out kept after the last musical beat
FADE = 0.4  # seconds of fade at the very end, so a cut cymbal never clicks
TIMEOUT = 90


def find_soundfont(explicit: str | os.PathLike | None = None) -> Path | None:
    """The soundfont to use: an explicit path, ``$MELODYMORPH_SOUNDFONT``, then the
    first ``.sf2``/``.sf3`` in the project's ``assets/soundfonts`` or the system dirs."""
    for candidate in (explicit, os.environ.get("MELODYMORPH_SOUNDFONT")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    for folder in SOUNDFONT_DIRS:
        if folder.is_dir():
            found = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".sf2", ".sf3"))
            if found:
                return found[0]
    return None


def available(soundfont: Path | None = None) -> bool:
    """True when FluidSynth, LAME and a soundfont are all present."""
    return bool(shutil.which("fluidsynth") and shutil.which("lame") and (soundfont or find_soundfont()))


def midi_to_wav(midi: bytes, soundfont: Path, *, sample_rate: int = SAMPLE_RATE, gain: float = GAIN) -> bytes:
    """Play a MIDI file through the soundfont, faster than real time."""
    with tempfile.TemporaryDirectory() as tmp:
        mid, wav = Path(tmp, "in.mid"), Path(tmp, "out.wav")
        mid.write_bytes(midi)
        cmd = [
            "fluidsynth",
            "-ni",
            "-g",
            str(gain),
            "-r",
            str(sample_rate),
            "-F",
            str(wav),
            str(soundfont),
            str(mid),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=TIMEOUT)
        return wav.read_bytes()


def fit_wav(wav: bytes, seconds: float, fade: float = FADE) -> bytes:
    """Trim or zero-pad a WAV to exactly ``seconds`` and fade out the last ``fade``.

    FluidSynth renders until every voice has died, and a crash cymbal can ring for
    half a minute, so stems would otherwise differ wildly in length and drift apart
    in the mixer. Fitting them all to the musical length keeps them aligned.
    """
    with wave.open(io.BytesIO(wav)) as r:
        ch, width, rate = r.getnchannels(), r.getsampwidth(), r.getframerate()
        frames = np.frombuffer(r.readframes(r.getnframes()), dtype=np.int16).reshape(-1, ch)
    if width != 2:
        raise ValueError("expected 16-bit PCM from FluidSynth")
    n = int(round(seconds * rate))
    out = np.zeros((n, ch), dtype=np.float32)
    keep = min(n, len(frames))
    out[:keep] = frames[:keep]
    k = min(int(fade * rate), n)
    if k > 1:
        out[n - k :] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(out, -32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def duration_seconds(arr: Arrangement) -> float:
    """Musical length of the arrangement plus the ring-out tail."""
    return arr.total_ticks / PPQ * 60.0 / arr.tempo + TAIL


def wav_to_mp3(wav: bytes, bitrate: int = 96) -> bytes:
    """Encode WAV bytes to a joint-stereo MP3."""
    out = subprocess.run(
        ["lame", "--quiet", "-b", str(bitrate), "-m", "j", "-", "-"],
        input=wav,
        check=True,
        capture_output=True,
        timeout=TIMEOUT,
    )
    return out.stdout


def render_midi(
    midi: bytes, soundfont: Path | None = None, *, bitrate: int = 96, seconds: float | None = None
) -> bytes:
    """One MIDI file to MP3 bytes, fitted to ``seconds`` when given."""
    sf = soundfont or find_soundfont()
    if sf is None:
        raise FileNotFoundError("no soundfont found; run `python -m scripts.get_soundfont`")
    wav = midi_to_wav(midi, sf)
    return wav_to_mp3(fit_wav(wav, seconds) if seconds else wav, bitrate)


def render_stems(arr: Arrangement, soundfont: Path | None = None, *, bitrate: int = 96) -> dict[str, bytes]:
    """Every instrument of the arrangement as its own MP3, rendered in parallel."""
    sf = soundfont or find_soundfont()
    if sf is None:
        raise FileNotFoundError("no soundfont found; run `python -m scripts.get_soundfont`")
    with ThreadPoolExecutor(max_workers=min(len(arr.tracks), os.cpu_count() or 2)) as pool:
        secs = duration_seconds(arr)
        futures = {
            t.key: pool.submit(render_midi, to_midi_bytes(arr, only=t.key), sf, bitrate=bitrate, seconds=secs)
            for t in arr.tracks
        }
        return {key: fut.result() for key, fut in futures.items()}


def render_mix(arr: Arrangement, soundfont: Path | None = None, *, bitrate: int = 128) -> bytes:
    """The whole band as a single MP3 (for export)."""
    return render_midi(to_midi_bytes(arr), soundfont, bitrate=bitrate, seconds=duration_seconds(arr))
