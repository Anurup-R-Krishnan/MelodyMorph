"""Pure-numpy synthesis so melodies play back without a soundfont dependency."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .tokenizer import STEPS_PER_BEAT, Melody

SAMPLE_RATE = 44100


def _adsr_envelope(n_samples: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float64)
    if n_samples == 1:
        return np.ones(1, dtype=np.float64)

    # Proportional allocation for short notes, absolute for longer notes
    a_len = min(int(0.015 * sr), max(1, n_samples // 6))
    d_len = min(int(0.03 * sr), max(1, n_samples // 6))
    r_len = min(int(0.05 * sr), max(1, n_samples // 4))

    # If combined lengths exceed n_samples, scale them down proportionally
    total = a_len + d_len + r_len
    if total > n_samples:
        scale = n_samples / total
        a_len = max(1, int(a_len * scale))
        d_len = max(1, int(d_len * scale))
        r_len = max(1, n_samples - a_len - d_len)
        sustain_len = 0
    else:
        sustain_len = n_samples - (a_len + d_len + r_len)

    sustain_level = 0.75

    attack = np.linspace(0.0, 1.0, a_len, endpoint=False)
    decay = np.linspace(1.0, sustain_level, d_len, endpoint=False)
    sustain = np.full(sustain_len, sustain_level)
    release = np.linspace(sustain_level, 0.0, r_len)

    env = np.concatenate([attack, decay, sustain, release])
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples]


def _note_wave(freq: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = max(int(duration_s * sr), 1)
    t = np.arange(n) / sr
    # Richer harmonic series with natural roll-off
    wave_ = (
        1.00 * np.sin(2 * np.pi * freq * t)
        + 0.40 * np.sin(2 * np.pi * 2 * freq * t)
        + 0.18 * np.sin(2 * np.pi * 3 * freq * t)
        + 0.08 * np.sin(2 * np.pi * 4 * freq * t)
    )
    wave_ *= _adsr_envelope(n, sr)
    return wave_


def melody_to_wave(melody: Melody, tempo_bpm: int = 100, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Render a melody to a mono float32 waveform in [-1, 1]."""
    if not melody:
        return np.zeros(1, dtype=np.float32)

    seconds_per_step = 60.0 / tempo_bpm / STEPS_PER_BEAT
    total_steps = max(n.onset + n.dur for n in melody)
    total_samples = int(total_steps * seconds_per_step * sr) + int(0.2 * sr)
    audio = np.zeros(total_samples, dtype=np.float64)

    for note in melody:
        freq = 440.0 * 2 ** ((note.pitch - 69) / 12)
        start = int(note.onset * seconds_per_step * sr)
        dur_s = note.dur * seconds_per_step * 0.95  # slight gap so repeats are audible
        tone = _note_wave(freq, dur_s, sr)
        end = start + len(tone)
        if end > len(audio):
            audio = np.pad(audio, (0, end - len(audio)))
        audio[start:end] += tone

    peak = np.max(np.abs(audio)) or 1.0
    audio = audio / peak * 0.9
    return audio.astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return path


def melody_to_wav_bytes(melody: Melody, tempo_bpm: int = 100, sr: int = SAMPLE_RATE) -> bytes:
    """Render straight to WAV bytes, for Streamlit's in-memory audio player."""
    import io

    audio = melody_to_wave(melody, tempo_bpm=tempo_bpm, sr=sr)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()
