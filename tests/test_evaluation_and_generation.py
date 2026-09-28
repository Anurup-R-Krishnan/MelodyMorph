import torch
from melodymorph.audio import melody_to_wave
from melodymorph.evaluate import (
    evaluate_generations,
    in_scale_ratio,
    jensen_shannon,
    pairwise_distinctness,
    repetition_rate,
)
from melodymorph.generate import generate_continuations, generate_variations
from melodymorph.midi_io import parse_note_string
from melodymorph.model import MelodyTransformer, ModelConfig
from melodymorph.tokenizer import MelodyTokenizer, Note


def test_audio_synthesis_no_nans_and_bounded():
    melody = parse_note_string("C4/s D4/s E4/s F4/s G4/w")
    wave = melody_to_wave(melody, tempo_bpm=120)
    assert len(wave) > 0
    assert not torch.isnan(torch.tensor(wave)).any()
    assert (wave >= -1.0).all() and (wave <= 1.0).all()


def test_evaluation_metrics():
    # In C major scale
    c_maj_melody = parse_note_string("C4/q E4/q G4/q B4/q")
    assert in_scale_ratio(c_maj_melody, tonic_pc=0) == 1.0

    # With chromatic non-scale tone (C#)
    chromatic = parse_note_string("C4/q C#4/q G4/q B4/q")
    assert in_scale_ratio(chromatic, tonic_pc=0) == 0.75

    # Repetition rate
    repeating = [Note(i * 4, 60, 4) for i in range(8)]
    assert repetition_rate(repeating, n=2) == 1.0

    # Pairwise distinctness
    m1 = parse_note_string("C4/q E4/q G4/q")
    m2 = parse_note_string("D4/q F4/q A4/q")
    assert pairwise_distinctness([m1, m2]) == 1.0

    # Jensen Shannon bounded
    p = [1.0 / 12] * 12
    q = [1.0 / 12] * 12
    assert jensen_shannon(p, q) == 0.0


def test_pitch_markov_baseline():
    from melodymorph.evaluate import pitch_markov_baseline_perplexity
    import math

    train_melodies = [parse_note_string("C4/q D4/q E4/q F4/q G4/q")]
    val_melodies = [parse_note_string("C4/q D4/q E4/q")]
    ppl = pitch_markov_baseline_perplexity(train_melodies, val_melodies, order=1)
    assert ppl > 0
    assert not math.isnan(ppl)


def test_rhythm_entropy():
    from melodymorph.evaluate import rhythm_entropy

    # Monotone rhythms have 0 entropy
    mono = parse_note_string("C4/q E4/q G4/q C5/q")
    assert rhythm_entropy(mono) == 0.0

    # Diverse rhythms have positive entropy
    varied = parse_note_string("C4/s E4/e G4/q C5/h")
    assert rhythm_entropy(varied) > 1.5


def test_contour_strip_rendering(tmp_path):
    from melodymorph.viz import plot_contour_strip, save_contour_strip
    import matplotlib.pyplot as plt

    melody = parse_note_string("C4/q E4/q G4/h")
    fig = plot_contour_strip(melody, seed_len=4, theme="dark")
    assert fig is not None
    plt.close(fig)

    out_file = tmp_path / "contour.png"
    save_contour_strip(melody, str(out_file), seed_len=4, theme="light")
    assert out_file.exists() and out_file.stat().st_size > 0


def test_train_resumption_checkpoint(tmp_path):
    from pathlib import Path
    from melodymorph.train import TrainConfig, _save_checkpoint

    tok = MelodyTokenizer()
    m_cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=32, dropout=0.0)
    model = MelodyTransformer(m_cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    ckpt_path = str(tmp_path / "resume_test.pt")
    dummy_cfg = TrainConfig(
        data_cache=str(tmp_path / "dummy.jsonl"),
        epochs=3,
        checkpoint_path=ckpt_path,
        run_dir=str(tmp_path / "runs"),
    )
    _save_checkpoint(dummy_cfg, m_cfg, model, tok, ckpt_path, optimizer=opt, epoch=1, step=10, val_loss=2.5, best_val=2.5)

    assert Path(ckpt_path).exists()
    loaded_ckpt = torch.load(ckpt_path, weights_only=False)
    assert loaded_ckpt["epoch"] == 1
    assert loaded_ckpt["step"] == 10
    assert "optimizer_state" in loaded_ckpt
    assert loaded_ckpt["val_loss"] == 2.5
    assert loaded_ckpt["best_val_loss"] == 2.5

    from melodymorph.train import load_checkpoint
    import pytest
    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint(str(tmp_path / "non_existent.pt"))



def test_generation_continuations_returns_k_unique():
    tok = MelodyTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=64, dropout=0.0)
    model = MelodyTransformer(cfg)
    model.eval()

    seed = parse_note_string("C4/q E4/q")
    cands = generate_continuations(model, tok, seed, n_bars=2, k=3, device="cpu", max_attempts=15)
    assert len(cands) > 0
    # Seed length is preserved accurately
    for c in cands:
        assert c.seed_len_steps == 8


def test_piano_roll_measure_ticks_and_wide_span():
    from melodymorph.viz import plot_piano_roll
    import matplotlib.pyplot as plt

    # Wide pitch span (> 20 semitones)
    wide_melody = parse_note_string("C3/q G4/q C6/q")
    fig = plot_piano_roll(wide_melody, title="Wide span", theme="dark")
    assert fig is not None
    ax = fig.axes[0]
    # Verify x ticks correspond to measure markers
    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any("m.1" in lbl for lbl in x_labels)
    plt.close(fig)


def test_contour_strip_empty_melody():
    from melodymorph.viz import plot_contour_strip
    import matplotlib.pyplot as plt

    fig = plot_contour_strip([], seed_len=0, theme="dark")
    assert fig is not None
    plt.close(fig)


def test_midi_fit_range_wide_span():
    from melodymorph.midi_io import _fit_range, Note

    # Melody spanning far above MAX_PITCH
    melody = [Note(0, 96, 4), Note(4, 100, 4)]
    fitted = _fit_range(melody)
    assert len(fitted) == 2
    assert all(48 <= n.pitch <= 84 for n in fitted)


def test_audio_cd_quality_sample_rate():
    from melodymorph.audio import SAMPLE_RATE, melody_to_wave

    assert SAMPLE_RATE == 44100
    melody = parse_note_string("C4/q")
    wave = melody_to_wave(melody, tempo_bpm=120)
    # At 120 bpm, quarter note is 0.5s -> ~22050 samples at 44.1kHz (+ padding)
    assert len(wave) > 20000


def test_variations_ranked_by_motif_closeness():
    from melodymorph.generate import generate_variations

    tok = MelodyTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=64, dropout=0.0)
    model = MelodyTransformer(cfg)
    model.eval()

    seed = parse_note_string("C4/q E4/q G4/h")
    cands = generate_variations(model, tok, seed, n_bars=2, k=2, device="cpu", max_attempts=15)
    assert len(cands) > 0
    for c in cands:
        assert c.motif_score is not None


