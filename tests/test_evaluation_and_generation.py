import math

import pytest
import torch

from melodymorph.evaluate import (
    KneserNey,
    in_scale_ratio,
    jensen_shannon,
    log_human_rating,
    pairwise_distinctness,
    repetition_rate,
    rhythm_entropy,
)
from melodymorph.generate import generate_continuations, generate_variations
from melodymorph.midi_io import parse_note_string
from melodymorph.model import MelodyTransformer, ModelConfig
from melodymorph.tokenizer import MelodyTokenizer, Note, melody_duration


def _model(tok):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=64, dropout=0.0)
    return MelodyTransformer(cfg).eval()


def test_evaluation_metrics():
    assert in_scale_ratio(parse_note_string("C4/q E4/q G4/q B4/q")) == 1.0
    assert in_scale_ratio(parse_note_string("C4/q C#4/q G4/q B4/q")) == 0.75
    # A natural minor shares C major's notes; G# (harmonic minor) does not
    assert in_scale_ratio(parse_note_string("A4/q G#4/q"), tonic_pc=9, mode="minor") == 0.5
    assert repetition_rate([Note(i * 4, 60, 4) for i in range(8)], n=2) == 1.0
    m1, m2 = parse_note_string("C4/q E4/q G4/q"), parse_note_string("D4/q F4/q A4/q")
    assert pairwise_distinctness([m1, m2]) == 1.0
    assert jensen_shannon([1 / 12] * 12, [1 / 12] * 12) == 0.0
    assert rhythm_entropy(parse_note_string("C4/q E4/q G4/q C5/q")) == 0.0
    assert rhythm_entropy(parse_note_string("C4/s E4/e G4/q C5/h")) > 1.5


def test_kneser_ney_is_a_normalised_distribution():
    stream = [1, 3, 4, 5, 6, 3, 4, 7, 6, 3, 4, 5, 6, 2] * 20
    kn = KneserNey(order=3).fit(stream, vocab_size=10)
    for ctx in [(3, 4), (4, 5), (9, 9), ()]:
        assert math.isclose(sum(kn.prob(ctx, w) for w in range(10)), 1.0, rel_tol=1e-9)
    assert kn.prob((3, 4), 5) > kn.prob((3, 4), 2)


def test_continuations_stop_on_the_target_bar():
    tok = MelodyTokenizer()
    seed = parse_note_string("C4/q E4/q G4/q E4/q")  # one bar
    cands = generate_continuations(_model(tok), tok, seed, n_bars=2, k=3)
    assert cands
    for c in cands:
        assert c.seed_len_steps == 16
        assert c.melody[: len(seed)] == seed          # seed kept verbatim
        assert all(n.onset < 48 for n in c.melody)    # nothing past bar 3
        assert c.completed


def test_continuations_come_back_in_the_seeds_key():
    tok = MelodyTokenizer()
    seed = parse_note_string("F#4/q A#4/q C#5/q A#4/q")
    for c in generate_continuations(_model(tok), tok, seed, n_bars=1, k=2):
        assert c.melody[: len(seed)] == seed


def test_variations_keep_rhythm_and_never_copy_the_seed():
    tok = MelodyTokenizer()
    seed = parse_note_string("E4/q E4/q F4/q G4/q G4/q F4/e E4/e D4/h")
    cands = generate_variations(_model(tok), tok, seed, k=4)
    assert cands
    for c in cands:
        assert [(n.onset, n.dur) for n in c.melody] == [(n.onset, n.dur) for n in seed]
        assert [n.pitch for n in c.melody] != [n.pitch for n in seed]
        assert 0.0 <= c.motif_score <= 1.0
        assert melody_duration(c.melody) == melody_duration(seed)


def test_human_rating_log_has_context(tmp_path):
    path = tmp_path / "ratings.csv"
    log_human_rating(path, {"checkpoint": "ck.pt", "mode": "variation", "take": 1, "seed": "C4/q",
                            "melody": "D4/q", "params": "()", "musicality_1_5": 4,
                            "motif_preservation_1_5": 3})
    header, row = path.read_text().splitlines()
    assert header.startswith("timestamp,checkpoint,mode,take,seed,melody")
    assert "variation" in row and "ck.pt" in row


def test_checkpoint_roundtrip_is_weights_only(tmp_path):
    from melodymorph.train import TrainConfig, _save_checkpoint, load_checkpoint

    tok = MelodyTokenizer()
    model = _model(tok)
    path = str(tmp_path / "ck.pt")
    _save_checkpoint(TrainConfig(), model.cfg, model, tok, path, epoch=1, step=10, val_loss=2.5, best_val=2.5)
    loaded, _ = load_checkpoint(path, device="cpu")
    assert torch.equal(loaded.tok_emb.weight, model.tok_emb.weight)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_checkpoint(str(tmp_path / "missing.pt"))


def test_long_seeds_are_trimmed_on_whole_bars():
    from melodymorph.generate import trim_seed
    seed = [Note(4 + i * 4, 60 + i % 7, 4) for i in range(50)]   # starts on beat 2
    last = trim_seed(seed, 32, keep="last")
    first = trim_seed(seed, 32, keep="first")
    assert len(last) == len(first) == 32
    assert all(a.onset % 16 == b.onset % 16 for a, b in zip(last, seed[-32:]))
    assert first == seed[:32]
    tok = MelodyTokenizer()
    assert generate_continuations(_model(tok), tok, seed, n_bars=1, k=1)
