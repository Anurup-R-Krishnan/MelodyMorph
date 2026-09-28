import torch

from melodymorph.model import MelodyTransformer, ModelConfig


def _small_model():
    cfg = ModelConfig(vocab_size=73, n_layer=2, n_head=2, d_model=32, block_size=16, dropout=0.0)
    return MelodyTransformer(cfg), cfg


def test_forward_shapes():
    model, cfg = _small_model()
    x = torch.randint(0, cfg.vocab_size, (4, 16))
    logits, loss = model(x, x)
    assert logits.shape == (4, 16, cfg.vocab_size)
    assert loss is not None and loss.ndim == 0


def test_forward_without_targets_has_no_loss():
    model, cfg = _small_model()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(x)
    assert loss is None
    assert logits.shape == (2, 8, cfg.vocab_size)


def test_backward_pass_updates_params():
    model, cfg = _small_model()
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    before = model.tok_emb.weight.clone()
    _, loss = model(x, x)
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_causality_future_tokens_do_not_affect_earlier_logits():
    """A hallmark of a correctly causal-masked model: changing a later token
    must not change the logits produced at an earlier position."""
    model, cfg = _small_model()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        logits_a, _ = model(x)

    x2 = x.clone()
    x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # perturb only the last token
    with torch.no_grad():
        logits_b, _ = model(x2)

    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


import pytest


def test_exceeding_block_size_raises():
    model, cfg = _small_model()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block size"):
        model(x)


def test_num_parameters_accurate():
    model, _ = _small_model()
    # Unique parameter count should match sum of unique module parameters
    expected = sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == expected


def test_next_token_logits_matches_full_forward():
    model, cfg = _small_model()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    full_logits, _ = model(x)
    fast_logits = model.next_token_logits(x)
    assert torch.allclose(fast_logits, full_logits[:, -1, :], atol=1e-5)
