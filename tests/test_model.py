import pytest
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
    x = torch.randint(1, cfg.vocab_size, (2, 16))
    before = model.tok_emb.weight.clone()
    _, loss = model(x, x)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert not torch.equal(before, model.tok_emb.weight)


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



def test_exceeding_block_size_raises():
    model, cfg = _small_model()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    with pytest.raises(ValueError, match="exceeds block size"):
        model(x)


def test_num_parameters_counts_tied_weights_once():
    model, cfg = _small_model()
    untied = sum(p.numel() for _, p in model.named_parameters(remove_duplicate=False))
    assert model.num_parameters() == untied - cfg.vocab_size * cfg.d_model


def test_next_token_logits_matches_full_forward():
    model, cfg = _small_model()
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (2, 10))
    full_logits, _ = model(x)
    fast_logits = model.next_token_logits(x)
    assert torch.allclose(fast_logits, full_logits[:, -1, :], atol=1e-5)


def test_kv_cache_matches_full_forward():
    model, cfg = _small_model()
    model.eval()
    x = torch.randint(1, cfg.vocab_size, (3, 12))
    full, _ = model(x)
    cache = model.new_cache()
    assert torch.allclose(model.step(x[:, :5], cache), full[:, 4], atol=1e-5)
    for t in range(5, 12):
        assert torch.allclose(model.step(x[:, t : t + 1], cache), full[:, t], atol=1e-5)


def test_kv_cache_refuses_to_overflow_block():
    model, cfg = _small_model()
    cache = model.new_cache()
    model.step(torch.randint(1, cfg.vocab_size, (1, cfg.block_size)), cache)
    with pytest.raises(ValueError, match="exceeds block size"):
        model.step(torch.randint(1, cfg.vocab_size, (1, 1)), cache)


def test_weight_decay_skips_norms_biases_and_embeddings():
    model, _ = _small_model()
    decay, no_decay = model.param_groups(0.1)
    decayed = {id(p) for p in decay["params"]}
    for name, p in model.named_parameters():
        expect = p.dim() >= 2 and not name.startswith(("tok_emb", "pos_emb"))
        assert (id(p) in decayed) == expect, name
