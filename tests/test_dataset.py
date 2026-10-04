import torch

from melodymorph.dataset import MelodyWindowDataset, melody_windows
from melodymorph.tokenizer import MelodyTokenizer, Note


def _melody(n_notes: int):
    return [Note(i * 4, 60 + i % 12, 4) for i in range(n_notes)]


def test_every_target_scored_exactly_once():
    ids = list(range(1000))
    windows = melody_windows(ids, block_size=64, stride=32)
    scored = []
    for score_from, w in windows:
        assert len(w) <= 65
        start = ids.index(w[0])
        scored += [start + 1 + j for j in range(score_from, len(w) - 1)]
    assert scored == list(range(1, 1000))


def test_windows_start_at_bos_and_never_mix_tunes():
    tok = MelodyTokenizer()
    ds = MelodyWindowDataset([_melody(5), _melody(7)], tok, block_size=64)
    assert len(ds) == 2
    for x, y in (ds[0], ds[1]):
        assert x[0] == tok.bos_id
        eos = (y == tok.eos_id).nonzero()
        assert len(eos) == 1
        assert (y[eos[0, 0] + 1 :] == tok.pad_id).all()   # nothing after EOS is scored


def test_long_melody_overlap_is_not_scored_twice():
    tok = MelodyTokenizer()
    ds = MelodyWindowDataset([_melody(60)], tok, block_size=64, stride=32)  # ~182 tokens
    n_scored = int((ds.y != tok.pad_id).sum())
    assert len(ds) > 1
    assert n_scored == len(tok.encode(_melody(60))) - 1


def test_padding_is_ignored_by_the_loss():
    from melodymorph.model import MelodyTransformer, ModelConfig

    tok = MelodyTokenizer()
    model = MelodyTransformer(ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16,
                                          block_size=64, dropout=0.0, pad_id=tok.pad_id)).eval()
    ds = MelodyWindowDataset([_melody(5)], tok, block_size=64)
    x, y = ds[0]
    n = int((y != tok.pad_id).sum())
    _, loss_padded = model(x[None], y[None])
    _, loss_trim = model(x[None, :n], y[None, :n])
    assert torch.allclose(loss_padded, loss_trim, atol=1e-5)
