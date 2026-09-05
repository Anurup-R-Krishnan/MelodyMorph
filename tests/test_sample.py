import torch

from melodymorph.model import MelodyTransformer, ModelConfig
from melodymorph.sample import GrammarState, _apply_grammar_mask, generate
from melodymorph.tokenizer import MelodyTokenizer


def test_grammar_state_transitions():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    assert state.stage == GrammarState.POS

    state.update(tok.pos_token(0))
    assert state.stage == GrammarState.PITCH

    state.update(tok.pitch_token(60))
    assert state.stage == GrammarState.DUR

    state.update(tok.dur_token(4))
    assert state.stage == GrammarState.POS

    state.update(tok.bar_id)
    assert state.stage == GrammarState.POS


def test_grammar_mask_only_allows_legal_family():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    state.update(tok.pos_token(0))  # now expects a PITCH
    logits = torch.zeros(1, tok.vocab_size)
    masked = _apply_grammar_mask(logits, state.allowed_ids(allow_eos=False))
    allowed = set(tok.pitch_ids)
    for i in range(tok.vocab_size):
        val = masked[0, i].item()
        if i in allowed:
            assert val == 0.0
        else:
            assert val == float("-inf")


def test_generate_produces_only_grammatical_sequences():
    tok = MelodyTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=64, dropout=0.0)
    model = MelodyTransformer(cfg)
    model.eval()

    prompt = tok.encode([], add_special=True)[:-1]  # just BOS
    ids = generate(model, tok, prompt, max_new_tokens=40, temperature=1.0, device="cpu")

    # decoding must not raise and every decoded note must be in-range
    melody = tok.decode(ids)
    for note in melody:
        assert 0 <= note.onset
        assert note.dur >= 1
