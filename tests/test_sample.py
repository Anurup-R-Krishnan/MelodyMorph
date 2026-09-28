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


def test_grammar_enforces_monotonic_positions():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    # Start of bar
    state.update(tok.bar_id)
    # Note at pos 8
    state.update(tok.pos_token(8))
    state.update(tok.pitch_token(60))
    state.update(tok.dur_token(4))
    # Now in POS stage: positions 0 through 7 must NOT be allowed!
    allowed = set(state.allowed_ids(allow_eos=False))
    for p in range(0, 8):
        assert tok.pos_token(p) not in allowed
    # Positions 8 through 15 MUST be allowed
    for p in range(8, 16):
        assert tok.pos_token(p) in allowed


def test_generate_greedy_temperature_zero():
    tok = MelodyTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=64, dropout=0.0)
    model = MelodyTransformer(cfg)
    model.eval()

    prompt = [tok.bos_id, tok.bar_id, tok.pos_token(0), tok.pitch_token(60), tok.dur_token(4)]
    # Two greedy runs with temp=0 must be 100% identical
    run1 = generate(model, tok, prompt, max_new_tokens=20, temperature=0.0, device="cpu")
    run2 = generate(model, tok, prompt, max_new_tokens=20, temperature=0.0, device="cpu")
    assert run1 == run2


def test_repetition_penalty_dampens_logits():
    from melodymorph.sample import _apply_repetition_penalty
    tok = MelodyTokenizer()
    logits = torch.ones(1, tok.vocab_size)
    target_token = tok.pitch_token(60)
    penalized = _apply_repetition_penalty(logits.clone(), [target_token], penalty=1.5)
    assert penalized[0, target_token] < logits[0, target_token]
    # Other tokens unaffected
    other_token = tok.pitch_token(62)
    assert penalized[0, other_token] == logits[0, other_token]
