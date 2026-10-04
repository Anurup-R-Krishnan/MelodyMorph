import torch

from melodymorph.model import MelodyTransformer, ModelConfig
from melodymorph.sample import (
    GrammarState,
    _apply_grammar_mask,
    _apply_repetition_penalty,
    generate,
    generate_batch,
)
from melodymorph.tokenizer import MelodyTokenizer


def _model(tok):
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=64, dropout=0.0)
    return MelodyTransformer(cfg).eval()


def _note(state, tok, pos, pitch, dur):
    for t in (tok.pos_token(pos), tok.pitch_token(pitch), tok.dur_token(dur)):
        state.update(t)


def test_grammar_state_transitions():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    assert state.allowed_ids(allow_eos=True) == list(tok.ts_ids.values())  # opens with a TS
    state.update(tok.ts_ids[16])
    assert state.allowed_ids(allow_eos=True) == [tok.bar_id]               # then a BAR
    state.update(tok.bar_id)
    state.update(tok.pos_token(0))
    assert state.stage == GrammarState.PITCH
    state.update(tok.pitch_token(60))
    assert state.stage == GrammarState.DUR
    state.update(tok.dur_token(4))
    assert state.stage == GrammarState.POS


def test_grammar_mask_only_allows_legal_family():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    state.update(tok.bar_id)
    state.update(tok.pos_token(0))  # now expects a PITCH
    masked = _apply_grammar_mask(torch.zeros(1, tok.vocab_size), state.allowed_ids(allow_eos=False))
    allowed = set(tok.pitch_ids)
    for i in range(tok.vocab_size):
        assert (masked[0, i].item() == 0.0) == (i in allowed)


def test_grammar_forbids_overlapping_notes():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    state.update(tok.bar_id)
    _note(state, tok, 4, 60, 8)  # sounds until step 12
    allowed = set(state.allowed_ids(allow_eos=False))
    assert all(tok.pos_token(p) not in allowed for p in range(0, 12))
    assert all(tok.pos_token(p) in allowed for p in range(12, 16))


def test_grammar_carries_sustain_across_barline():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    state.update(tok.bar_id)
    _note(state, tok, 12, 60, 8)  # ends at step 20 = bar 1, pos 4
    state.update(tok.bar_id)
    allowed = set(state.allowed_ids(allow_eos=False))
    assert tok.pos_token(3) not in allowed and tok.pos_token(4) in allowed


def test_generated_sequences_are_monophonic_and_decodable():
    tok = MelodyTokenizer()
    torch.manual_seed(0)
    rows = generate_batch(_model(tok), tok, [tok.bos_id], n=8, max_new_tokens=60)
    for ids, _ in rows:
        melody = tok.decode(ids)
        for a, b in zip(melody, melody[1:]):
            assert b.onset >= a.end


def test_stop_step_ends_rows_on_the_target_bar():
    tok = MelodyTokenizer()
    torch.manual_seed(0)
    rows = generate_batch(_model(tok), tok, [tok.bos_id, tok.ts_ids[16]], n=6, max_new_tokens=400,
                          stop_step=32)
    for ids, completed in rows:
        assert completed
        assert tok.eos_id not in ids
        assert all(n.onset < 32 for n in tok.decode(ids))


def test_template_forces_structure_and_samples_pitches():
    tok = MelodyTokenizer()
    template = [tok.bar_id, tok.pos_token(0), None, tok.dur_token(4),
                tok.pos_token(4), None, tok.dur_token(8)]
    torch.manual_seed(0)
    for ids, _ in generate_batch(_model(tok), tok, [tok.bos_id], n=4, template=template):
        melody = tok.decode(ids)
        assert [(n.onset, n.dur) for n in melody] == [(0, 4), (4, 8)]


def test_generate_greedy_is_deterministic():
    tok = MelodyTokenizer()
    model = _model(tok)
    prompt = [tok.bos_id, tok.bar_id, tok.pos_token(0), tok.pitch_token(60), tok.dur_token(4)]
    run1 = generate(model, tok, prompt, max_new_tokens=20, temperature=0.0)
    run2 = generate(model, tok, prompt, max_new_tokens=20, temperature=0.0)
    assert run1 == run2


def test_repetition_penalty_touches_only_the_given_pitches():
    tok = MelodyTokenizer()
    logits = torch.ones(tok.vocab_size)
    penalised = _apply_repetition_penalty(logits.clone(), [tok.pitch_token(60)], penalty=1.5)
    assert penalised[tok.pitch_token(60)] < 1.0
    changed = (penalised != logits).nonzero().flatten().tolist()
    assert changed == [tok.pitch_token(60)]



def test_sampling_past_the_block_size_reprefills_the_cache():
    tok = MelodyTokenizer()
    cfg = ModelConfig(vocab_size=tok.vocab_size, n_layer=1, n_head=1, d_model=16, block_size=24, dropout=0.0)
    model = MelodyTransformer(cfg).eval()
    torch.manual_seed(0)
    (ids, _), = generate_batch(model, tok, [tok.bos_id], n=1, max_new_tokens=80, stop_on_eos=False)
    assert len(ids) > 3 * cfg.block_size
    melody = tok.decode(ids)
    assert all(b.onset >= a.end for a, b in zip(melody, melody[1:]))


def test_triple_bars_limit_positions_to_twelve():
    tok = MelodyTokenizer()
    state = GrammarState(tok)
    state.update(tok.ts_ids[12])
    state.update(tok.bar_id)
    allowed = set(state.allowed_ids(allow_eos=False))
    assert tok.pos_token(11) in allowed and tok.pos_token(12) not in allowed
    _note(state, tok, 8, 60, 8)              # sounds into the next 12-step bar, until step 16
    state.update(tok.bar_id)
    allowed = set(state.allowed_ids(allow_eos=False))
    assert tok.pos_token(3) not in allowed and tok.pos_token(4) in allowed
