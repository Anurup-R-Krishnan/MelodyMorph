"""Autoregressive sampling: temperature/top-k/top-p decoding under a grammar mask,
batched so that all candidates for a request share one forward pass per token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import MelodyTransformer
from .tokenizer import STEPS_PER_BAR, Melody, MelodyTokenizer


# --------------------------------------------------------------------------
# grammar mask: a REMI stream is BOS [TS] (BAR (POS PITCH DUR)*)*, so at any point
# only one token family is legal next. The state also tracks absolute time, so
# a note can never start before the previous one has ended (the corpus is
# strictly monophonic) and generation can stop exactly at a target bar.
# --------------------------------------------------------------------------
class GrammarState:
    METER, START, POS, PITCH, DUR = "meter", "start", "pos", "pitch", "dur"

    def __init__(self, tokenizer: MelodyTokenizer, max_empty_bars: int = 2):
        self.tok = tokenizer
        self.max_empty_bars = max_empty_bars
        # after BOS: the time signature (if the vocabulary has one), then a BAR
        self.stage = self.METER if tokenizer.meter_tokens else self.START
        self.bar_len = STEPS_PER_BAR
        self.bar = -1
        self.last_end = 0         # absolute step at which the last note ends
        self.onset = 0            # onset of the note being emitted
        self.notes_in_bar = 0
        self.empty_run = 0
        self.n_notes = 0

    def update(self, token_id: int) -> None:
        tok = self.tok
        if tok.is_ts(token_id):
            self.bar_len = tok.bar_of_ts(token_id)
            self.stage = self.START
        elif token_id == tok.bar_id:
            if self.bar >= 0:
                self.empty_run = self.empty_run + 1 if self.notes_in_bar == 0 else 0
            self.bar += 1
            self.notes_in_bar = 0
            self.stage = self.POS
        elif tok.is_pos(token_id):
            self.onset = self.bar * self.bar_len + tok.pos_of(token_id)
            self.stage = self.PITCH
        elif tok.is_pitch(token_id):
            self.stage = self.DUR
        elif tok.is_dur(token_id):
            self.last_end = self.onset + tok.dur_of(token_id)
            self.notes_in_bar += 1
            self.n_notes += 1
            self.stage = self.POS
        # BOS/PAD/EOS don't change the grammar state

    def allowed_ids(self, allow_eos: bool) -> list[int]:
        tok = self.tok
        if self.stage == self.METER:
            return list(tok.ts_ids.values())
        if self.stage == self.START:
            return [tok.bar_id]
        if self.stage == self.PITCH:
            return tok.pitch_ids
        if self.stage == self.DUR:
            return tok.dur_ids

        # POS stage: no position before the previous note has ended
        min_pos = max(0, self.last_end - self.bar * self.bar_len)
        ids = [tok.pos_ids[p] for p in range(min_pos, self.bar_len)]
        if self.notes_in_bar > 0 or self.empty_run < self.max_empty_bars or not ids:
            ids.append(tok.bar_id)
        if allow_eos and self.n_notes > 0:
            ids.append(tok.eos_id)
        return ids


def _apply_grammar_mask(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed] = 0.0
    return logits + mask


def _apply_repetition_penalty(
    logits: torch.Tensor, recent_pitches: list[int], penalty: float = 1.0,
) -> torch.Tensor:
    """Keskar-style penalty, restricted to PITCH tokens.

    Penalising structural tokens (POS/DUR/BAR) would push the model away from
    ordinary rhythms: a 4/4 melody *should* reuse DUR_4 and POS_0 constantly.
    """
    if penalty == 1.0 or not recent_pitches:
        return logits
    idx = torch.tensor(sorted(set(recent_pitches)), device=logits.device)
    vals = logits[..., idx]
    logits[..., idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(probs, dim=-1) - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)

    return logits


@torch.no_grad()
def generate_batch(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    prompt_ids: list[int],
    n: int,
    max_new_tokens: int = 192,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    stop_step: int | None = None,
    stop_on_eos: bool = True,
    min_new_tokens: int = 12,
    template: list[int | None] | None = None,
    penalty_window: int = 16,
) -> list[tuple[list[int], bool]]:
    """Sample ``n`` extensions of ``prompt_ids`` in one batch.

    ``stop_step``: stop a row cleanly when it would open a bar at or past this
    absolute step (EOS is then disallowed, so the row always gets there).
    ``template``: a token schedule to teacher-force; ``None`` slots are sampled.
    Returns ``(ids, completed)`` per row; ``completed`` is False if the row ran
    out of ``max_new_tokens`` before reaching ``stop_step`` / EOS.
    """
    model.eval()
    device = next(model.parameters()).device
    states = [GrammarState(tokenizer) for _ in range(n)]
    for st in states:
        for pid in prompt_ids:
            st.update(pid)
    rows = [list(prompt_ids) for _ in range(n)]
    done = [False] * n
    completed = [False] * n
    x = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)
    steps = len(template) if template is not None else max_new_tokens
    pitch_set = set(tokenizer.pitch_ids)
    block = model.cfg.block_size
    cache = None
    feed = x

    for step in range(steps):
        if cache is None or cache[0].k.size(2) + feed.size(1) > block:
            # (re)prefill the KV cache; when the context outgrows the block, keep
            # the most recent 3/4 so re-prefills stay rare
            keep = block if cache is None else (3 * block) // 4
            cache = model.new_cache()
            feed = x[:, -keep:]
        logits = model.step(feed, cache).float()
        mask = torch.full_like(logits, float("-inf"))
        for r, st in enumerate(states):
            if done[r]:
                mask[r, tokenizer.pad_id] = 0.0
                continue
            allow_eos = (stop_on_eos and stop_step is None
                         and len(rows[r]) - len(prompt_ids) >= min_new_tokens)
            mask[r, st.allowed_ids(allow_eos)] = 0.0
            if repetition_penalty != 1.0:
                recent = [t for t in rows[r][-3 * penalty_window:] if t in pitch_set]
                logits[r] = _apply_repetition_penalty(logits[r], recent, repetition_penalty)
        logits = logits + mask

        if temperature <= 1e-4:
            next_ids = torch.argmax(logits, dim=-1)
        else:
            logits = _top_k_top_p(logits / temperature, top_k, top_p)
            next_ids = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(1)
        if template is not None and template[step] is not None:
            next_ids = torch.full_like(next_ids, template[step])

        for r in range(n):
            if done[r]:
                next_ids[r] = tokenizer.pad_id
                continue
            tid = int(next_ids[r])
            st = states[r]
            if tid == tokenizer.eos_id:
                done[r] = completed[r] = True
                next_ids[r] = tokenizer.pad_id
                continue
            if (tid == tokenizer.bar_id and stop_step is not None
                    and (st.bar + 1) * st.bar_len >= stop_step):
                done[r] = completed[r] = True
                next_ids[r] = tokenizer.pad_id
                continue
            rows[r].append(tid)
            st.update(tid)

        if all(done):
            break
        feed = next_ids[:, None]
        x = torch.cat([x, feed], dim=1)

    if template is not None:
        completed = [True] * n
    return list(zip(rows, completed))


def generate(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 192,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
    device: str | None = None,
    stop_on_eos: bool = True,
    min_new_tokens: int = 12,
) -> list[int]:
    """Single-sequence convenience wrapper around :func:`generate_batch`."""
    if device is not None:
        model = model.to(device)
    ids, _ = generate_batch(
        model, tokenizer, prompt_ids, 1, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repetition_penalty=repetition_penalty, stop_on_eos=stop_on_eos,
        min_new_tokens=min_new_tokens,
    )[0]
    return ids


# --------------------------------------------------------------------------
# motif similarity
# --------------------------------------------------------------------------
def interval_contour(melody: Melody) -> list[int]:
    """Sign of each successive pitch interval: -1 / 0 / +1."""
    notes = sorted(melody, key=lambda n: n.onset)
    return [(b.pitch > a.pitch) - (b.pitch < a.pitch) for a, b in zip(notes, notes[1:])]


def levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def motif_similarity(seed: Melody, candidate: Melody) -> float:
    """1 - normalised edit distance between interval contours, in [0, 1]."""
    a, b = interval_contour(seed), interval_contour(candidate)
    if not a and not b:
        return 1.0
    return max(0.0, 1.0 - levenshtein(a, b) / max(len(a), len(b), 1))


def region_within(melody: Melody, start_step: int, end_step: int) -> Melody:
    return [n for n in melody if start_step <= n.onset < end_step]
