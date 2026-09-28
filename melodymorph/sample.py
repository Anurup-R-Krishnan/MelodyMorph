"""Autoregressive sampling: temperature/top-k/top-p decoding with a grammar mask,
plus the two user-facing generation modes (continuation and variation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import MelodyTransformer
from .tokenizer import Melody, MelodyTokenizer


# --------------------------------------------------------------------------
# grammar mask: a REMI stream is always (BAR | POS PITCH DUR)*, so at any point
# only one token family is legal next. This alone eliminates most structurally
# broken output, independent of how well-trained the model is.
# --------------------------------------------------------------------------
class GrammarState:
    """Tracks whether the next legal token is a POS, a PITCH, or a DUR,
    and enforces strictly monotonic position within each bar."""

    POS, PITCH, DUR = "pos", "pitch", "dur"

    def __init__(self, tokenizer: MelodyTokenizer):
        self.tok = tokenizer
        self.stage = self.POS  # after BOS or BAR, a POS (or another BAR) is next
        self.current_pos = -1
        self.notes_in_bar = 0

    def update(self, token_id: int) -> None:
        tok = self.tok
        if token_id == tok.bar_id:
            self.stage = self.POS
            self.current_pos = -1
            self.notes_in_bar = 0
        elif tok.is_pos(token_id):
            self.stage = self.PITCH
            self.current_pos = tok.pos_of(token_id)
        elif tok.is_pitch(token_id):
            self.stage = self.DUR
        elif tok.is_dur(token_id):
            self.stage = self.POS
            self.notes_in_bar += 1
        # BOS/PAD/EOS don't change the grammar state (EOS ends generation anyway)

    def allowed_ids(self, allow_eos: bool) -> list[int]:
        tok = self.tok
        if self.stage == self.PITCH:
            return tok.pitch_ids
        if self.stage == self.DUR:
            return tok.dur_ids

        # POS stage: only allow positions >= current_pos to guarantee monotonic time
        start_pos = max(0, self.current_pos)
        pos_allowed = [tok.pos_ids[p] for p in range(start_pos, 16)]
        ids = list(pos_allowed)

        # Allow BAR token if we've emitted notes in this bar or haven't emitted consecutive empty bars
        if self.notes_in_bar > 0 or self.current_pos >= 0:
            ids.append(tok.bar_id)

        if allow_eos:
            ids.append(tok.eos_id)
        return ids if ids else tok.pos_ids


def _apply_grammar_mask(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed] = 0.0
    return logits + mask


def _apply_repetition_penalty(
    logits: torch.Tensor, generated_tokens: list[int], penalty: float = 1.0, window: int = 64
) -> torch.Tensor:
    """Standard repetition penalty (Keskar et al.): dampens repetitive loops."""
    if penalty == 1.0 or not generated_tokens:
        return logits
    recent = set(generated_tokens[-window:])
    for tid in recent:
        val = logits[..., tid]
        logits[..., tid] = torch.where(val > 0, val / penalty, val * penalty)
    return logits


def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative - probs > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)

    return logits


@torch.no_grad()
def generate(
    model: MelodyTransformer,
    tokenizer: MelodyTokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 192,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    device: str | None = None,
    stop_on_eos: bool = True,
    min_new_tokens: int = 12,
) -> list[int]:
    """Autoregressively extend ``prompt_ids`` under the REMI grammar mask."""
    model.eval()
    if device is not None:
        target_device = torch.device(device)
        model_device = next(model.parameters()).device
        if model_device != target_device:
            model = model.to(target_device)
    device = str(next(model.parameters()).device)
    ids = list(prompt_ids)

    state = GrammarState(tokenizer)
    for pid in ids:
        state.update(pid)

    x = torch.tensor([ids], dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        raw_logits = model.next_token_logits(x)
        allow_eos = stop_on_eos and (len(ids) - len(prompt_ids)) >= min_new_tokens
        allowed = state.allowed_ids(allow_eos)
        logits = _apply_grammar_mask(raw_logits, allowed)

        if repetition_penalty > 1.0:
            logits = _apply_repetition_penalty(logits, ids, penalty=repetition_penalty)

        if temperature <= 1e-4:
            # Deterministic greedy decoding
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-4)
            logits = _top_k_top_p(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)

            # Defensive guard against NaN or total probability collapse
            if torch.isnan(probs).any() or probs.sum() <= 0:
                valid_mask = torch.isfinite(logits)
                if valid_mask.any():
                    probs = valid_mask.float() / valid_mask.float().sum()
                else:
                    # fallback to allowed ids uniform
                    probs = torch.zeros_like(logits)
                    probs[..., allowed] = 1.0 / len(allowed)

            next_id = torch.multinomial(probs, num_samples=1)

        next_val = int(next_id.item())

        if next_val == tokenizer.eos_id:
            break

        ids.append(next_val)
        state.update(next_val)
        x = torch.cat([x, next_id], dim=1)
        if x.size(1) > model.cfg.block_size:
            x = x[:, -model.cfg.block_size:]

    return ids


# --------------------------------------------------------------------------
# motif preservation
# --------------------------------------------------------------------------
def interval_contour(melody: Melody) -> list[int]:
    """Sign of each successive pitch interval: -1 / 0 / +1."""
    notes = sorted(melody, key=lambda n: n.onset)
    contour = []
    for a, b in zip(notes, notes[1:]):
        diff = b.pitch - a.pitch
        contour.append(1 if diff > 0 else (-1 if diff < 0 else 0))
    return contour


def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def motif_similarity(seed: Melody, candidate_region: Melody) -> float:
    """1 - normalised edit distance between interval contours, in [0, 1].

    1.0 means an identical contour to the seed; 0.0 means completely different.
    """
    a, b = interval_contour(seed), interval_contour(candidate_region)
    if not a and not b:
        return 1.0
    dist = _levenshtein(a, b)
    return max(0.0, 1.0 - dist / max(len(a), len(b), 1))


def region_within(melody: Melody, start_step: int, end_step: int) -> Melody:
    return [n for n in melody if start_step <= n.onset < end_step]
