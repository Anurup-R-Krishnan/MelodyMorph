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
    """Tracks whether the next legal token is a POS, a PITCH, or a DUR."""

    POS, PITCH, DUR = "pos", "pitch", "dur"

    def __init__(self, tokenizer: MelodyTokenizer):
        self.tok = tokenizer
        self.stage = self.POS  # after BOS or BAR, a POS (or another BAR) is next

    def update(self, token_id: int) -> None:
        tok = self.tok
        if token_id == tok.bar_id:
            self.stage = self.POS
        elif tok.is_pos(token_id):
            self.stage = self.PITCH
        elif tok.is_pitch(token_id):
            self.stage = self.DUR
        elif tok.is_dur(token_id):
            self.stage = self.POS
        # BOS/PAD/EOS don't change the grammar state (EOS ends generation anyway)

    def allowed_ids(self, allow_eos: bool) -> list[int]:
        tok = self.tok
        if self.stage == self.PITCH:
            return tok.pitch_ids
        if self.stage == self.DUR:
            return tok.dur_ids
        # POS stage: a POS token, a fresh BAR, or (optionally) EOS
        ids = list(tok.pos_ids) + [tok.bar_id]
        if allow_eos:
            ids.append(tok.eos_id)
        return ids


def _apply_grammar_mask(logits: torch.Tensor, allowed: list[int]) -> torch.Tensor:
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed] = 0.0
    return logits + mask


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
    device: str | None = None,
    stop_on_eos: bool = True,
    min_new_tokens: int = 12,
) -> list[int]:
    """Autoregressively extend ``prompt_ids`` under the REMI grammar mask."""
    model.eval()
    # The sampling tensors must live on the model's own device. A stale
    # "cpu" default used to crash CUDA-loaded checkpoints with a
    # device-mismatch RuntimeError, so the model always wins.
    device = str(next(model.parameters()).device)
    ids = list(prompt_ids)

    state = GrammarState(tokenizer)
    for pid in ids:
        state.update(pid)

    x = torch.tensor([ids], dtype=torch.long, device=device)

    for step in range(max_new_tokens):
        logits = model.next_token_logits(x) / max(temperature, 1e-5)
        allow_eos = stop_on_eos and (len(ids) - len(prompt_ids)) >= min_new_tokens
        logits = _apply_grammar_mask(logits, state.allowed_ids(allow_eos))
        logits = _top_k_top_p(logits, top_k, top_p)

        probs = F.softmax(logits, dim=-1)
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
