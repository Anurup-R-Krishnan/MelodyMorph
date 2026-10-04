"""A small decoder-only Transformer for autoregressive melody modelling.

The blocks are written out explicitly rather than using ``nn.TransformerDecoder``:
the causal-attention mechanism is the subject of this project, and an explicit
implementation keeps the sampling loop transparent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    block_size: int = 256
    dropout: float = 0.1
    pad_id: int = 0  # targets equal to this are ignored by the loss

    def to_dict(self) -> dict:
        return asdict(self)


class KVCache:
    """Per-layer keys and values of the positions seen so far."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """``cache`` (inference only): keys/values of earlier positions for this
        layer; the new ones are appended to it in place."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        past = 0
        if cache is not None:
            past = cache.k.size(2) if cache.k is not None else 0
            if past:
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v = k, v

        if past == 0:
            mask, causal = None, True
        else:
            # queries sit at positions past..past+T-1 and may see every key up to themselves
            q_pos = torch.arange(past, past + T, device=x.device)[:, None]
            mask, causal = torch.arange(past + T, device=x.device)[None, :] <= q_pos, False
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=causal,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cache)
        x = x + self.mlp(self.ln2(x))
        return x


class MelodyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 recipe)
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        # nn.Module.parameters() already deduplicates tied tensors via internal memo set.
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.cfg.block_size}")

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_id,
            )
        return logits, loss

    def param_groups(self, weight_decay: float) -> list[dict]:
        """AdamW groups: decay matrices only, never biases, LayerNorms or embeddings."""
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            is_matrix = p.dim() >= 2 and not name.startswith(("tok_emb", "pos_emb", "head"))
            (decay if is_matrix else no_decay).append(p)
        return [{"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0}]

    def new_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    @torch.no_grad()
    def step(self, idx: torch.Tensor, cache: list[KVCache]) -> torch.Tensor:
        """Incremental decoding: feed the next token(s) ``idx`` (B, T) after the
        positions already in ``cache`` and return the last position's logits.

        The caller must restart with a fresh cache (and a cropped context) once
        the total length would exceed ``block_size``.
        """
        past = cache[0].k.size(2) if cache[0].k is not None else 0
        T = idx.size(1)
        if past + T > self.cfg.block_size:
            raise ValueError("cached context exceeds block size")
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block, layer_cache in zip(self.blocks, cache):
            x = block(x, layer_cache)
        return self.head(self.ln_f(x[:, -1, :]))

    @torch.no_grad()
    def next_token_logits(self, idx: torch.Tensor) -> torch.Tensor:
        """Logits for the position after the (right-cropped) context."""
        idx = idx[:, -self.cfg.block_size:]
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        # Project only the final position through the unembedding head
        last_hidden = self.ln_f(x[:, -1:, :])
        logits = self.head(last_hidden)
        return logits[:, 0, :]
