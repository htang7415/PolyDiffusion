"""Neural network building blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

# Standard RoPE (Rotary Position Embedding) base frequency
ROTARY_EMBEDDING_BASE = 10000


class AdaLayerNorm(nn.Module):
    """LayerNorm reparameterised by a conditioning embedding."""

    def __init__(self, hidden_size: int, cond_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(cond_dim, hidden_size * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(cond).chunk(2, dim=-1)
        x = self.layer_norm(x)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FeedForward(nn.Module):
    """Transformer feed-forward network with GELU activation."""

    def __init__(self, hidden_size: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        inner = hidden_size * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RotaryEmbedding(nn.Module):
    """Sinusoidal rotary positional embeddings."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (ROTARY_EMBEDDING_BASE ** (torch.arange(0, self.dim, device=device, dtype=torch.float32) / self.dim))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        sin = freqs.sin()[None, None, :, :]
        cos = freqs.cos()[None, None, :, :]
        return sin, cos


def apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to query/key projections."""
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_odd * cos + x_even * sin
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def build_alibi_bias(num_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """Construct ALiBi slopes (simple linear bias)."""
    slopes = torch.logspace(-2, 0, steps=num_heads, device=device)
    position_ids = torch.arange(seq_len, device=device)
    bias = torch.outer(slopes, position_ids)
    return bias.unsqueeze(0)


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, seq_len, hidden = x.size()
    head_dim = hidden // num_heads
    x = x.view(batch, seq_len, num_heads, head_dim)
    return x.transpose(1, 2)  # (batch, heads, seq, head_dim)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, head_dim = x.size()
    return x.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)


class TransformerBlock(nn.Module):
    """Transformer block with AdaLayerNorm conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        cond_dim: int,
        use_rotary: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Head dimension must be even for rotary embeddings.")
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.ada_norm1 = AdaLayerNorm(hidden_size, cond_dim)
        self.ada_norm2 = AdaLayerNorm(hidden_size, cond_dim)
        self.ff = FeedForward(hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.use_rotary = use_rotary
        self.rotary = RotaryEmbedding(self.head_dim // 2) if use_rotary else None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.ada_norm1(x, cond)
        q = _split_heads(self.q_proj(x), self.num_heads)
        k = _split_heads(self.k_proj(x), self.num_heads)
        v = _split_heads(self.v_proj(x), self.num_heads)

        if self.use_rotary and self.rotary is not None:
            sin, cos = self.rotary(x.size(1), x.device)
            q = apply_rotary(q, sin, cos)
            k = apply_rotary(k, sin, cos)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v)
        attn_out = _merge_heads(attn_out)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        x = residual + attn_out

        residual = x
        x = self.ada_norm2(x, cond)
        x = residual + self.ff(x)
        return x


class TransformerBackbone(nn.Module):
    """Stack of Transformer blocks."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        dropout: float,
        cond_dim: int,
        use_rotary: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    cond_dim=cond_dim,
                    use_rotary=use_rotary,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond, attn_mask=attn_mask)
        return self.final_norm(x)
