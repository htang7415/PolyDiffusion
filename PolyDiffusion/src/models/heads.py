"""Prediction heads for guided polymer diffusion."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn


def pooled_representation(hidden: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average pool hidden states across valid tokens."""
    if mask is None:
        return hidden.mean(dim=1)
    weights = mask.float()
    pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
    denom = weights.sum(dim=1, keepdim=False).clamp_min(1.0)
    return pooled / denom.unsqueeze(-1)


class MLPHead(nn.Module):
    """Two-layer perceptron head."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden = max(128, input_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PropertyHeads(nn.Module):
    """Predict glass transition, melting, etc."""

    def __init__(self, input_dim: int, properties: Iterable[str]) -> None:
        super().__init__()
        self.properties = list(properties)
        self.heads = nn.ModuleDict({name: MLPHead(input_dim, 1) for name in self.properties})

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(pooled).squeeze(-1) for name, head in self.heads.items()}


class GrammarHead(nn.Module):
    """Auxiliary head to predict grammar feasibility."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.head = MLPHead(input_dim, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.head(pooled).squeeze(-1)
