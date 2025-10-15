"""Optional rectified flow matching utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FlowMatchingConfig:
    hidden_size: int
    loss_weight: float = 1.0


class FlowMatchingObjective(nn.Module):
    """Simple L2 regression on velocity predictions."""

    def __init__(self, config: FlowMatchingConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, velocity_pred: torch.Tensor, velocity_target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(velocity_pred, velocity_target) * self.config.loss_weight
