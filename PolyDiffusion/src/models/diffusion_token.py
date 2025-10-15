"""Categorical diffusion utilities for token sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class DiffusionConfig:
    vocab_size: int
    num_steps: int
    mask_token_id: int
    schedule: str = "linear"
    min_noise: float = 0.01
    max_noise: float = 0.5


class CategoricalDiffusion(nn.Module):
    """Simple discrete diffusion process implemented as token corruption."""

    def __init__(self, config: DiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("noise_schedule", self._build_schedule(config), persistent=False)

    def _build_schedule(self, config: DiffusionConfig) -> torch.Tensor:
        if config.schedule == "linear":
            levels = torch.linspace(config.min_noise, config.max_noise, config.num_steps)
        elif config.schedule == "cosine":
            steps = torch.arange(config.num_steps, dtype=torch.float32)
            alpha = torch.cos((steps / config.num_steps) * 0.5 * torch.pi)
            levels = config.min_noise + (config.max_noise - config.min_noise) * (1 - alpha / alpha.max())
        else:
            raise ValueError(f"Unknown schedule {config.schedule}")
        return levels

    @property
    def device(self) -> torch.device:
        return self.noise_schedule.device

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.config.num_steps, (batch_size,), device=self.device, dtype=torch.long)

    def noise_level(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.noise_schedule.to(self.device)[timesteps]

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_level = self.noise_level(timesteps).unsqueeze(1)
        # torch.rand_like doesn't support generator in all PyTorch versions
        probs = torch.rand(x0.shape, dtype=torch.float32, device=x0.device, generator=generator)
        mask = probs < noise_level
        x_t = x0.clone()
        random_tokens = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=x0.shape,
            device=x0.device,
            generator=generator,
        )
        x_t[mask] = random_tokens[mask]
        return x_t, mask

    def loss(
        self,
        model_logits: torch.Tensor,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy with original tokens on corrupted positions."""
        vocab_size = model_logits.size(-1)
        target = x0.clone()
        if mask is not None:
            target = target.where(mask, torch.full_like(target, self.config.mask_token_id))
        loss = nn.functional.cross_entropy(
            model_logits.view(-1, vocab_size),
            target.view(-1),
            reduction="none",
        )
        if mask is not None:
            loss = loss.view_as(x0)
            loss = loss * mask.float()
            loss = loss.sum() / mask.float().sum().clamp_min(1.0)
        else:
            loss = loss.mean()
        return loss
