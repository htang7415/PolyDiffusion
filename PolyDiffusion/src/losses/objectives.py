"""Loss functions for different training stages."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn


def diffusion_ce(model, outputs, x0, timesteps, mask) -> torch.Tensor:
    return model.diffusion.loss(outputs["logits"], x0, timesteps, mask)


def synthesis_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred, target)


def property_loss(
    property_preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_property: Optional[str] = None,
) -> torch.Tensor:
    """Compute property prediction loss.

    Args:
        property_preds: Model predictions for each property.
        targets: Ground truth values for each property.
        target_property: If specified, only compute loss for this property (single-property training).
            Otherwise, compute average loss across all properties.

    Returns:
        Property loss tensor.
    """
    losses = []

    # Single-property mode (Stage C)
    if target_property is not None:
        if target_property in property_preds and target_property in targets:
            return nn.functional.mse_loss(property_preds[target_property], targets[target_property])
        else:
            device = next(iter(property_preds.values())).device if property_preds else "cpu"
            return torch.tensor(0.0, device=device)

    # Multi-property mode (for reference or multi-task learning)
    for name, pred in property_preds.items():
        if name not in targets:
            continue
        losses.append(nn.functional.mse_loss(pred, targets[name]))

    if not losses:
        device = next(iter(property_preds.values())).device if property_preds else "cpu"
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def grammar_penalty(anchor_count: torch.Tensor, valence_flag: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(anchor_count, dtype=torch.float32)
    zeros = torch.zeros_like(anchor_count, dtype=torch.float32)
    anchors_term = torch.where(anchor_count == 2, zeros, ones)
    valence_term = 1.0 - valence_flag.float()
    return (anchors_term + valence_term).mean()


def stage_a_objective(
    model,
    outputs,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    mask: torch.Tensor,
    synth_target: torch.Tensor,
    lambda_syn: float,
) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    losses["diffusion"] = diffusion_ce(model, outputs, x0, timesteps, mask)
    losses["synth"] = synthesis_loss(outputs["synth_pred"], synth_target)
    total = losses["diffusion"] + lambda_syn * losses["synth"]
    losses["total"] = total
    return losses


def stage_b_objective(
    model,
    outputs,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    mask: torch.Tensor,
    synth_target: torch.Tensor,
    anchor_count: torch.Tensor,
    valence_flag: torch.Tensor,
    lambda_syn: float,
    lambda_gram: float,
) -> Dict[str, torch.Tensor]:
    losses = stage_a_objective(model, outputs, x0, timesteps, mask, synth_target, lambda_syn)
    gram = grammar_penalty(anchor_count, valence_flag)
    losses["grammar"] = gram
    losses["total"] = losses["total"] + lambda_gram * gram
    return losses


def stage_c_objective(
    model,
    outputs,
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    mask: torch.Tensor,
    synth_target: torch.Tensor,
    property_targets: Dict[str, torch.Tensor],
    anchor_count: torch.Tensor,
    valence_flag: torch.Tensor,
    lambda_syn: float,
    lambda_prop: float,
    lambda_gram: float,
    target_property: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Stage C objective for property-guided generation.

    Args:
        target_property: If specified, trains only on this property (recommended).
            Train separate models for Tg, Tm, Td, Eg, chi.
            If None, trains on all properties simultaneously (multi-task).
    """
    losses = stage_a_objective(model, outputs, x0, timesteps, mask, synth_target, lambda_syn)
    prop = property_loss(outputs["property_preds"], property_targets, target_property)
    losses["properties"] = prop
    losses["total"] = losses["total"] + lambda_prop * prop
    gram = grammar_penalty(anchor_count, valence_flag)
    losses["grammar"] = gram
    losses["total"] = losses["total"] + lambda_gram * gram
    return losses
