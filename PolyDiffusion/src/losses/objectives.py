"""Loss functions for different training stages."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..chem.ap_smiles import SHIELD1, SHIELD2


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
            device = next(iter(property_preds.values())).device if property_preds else torch.device("cpu")
            return torch.tensor(0.0, device=device)

    # Multi-property mode (for reference or multi-task learning)
    for name, pred in property_preds.items():
        if name not in targets:
            continue
        losses.append(nn.functional.mse_loss(pred, targets[name]))

    if not losses:
        device = next(iter(property_preds.values())).device if property_preds else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def _anchor_supervision_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    vocab,
) -> torch.Tensor:
    shield1_id = vocab.token_to_id[SHIELD1]
    shield2_id = vocab.token_to_id[SHIELD2]
    mask = (targets == shield1_id) | (targets == shield2_id)
    if not torch.any(mask):
        return logits.new_tensor(0.0)
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    return F.cross_entropy(selected_logits, selected_targets)


def _grammar_head_loss(
    grammar_logits: Optional[torch.Tensor],
    anchor_count: torch.Tensor,
    valence_flag: torch.Tensor,
) -> torch.Tensor:
    if grammar_logits is None:
        return anchor_count.new_tensor(0.0, dtype=torch.float32)
    target = ((anchor_count == 2) & (valence_flag > 0.5)).float()
    target = target.to(grammar_logits.dtype)
    return F.binary_cross_entropy_with_logits(grammar_logits, target)


def grammar_penalty(
    outputs: Dict[str, torch.Tensor],
    x0: torch.Tensor,
    anchor_count: torch.Tensor,
    valence_flag: torch.Tensor,
    vocab,
) -> torch.Tensor:
    anchor_loss = _anchor_supervision_loss(outputs["logits"], x0, vocab)
    grammar_logits = outputs.get("grammar_logits")
    if grammar_logits is None and "grammar_pred" in outputs:
        probs = torch.clamp(outputs["grammar_pred"], 1e-4, 1.0 - 1e-4)
        grammar_logits = torch.logit(probs)
    grammar_loss = _grammar_head_loss(grammar_logits, anchor_count, valence_flag)
    return anchor_loss + grammar_loss


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
    vocab,
) -> Dict[str, torch.Tensor]:
    losses = stage_a_objective(model, outputs, x0, timesteps, mask, synth_target, lambda_syn)
    gram = grammar_penalty(outputs, x0, anchor_count, valence_flag, vocab)
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
    vocab,
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
    gram = grammar_penalty(outputs, x0, anchor_count, valence_flag, vocab)
    losses["grammar"] = gram
    losses["total"] = losses["total"] + lambda_gram * gram
    return losses
