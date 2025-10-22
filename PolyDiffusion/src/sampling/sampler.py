"""Classifier-free and gradient guided sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from ..chem.ap_smiles import SHIELD1, SHIELD2
from ..chem.base_vocab import BaseVocabulary
from ..chem.plain_vocab import PlainVocab  # Backward compat
from ..chem.vocab import AnchorSafeVocab  # Backward compat
from ..sampling.decode import decode_tokens
from ..models.dit_token import DiffusionTransformer


@dataclass
class SamplerConfig:
    max_length: int = 64
    cfg_scale: float = 1.5
    gradient_weight: float = 0.0
    temperature: float = 1.0
    early_stop: bool = True
    min_tokens: int = 2
    target_length_min: Optional[int] = None
    target_length_max: Optional[int] = None


def _sample_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Sample token indices from logits with optional temperature scaling.

    When temperature <= 0, falls back to greedy argmax sampling so callers
    can disable stochasticity.
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    # Detach to avoid autograd tracking during sampling.
    logits = logits.detach()
    if temperature != 1.0:
        logits = logits / float(temperature)
    # Improve numerical stability before exponentiation.
    logits = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(logits, dim=-1)
    probs = probs.reshape(-1, probs.shape[-1])
    probs_sum = probs.sum(dim=-1, keepdim=True)
    normalized = probs / torch.clamp(probs_sum, min=1e-12)
    # Replace zero-sum rows with a uniform distribution to keep multinomial happy.
    uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
    probs = torch.where(probs_sum > 1e-12, normalized, uniform)
    sampled = torch.multinomial(probs, num_samples=1)
    return sampled.view(*logits.shape[:-1])


def _mask_logits(logits: torch.Tensor, forbidden_ids: Sequence[int]) -> torch.Tensor:
    """Apply a large negative bias to logits for forbidden token ids."""
    if not forbidden_ids:
        return logits
    for token_id in forbidden_ids:
        if token_id is None or token_id < 0 or token_id >= logits.size(-1):
            continue
        logits[..., token_id] = float("-inf")
    return logits


def _apply_early_stopping(
    tokens: torch.Tensor,
    frozen_mask: torch.Tensor,
    finished_rows: torch.Tensor,
    eos_id: int,
    pad_id: int,
    prefix_length: int,
    min_tokens: int,
    required_token_ids: Optional[Sequence[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Freeze sequences once the first EOS token appears by padding the tail.

    Args:
        tokens: Current token matrix (batch, seq_len).
        frozen_mask: Boolean mask marking positions frozen for future updates.
        finished_rows: Boolean vector indicating which rows already terminated.
        eos_id: Vocabulary id for EOS.
        pad_id: Vocabulary id for PAD.
        required_token_ids: Optional set of token ids that must appear before termination.

    Returns:
        Updated (tokens, frozen_mask, finished_rows).
    """
    if tokens.numel() == 0:
        return tokens, frozen_mask, finished_rows

    required_set = (
        {int(token_id) for token_id in required_token_ids if token_id is not None and token_id >= 0}
        if required_token_ids
        else set()
    )

    batch, seq_len = tokens.shape
    device = tokens.device

    eos_mask = tokens == eos_id
    if not torch.any(eos_mask):
        return tokens, frozen_mask, finished_rows

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
    first_pos = torch.where(eos_mask, positions, seq_len)
    first_idx = torch.min(first_pos, dim=1).values

    new_finished = (~finished_rows) & (first_idx < seq_len)
    if not torch.any(new_finished):
        return tokens, frozen_mask, finished_rows

    row_indices = new_finished.nonzero(as_tuple=True)[0]
    for row in row_indices.tolist():
        eos_position = int(first_idx[row].item())
        content_tokens = max(0, eos_position - prefix_length)
        if content_tokens < min_tokens:
            continue
        if required_set:
            row_tokens = tokens[row, : eos_position + 1].tolist()
            row_token_set = {int(token) for token in row_tokens}
            if not required_set.issubset(row_token_set):
                continue
        if eos_position + 1 < seq_len:
            tokens[row, eos_position + 1 :] = pad_id
        frozen_mask[row, eos_position:] = True
        finished_rows[row] = True

    return tokens, frozen_mask, finished_rows


class GuidedSampler:
    """Sampling wrapper supporting CFG and gradient guidance."""

    def __init__(self, model: DiffusionTransformer, vocab: BaseVocabulary, config: Optional[SamplerConfig] = None) -> None:
        self.model = model.eval()
        self.vocab = vocab
        self.config = config or SamplerConfig()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_conditions(
        self,
        num_samples: int,
        property_targets: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        prop_tensors: Optional[Dict[str, torch.Tensor]] = None
        if property_targets:
            valid_props = set(getattr(self.model.config, "property_names", []))
            invalid = set(property_targets) - valid_props
            if invalid:
                raise ValueError(f"Unknown property targets: {', '.join(sorted(invalid))}")
            prop_tensors = {
                name: torch.full((num_samples,), float(value), device=self.device)
                for name, value in property_targets.items()
            }
        return prop_tensors

    def sample(
        self,
        num_samples: int,
        num_steps: int,
        property_targets: Optional[Dict[str, float]] = None,
        cfg_scale: Optional[float] = None,
        gradient_weight: Optional[float] = None,
        include_properties: bool = True,
    ) -> List[Dict[str, object]]:
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale
        gradient_weight = gradient_weight if gradient_weight is not None else self.config.gradient_weight

        properties = self._prepare_conditions(num_samples, property_targets)

        seq_len = self.config.max_length
        tokens = torch.full((num_samples, seq_len), self.vocab.pad_id, device=self.device, dtype=torch.long)
        bos_id = self.vocab.bos_id
        eos_id = self.vocab.eos_id
        pad_id = self.vocab.pad_id
        shield1_id = self.vocab.token_to_id[SHIELD1]
        shield2_id = self.vocab.token_to_id[SHIELD2]

        tokens[:, 0] = bos_id
        tokens[:, 1] = shield1_id

        min_content = max(self.config.min_tokens, 1)
        if self.config.target_length_min is not None:
            min_content = max(min_content, int(self.config.target_length_min))
        max_content = max(1, seq_len - 2)
        if self.config.target_length_max is not None:
            max_content = min(max_content, int(self.config.target_length_max))
        if max_content <= min_content:
            length_targets = torch.full((num_samples,), min_content, device=self.device, dtype=torch.long)
        else:
            length_targets = torch.randint(
                low=min_content,
                high=max_content + 1,
                size=(num_samples,),
                device=self.device,
            )

        anchor_mask = torch.zeros_like(tokens, dtype=torch.bool)
        anchor_mask[:, 0] = True
        anchor_mask[:, 1] = True

        frozen_mask = torch.zeros_like(tokens, dtype=torch.bool)
        finished_rows = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        uncond_mask = torch.ones(num_samples, device=self.device, dtype=torch.bool)

        attention_mask = tokens != pad_id
        final_outputs = None

        total_steps = max(1, min(num_steps, self.model.diffusion.config.num_steps))
        for step in reversed(range(total_steps)):
            timesteps = torch.full((num_samples,), step, device=self.device, dtype=torch.long)
            use_grad = gradient_weight > 0.0
            if use_grad:
                outputs_cond = self.model(tokens, timesteps, attention_mask=attention_mask, properties=properties)
            else:
                with torch.no_grad():
                    outputs_cond = self.model(tokens, timesteps, attention_mask=attention_mask, properties=properties)
            with torch.no_grad():
                outputs_uncond = self.model(
                    tokens,
                    timesteps,
                    attention_mask=attention_mask,
                    properties=properties,
                    condition_dropout_mask=uncond_mask,
                )
            logits_cond = outputs_cond["logits"]
            logits_uncond = outputs_uncond["logits"]
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            if gradient_weight > 0.0:
                # Get hidden states (which connect to property predictions)
                hidden = outputs_cond["hidden"]

                # Compute guidance loss
                guidance_loss = torch.zeros((), device=self.device)
                if property_targets:
                    for name, target in property_targets.items():
                        pred = outputs_cond["property_preds"].get(name)
                        if pred is not None:
                            guidance_loss = guidance_loss + (pred - float(target)).pow(2).mean()

                if guidance_loss.requires_grad:
                    # Take gradient w.r.t. hidden states (not logits!)
                    grad_hidden = torch.autograd.grad(guidance_loss, hidden, retain_graph=False, allow_unused=True)[0]
                    if grad_hidden is not None:
                        # Project hidden gradients to logit space via output head
                        # hidden: (batch, seq_len, hidden_dim)
                        # grad_hidden: (batch, seq_len, hidden_dim)
                        # head.weight: (vocab_size, hidden_dim)
                        # Result: (batch, seq_len, vocab_size)
                        grad_logits = torch.matmul(grad_hidden, self.model.head.weight.T)
                        logits = logits - gradient_weight * grad_logits

            forbidden_ids = [
                pad_id,
                self.vocab.mask_id,
                self.vocab.token_to_id.get("<UNK>"),
            ]
            logits = _mask_logits(logits, forbidden_ids)

            if eos_id is not None:
                has_anchor2 = torch.any(tokens == shield2_id, dim=1)
                content_lengths = torch.clamp((tokens != pad_id).sum(dim=1) - 2, min=0)
                eos_ready = has_anchor2 & (content_lengths >= length_targets)

                # Single clone if any modifications needed
                needs_modification = not torch.all(has_anchor2) or torch.any(eos_ready)
                if needs_modification:
                    logits = logits.clone()

                    if not torch.all(has_anchor2):
                        missing_rows = (~has_anchor2).nonzero(as_tuple=True)[0]
                        if missing_rows.numel() > 0:
                            logits[missing_rows, :, eos_id] = float("-inf")

                    if torch.any(eos_ready):
                        boosts = torch.zeros(num_samples, device=self.device, dtype=logits.dtype)
                        diff = content_lengths.float() - length_targets.float()
                        boosts[eos_ready] = torch.clamp(diff[eos_ready] * 0.5, min=0.0, max=6.0)
                        logits[:, :, eos_id] = logits[:, :, eos_id] + boosts.unsqueeze(-1)

            sampled = _sample_from_logits(logits, self.config.temperature)
            update_mask = ~(anchor_mask | frozen_mask)
            tokens = torch.where(update_mask, sampled, tokens).detach()
            tokens[:, 0] = bos_id
            tokens[:, 1] = shield1_id
            anchor2_positions = tokens == shield2_id
            new_anchor2 = anchor2_positions & ~frozen_mask
            if torch.any(new_anchor2):
                frozen_mask = frozen_mask | new_anchor2
            if self.config.early_stop:
                tokens, frozen_mask, finished_rows = _apply_early_stopping(
                    tokens,
                    frozen_mask,
                    finished_rows,
                    eos_id,
                    pad_id,
                    prefix_length=2,
                    min_tokens=self.config.min_tokens,
                    required_token_ids=(shield2_id,),
                )
            attention_mask = tokens != pad_id
            final_outputs = {
                "property_preds": {name: tensor.detach() for name, tensor in outputs_cond["property_preds"].items()},
            }
            if self.config.early_stop and torch.all(finished_rows):
                break

        sequences = tokens.detach().cpu().tolist()
        smiles = decode_tokens(self.vocab, sequences)
        results: List[Dict[str, object]] = []
        if final_outputs is None:
            dummy = torch.zeros(num_samples, device=self.device)
            final_outputs = {
                "property_preds": {name: dummy for name in (property_targets or {}).keys()},
            }
        for idx, smile in enumerate(smiles):
            result = {
                "ap_smiles": smile,
                "logits": None,
                "prediction": {},
            }
            if include_properties:
                for name, tensor in final_outputs["property_preds"].items():
                    result["prediction"][name] = float(tensor[idx].detach().cpu())
            results.append(result)
        return results


class PlainSampler:
    """Sampling wrapper for Stage A models without anchor tokens."""

    def __init__(self, model: DiffusionTransformer, vocab: BaseVocabulary, config: Optional[SamplerConfig] = None) -> None:
        self.model = model.eval()
        self.vocab = vocab
        self.config = config or SamplerConfig()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def sample(
        self,
        num_samples: int,
        num_steps: int,
        cfg_scale: Optional[float] = None,
        gradient_weight: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale
        gradient_weight = gradient_weight if gradient_weight is not None else self.config.gradient_weight

        seq_len = self.config.max_length
        tokens = torch.full((num_samples, seq_len), self.vocab.pad_id, device=self.device, dtype=torch.long)
        bos_id = self.vocab.bos_id
        eos_id = self.vocab.eos_id
        pad_id = self.vocab.pad_id

        tokens[:, 0] = bos_id

        fixed_mask = torch.zeros_like(tokens, dtype=torch.bool)
        fixed_mask[:, 0] = True

        frozen_mask = torch.zeros_like(tokens, dtype=torch.bool)
        finished_rows = torch.zeros(num_samples, dtype=torch.bool, device=self.device)

        uncond_mask = torch.ones(num_samples, device=self.device, dtype=torch.bool)
        attention_mask = tokens != pad_id
        final_outputs = None

        total_steps = max(1, min(num_steps, self.model.diffusion.config.num_steps))
        for step in reversed(range(total_steps)):
            timesteps = torch.full((num_samples,), step, device=self.device, dtype=torch.long)
            use_grad = gradient_weight > 0.0
            if use_grad:
                outputs_cond = self.model(tokens, timesteps, attention_mask=attention_mask, properties=None)
            else:
                with torch.no_grad():
                    outputs_cond = self.model(tokens, timesteps, attention_mask=attention_mask, properties=None)
            with torch.no_grad():
                outputs_uncond = self.model(
                    tokens,
                    timesteps,
                    attention_mask=attention_mask,
                    properties=None,
                    condition_dropout_mask=uncond_mask,
                )
            logits_cond = outputs_cond["logits"]
            logits_uncond = outputs_uncond["logits"]
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            # Gradient guidance removed for PlainSampler (Stage A has no property targets)
            # If property-guided sampling is needed for Stage A, use GuidedSampler instead

            forbidden_ids = [
                pad_id,
                self.vocab.mask_id,
                self.vocab.token_to_id.get("<UNK>"),
            ]
            logits = _mask_logits(logits, forbidden_ids)
            sampled = _sample_from_logits(logits, self.config.temperature)
            update_mask = ~(fixed_mask | frozen_mask)
            tokens = torch.where(update_mask, sampled, tokens).detach()
            tokens[:, 0] = bos_id
            if self.config.early_stop:
                tokens, frozen_mask, finished_rows = _apply_early_stopping(
                    tokens,
                    frozen_mask,
                    finished_rows,
                    eos_id,
                    pad_id,
                    prefix_length=1,
                    min_tokens=self.config.min_tokens,
                    required_token_ids=None,
                )
            attention_mask = tokens != pad_id
            final_outputs = None
            if self.config.early_stop and torch.all(finished_rows):
                break

        sequences = tokens.detach().cpu().tolist()
        smiles = [self.vocab.detokenize(seq) for seq in sequences]
        results: List[Dict[str, object]] = []
        for idx, smile in enumerate(smiles):
            result = {
                "smiles": smile,
                "logits": None,
                "prediction": {},
            }
            results.append(result)
        return results
