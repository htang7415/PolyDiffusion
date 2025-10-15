"""Classifier-free and gradient guided sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from ..chem.ap_smiles import SHIELD1, SHIELD2
from ..chem.vocab import AnchorSafeVocab
from ..sampling.decode import decode_tokens
from ..models.dit_token import DiffusionTransformer


@dataclass
class SamplerConfig:
    max_length: int = 64
    cfg_scale: float = 1.5
    gradient_weight: float = 0.0


class GuidedSampler:
    """Sampling wrapper supporting CFG and gradient guidance."""

    def __init__(self, model: DiffusionTransformer, vocab: AnchorSafeVocab, config: Optional[SamplerConfig] = None) -> None:
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
        synth_target: Optional[float],
    ) -> Dict[str, torch.Tensor]:
        prop_tensors: Dict[str, torch.Tensor] = {}
        if property_targets:
            for name, value in property_targets.items():
                prop_tensors[name] = torch.full((num_samples,), float(value), device=self.device)
        if synth_target is not None:
            prop_tensors["_s_target"] = torch.full((num_samples,), float(synth_target), device=self.device)
        return prop_tensors

    def sample(
        self,
        num_samples: int,
        num_steps: int,
        property_targets: Optional[Dict[str, float]] = None,
        synth_target: Optional[float] = None,
        cfg_scale: Optional[float] = None,
        gradient_weight: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        cfg_scale = cfg_scale if cfg_scale is not None else self.config.cfg_scale
        gradient_weight = gradient_weight if gradient_weight is not None else self.config.gradient_weight

        props = {k: torch.full((num_samples,), float(v), device=self.device) for k, v in (property_targets or {}).items()}
        s_target_tensor = torch.full((num_samples,), float(synth_target), device=self.device) if synth_target is not None else None

        seq_len = self.config.max_length
        tokens = torch.full((num_samples, seq_len), self.vocab.pad_id, device=self.device, dtype=torch.long)
        tokens[:, 0] = self.vocab.bos_id
        tokens[:, 1] = self.vocab.token_to_id[SHIELD1]
        tokens[:, -2] = self.vocab.token_to_id[SHIELD2]
        tokens[:, -1] = self.vocab.eos_id
        anchor_mask = torch.zeros_like(tokens, dtype=torch.bool)
        anchor_mask[:, 0] = True
        anchor_mask[:, 1] = True
        anchor_mask[:, -2] = True
        anchor_mask[:, -1] = True

        uncond_mask = torch.ones(num_samples, device=self.device, dtype=torch.bool)

        attention_mask = tokens != self.vocab.pad_id
        final_outputs = None

        for step in reversed(range(max(num_steps, 1))):
            timesteps = torch.full((num_samples,), step, device=self.device, dtype=torch.long)
            use_grad = gradient_weight > 0.0
            if use_grad:
                outputs_cond = self.model(tokens, timesteps, attention_mask=attention_mask, properties=props if props else None, s_target=s_target_tensor)
            else:
                with torch.no_grad():
                    outputs_cond = self.model(tokens, timesteps, attention_mask=attention_mask, properties=props if props else None, s_target=s_target_tensor)
            with torch.no_grad():
                outputs_uncond = self.model(
                    tokens,
                    timesteps,
                    attention_mask=attention_mask,
                    properties=props if props else None,
                    s_target=s_target_tensor,
                    condition_dropout_mask=uncond_mask,
                )
            logits_cond = outputs_cond["logits"]
            logits_uncond = outputs_uncond["logits"]
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            if gradient_weight > 0.0:
                logits_cond.requires_grad_(True)
                guidance_loss = 0.0
                if property_targets:
                    for name, target in property_targets.items():
                        pred = outputs_cond["property_preds"].get(name)
                        if pred is not None:
                            guidance_loss = guidance_loss + (pred - float(target)).pow(2).mean()
                if synth_target is not None:
                    guidance_loss = guidance_loss + torch.relu(outputs_cond["synth_pred"] - float(synth_target)).mean()
                if isinstance(guidance_loss, torch.Tensor) and guidance_loss.requires_grad:
                    grad = torch.autograd.grad(guidance_loss, logits_cond, retain_graph=False, allow_unused=True)[0]
                    if grad is not None:
                        logits = logits - gradient_weight * grad
                logits_cond.requires_grad_(False)

            probs = torch.softmax(logits, dim=-1)
            sampled = torch.argmax(probs, dim=-1)
            update_mask = ~anchor_mask
            tokens = torch.where(update_mask, sampled, tokens).detach()
            attention_mask = tokens != self.vocab.pad_id
            final_outputs = {
                "synth_pred": outputs_cond["synth_pred"].detach(),
                "property_preds": {name: tensor.detach() for name, tensor in outputs_cond["property_preds"].items()},
            }

        sequences = tokens.detach().cpu().tolist()
        smiles = decode_tokens(self.vocab, sequences)
        results: List[Dict[str, object]] = []
        if final_outputs is None:
            dummy = torch.zeros(num_samples, device=self.device)
            final_outputs = {
                "synth_pred": dummy,
                "property_preds": {name: dummy for name in (property_targets or {}).keys()},
            }
        for idx, smile in enumerate(smiles):
            result = {
                "ap_smiles": smile,
                "logits": None,
                "prediction": {
                    "synth": float(final_outputs["synth_pred"][idx].detach().cpu()),
                },
            }
            for name, tensor in final_outputs["property_preds"].items():
                result["prediction"][name] = float(tensor[idx].detach().cpu())
            results.append(result)
        return results
