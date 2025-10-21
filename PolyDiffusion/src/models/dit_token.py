"""Diffusion Transformer wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
from torch import nn

from .diffusion_token import CategoricalDiffusion, DiffusionConfig
from .heads import GrammarHead, PropertyHeads, SynthesisHead, pooled_representation
from .modules import TransformerBackbone


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    dropout: float
    diffusion_steps: int
    property_names: Iterable[str]
    cfg_dropout: float = 0.1
    use_flow_matching: bool = False
    self_conditioning: bool = True


class ConditionNet(nn.Module):
    """Encode numeric property targets for conditioning."""

    def __init__(self, property_names: Iterable[str], hidden_size: int) -> None:
        super().__init__()
        self.property_names = list(property_names)
        input_dim = len(self.property_names) + 1  # plus synthesis target
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(
        self,
        properties: Optional[Dict[str, torch.Tensor]],
        s_target: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if properties is None:
            properties = {}
        feats = []
        for name in self.property_names:
            if name in properties and properties[name] is not None:
                feats.append(properties[name].unsqueeze(-1).float())
            else:
                feats.append(torch.zeros(batch_size, 1, device=device))
        if s_target is None:
            s_target = torch.zeros(batch_size, device=device)
        feats.append(s_target.unsqueeze(-1).float())
        features = torch.cat(feats, dim=-1)
        return self.net(features)


class DiffusionTransformer(nn.Module):
    """Full model containing token backbone and supervision heads."""

    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.config = model_config
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(model_config.vocab_size, model_config.hidden_size)
        # Preallocate positional embeddings for maximum sequence length to avoid dynamic expansion
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, model_config.hidden_size) * 0.01)
        self.time_embed = nn.Embedding(diffusion_config.num_steps + 1, model_config.hidden_size)
        self.dropout = nn.Dropout(model_config.dropout)
        self.condition_net = ConditionNet(model_config.property_names, model_config.hidden_size)
        self.backbone = TransformerBackbone(
            num_layers=model_config.num_layers,
            hidden_size=model_config.hidden_size,
            num_heads=model_config.num_heads,
            dropout=model_config.dropout,
            cond_dim=model_config.hidden_size,
        )
        self.head = nn.Linear(model_config.hidden_size, model_config.vocab_size)
        self.property_heads = PropertyHeads(model_config.hidden_size, model_config.property_names)
        self.synth_head = SynthesisHead(model_config.hidden_size)
        self.grammar_head = GrammarHead(model_config.hidden_size)
        self.diffusion = CategoricalDiffusion(diffusion_config)
        self.cfg_dropout = model_config.cfg_dropout

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        properties: Optional[Dict[str, torch.Tensor]] = None,
        s_target: Optional[torch.Tensor] = None,
        condition_dropout_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len = tokens.shape
        device = tokens.device

        # Validate sequence length
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum allowed length {self.max_seq_len}. "
                f"Consider increasing max_seq_len in model initialization."
            )

        timed = self.time_embed(timesteps).unsqueeze(1)
        x = self.token_embed(tokens) + timed + self.pos_embed[:, :seq_len, :]
        x = self.dropout(x)

        cond_emb = self.condition_net(properties, s_target, batch, device)
        if condition_dropout_mask is None and self.training and self.cfg_dropout > 0.0:
            condition_dropout_mask = torch.rand(batch, device=device) < self.cfg_dropout
        if condition_dropout_mask is not None:
            keep = (~condition_dropout_mask.bool()).float().unsqueeze(-1)
            cond_emb = cond_emb * keep

        if attention_mask is None:
            attention_mask = torch.ones(batch, seq_len, device=device, dtype=torch.bool)

        hidden = self.backbone(x, cond_emb, attn_mask=attention_mask)
        logits = self.head(hidden)
        pooled = pooled_representation(hidden, attention_mask)
        property_preds = self.property_heads(pooled)
        synth_pred = self.synth_head(pooled)
        grammar_logits = self.grammar_head(pooled)
        grammar_pred = torch.sigmoid(grammar_logits)

        return {
            "logits": logits,
            "pooled": pooled,
            "property_preds": property_preds,
            "synth_pred": synth_pred,
            "grammar_logits": grammar_logits,
            "grammar_pred": grammar_pred,
            "cond_emb": cond_emb,
        }
