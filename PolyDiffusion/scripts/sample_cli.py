#!/usr/bin/env python
"""CLI for guided sampling."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import torch

from PolyDiffusion.src.chem.vocab import AnchorSafeVocab
from PolyDiffusion.src.models.dit_token import DiffusionTransformer, ModelConfig
from PolyDiffusion.src.models.diffusion_token import DiffusionConfig
from PolyDiffusion.src.sampling.sampler import GuidedSampler, SamplerConfig
from PolyDiffusion.src.train.common import load_yaml


def parse_targets(target_str: Optional[str]) -> Dict[str, float]:
    if not target_str:
        return {}
    targets: Dict[str, float] = {}
    for item in target_str.split(","):
        item = item.strip()
        if not item:
            continue
        if "in" in item:
            match = re.match(r"(\w+)\s+in\s+\[(.+),(.+)\]", item)
            if match:
                name, low, high = match.groups()
                targets[name] = (float(low) + float(high)) / 2.0
                continue
        for op in ("<=", ">=", "="):
            if op in item:
                name, value = item.split(op)
                targets[name.strip()] = float(value)
                break
    return targets


def build_model(vocab: AnchorSafeVocab, config_path: Path) -> DiffusionTransformer:
    cfg = load_yaml(config_path)
    model_cfg = ModelConfig(
        vocab_size=len(vocab),
        hidden_size=cfg["d_model"],
        num_layers=cfg["n_layers"],
        num_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.1),
        diffusion_steps=cfg.get("diffusion_steps", 8),
        property_names=cfg.get("property_names", ["Tg", "Tm", "Td", "Eg", "chi"]),
        cfg_dropout=cfg.get("cfg_dropout", 0.1),
        use_flow_matching=cfg.get("use_flow_matching", False),
        self_conditioning=cfg.get("self_conditioning", True),
    )
    diffusion_cfg = DiffusionConfig(
        vocab_size=len(vocab),
        num_steps=model_cfg.diffusion_steps,
        mask_token_id=vocab.mask_id,
        schedule=cfg.get("schedule", "linear"),
        min_noise=cfg.get("min_noise", 0.05),
        max_noise=cfg.get("max_noise", 0.4),
    )
    return DiffusionTransformer(model_cfg, diffusion_cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str, help="Checkpoint path.")
    parser.add_argument("--vocab", required=True, type=str, help="Vocabulary file.")
    parser.add_argument("--config", default="configs/model_base.yaml", type=str, help="Model config YAML.")
    parser.add_argument("--num", default=10, type=int, help="Number of samples.")
    parser.add_argument("--steps", default=10, type=int, help="Diffusion steps.")
    parser.add_argument("--targets", default="", type=str, help="Comma separated property targets.")
    parser.add_argument("--s_target", default=None, type=float, help="Synthesis score target.")
    parser.add_argument("--cfg", default=1.5, type=float, help="CFG scale.")
    parser.add_argument("--grad", default=0.0, type=float, help="Gradient guidance weight.")
    args = parser.parse_args()

    vocab = AnchorSafeVocab.load(Path(args.vocab))
    model = build_model(vocab, Path(args.config))
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    sampler = GuidedSampler(model, vocab, SamplerConfig(cfg_scale=args.cfg, gradient_weight=args.grad))

    property_targets = parse_targets(args.targets)
    results = sampler.sample(
        num_samples=args.num,
        num_steps=args.steps,
        property_targets=property_targets,
        synth_target=args.s_target,
        cfg_scale=args.cfg,
        gradient_weight=args.grad,
    )

    for item in results:
        print(item["ap_smiles"], item["prediction"])


if __name__ == "__main__":
    main()
