#!/usr/bin/env python
"""CLI for guided sampling."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional

import torch

from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.sampling.sampler import GuidedSampler, SamplerConfig
from PolyDiffusion.train.common import build_model, load_yaml


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
    parser.add_argument("--device", default="auto", type=str, help="Target device (auto/cpu/cuda).")
    args = parser.parse_args()

    vocab = AnchorSafeVocab.load(Path(args.vocab))
    model_cfg = load_yaml(Path(args.config))
    model = build_model(vocab, model_cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
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
