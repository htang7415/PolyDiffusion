#!/usr/bin/env python
"""CLI for stage-aware sampling with optional output logging and metrics."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch

from PolyDiffusion.chem.plain_vocab import PlainVocab
from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.sampling.sampler import GuidedSampler, PlainSampler, SamplerConfig
from PolyDiffusion.train.common import build_model, load_yaml
from PolyDiffusion.scripts.evaluate_stage import (
    compute_stage_a_metrics,
    compute_stage_b_metrics,
    compute_stage_c_metrics,
)

try:  # pragma: no cover - optional dependency
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None

try:  # pragma: no cover - optional dependency
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


STAGE_METRIC_KEYS = {
    "a": [
        "validity_rdkit",
        "synthesizability_mean",
        "synthesizability_std",
        "uniqueness",
        "internal_diversity_mean",
    ],
    "b": [
        "validity_rdkit",
        "synthesizability_mean",
        "synthesizability_std",
        "uniqueness",
        "internal_diversity_mean",
        "anchor_correctness",
    ],
    "c": [
        "anchor_correctness",
        "synthesizability_mean",
        "synthesizability_std",
    ],
}


def _get_peak_memory_mb() -> float:
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage:
            # On macOS ru_maxrss is in bytes, on Linux it's kilobytes.
            if sys.platform.startswith("darwin"):
                usage /= 1024.0
            return usage / 1024.0
    if psutil is not None:
        try:
            rss = psutil.Process().memory_info().rss
            return rss / (1024.0 * 1024.0)
        except Exception:  # pragma: no cover
            pass
    return 0.0


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
    parser = argparse.ArgumentParser(description="Sample molecules/polymers from trained PolyDiffusion checkpoints.")
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
    parser.add_argument("--stage", type=str, choices=["a", "b", "c"], required=True, help="Training stage of the checkpoint.")
    parser.add_argument("--output", type=str, help="Write sampled structures (one per line).")
    parser.add_argument("--max-length", type=int, default=64, help="Maximum token length during sampling (default: 64).")
    parser.add_argument("--print-samples", action="store_true", help="Print sampled structures to stdout.")
    args = parser.parse_args()

    start_time = time.perf_counter()

    stage = args.stage.lower()
    vocab_path = Path(args.vocab)
    model_cfg = load_yaml(Path(args.config))

    if stage == "a":
        vocab = PlainVocab.load(vocab_path)
    else:
        vocab = AnchorSafeVocab.load(vocab_path)

    model = build_model(vocab, model_cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    if device.type == "cuda":  # reset peak tracking for clean measurement
        torch.cuda.reset_peak_memory_stats(device)
    sampler_cfg = SamplerConfig(max_length=args.max_length, cfg_scale=args.cfg, gradient_weight=args.grad)

    property_targets = parse_targets(args.targets)

    if stage == "a":
        if property_targets:
            raise ValueError("Stage A sampling does not support property targets.")
        sampler = PlainSampler(model, vocab, sampler_cfg)
        results = sampler.sample(
            num_samples=args.num,
            num_steps=args.steps,
            synth_target=args.s_target,
            cfg_scale=args.cfg,
            gradient_weight=args.grad,
        )
        display_key = "smiles"
    else:
        sampler = GuidedSampler(model, vocab, sampler_cfg)
        include_properties = stage == "c"
        if stage == "b" and property_targets:
            raise ValueError("Stage B models do not support property targets.")
        results = sampler.sample(
            num_samples=args.num,
            num_steps=args.steps,
            property_targets=property_targets if stage == "c" else None,
            synth_target=args.s_target,
            cfg_scale=args.cfg,
            gradient_weight=args.grad,
            include_properties=include_properties,
        )
        display_key = "ap_smiles"

    structures = [item.get(display_key, "") for item in results]
    predictions = [item.get("prediction", {}) for item in results]

    # Compute stage-specific metrics from sampled structures.
    if stage == "a":
        metrics = compute_stage_a_metrics(structures, predictions, training_set=None, stage="a")
    elif stage == "b":
        metrics = compute_stage_b_metrics(structures, predictions, training_set=None)
    else:
        metrics = compute_stage_c_metrics(structures, predictions, property_targets=None, target_property=None)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for structure in structures:
                handle.write(f"{structure}\n")
        metrics_path = out_path.with_name("metrics.csv")
        metric_keys = STAGE_METRIC_KEYS.get(stage, sorted(metrics.keys()))
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key in metric_keys:
                value = metrics.get(key)
                writer.writerow([key, "" if value is None else value])
    else:
        metrics_path = None

    elapsed = time.perf_counter() - start_time
    peak_ram = _get_peak_memory_mb()
    peak_cuda = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    if args.print_samples:
        for structure, prediction in zip(structures, predictions):
            synth = prediction.get("synth")
            extras = {k: v for k, v in prediction.items() if k != "synth"}
            if extras:
                extra_str = ", ".join(f"{k}={v:.3f}" for k, v in extras.items())
                print(f"{structure} | synth={synth:.3f} | {extra_str}")
            else:
                print(f"{structure} | synth={synth:.3f}")

    summary_parts = [f"{len(structures)} samples"]
    if args.output:
        summary_parts.append(f"samples → {args.output}")
        if metrics_path:
            summary_parts.append(f"metrics → {metrics_path}")
    summary_parts.append(f"elapsed {elapsed:.2f}s")
    summary_parts.append(f"peak RAM {peak_ram:.1f} MB")
    if device.type == "cuda":
        summary_parts.append(f"peak CUDA {peak_cuda:.1f} MB")
    print(" | ".join(summary_parts))


if __name__ == "__main__":
    main()
