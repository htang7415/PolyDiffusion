#!/usr/bin/env python
"""CLI for stage-aware sampling with optional output logging and metrics."""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from PolyDiffusion.chem.ap_smiles import SHIELD1, SHIELD2, convert_polymer_to_ap_smiles
from PolyDiffusion.chem.plain_vocab import PlainVocab
from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.sampling.sampler import GuidedSampler, PlainSampler, SamplerConfig
from PolyDiffusion.utils.sa_score import RDKIT_AVAILABLE, calculate_sa_score_batch
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

_SA_WARNING_EMITTED = False


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


def _collect_vocab_files(root: Path, recursive: bool = False) -> List[Path]:
    """Return sorted list of vocabulary files under a directory tree."""
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.name.startswith("vocab") and root.suffix == ".txt" else []
    pattern = "**/vocab*.txt" if recursive else "vocab*.txt"
    return sorted(path for path in root.glob(pattern) if path.is_file())


def _prepare_sa_inputs(structures: List[str], stage: str) -> List[str]:
    stage = stage.lower()
    if stage == "a":
        return [s or "" for s in structures]
    capped: List[str] = []
    for structure in structures:
        if not structure:
            capped.append("")
        else:
            capped.append(
                structure.replace(SHIELD1, "C")
                .replace(SHIELD2, "C")
                .replace("[*:1]", "C")
                .replace("[*:2]", "C")
                .replace("[*]", "C")
                .replace("*", "C")
            )
    return capped


def _attach_sa_scores(
    structures: List[str],
    predictions: List[Dict[str, float]],
    stage: str,
) -> Optional[List[float]]:
    global _SA_WARNING_EMITTED
    if not RDKIT_AVAILABLE:
        if not _SA_WARNING_EMITTED:
            print("RDKit not available; skipping SA score calculation.", file=sys.stderr)
            _SA_WARNING_EMITTED = True
        return None

    sa_inputs = _prepare_sa_inputs(structures, stage)
    scores = calculate_sa_score_batch(sa_inputs, invalid_value=-1.0, show_progress=False)
    for pred, score in zip(predictions, scores):
        if pred is None:
            continue
        if "synth_model" not in pred and pred.get("synth") is not None:
            pred["synth_model"] = pred.get("synth")
        if score >= 0:
            pred["synth"] = float(score)
            pred["synth_source"] = "rdkit_sa"
            pred["sa_score_rdkit"] = float(score)
        else:
            pred.setdefault("synth_source", "model_prediction" if pred.get("synth") is not None else "unavailable")
            pred["sa_score_rdkit"] = None
            if pred.get("synth") is None and pred.get("synth_model") is not None:
                pred["synth"] = pred["synth_model"]
    return scores


def _ap_to_plain(ap_smiles: str) -> str:
    if not ap_smiles:
        return ap_smiles
    return ap_smiles.replace("[*:1]", "*").replace("[*:2]", "*")


def _plain_to_ap(smiles: str) -> str:
    if not smiles:
        return ""
    try:
        return convert_polymer_to_ap_smiles(smiles)
    except ValueError:
        return ""


def _format_structure(structure: str, stage: str, anchor_format: str) -> str:
    if not structure:
        return structure
    if stage in {"b", "c"} and anchor_format == "plain":
        return structure.replace("[*:1]", "*").replace("[*:2]", "*")
    return structure


def _resolve_vocab_path(stage: str, ckpt_path: Path, vocab_arg: Optional[str]) -> Path:
    stage = stage.lower()
    candidates: List[Path] = []
    if vocab_arg:
        candidates.append(Path(vocab_arg))

    if ckpt_path:
        ckpt_dir = ckpt_path.parent
        candidates.append(ckpt_dir / "vocab.txt")
        candidates.extend(_collect_vocab_files(ckpt_dir, recursive=False))

    stage_dirs: Dict[str, List[tuple[Path, bool]]] = {
        "a": [(Path("Results/stage_a"), True)],
        "b": [(Path("Results/stage_b"), True)],
        "c": [(Path("Results/stage_c"), True), (Path("Results/stage_b"), True)],
    }
    for directory, recursive in stage_dirs.get(stage, []):
        candidates.extend(_collect_vocab_files(directory, recursive=recursive))

    repo_dirs: List[tuple[Path, bool]] = [
        (Path("PolyDiffusion"), False),
    ]
    for directory, recursive in repo_dirs:
        candidates.extend(_collect_vocab_files(directory, recursive=recursive))

    # Preserve order while removing duplicate string representations
    seen: set[str] = set()
    ordered_candidates: List[Path] = []
    for candidate in candidates:
        expanded = candidate.expanduser()
        key = str(expanded)
        if key not in seen:
            seen.add(key)
            ordered_candidates.append(expanded)

    for idx, candidate in enumerate(ordered_candidates):
        if candidate.exists():
            if idx == 0:
                return candidate
            if ckpt_path and vocab_arg and idx == 1:
                message = (
                    "[sample_cli] Provided vocabulary path was not found. "
                    f"Using checkpoint directory fallback: {candidate}"
                )
            elif ckpt_path and idx == 1:
                message = (
                    "[sample_cli] Vocabulary file not found in the expected checkpoint directory; "
                    f"using fallback: {candidate}"
                )
            else:
                prefix = "Provided vocabulary path was not found. " if vocab_arg else ""
                message = f"[sample_cli] {prefix}Using stage default vocabulary: {candidate}"
            print(message, file=sys.stderr)
            return candidate

    searched = "\n  ".join(str(path) for path in ordered_candidates)
    raise FileNotFoundError(
        "Unable to locate a vocabulary file for sampling. Checked the following locations:\n"
        f"  {searched}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample molecules/polymers from trained PolyDiffusion checkpoints.")
    parser.add_argument("--ckpt", required=True, type=str, help="Checkpoint path.")
    parser.add_argument(
        "--vocab",
        type=str,
        help="Vocabulary file. If omitted, attempts to infer from the checkpoint folder or stage defaults.",
    )
    parser.add_argument("--config", default="configs/model_base.yaml", type=str, help="Model config YAML.")
    parser.add_argument("--num", default=10, type=int, help="Number of samples.")
    parser.add_argument("--steps", default=10, type=int, help="Diffusion steps.")
    parser.add_argument("--targets", default="", type=str, help="Comma separated property targets.")
    parser.add_argument("--cfg", default=1.5, type=float, help="CFG scale.")
    parser.add_argument("--grad", default=0.0, type=float, help="Gradient guidance weight.")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature (<=0 for greedy decoding).")
    parser.add_argument("--device", default="auto", type=str, help="Target device (auto/cpu/cuda).")
    parser.add_argument("--stage", type=str, choices=["a", "b", "c"], required=True, help="Training stage of the checkpoint.")
    parser.add_argument("--output", type=str, help="Write sampled structures (one per line).")
    parser.add_argument("--max-length", type=int, default=96, help="Maximum token length during sampling (default: 64).")
    parser.add_argument("--min-tokens", type=int, default=2, help="Minimum non-special tokens required before terminating a sequence.")
    parser.add_argument(
        "--min-sample-length",
        type=int,
        help="Minimum non-special token count (excluding BOS and first anchor) before EOS is encouraged.",
    )
    parser.add_argument(
        "--max-sample-length",
        type=int,
        help="Maximum non-special token count (excluding BOS and first anchor) before EOS is encouraged.",
    )
    parser.add_argument(
        "--anchor-format",
        type=str,
        default="labeled",
        choices=["labeled", "plain"],
        help="Stage B/C only: choose 'labeled' for [*:1]/[*:2] anchors (default) or 'plain' for bare '*' attachment points in outputs.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()

    stage = args.stage.lower()
    ckpt_path = Path(args.ckpt)
    vocab_path = _resolve_vocab_path(stage, ckpt_path, args.vocab)
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
    min_tokens = max(0, args.min_tokens)
    if args.min_tokens == parser.get_default("min_tokens") and stage in {"b", "c"}:
        min_tokens = max(min_tokens, 6)  # ensure polymer outputs emit anchor + payload tokens

    if (
        args.min_sample_length is not None
        and args.max_sample_length is not None
        and args.max_sample_length < args.min_sample_length
    ):
        parser.error("--max-sample-length must be >= --min-sample-length")

    sampler_cfg = SamplerConfig(
        max_length=args.max_length,
        cfg_scale=args.cfg,
        gradient_weight=args.grad,
        temperature=args.temperature,
        min_tokens=min_tokens,
        target_length_min=args.min_sample_length,
        target_length_max=args.max_sample_length,
    )

    property_targets = parse_targets(args.targets)

    if stage == "a":
        if property_targets:
            raise ValueError("Stage A sampling does not support property targets.")
        sampler = PlainSampler(model, vocab, sampler_cfg)
        results = sampler.sample(
            num_samples=args.num,
            num_steps=args.steps,
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
            cfg_scale=args.cfg,
            gradient_weight=args.grad,
            include_properties=include_properties,
        )
        display_key = "ap_smiles"

    structures = [item.get(display_key, "") for item in results]
    predictions = [item.get("prediction", {}) for item in results]
    ap_structures = structures
    if stage == "a":
        plain_structures = ap_structures
    else:
        plain_structures = [_ap_to_plain(s) for s in ap_structures]

    if stage == "a":
        metrics_structures = plain_structures
    else:
        metrics_structures = [_plain_to_ap(s) for s in plain_structures]

    sa_structures = metrics_structures if stage == "a" else plain_structures
    sa_scores = _attach_sa_scores(sa_structures, predictions, stage)

    # Compute stage-specific metrics from sampled structures.
    if stage == "a":
        metrics = compute_stage_a_metrics(metrics_structures, predictions, training_set=None, stage="a")
    elif stage == "b":
        metrics = compute_stage_b_metrics(metrics_structures, predictions, training_set=None)
    else:
        metrics = compute_stage_c_metrics(metrics_structures, predictions, property_targets=None, target_property=None)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            for structure in plain_structures:
                handle.write(f"{structure}\n")
        ap_output_path: Optional[Path] = None
        if stage in {"b", "c"}:
            if out_path.name.lower() == "samples.smi":
                ap_output_path = out_path.with_name("AP-samples.smi")
            else:
                ap_output_path = out_path.with_name(f"{out_path.stem}_ap{out_path.suffix}")
            with ap_output_path.open("w", encoding="utf-8") as handle:
                for ap_smiles in ap_structures:
                    handle.write(f"{ap_smiles}\n")
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
        ap_output_path = None

    elapsed = time.perf_counter() - start_time
    peak_ram = _get_peak_memory_mb()
    peak_cuda = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)

    summary_parts = [f"{len(plain_structures)} samples"]
    if args.output:
        summary_parts.append(f"samples → {args.output}")
        if ap_output_path is not None:
            summary_parts.append(f"ap-samples → {ap_output_path}")
        if metrics_path:
            summary_parts.append(f"metrics → {metrics_path}")
    summary_parts.append(f"elapsed {elapsed:.2f}s")
    summary_parts.append(f"peak RAM {peak_ram:.1f} MB")
    if device.type == "cuda":
        summary_parts.append(f"peak CUDA {peak_cuda:.1f} MB")
    if sa_scores is not None:
        valid_sa = sum(1 for score in sa_scores if score is not None and score >= 0)
        summary_parts.append(f"RDKit SA {valid_sa}/{len(sa_scores)}")
    print(" | ".join(summary_parts))


if __name__ == "__main__":
    main()
