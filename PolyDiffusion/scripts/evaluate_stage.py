#!/usr/bin/env python
"""Evaluate model performance at any training stage with stage-specific metrics.

Stage A (Small Molecules):
- Validity (RDKit sanitize success rate)
- Synthesizability (mean synthesis score)
- Uniqueness
- Novelty (vs training set)
- Internal Diversity (Tanimoto distance)

Stage B (Polymers):
- Anchor Correctness
- Validity (RDKit sanitize success rate)
- Synthesizability (mean synthesis score)
- Uniqueness
- Novelty (vs training set)
- Internal Diversity (Tanimoto distance)

Stage C (Property-Guided Polymers):
- Property Hit Rate (fraction meeting target constraints)
- Property-specific metrics per trained property
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch
import numpy as np

from PolyDiffusion.chem.ap_smiles import ANCHOR1, ANCHOR2, unshield_anchors
from PolyDiffusion.chem.plain_vocab import PlainVocab
from PolyDiffusion.chem.valence import has_two_anchors
from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.utils.sa_score import calculate_sa_score_batch
from PolyDiffusion.sampling.sampler import GuidedSampler, PlainSampler, SamplerConfig
from PolyDiffusion.train.common import build_model, load_yaml

# Try to import RDKit for advanced validation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    from rdkit import DataStructs
    from rdkit import RDLogger
    HAS_RDKIT = True
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
except ImportError:
    HAS_RDKIT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def smiles_to_mol(smiles: str) -> Optional[object]:
    """Convert SMILES to RDKit mol, handling AP-SMILES by capping anchors."""
    if not HAS_RDKIT:
        return None

    # Cap anchors with carbon for validation
    capped = smiles.replace(ANCHOR1, "C").replace(ANCHOR2, "C")
    try:
        mol = Chem.MolFromSmiles(capped, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def load_samples_from_file(path: Path, stage: str) -> tuple[List[str], List[Dict[str, float]]]:
    """Load previously generated samples for evaluation."""
    if not path.exists():
        raise FileNotFoundError(f"Samples file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if stage == "a":
        return lines, [{} for _ in lines]

    if stage == "b":
        ap_smiles: List[str] = []
        for line in lines:
            try:
                ap = unshield_anchors(line)
            except ValueError:
                ap = line
            ap_smiles.append(ap)
        return ap_smiles, [{} for _ in ap_smiles]

    raise ValueError("Stage C metrics require model predictions; omit --samples for Stage C evaluation.")


def prepare_sa_inputs(samples: List[str], stage: str) -> List[str]:
    if stage == "a":
        return samples
    if stage == "b":
        return [s.replace(ANCHOR1, "C").replace(ANCHOR2, "C") for s in samples]
    return samples


def attach_sa_scores(
    samples: List[str],
    predictions: List[Dict[str, float]],
    stage: str,
) -> Optional[List[float]]:
    """Attach RDKit SA scores to prediction dictionaries when available."""
    if not samples:
        return None
    sa_inputs = prepare_sa_inputs(samples, stage)
    scores = calculate_sa_score_batch(sa_inputs, invalid_value=-1.0, show_progress=False)
    for pred, score in zip(predictions, scores):
        if pred is None:
            continue
        synth_pred = pred.get("synth")
        if synth_pred is not None and "synth_model" not in pred:
            pred["synth_model"] = synth_pred
        if score >= 0:
            pred["synth"] = float(score)
            pred["synth_source"] = "rdkit_sa"
            pred["sa_score_rdkit"] = float(score)
        else:
            # For invalid SMILES, don't use model prediction as fallback
            # Metrics should only use valid RDKit SA scores
            pred["synth_source"] = "invalid_smiles"
            pred["sa_score_rdkit"] = None
            pred["synth"] = None  # Don't fallback to model prediction
    return scores


def compute_sa_statistics(samples: List[str], stage: str) -> tuple[Optional[float], Optional[float], int]:
    if not HAS_RDKIT:
        return None, None, len(samples)
    sa_inputs = prepare_sa_inputs(samples, stage)
    scores = calculate_sa_score_batch(sa_inputs, invalid_value=-1.0, show_progress=False)
    valid_scores = [score for score in scores if score >= 0]
    invalid = len(scores) - len(valid_scores)
    if not valid_scores:
        return None, None, invalid
    return float(np.mean(valid_scores)), float(np.std(valid_scores)), invalid


def format_stat(value: Optional[float], precision: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def compute_stage_a_metrics(
    samples: List[str],
    predictions: List[Dict[str, float]],
    training_set: Optional[Set[str]] = None,
    stage: str = "a",
) -> Dict[str, float]:
    """Compute Stage A (small molecule) metrics.

    Metrics:
    - Validity: RDKit sanitize success rate
    - Synthesizability: Mean synthesis score
    - Uniqueness: Fraction of unique SMILES
    - Novelty: Fraction not in training set
    - Internal Diversity: Mean Tanimoto distance
    """
    total = len(samples)
    if total == 0:
        return {}

    metrics = {}

    # 1. Validity (RDKit sanitize)
    if HAS_RDKIT:
        valid_count = sum(1 for s in samples if smiles_to_mol(s) is not None)
        metrics["validity_rdkit"] = valid_count / total
    else:
        logger.warning("RDKit not available, validity check skipped")
        metrics["validity_rdkit"] = None

    # 2. Synthesizability (mean synthesis score)
    # Only use RDKit SA scores, not model predictions
    synth_scores: List[float] = []
    rdkit_count = 0
    invalid_count = 0

    for pred in predictions:
        if pred is None:
            continue
        # Only accept RDKit-calculated SA scores
        if pred.get("synth_source") == "rdkit_sa":
            synth = pred.get("synth")
            if synth is not None and synth >= 0:
                synth_scores.append(float(synth))
                rdkit_count += 1
        elif pred.get("synth_source") == "invalid_smiles":
            invalid_count += 1

    if synth_scores:
        metrics["synthesizability_mean"] = float(np.mean(synth_scores))
        metrics["synthesizability_std"] = float(np.std(synth_scores))
        metrics["synth_source"] = "rdkit_sa"
    else:
        # Fallback: compute directly if no valid predictions
        sa_mean, sa_std, invalid = compute_sa_statistics(samples, stage)
        metrics["synthesizability_mean"] = sa_mean
        metrics["synthesizability_std"] = sa_std
        metrics["synth_source"] = "rdkit_sa_fallback" if sa_mean is not None else "unavailable"
        invalid_count = invalid

    # Add validity statistics
    metrics["sa_valid_count"] = rdkit_count
    metrics["sa_invalid_count"] = invalid_count
    metrics["sa_valid_fraction"] = rdkit_count / total if total > 0 else 0.0

    # 3. Uniqueness
    unique_samples = set(samples)
    metrics["uniqueness"] = len(unique_samples) / total
    metrics["unique_count"] = len(unique_samples)

    # 4. Novelty (if training set provided)
    if training_set is not None:
        novel_samples = [s for s in unique_samples if s not in training_set]
        metrics["novelty"] = len(novel_samples) / len(unique_samples) if unique_samples else 0.0
        metrics["novel_count"] = len(novel_samples)
    else:
        metrics["novelty"] = None
        metrics["novel_count"] = None

    # 5. Internal Diversity (Tanimoto distance)
    if HAS_RDKIT and len(unique_samples) > 1:
        mols = [smiles_to_mol(s) for s in unique_samples]
        valid_mols = [m for m in mols if m is not None]

        if len(valid_mols) > 1:
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]
            distances = []
            for i in range(len(fps)):
                for j in range(i + 1, len(fps)):
                    distances.append(1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))

            metrics["internal_diversity_mean"] = np.mean(distances)
            metrics["internal_diversity_std"] = np.std(distances)
        else:
            metrics["internal_diversity_mean"] = None
            metrics["internal_diversity_std"] = None
    else:
        metrics["internal_diversity_mean"] = None
        metrics["internal_diversity_std"] = None

    return metrics


def compute_stage_b_metrics(
    samples: List[str],
    predictions: List[Dict[str, float]],
    training_set: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute Stage B (polymer) metrics.

    Includes all Stage A metrics plus:
    - Anchor Correctness: Fraction with exactly two anchors [*:1] and [*:2]
    """
    # Start with Stage A metrics
    metrics = compute_stage_a_metrics(samples, predictions, training_set, stage="b")

    # Add anchor correctness
    total = len(samples)
    if total > 0:
        anchor_correct = sum(1 for s in samples if has_two_anchors(s))
        metrics["anchor_correctness"] = anchor_correct / total
        metrics["anchor_correct_count"] = anchor_correct

    return metrics


def compute_stage_c_metrics(
    samples: List[str],
    predictions: List[Dict[str, float]],
    property_targets: Optional[Dict[str, tuple]] = None,
    target_property: Optional[str] = None,
) -> Dict[str, float]:
    """Compute Stage C (property-guided) metrics.

    Metrics:
    - Property Hit Rate: Fraction of samples meeting target constraints
    - Property-specific MAE/RMSE if ground truth available

    Args:
        samples: Generated AP-SMILES strings.
        predictions: Model predictions for each sample.
        property_targets: Dict mapping property name to (min, max) bounds.
        target_property: Name of the specific property this model was trained on.
    """
    total = len(samples)
    if total == 0:
        return {}

    metrics = {}

    # Add basic anchor correctness
    if total > 0:
        anchor_correct = sum(1 for s in samples if has_two_anchors(s))
        metrics["anchor_correctness"] = anchor_correct / total

    # Compute hit rate for target property
    if property_targets and target_property:
        if target_property in property_targets:
            min_val, max_val = property_targets[target_property]
            hits = 0
            for pred in predictions:
                val = pred.get(target_property)
                if val is not None and min_val <= val <= max_val:
                    hits += 1

            metrics[f"{target_property}_hit_rate"] = hits / total
            metrics[f"{target_property}_hits"] = hits
            metrics[f"{target_property}_target_range"] = f"[{min_val}, {max_val}]"

    # Property statistics
    if target_property:
        values = [p.get(target_property, 0.0) for p in predictions if target_property in p]
        if values:
            metrics[f"{target_property}_mean"] = np.mean(values)
            metrics[f"{target_property}_std"] = np.std(values)
            metrics[f"{target_property}_min"] = np.min(values)
            metrics[f"{target_property}_max"] = np.max(values)

    # Synthesizability - only use RDKit SA scores, ignore invalid SMILES
    synth_scores: List[float] = []
    rdkit_count = 0
    invalid_count = 0

    for pred in predictions:
        if pred is None:
            continue
        # Only accept RDKit-calculated SA scores
        if pred.get("synth_source") == "rdkit_sa":
            synth = pred.get("synth")
            if synth is not None and synth >= 0:
                synth_scores.append(float(synth))
                rdkit_count += 1
        elif pred.get("synth_source") == "invalid_smiles":
            invalid_count += 1

    if synth_scores:
        metrics["synthesizability_mean"] = float(np.mean(synth_scores))
        metrics["synthesizability_std"] = float(np.std(synth_scores))
        metrics["synth_source"] = "rdkit_sa"
    else:
        metrics["synthesizability_mean"] = None
        metrics["synthesizability_std"] = None
        metrics["synth_source"] = "unavailable"

    # Add validity statistics
    metrics["sa_valid_count"] = rdkit_count
    metrics["sa_invalid_count"] = invalid_count
    metrics["sa_valid_fraction"] = rdkit_count / total if total > 0 else 0.0

    return metrics


def load_training_set(path: Optional[Path], field: str = "smiles") -> Set[str]:
    """Load training set SMILES for novelty computation."""
    if path is None or not path.exists():
        return set()

    training_smiles = set()

    if path.suffix == ".csv":
        import csv
        with path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if field in row:
                    training_smiles.add(row[field])
    elif path.suffix in [".jsonl", ".gz"]:
        from PolyDiffusion.utils.fileio import stream_jsonl
        for record in stream_jsonl(path):
            if field in record:
                training_smiles.add(record[field])

    logger.info(f"Loaded {len(training_smiles)} training set SMILES from {path}")
    return training_smiles


def sample_and_evaluate(
    checkpoint_path: Path,
    vocab_path: Path,
    config_path: Path,
    num_samples: int,
    num_steps: int,
    max_length: int,
    min_tokens: int,
    stage: str,
    temperature: float,
    output_path: Optional[Path] = None,
    property_targets: Optional[Dict[str, tuple]] = None,
    target_property: Optional[str] = None,
    synth_target: Optional[float] = None,
    cfg_scale: float = 1.0,
    gradient_weight: float = 0.0,
    training_data_path: Optional[Path] = None,
    samples_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Sample from a checkpoint and compute stage-specific evaluation metrics."""

    logger.info(f"Loading model from {checkpoint_path}")
    stage = stage.lower()
    if stage == "a":
        vocab = PlainVocab.load(vocab_path)
    else:
        vocab = AnchorSafeVocab.load(vocab_path)
    model_cfg = load_yaml(config_path)
    model = build_model(vocab, model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    sampler_config = SamplerConfig(
        max_length=max_length,
        cfg_scale=cfg_scale,
        gradient_weight=gradient_weight,
        temperature=temperature,
        min_tokens=min_tokens,
    )

    smiles_list: List[str]
    predictions: List[Dict[str, float]]
    results: List[Dict[str, object]] = []
    sample_property_targets = None
    used_saved_samples = samples_path is not None

    if samples_path is not None:
        if stage == "c":
            raise ValueError("Stage C evaluation requires fresh sampling to obtain property predictions.")
        smiles_list, predictions = load_samples_from_file(samples_path, stage)
        logger.info(f"Loaded {len(smiles_list)} samples from {samples_path}")
    else:
        logger.info(f"Sampling {num_samples} molecules/polymers (steps={num_steps})...")
        if stage == "a":
            sampler = PlainSampler(model, vocab, sampler_config)
            results = sampler.sample(
                num_samples=num_samples,
                num_steps=num_steps,
                synth_target=synth_target,
                cfg_scale=cfg_scale,
                gradient_weight=gradient_weight,
            )
            smiles_list = [r["smiles"] for r in results]
        else:
            sampler = GuidedSampler(model, vocab, sampler_config)
            include_properties = stage == "c"
            if stage == "c" and target_property and property_targets and target_property in property_targets:
                min_val, max_val = property_targets[target_property]
                midpoint = (min_val + max_val) / 2.0
                sample_property_targets = {target_property: midpoint}
            if stage == "b" and property_targets:
                logger.warning("Property targets ignored for Stage B evaluation.")
            results = sampler.sample(
                num_samples=num_samples,
                num_steps=num_steps,
                property_targets=sample_property_targets if stage == "c" else None,
                synth_target=synth_target,
                cfg_scale=cfg_scale,
                gradient_weight=gradient_weight,
                include_properties=include_properties,
            )
            smiles_list = [r["ap_smiles"] for r in results]
        predictions = [r.get("prediction", {}) for r in results]

    attach_sa_scores(smiles_list, predictions, stage)

    effective_num_samples = len(smiles_list)

    # Load training set for novelty
    training_set = None
    if training_data_path:
        field = "smiles" if stage == "a" else "ap_smiles"
        training_set = load_training_set(training_data_path, field)

    # Compute stage-specific metrics
    logger.info(f"Computing Stage {stage.upper()} metrics...")

    if stage == "a":
        metrics = compute_stage_a_metrics(smiles_list, predictions, training_set, stage="a")
    elif stage == "b":
        metrics = compute_stage_b_metrics(smiles_list, predictions, training_set)
    elif stage == "c":
        metrics = compute_stage_c_metrics(smiles_list, predictions, property_targets, target_property)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Compile results
    sample_key = "smiles" if stage == "a" else "ap_smiles"
    evaluation = {
        "stage": stage,
        "checkpoint": str(checkpoint_path),
        "num_samples": effective_num_samples,
        "num_steps": num_steps if not used_saved_samples else None,
        "cfg_scale": cfg_scale if not used_saved_samples else None,
        "gradient_weight": gradient_weight if not used_saved_samples else None,
        "target_property": target_property,
        "property_targets": {k: list(v) for k, v in property_targets.items()} if property_targets else None,
        "metrics": metrics,
        "samples_path": str(samples_path) if samples_path else None,
        "used_saved_samples": used_saved_samples,
        "samples": [
            {sample_key: s, "prediction": p}
            for s, p in zip(smiles_list[:10], predictions[:10])
        ],
    }

    # Print summary
    print_evaluation_summary(stage, metrics, target_property)

    # Save to file
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(evaluation, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    return evaluation


def print_evaluation_summary(stage: str, metrics: Dict, target_property: Optional[str] = None):
    """Print formatted evaluation summary."""
    print("\n" + "="*60)
    print(f"STAGE {stage.upper()} EVALUATION RESULTS")
    print("="*60)

    if stage in ["a", "b"]:
        # Small molecules / Polymers
        if metrics.get("validity_rdkit") is not None:
            print(f"Validity (RDKit): {metrics['validity_rdkit']:.2%}")

        if stage == "b":
            print(f"Anchor Correctness: {metrics.get('anchor_correctness', 0):.2%}")

        # Show SA validity statistics
        sa_valid_fraction = metrics.get("sa_valid_fraction", 0)
        sa_valid_count = metrics.get("sa_valid_count", 0)
        sa_invalid_count = metrics.get("sa_invalid_count", 0)
        print(f"SA Valid Fraction: {sa_valid_fraction:.2%} ({sa_valid_count} valid, {sa_invalid_count} invalid)")

        synth_mean = metrics.get("synthesizability_mean")
        synth_std = metrics.get("synthesizability_std")
        if synth_mean is not None:
            print(f"Synthesizability: {format_stat(synth_mean)} ± {format_stat(synth_std)} (RDKit SA score for valid SMILES only)")
        else:
            print(f"Synthesizability: N/A (no valid SMILES generated)")
        print(f"Uniqueness: {metrics.get('uniqueness', 0):.2%} ({metrics.get('unique_count', 0)})")

        if metrics.get("novelty") is not None:
            print(f"Novelty: {metrics['novelty']:.2%} ({metrics.get('novel_count', 0)})")

        if metrics.get("internal_diversity_mean") is not None:
            print(f"Internal Diversity: {metrics['internal_diversity_mean']:.3f} ± {metrics.get('internal_diversity_std', 0):.3f}")

    elif stage == "c":
        # Property-guided polymers
        print(f"Anchor Correctness: {metrics.get('anchor_correctness', 0):.2%}")

        if target_property:
            hit_rate_key = f"{target_property}_hit_rate"
            if hit_rate_key in metrics:
                print(f"\n{target_property.upper()} Hit Rate: {metrics[hit_rate_key]:.2%}")
                print(f"  Target Range: {metrics.get(f'{target_property}_target_range', 'N/A')}")
                print(f"  Hits: {metrics.get(f'{target_property}_hits', 0)}")

            mean_key = f"{target_property}_mean"
            if mean_key in metrics:
                print(f"  Mean: {metrics[mean_key]:.2f} ± {metrics.get(f'{target_property}_std', 0):.2f}")
                print(f"  Range: [{metrics.get(f'{target_property}_min', 0):.2f}, {metrics.get(f'{target_property}_max', 0):.2f}]")

        synth_mean = metrics.get("synthesizability_mean")
        synth_std = metrics.get("synthesizability_std")
        print(f"\nSynthesizability: {format_stat(synth_mean)} ± {format_stat(synth_std)}")

    print("="*60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model performance with stage-specific metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage A: Small molecules
  polydiff-eval --stage a --ckpt stage_a.pt --vocab vocab.txt --config config.yaml --num 1000 --training-data data/molecules.jsonl.gz

  # Stage B: Polymers
  polydiff-eval --stage b --ckpt stage_b.pt --vocab vocab.txt --config config.yaml --num 1000 --training-data data/polymers.jsonl.gz

  # Stage C: Property-guided (Tg model)
  polydiff-eval --stage c --ckpt stage_c_tg.pt --vocab vocab.txt --config config.yaml --num 1000 \\
    --target-property Tg --property-range Tg 200 400 --cfg 2.0 --grad 0.3
        """,
    )

    # Required arguments
    parser.add_argument("--stage", type=str, choices=["a", "b", "c"], required=True, help="Training stage")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary file")
    parser.add_argument("--config", type=str, required=True, help="Model config YAML")

    # Sampling parameters
    parser.add_argument("--num", type=int, default=1000, help="Number of samples")
    parser.add_argument("--steps", type=int, default=10, help="Diffusion steps")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--samples", type=str, help="Path to pre-generated samples (.smi) to evaluate")
    parser.add_argument("--max-length", type=int, default=96, help="Maximum token length during sampling")
    parser.add_argument("--min-tokens", type=int, default=2, help="Minimum non-special tokens before allowing EOS")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (<=0 for greedy decoding)")

    # Stage C specific
    parser.add_argument("--target-property", type=str, help="Target property name (Tg, Tm, Td, Eg, chi)")
    parser.add_argument(
        "--property-range",
        nargs=3,
        action="append",
        metavar=("PROP", "MIN", "MAX"),
        help="Property range: PROP MIN MAX (can specify multiple)"
    )

    # Guidance parameters
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    parser.add_argument("--grad", type=float, default=0.0, help="Gradient guidance weight")
    parser.add_argument("--s_target", type=float, help="Synthesis score target")

    # Training data for novelty
    parser.add_argument("--training-data", type=str, help="Training data path for novelty computation")

    args = parser.parse_args()

    # Parse property ranges
    property_targets = None
    if args.property_range:
        property_targets = {}
        for prop, min_val, max_val in args.property_range:
            property_targets[prop] = (float(min_val), float(max_val))

    # Validate Stage C arguments
    if args.stage == "c" and not args.target_property:
        parser.error("Stage C requires --target-property")

    checkpoint_path = Path(args.ckpt)
    vocab_path = Path(args.vocab)
    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else None
    training_data_path = Path(args.training_data) if args.training_data else None
    samples_path = Path(args.samples) if args.samples else None

    min_tokens = max(0, args.min_tokens)
    default_min_tokens = parser.get_default("min_tokens")
    if args.min_tokens == default_min_tokens and args.stage in {"b", "c"}:
        min_tokens = max(min_tokens, 6)

    sample_and_evaluate(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        config_path=config_path,
        num_samples=args.num,
        num_steps=args.steps,
        max_length=args.max_length,
        min_tokens=min_tokens,
        stage=args.stage,
        temperature=args.temperature,
        output_path=output_path,
        property_targets=property_targets,
        target_property=args.target_property,
        synth_target=args.s_target,
        cfg_scale=args.cfg,
        gradient_weight=args.grad,
        training_data_path=training_data_path,
        samples_path=samples_path,
    )


if __name__ == "__main__":
    main()
