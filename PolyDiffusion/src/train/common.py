"""Shared helpers for training stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import json
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..chem import valence as valence_utils
from ..chem.ap_smiles import ANCHOR1, ANCHOR2
from ..chem import convert_polymer_to_ap_smiles
from ..chem.base_vocab import BaseVocabulary
from ..chem.vocab import AnchorSafeVocab  # Backward compat
from ..chem.plain_vocab import PlainVocab  # Backward compat
from ..chem.vocab_config import TokenizationConfig, load_tokenization_config
from ..chem.vocab_factory import create_vocabulary, load_vocabulary_auto
from ..data.collate import collate_token_batch
from ..data.datasets import CsvDataset, DatasetConfig, JsonlDataset
from ..models.dit_token import DiffusionTransformer, ModelConfig
from ..models.diffusion_token import DiffusionConfig
from ..utils.fileio import open_compressed


PROPERTY_NAMES = ["Tg", "Tm", "Td", "Eg", "chi"]
COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".zip"}
SMILES_KEYS_STAGE_A: Sequence[str] = ("SMILES", "smiles", "Smiles")
RAW_POLYMER_SMILES_KEYS: Sequence[str] = ("SMILES", "smiles", "Smiles")
SYNTH_SCORE_KEYS: Sequence[str] = ("synth_score", "SA_Score", "SA Score")
Record = MutableMapping[str, object]


def _sniff_dataset_format(path: Path) -> str:
    """Fallback format detection by peeking at the file contents."""
    with open_compressed(path, "rt") as handle:
        for _ in range(32):
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("{") or stripped.startswith("["):
                return "jsonl"
            if "," in stripped or "\t" in stripped:
                return "csv"
    raise ValueError(
        "Unable to infer dataset format from contents. "
        f"Please rename the file with a .csv/.tsv/.jsonl extension or specify 'format' in the config. (path={path})"
    )


def _infer_dataset_format(path: Path) -> str:
    """Infer whether a dataset should be parsed as CSV or JSONL."""
    suffixes = [suffix.lower() for suffix in path.suffixes]
    while suffixes and suffixes[-1] in COMPRESSION_SUFFIXES:
        suffixes.pop()
    if suffixes:
        last = suffixes[-1]
        if last in (".jsonl", ".json"):
            return "jsonl"
        if last in (".csv", ".tsv"):
            return "csv"
        raise ValueError(f"Unsupported dataset format for path: {path}")
    return _sniff_dataset_format(path)


def _find_first_key(record: Record, candidates: Sequence[str]) -> Optional[str]:
    """Return the first key present in a record from a candidate list."""
    for key in candidates:
        if key in record:
            return key
    return None


def _extract_optional_float(record: Record, keys: Sequence[str], default: float = 0.0) -> float:
    """Extract a numeric field from a record, tolerating missing or invalid values."""
    log = logging.getLogger(__name__)
    for key in keys:
        if key not in record:
            continue
        value = record[key]
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                return float(stripped)
            except ValueError:
                log.warning("Failed to parse %s=%r as float. Using default %.3f.", key, value, default)
                return default
        try:
            return float(value)
        except (TypeError, ValueError):
            log.warning("Failed to parse %s=%r as float. Using default %.3f.", key, value, default)
            return default
    return default


def build_vocab_from_dataset(
    dataset,
    stage: str,
    tokenization_config: TokenizationConfig,
    limit: Optional[int] = None,
) -> BaseVocabulary:
    """
    Build vocabulary automatically from dataset using configured tokenization method.

    Args:
        dataset: Dataset to build vocab from
        stage: "a", "b", or "c"
        tokenization_config: Tokenization configuration
        limit: Max number of samples to use (overrides config if provided)

    Returns:
        BaseVocabulary instance (method determined by tokenization_config)
    """
    log = logging.getLogger(__name__)

    # Use provided limit or fall back to config
    if limit is None:
        limit = tokenization_config.vocab_limit_samples

    log.info(f"Building {tokenization_config.method} vocabulary from dataset (limit={limit})...")

    # Extract corpus from dataset
    corpus = []
    count = 0
    for record in dataset:
        if limit is not None and count >= limit:
            break

        if stage == "a":
            # Stage A: plain SMILES
            smiles_key = None
            for key in ["SMILES", "smiles"]:
                if key in record:
                    smiles_key = key
                    break
            if smiles_key:
                corpus.append(str(record[smiles_key]))
                count += 1
        else:
            # Stage B/C: polymer SMILES with attachment points
            if "ap_smiles" in record:
                corpus.append(str(record["ap_smiles"]))
                count += 1
            else:
                # Raw polymer SMILES - convert
                smiles_key = None
                for key in ["SMILES", "smiles", "Smiles"]:
                    if key in record:
                        smiles_key = key
                        break
                if smiles_key:
                    raw = str(record[smiles_key])
                    try:
                        ap = convert_polymer_to_ap_smiles(raw)
                        corpus.append(ap)
                        count += 1
                    except ValueError as e:
                        log.warning(f"Skipping invalid polymer SMILES '{raw}': {e}")

    if not corpus:
        raise RuntimeError("No valid SMILES found in dataset to build vocabulary")

    log.info(f"Extracted {len(corpus)} SMILES strings from dataset")

    # Use factory to create vocabulary
    vocab = create_vocabulary(
        config=tokenization_config,
        stage=stage,
        corpus=corpus
    )

    log.info(f"Built vocabulary with {len(vocab)} tokens using {tokenization_config.method} tokenization")

    return vocab


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(vocab: AnchorSafeVocab | PlainVocab, model_cfg: dict, max_seq_len: int = 2048) -> DiffusionTransformer:
    """Build DiffusionTransformer model from configuration.

    Args:
        vocab: Vocabulary (PlainVocab for Stage A, AnchorSafeVocab for Stage B/C).
        model_cfg: Model configuration dictionary from YAML.
        max_seq_len: Maximum sequence length for positional embeddings (default: 2048).

    Returns:
        DiffusionTransformer model instance.
    """
    model_config = ModelConfig(
        vocab_size=len(vocab),
        hidden_size=model_cfg["d_model"],
        num_layers=model_cfg["n_layers"],
        num_heads=model_cfg["n_heads"],
        dropout=model_cfg.get("dropout", 0.1),
        diffusion_steps=model_cfg.get("diffusion_steps", 200),
        property_names=model_cfg.get("property_names", PROPERTY_NAMES),
        cfg_dropout=model_cfg.get("cfg_dropout", 0.1),
        use_flow_matching=model_cfg.get("use_flow_matching", False),
        self_conditioning=model_cfg.get("self_conditioning", True),
    )
    diffusion_config = DiffusionConfig(
        vocab_size=len(vocab),
        num_steps=model_config.diffusion_steps,
        mask_token_id=vocab.mask_id,
        schedule=model_cfg.get("schedule", "linear"),
        min_noise=model_cfg.get("min_noise", 0.01),
        max_noise=model_cfg.get("max_noise", 0.5),
    )
    return DiffusionTransformer(model_config, diffusion_config, max_seq_len=max_seq_len)


def split_dataset(
    dataset,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple:
    """
    Split a dataset into train/validation/test subsets using deterministic random splitting.

    Args:
        dataset: Dataset to split
        ratios: Tuple of (train_ratio, val_ratio, test_ratio). Must sum to 1.0.
        seed: Random seed for reproducible splits (default: 42)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    total_size = len(dataset)
    train_size = int(ratios[0] * total_size)
    val_size = int(ratios[1] * total_size)
    test_size = total_size - train_size - val_size  # Ensure all samples are used

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    log = logging.getLogger(__name__)
    log.info(
        f"Split dataset: train={train_size} ({ratios[0]*100:.0f}%), "
        f"val={val_size} ({ratios[1]*100:.0f}%), "
        f"test={test_size} ({ratios[2]*100:.0f}%), "
        f"total={total_size} (seed={seed})"
    )

    return train_dataset, val_dataset, test_dataset


def build_stage_dataset(
    stage: str,
    data_cfg: dict,
    property_names: Sequence[str] | None = None,
) -> JsonlDataset | CsvDataset:
    config = DatasetConfig(
        path=Path(data_cfg["path"]),
        limit=data_cfg.get("limit"),
        shuffle=data_cfg.get("shuffle", True),
        cache_in_memory=data_cfg.get("cache_in_memory", True),
        seed=data_cfg.get("seed"),
        delimiter=data_cfg.get("delimiter"),
        fieldnames=data_cfg.get("fieldnames"),
        has_header=data_cfg.get("has_header", True),
    )
    dataset_format = data_cfg.get("format")
    if dataset_format is None:
        dataset_format = _infer_dataset_format(config.path)
    dataset_format = dataset_format.lower()
    if dataset_format not in {"csv", "jsonl"}:
        raise ValueError(f"Unsupported dataset format '{dataset_format}'. Expected 'csv' or 'jsonl'.")
    dataset_cls = CsvDataset if dataset_format == "csv" else JsonlDataset

    if stage == "a":
        # No required fields - collate will auto-detect and handle missing SA_Score
        return dataset_cls(config, required_fields=set())
    if stage == "b":
        # No required fields - collate will auto-detect and handle missing SA_Score
        return dataset_cls(config, required_fields=set())

    property_fields = set(property_names or PROPERTY_NAMES)
    required = {"ap_smiles"} | property_fields
    return dataset_cls(config, required_fields=required)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_stage_a(records: List[Dict[str, object]], vocab: BaseVocabulary) -> Dict[str, torch.Tensor]:
    """
    Collate Stage A batch (small molecules without attachment points).

    Supports multiple formats:
    - CSV with SA_Score: SMILES, SA_Score
    - CSV without score: SMILES (uses default 0.0)
    - TSV/GZ: index<tab>SMILES (uses default 0.0)
    - JSONL: smiles, synth_score
    """
    # Detect SMILES column name
    smiles_key = _find_first_key(records[0], SMILES_KEYS_STAGE_A)
    if smiles_key is None:
        raise KeyError("No SMILES column found in dataset. Expected one of: 'SMILES', 'Smiles', 'smiles'.")

    # Tokenize plain SMILES (no attachment points)
    tokens = []
    for record in records:
        if smiles_key not in record:
            raise KeyError(f"Record missing SMILES column '{smiles_key}'.")
        tokens.append(vocab.tokenize(str(record[smiles_key])))
    batch = collate_token_batch(tokens, vocab.pad_id)

    return batch


def collate_stage_b(records: List[Dict[str, object]], vocab: BaseVocabulary) -> Dict[str, torch.Tensor]:
    """
    Collate Stage B batch (polymers with attachment points).

    Supports both formats:
    - Preprocessed JSONL: {"ap_smiles": "[*:1]CCC[*:2]", "synth_score": 6.88}
    - Raw CSV/GZ: {"SMILES": "*CCC*", "SA Score": 6.88} (auto-converts to AP-SMILES)
    - Raw without score: {"SMILES": "*CCC*"} or {"Smiles": "*CCC*"} (uses default 0.0)
    """
    # Check format: preprocessed (ap_smiles) or raw (SMILES)
    if "ap_smiles" in records[0]:
        # Preprocessed format
        aps = [str(r["ap_smiles"]) for r in records]
    else:
        # Raw CSV/GZ format - need to convert
        smiles_key = _find_first_key(records[0], RAW_POLYMER_SMILES_KEYS)
        if smiles_key is None:
            raise KeyError("No SMILES column found. Expected 'SMILES', 'smiles', or 'Smiles'.")

        # Convert raw polymer SMILES to AP-SMILES
        raw_smiles = [str(r[smiles_key]) for r in records]
        aps = []
        log = logging.getLogger(__name__)
        for raw in raw_smiles:
            try:
                ap = convert_polymer_to_ap_smiles(raw)
                aps.append(ap)
            except ValueError as e:
                log.error("Failed to convert polymer SMILES '%s': %s", raw, e)
                raise ValueError(
                    f"Unable to convert polymer SMILES '{raw}' to AP-SMILES. "
                    "Ensure each repeat unit has exactly two attachment points ('*' or '[*]')."
                ) from e

    # Tokenize AP-SMILES (use unified tokenize() interface)
    tokens = [vocab.tokenize(ap) for ap in aps]
    batch = collate_token_batch(tokens, vocab.pad_id)
    anchor_count = torch.tensor([ap.count(ANCHOR1) + ap.count(ANCHOR2) for ap in aps], dtype=torch.int64)
    valence = torch.tensor([1.0 if valence_utils.valence_ok(ap) else 0.0 for ap in aps], dtype=torch.float32)
    batch["anchor_count"] = anchor_count
    batch["valence"] = valence
    return batch


def collate_stage_c(
    records: List[Dict[str, object]],
    vocab: BaseVocabulary,
    property_names: Sequence[str] = PROPERTY_NAMES,
) -> Dict[str, torch.Tensor]:
    aps = [str(r["ap_smiles"]) for r in records]
    tokens = [vocab.tokenize(ap) for ap in aps]  # Use unified tokenize() interface
    batch = collate_token_batch(tokens, vocab.pad_id)
    properties: Dict[str, torch.Tensor] = {}
    for name in property_names:
        values = torch.tensor([float(r[name]) for r in records], dtype=torch.float32)
        properties[name] = values
    anchor_count = torch.tensor([ap.count(ANCHOR1) + ap.count(ANCHOR2) for ap in aps], dtype=torch.int64)
    valence = torch.tensor([1.0 if valence_utils.valence_ok(ap) else 0.0 for ap in aps], dtype=torch.float32)
    batch["properties"] = properties
    batch["anchor_count"] = anchor_count
    batch["valence"] = valence
    return batch


def make_dataloader(
    dataset,
    batch_size: int,
    collate_fn,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for training or evaluation.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        collate_fn: Collation function
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory (default: auto-detect CUDA)
        shuffle: Whether to shuffle the data (default: True for training, False for eval)

    Returns:
        DataLoader instance
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def save_checkpoint(
    path: Path,
    model: DiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object] = None,
    step: int = 0,
    loss: float = 0.0,
    metadata: Optional[Dict] = None,
) -> None:
    """Save complete checkpoint with model, optimizer, scheduler state."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
        "model_config": {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_layers": model.config.num_layers,
            "num_heads": model.config.num_heads,
            "dropout": model.config.dropout,
            "diffusion_steps": model.config.diffusion_steps,
            "property_names": list(model.config.property_names),
            "cfg_dropout": model.config.cfg_dropout,
            "use_flow_matching": model.config.use_flow_matching,
            "self_conditioning": model.config.self_conditioning,
        },
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if metadata is not None:
        checkpoint["metadata"] = metadata

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def compute_eval_loss_stage_a(
    model,
    dataloader: DataLoader,
    device: torch.device,
    vocab: BaseVocabulary,
) -> Dict[str, float]:
    """
    Compute evaluation loss for Stage A on a given dataset split.

    Args:
        model: DiffusionTransformer model
        dataloader: DataLoader for evaluation (validation or test)
        device: Device to run evaluation on
        vocab: Vocabulary

    Returns:
        Dictionary of loss components (e.g., {"total": X, "diffusion": Y})
    """
    from ..losses.objectives import stage_a_objective

    model.eval()
    loss_sums: Dict[str, float] = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)

            timesteps = model.diffusion.sample_timesteps(tokens.size(0))
            noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)

            outputs = model(noisy_tokens, timesteps, attention_mask=mask)
            losses = stage_a_objective(model, outputs, tokens, timesteps, noise_mask)

            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())

            num_batches += 1

    # Average losses
    avg_losses = {name: value / num_batches for name, value in loss_sums.items()} if num_batches > 0 else {}
    model.train()
    return avg_losses


def compute_eval_loss_stage_b(
    model,
    dataloader: DataLoader,
    device: torch.device,
    vocab: BaseVocabulary,
    lambda_gram: float,
) -> Dict[str, float]:
    """
    Compute evaluation loss for Stage B on a given dataset split.

    Args:
        model: DiffusionTransformer model
        dataloader: DataLoader for evaluation (validation or test)
        device: Device to run evaluation on
        vocab: Vocabulary
        lambda_gram: Grammar loss weight

    Returns:
        Dictionary of loss components (e.g., {"total": X, "diffusion": Y, "grammar": Z})
    """
    from ..losses.objectives import stage_b_objective

    model.eval()
    loss_sums: Dict[str, float] = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            anchor_count = batch["anchor_count"].to(device)
            valence = batch["valence"].to(device)

            timesteps = model.diffusion.sample_timesteps(tokens.size(0))
            noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
            anchor_mask_tokens = (tokens == vocab.token_to_id[ANCHOR1]) | (tokens == vocab.token_to_id[ANCHOR2])
            if torch.any(anchor_mask_tokens):
                noise_mask = noise_mask | anchor_mask_tokens
                noisy_tokens = noisy_tokens.masked_fill(anchor_mask_tokens, vocab.mask_id)

            outputs = model(noisy_tokens, timesteps, attention_mask=mask)
            losses = stage_b_objective(
                model,
                outputs,
                tokens,
                timesteps,
                noise_mask,
                anchor_count,
                valence,
                lambda_gram,
                vocab,
            )

            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())

            num_batches += 1

    # Average losses
    avg_losses = {name: value / num_batches for name, value in loss_sums.items()} if num_batches > 0 else {}
    model.train()
    return avg_losses


def compute_eval_loss_stage_c(
    model,
    dataloader: DataLoader,
    device: torch.device,
    vocab: BaseVocabulary,
    lambda_prop: float,
    lambda_gram: float,
    target_property: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute evaluation loss for Stage C on a given dataset split.

    Args:
        model: DiffusionTransformer model
        dataloader: DataLoader for evaluation (validation or test)
        device: Device to run evaluation on
        vocab: Vocabulary
        lambda_prop: Property loss weight
        lambda_gram: Grammar loss weight
        target_property: If specified, only evaluate on this property

    Returns:
        Dictionary of loss components (e.g., {"total": X, "diffusion": Y, "properties": Z, "grammar": W})
    """
    from ..losses.objectives import stage_c_objective

    model.eval()
    loss_sums: Dict[str, float] = {}
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)
            anchor_count = batch["anchor_count"].to(device)
            valence = batch["valence"].to(device)
            properties = {name: tensor.to(device) for name, tensor in batch["properties"].items()}

            timesteps = model.diffusion.sample_timesteps(tokens.size(0))
            noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
            anchor_mask_tokens = (tokens == vocab.token_to_id[ANCHOR1]) | (tokens == vocab.token_to_id[ANCHOR2])
            if torch.any(anchor_mask_tokens):
                noise_mask = noise_mask | anchor_mask_tokens
                noisy_tokens = noisy_tokens.masked_fill(anchor_mask_tokens, vocab.mask_id)

            outputs = model(noisy_tokens, timesteps, attention_mask=mask, properties=properties)
            losses = stage_c_objective(
                model,
                outputs,
                tokens,
                timesteps,
                noise_mask,
                properties,
                anchor_count,
                valence,
                lambda_prop,
                lambda_gram,
                vocab,
                target_property=target_property,
            )

            for name, value in losses.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu())

            num_batches += 1

    # Average losses
    avg_losses = {name: value / num_batches for name, value in loss_sums.items()} if num_batches > 0 else {}
    model.train()
    return avg_losses


def save_metrics_json(
    path: Path,
    final_metrics: Dict[str, Dict[str, float]],
    best_metrics: Dict[str, Dict[str, float]],
    best_step: int,
    best_val_loss: float,
) -> None:
    """
    Save evaluation metrics to JSON file.

    Args:
        path: Path to save JSON file
        final_metrics: Metrics for final model (train/val/test splits)
        best_metrics: Metrics for best model (train/val/test splits)
        best_step: Training step where best model was saved
        best_val_loss: Validation loss of best model

    Example output:
        {
          "final_model": {
            "train": {"total": 2.34, "diffusion": 2.10, ...},
            "validation": {"total": 2.45, "diffusion": 2.18, ...},
            "test": {"total": 2.50, "diffusion": 2.22, ...}
          },
          "best_model": {
            "train": {"total": 2.20, "diffusion": 1.98, ...},
            "validation": {"total": 2.38, "diffusion": 2.12, ...},
            "test": {"total": 2.42, "diffusion": 2.15, ...}
          },
          "best_step": 12500,
          "best_val_loss": 2.38
        }
    """
    metrics_data = {
        "final_model": final_metrics,
        "best_model": best_metrics,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    log = logging.getLogger(__name__)
    log.info(f"Saved evaluation metrics to {path}")


def _load_vocab_auto(path: Path, tokenization_config: Optional[TokenizationConfig] = None) -> BaseVocabulary:
    """
    Load a vocabulary file without requiring the caller to know the class.

    Uses the new load_vocabulary_auto() from vocab_factory for proper detection.
    """
    return load_vocabulary_auto(path, tokenization_config)


def _transfer_vocab_parameters(
    model: DiffusionTransformer,
    state_dict: Dict[str, torch.Tensor],
    target_vocab: BaseVocabulary,
    source_vocab: BaseVocabulary,
    reuse_token_embeddings: bool,
    reuse_output_head: bool,
) -> Tuple[int, int]:
    """Map shared tokens between vocabularies for embeddings and decoder head."""
    log = logging.getLogger(__name__)

    transferred_embeddings = 0
    transferred_logits = 0

    if reuse_token_embeddings and "token_embed.weight" in state_dict:
        source_embed = state_dict["token_embed.weight"]
        target_embed = model.token_embed.weight.data
        if source_embed.shape[1] != target_embed.shape[1]:
            log.warning(
                "Skipping token embedding transfer due to hidden size mismatch "
                "(source=%d, target=%d).",
                source_embed.shape[1],
                target_embed.shape[1],
            )
        else:
            for token, target_idx in target_vocab.token_to_id.items():
                source_idx = source_vocab.token_to_id.get(token)
                if source_idx is None:
                    continue
                if source_idx >= source_embed.shape[0] or target_idx >= target_embed.shape[0]:
                    continue
                target_embed[target_idx].copy_(source_embed[source_idx])
                transferred_embeddings += 1
            log.info(
                "Transferred embeddings for %d/%d tokens from source vocabulary.",
                transferred_embeddings,
                len(target_vocab),
            )
    else:
        log.info("Skipping token embedding transfer (reuse_token_embeddings=%s or source missing weights).", reuse_token_embeddings)

    if reuse_output_head and "head.weight" in state_dict:
        source_weight = state_dict["head.weight"]
        target_weight = model.head.weight.data
        if source_weight.shape[1] != target_weight.shape[1]:
            log.warning(
                "Skipping decoder head transfer due to hidden size mismatch "
                "(source=%d, target=%d).",
                source_weight.shape[1],
                target_weight.shape[1],
            )
        else:
            for token, target_idx in target_vocab.token_to_id.items():
                source_idx = source_vocab.token_to_id.get(token)
                if source_idx is None:
                    continue
                if source_idx >= source_weight.shape[0] or target_idx >= target_weight.shape[0]:
                    continue
                target_weight[target_idx].copy_(source_weight[source_idx])
                transferred_logits += 1
            log.info("Transferred decoder weights for %d/%d tokens.", transferred_logits, len(target_vocab))
            if "head.bias" in state_dict:
                source_bias = state_dict["head.bias"]
                target_bias = model.head.bias.data
                for token, target_idx in target_vocab.token_to_id.items():
                    source_idx = source_vocab.token_to_id.get(token)
                    if source_idx is None or source_idx >= source_bias.shape[0] or target_idx >= target_bias.shape[0]:
                        continue
                    target_bias[target_idx] = source_bias[source_idx]
    else:
        log.info("Skipping decoder head transfer (reuse_output_head=%s or source missing weights).", reuse_output_head)

    return transferred_embeddings, transferred_logits


def load_checkpoint(
    path: Path,
    model: DiffusionTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    strict: bool = False,
) -> int:
    """Load checkpoint and return the step number."""
    log = logging.getLogger(__name__)

    if not path.exists():
        log.warning(f"Checkpoint not found at {path}")
        return 0

    checkpoint = torch.load(path, map_location="cpu")

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        log.info(f"Loaded model state from {path}")
    else:
        # Legacy checkpoint format (state_dict only)
        model.load_state_dict(checkpoint, strict=strict)
        log.info(f"Loaded legacy model state from {path}")

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            log.info("Loaded optimizer state")
        except Exception as e:
            log.warning(f"Failed to load optimizer state: {e}")

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            log.info("Loaded scheduler state")
        except Exception as e:
            log.warning(f"Failed to load scheduler state: {e}")

    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", 0.0)
    log.info(f"Checkpoint at step {step} with loss {loss:.4f}")

    return step


def load_pretrained_for_finetuning(
    checkpoint_path: Path,
    vocab: AnchorSafeVocab,
    model_cfg: dict,
    freeze_backbone: bool = False,
    source_vocab_path: Optional[Path] = None,
    reuse_token_embeddings: bool = True,
    reuse_output_head: bool = True,
) -> DiffusionTransformer:
    """Load a pretrained checkpoint for fine-tuning.

    Args:
        checkpoint_path: Path to checkpoint file.
        vocab: Vocabulary (must match checkpoint).
        model_cfg: Model configuration dict.
        freeze_backbone: If True, freeze transformer backbone parameters.
        source_vocab_path: Optional path to the vocabulary used when training the checkpoint.
            When provided, token/decoder weights are remapped for shared tokens.
        reuse_token_embeddings: Copy token embedding rows that exist in both vocabularies.
        reuse_output_head: Copy decoder weight/bias rows for shared tokens.

    Returns:
        Model with loaded weights.
    """
    log = logging.getLogger(__name__)

    model = build_model(vocab, model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if source_vocab_path is not None:
        source_vocab = _load_vocab_auto(source_vocab_path)
        log.info(
            "Remapping checkpoint weights using source vocabulary at %s (%d tokens) -> target vocabulary (%d tokens).",
            source_vocab_path,
            len(source_vocab),
            len(vocab),
        )
        _transfer_vocab_parameters(
            model,
            state_dict,
            target_vocab=vocab,
            source_vocab=source_vocab,
            reuse_token_embeddings=reuse_token_embeddings,
            reuse_output_head=reuse_output_head,
        )
        state_dict = {
            key: value for key, value in state_dict.items() if key not in {"token_embed.weight", "head.weight", "head.bias"}
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.info("Missing parameters while loading checkpoint: %s", ", ".join(sorted(missing)))
    if unexpected:
        log.info("Ignoring unexpected parameters from checkpoint: %s", ", ".join(sorted(unexpected)))

    log.info(f"Loaded pretrained model from {checkpoint_path}")

    if freeze_backbone:
        # Freeze all parameters except heads
        for name, param in model.named_parameters():
            if not any(head in name for head in ["synth_head", "property_heads", "grammar_head", "head"]):
                param.requires_grad = False
                log.info(f"Froze parameter: {name}")
        log.info("Froze backbone parameters for fine-tuning")

    return model
