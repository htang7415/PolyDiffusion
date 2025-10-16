"""Shared helpers for training stages."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from ..chem import valence as valence_utils
from ..chem.ap_smiles import ANCHOR1, ANCHOR2
from ..chem.vocab import AnchorSafeVocab
from ..data.collate import collate_token_batch
from ..data.datasets import CsvDataset, DatasetConfig, JsonlDataset
from ..models.dit_token import DiffusionTransformer, ModelConfig
from ..models.diffusion_token import DiffusionConfig


PROPERTY_NAMES = ["Tg", "Tm", "Td", "Eg", "chi"]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(vocab: AnchorSafeVocab, model_cfg: dict) -> DiffusionTransformer:
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
    return DiffusionTransformer(model_config, diffusion_config)


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
    )
    if stage == "a":
        return JsonlDataset(config, required_fields={"smiles", "synth_score"})
    if stage == "b":
        return JsonlDataset(config, required_fields={"ap_smiles", "synth_score"})

    property_fields = set(property_names or PROPERTY_NAMES)
    required = {"ap_smiles", "synth_score"} | property_fields
    return CsvDataset(config, required_fields=required)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smiles_to_ap(smiles: str) -> str:
    """Coerce plain SMILES to AP-SMILES by attaching anchors at ends."""
    return f"{ANCHOR1}{smiles}{ANCHOR2}"


def collate_stage_a(records: List[Dict[str, object]], vocab: AnchorSafeVocab) -> Dict[str, torch.Tensor]:
    tokens = [vocab.tokenize_ap(smiles_to_ap(str(r["smiles"]))) for r in records]
    batch = collate_token_batch(tokens, vocab.pad_id)
    synth = torch.tensor([float(r["synth_score"]) for r in records], dtype=torch.float32)
    batch["synth"] = synth
    return batch


def collate_stage_b(records: List[Dict[str, object]], vocab: AnchorSafeVocab) -> Dict[str, torch.Tensor]:
    aps = [str(r["ap_smiles"]) for r in records]
    tokens = [vocab.tokenize_ap(ap) for ap in aps]
    batch = collate_token_batch(tokens, vocab.pad_id)
    synth = torch.tensor([float(r["synth_score"]) for r in records], dtype=torch.float32)
    anchor_count = torch.tensor([ap.count(ANCHOR1) + ap.count(ANCHOR2) for ap in aps], dtype=torch.int64)
    valence = torch.tensor([1.0 if valence_utils.valence_ok(ap) else 0.0 for ap in aps], dtype=torch.float32)
    batch["synth"] = synth
    batch["anchor_count"] = anchor_count
    batch["valence"] = valence
    return batch


def collate_stage_c(
    records: List[Dict[str, object]],
    vocab: AnchorSafeVocab,
    property_names: Sequence[str] = PROPERTY_NAMES,
) -> Dict[str, torch.Tensor]:
    aps = [str(r["ap_smiles"]) for r in records]
    tokens = [vocab.tokenize_ap(ap) for ap in aps]
    batch = collate_token_batch(tokens, vocab.pad_id)
    synth = torch.tensor([float(r["synth_score"]) for r in records], dtype=torch.float32)
    properties: Dict[str, torch.Tensor] = {}
    for name in property_names:
        values = torch.tensor([float(r[name]) for r in records], dtype=torch.float32)
        properties[name] = values
    anchor_count = torch.tensor([ap.count(ANCHOR1) + ap.count(ANCHOR2) for ap in aps], dtype=torch.int64)
    valence = torch.tensor([1.0 if valence_utils.valence_ok(ap) else 0.0 for ap in aps], dtype=torch.float32)
    batch["synth"] = synth
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
) -> DataLoader:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
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
) -> DiffusionTransformer:
    """Load a pretrained checkpoint for fine-tuning.

    Args:
        checkpoint_path: Path to checkpoint file.
        vocab: Vocabulary (must match checkpoint).
        model_cfg: Model configuration dict.
        freeze_backbone: If True, freeze transformer backbone parameters.

    Returns:
        Model with loaded weights.
    """
    log = logging.getLogger(__name__)

    model = build_model(vocab, model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    log.info(f"Loaded pretrained model from {checkpoint_path}")

    if freeze_backbone:
        # Freeze all parameters except heads
        for name, param in model.named_parameters():
            if not any(head in name for head in ["synth_head", "property_heads", "grammar_head", "head"]):
                param.requires_grad = False
                log.info(f"Froze parameter: {name}")
        log.info("Froze backbone parameters for fine-tuning")

    return model
