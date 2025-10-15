"""Shared helpers for training stages."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

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


def build_stage_dataset(stage: str, data_cfg: dict) -> JsonlDataset | CsvDataset:
    config = DatasetConfig(path=Path(data_cfg["path"]), limit=data_cfg.get("limit"))
    if stage in {"a", "b"}:
        return JsonlDataset(config)
    return CsvDataset(config)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smiles_to_ap(smiles: str) -> str:
    """Coerce plain SMILES to AP-SMILES by attaching anchors at ends."""
    return f"{ANCHOR1}{smiles}{ANCHOR2}"


def collate_stage_a(records: List[Dict[str, object]], vocab: AnchorSafeVocab) -> Dict[str, torch.Tensor]:
    tokens = [vocab.tokenize_ap(smiles_to_ap(r["smiles"])) for r in records]
    batch = collate_token_batch(tokens, vocab.pad_id)
    synth = torch.tensor([float(r["synth_score"]) for r in records], dtype=torch.float32)
    batch["synth"] = synth
    return batch


def collate_stage_b(records: List[Dict[str, object]], vocab: AnchorSafeVocab) -> Dict[str, torch.Tensor]:
    tokens = [vocab.tokenize_ap(r["ap_smiles"]) for r in records]
    batch = collate_token_batch(tokens, vocab.pad_id)
    synth = torch.tensor([float(r["synth_score"]) for r in records], dtype=torch.float32)
    anchor_count = torch.tensor([r["ap_smiles"].count(ANCHOR1) + r["ap_smiles"].count(ANCHOR2) for r in records], dtype=torch.int64)
    valence = torch.tensor([1.0 if valence_utils.valence_ok(r["ap_smiles"]) else 0.0 for r in records], dtype=torch.float32)
    batch["synth"] = synth
    batch["anchor_count"] = anchor_count
    batch["valence"] = valence
    return batch


def collate_stage_c(records: List[Dict[str, object]], vocab: AnchorSafeVocab) -> Dict[str, torch.Tensor]:
    tokens = [vocab.tokenize_ap(r["ap_smiles"]) for r in records]
    batch = collate_token_batch(tokens, vocab.pad_id)
    synth = torch.tensor([float(r["synth_score"]) for r in records], dtype=torch.float32)
    properties: Dict[str, torch.Tensor] = {}
    for name in PROPERTY_NAMES:
        values = torch.tensor([float(r.get(name, 0.0)) for r in records], dtype=torch.float32)
        properties[name] = values
    anchor_count = torch.tensor([r["ap_smiles"].count(ANCHOR1) + r["ap_smiles"].count(ANCHOR2) for r in records], dtype=torch.int64)
    valence = torch.tensor([1.0 if valence_utils.valence_ok(r["ap_smiles"]) else 0.0 for r in records], dtype=torch.float32)
    batch["synth"] = synth
    batch["properties"] = properties
    batch["anchor_count"] = anchor_count
    batch["valence"] = valence
    return batch


def make_dataloader(dataset, batch_size: int, collate_fn) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
