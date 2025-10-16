import json
from pathlib import Path

import pytest

from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.data.collate import collate_token_batch
from PolyDiffusion.data.datasets import DatasetConfig, JsonlDataset
from PolyDiffusion.sampling.decode import decode_tokens
from PolyDiffusion.sampling.sampler import GuidedSampler, SamplerConfig
from PolyDiffusion.models.dit_token import DiffusionTransformer, ModelConfig
from PolyDiffusion.models.diffusion_token import DiffusionConfig


def test_collate_token_batch_shapes() -> None:
    batch = collate_token_batch([[1, 2, 3], [4, 5]], pad_token_id=0)
    assert batch["tokens"].shape == (2, 3)
    assert batch["mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["lengths"].tolist() == [3, 2]


def test_jsonl_dataset_required_fields(tmp_path: Path) -> None:
    data_path = tmp_path / "records.jsonl"
    records = [
        {"smiles": "CCO", "synth_score": 3.2},
        {"smiles": "CCC", "synth_score": 3.5},
    ]
    with data_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    config = DatasetConfig(path=data_path, shuffle=False)
    dataset = JsonlDataset(config, required_fields={"smiles", "synth_score"})
    assert len(dataset) == 2
    assert dataset[0]["smiles"] == "CCO"

    bad_path = tmp_path / "bad.jsonl"
    with bad_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"smiles": "CCO"}) + "\n")
    bad_config = DatasetConfig(path=bad_path, shuffle=False)
    with pytest.raises(KeyError):
        JsonlDataset(bad_config, required_fields={"smiles", "synth_score"})


def test_decode_tokens_non_strict_returns_blank() -> None:
    vocab = AnchorSafeVocab.build(["[*:1]CC[*:2]"])
    invalid_sequence = [[vocab.bos_id, vocab.eos_id]]
    decoded = decode_tokens(vocab, invalid_sequence)
    assert decoded == [""]
    with pytest.raises(ValueError):
        decode_tokens(vocab, invalid_sequence, strict=True)


def test_sampler_rejects_unknown_property() -> None:
    vocab = AnchorSafeVocab.build(["[*:1]C[*:2]"])
    model_cfg = ModelConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        diffusion_steps=2,
        property_names=["Tg"],
    )
    diffusion_cfg = DiffusionConfig(vocab_size=len(vocab), num_steps=2, mask_token_id=vocab.mask_id)
    model = DiffusionTransformer(model_cfg, diffusion_cfg)
    sampler = GuidedSampler(model, vocab, SamplerConfig(max_length=8, cfg_scale=1.0))

    with pytest.raises(ValueError):
        sampler.sample(num_samples=1, num_steps=1, property_targets={"invalid": 1.0})
