"""Stage C training: property-guided fine-tuning."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ..chem.vocab import AnchorSafeVocab
from ..losses.objectives import stage_c_objective
from ..utils.logging import configure_logging
from .common import (
    PROPERTY_NAMES,
    build_model,
    build_stage_dataset,
    collate_stage_c,
    default_device,
    load_yaml,
    make_dataloader,
)


def run_stage_c(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()

    vocab = AnchorSafeVocab.load(Path(cfg["vocab_path"]))
    model_cfg = load_yaml(Path(cfg["model_config"]))
    model = build_model(vocab, model_cfg)
    device = default_device()
    model.to(device)

    dataset = build_stage_dataset("c", cfg["data"])
    dataloader = make_dataloader(dataset, cfg["training"]["batch_size"], lambda batch: collate_stage_c(batch, vocab))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    steps = cfg["training"]["steps"]
    lambda_syn = cfg["loss"]["lambda_syn"]
    lambda_prop = cfg["loss"]["lambda_prop"]
    lambda_gram = cfg["loss"]["lambda_gram"]
    log = logging.getLogger(__name__)
    model.train()

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        synth = batch["synth"].to(device)
        anchor_count = batch["anchor_count"].to(device)
        valence = batch["valence"].to(device)
        properties = {name: batch["properties"][name].to(device) for name in PROPERTY_NAMES}

        timesteps = model.diffusion.sample_timesteps(tokens.size(0))
        noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
        outputs = model(noisy_tokens, timesteps, attention_mask=mask, properties=properties, s_target=synth)
        losses = stage_c_objective(
            model,
            outputs,
            tokens,
            timesteps,
            noise_mask,
            synth,
            properties,
            anchor_count,
            valence,
            lambda_syn,
            lambda_prop,
            lambda_gram,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        if step % cfg["training"].get("log_interval", 10) == 0:
            log.info(
                "step=%d loss_total=%.4f loss_diff=%.4f loss_syn=%.4f loss_prop=%.4f loss_gram=%.4f",
                step,
                float(losses["total"]),
                float(losses["diffusion"]),
                float(losses["synth"]),
                float(losses["properties"]),
                float(losses["grammar"]),
            )

    if "checkpoint_path" in cfg:
        torch.save(model.state_dict(), cfg["checkpoint_path"])
        log.info("Saved checkpoint to %s", cfg["checkpoint_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to stage C YAML config.")
    args = parser.parse_args()
    run_stage_c(args.config)


if __name__ == "__main__":
    main()
