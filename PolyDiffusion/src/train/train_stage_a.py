"""Stage A training loop: small molecules + synthesis regression."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from ..chem.vocab import AnchorSafeVocab
from ..losses.objectives import stage_a_objective
from ..utils.logging import configure_logging
from .common import (
    build_model,
    build_stage_dataset,
    collate_stage_a,
    default_device,
    load_yaml,
    make_dataloader,
)


def run_stage_a(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()

    vocab = AnchorSafeVocab.load(Path(cfg["vocab_path"]))
    model_cfg = load_yaml(Path(cfg["model_config"]))
    model = build_model(vocab, model_cfg)
    device = default_device()
    model.to(device)

    dataset = build_stage_dataset("a", cfg["data"])
    dataloader = make_dataloader(dataset, cfg["training"]["batch_size"], lambda batch: collate_stage_a(batch, vocab))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

    steps = cfg["training"]["steps"]
    lambda_syn = cfg["loss"]["lambda_syn"]
    log = logging.getLogger(__name__)
    model.train()

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        synth = batch["synth"].to(device)

        timesteps = model.diffusion.sample_timesteps(tokens.size(0))
        noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
        outputs = model(noisy_tokens.to(device), timesteps, attention_mask=mask, s_target=synth)
        losses = stage_a_objective(model, outputs, tokens, timesteps, noise_mask, synth, lambda_syn)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        if step % cfg["training"].get("log_interval", 10) == 0:
            log.info(
                "step=%d loss_total=%.4f loss_diff=%.4f loss_syn=%.4f",
                step,
                float(losses["total"]),
                float(losses["diffusion"]),
                float(losses["synth"]),
            )

    if "checkpoint_path" in cfg:
        torch.save(model.state_dict(), cfg["checkpoint_path"])
        log.info("Saved checkpoint to %s", cfg["checkpoint_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to stage A YAML config.")
    args = parser.parse_args()
    run_stage_a(args.config)


if __name__ == "__main__":
    main()
