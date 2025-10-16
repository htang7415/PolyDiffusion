"""Stage A training loop: small molecules + synthesis regression."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    save_checkpoint,
    load_checkpoint,
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
    train_cfg = cfg["training"]
    dataloader = make_dataloader(
        dataset,
        train_cfg["batch_size"],
        lambda batch: collate_stage_a(batch, vocab),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    # Learning rate scheduler
    steps = train_cfg["steps"]
    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=train_cfg.get("lr_min", 1e-6))

    # Resume from checkpoint if specified
    start_step = 0
    if "resume_checkpoint" in cfg and cfg["resume_checkpoint"]:
        resume_path = Path(cfg["resume_checkpoint"])
        if resume_path.exists():
            start_step = load_checkpoint(resume_path, model, optimizer, scheduler)
            log = logging.getLogger(__name__)
            log.info(f"Resumed from checkpoint {resume_path} at step {start_step}")

    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 500)
    lambda_syn = cfg["loss"]["lambda_syn"]
    log = logging.getLogger(__name__)

    # Setup Results directory
    results_dir = Path(cfg.get("results_dir", "Results/stage_a"))
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Results will be saved to {results_dir}")

    # Best model tracking
    best_loss = float('inf')
    best_checkpoint_path = results_dir / "best_model.pt"

    # Gradient clipping
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    model.train()

    data_iter = iter(dataloader)
    for step in range(start_step, steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        synth = batch["synth"].to(device)

        timesteps = model.diffusion.sample_timesteps(tokens.size(0))
        noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
        outputs = model(noisy_tokens, timesteps, attention_mask=mask, s_target=synth)
        losses = stage_a_objective(model, outputs, tokens, timesteps, noise_mask, synth, lambda_syn)

        optimizer.zero_grad()
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log.info(
                "step=%d loss_total=%.4f loss_diff=%.4f loss_syn=%.4f lr=%.2e",
                step,
                float(losses["total"]),
                float(losses["diffusion"]),
                float(losses["synth"]),
                current_lr,
            )

        # Save periodic checkpoints
        if (step + 1) % save_interval == 0:
            checkpoint_path = results_dir / f"checkpoint_step_{step+1}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step + 1, losses["total"].item())
            log.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if losses["total"].item() < best_loss:
            best_loss = losses["total"].item()
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, step + 1, best_loss)
            log.info(f"Saved best model with loss {best_loss:.4f}")

    # Save final checkpoint
    final_checkpoint_path = results_dir / "final_model.pt"
    save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, steps, losses["total"].item())
    log.info(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")

    # Legacy checkpoint path support
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
