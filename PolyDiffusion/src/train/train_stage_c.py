"""Stage C training: property-guided fine-tuning.

IMPORTANT: For best results, train separate models for each property (Tg, Tm, Td, Eg, chi).
Use the 'target_property' config field to specify which property to train on.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..chem.vocab import AnchorSafeVocab
from ..losses.objectives import stage_c_objective
from ..utils.logging import configure_logging
from .common import (
    build_model,
    build_stage_dataset,
    collate_stage_c,
    default_device,
    load_yaml,
    make_dataloader,
    save_checkpoint,
    load_checkpoint,
    load_pretrained_for_finetuning,
)


def run_stage_c(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()

    vocab = AnchorSafeVocab.load(Path(cfg["vocab_path"]))
    model_cfg = load_yaml(Path(cfg["model_config"]))

    # Load pretrained Stage B model if specified
    if "pretrained_checkpoint" in cfg and cfg["pretrained_checkpoint"]:
        pretrained_path = Path(cfg["pretrained_checkpoint"])
        freeze_backbone = cfg.get("freeze_backbone", False)
        model = load_pretrained_for_finetuning(pretrained_path, vocab, model_cfg, freeze_backbone)
        log = logging.getLogger(__name__)
        log.info(f"Loaded pretrained model from {pretrained_path}")
    else:
        model = build_model(vocab, model_cfg)

    device = default_device()
    model.to(device)

    # Single-property training mode (RECOMMENDED)
    target_property = cfg.get("target_property", None)
    if target_property:
        log = logging.getLogger(__name__)
        log.info(f"Training in SINGLE-PROPERTY mode for: {target_property}")
        property_names = [target_property]  # Only train on this one property
    else:
        log = logging.getLogger(__name__)
        log.warning("Training in MULTI-PROPERTY mode (all properties). Consider using 'target_property' for better results.")
        property_names = model.config.property_names

    dataset = build_stage_dataset("c", cfg["data"], property_names=model.config.property_names)
    train_cfg = cfg["training"]
    dataloader = make_dataloader(
        dataset,
        train_cfg["batch_size"],
        lambda batch: collate_stage_c(batch, vocab, model.config.property_names),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

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
    save_interval = train_cfg.get("save_interval", 200)
    lambda_syn = cfg["loss"]["lambda_syn"]
    lambda_prop = cfg["loss"]["lambda_prop"]
    lambda_gram = cfg["loss"]["lambda_gram"]
    log = logging.getLogger(__name__)

    # Setup Results directory with property name
    if target_property:
        results_dir = Path(cfg.get("results_dir", f"Results/stage_c/{target_property}"))
    else:
        results_dir = Path(cfg.get("results_dir", "Results/stage_c/multi_property"))
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
        anchor_count = batch["anchor_count"].to(device)
        valence = batch["valence"].to(device)

        # Only use target property for conditioning if single-property mode
        if target_property:
            properties = {target_property: batch["properties"][target_property].to(device)}
        else:
            properties = {name: batch["properties"][name].to(device) for name in model.config.property_names}

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
            target_property=target_property,  # Pass target_property to loss function
        )

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        if step % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log.info(
                "step=%d loss_total=%.4f loss_diff=%.4f loss_syn=%.4f loss_prop=%.4f loss_gram=%.4f lr=%.2e",
                step,
                float(losses["total"]),
                float(losses["diffusion"]),
                float(losses["synth"]),
                float(losses["properties"]),
                float(losses["grammar"]),
                current_lr,
            )

        # Save periodic checkpoints
        if (step + 1) % save_interval == 0:
            checkpoint_path = results_dir / f"checkpoint_step_{step+1}.pt"
            metadata = {"target_property": target_property} if target_property else {}
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step + 1, losses["total"].item(), metadata)
            log.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if losses["total"].item() < best_loss:
            best_loss = losses["total"].item()
            metadata = {"target_property": target_property} if target_property else {}
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, step + 1, best_loss, metadata)
            log.info(f"Saved best model with loss {best_loss:.4f}")

    # Save final checkpoint
    final_checkpoint_path = results_dir / "final_model.pt"
    metadata = {"target_property": target_property} if target_property else {}
    save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, steps, losses["total"].item(), metadata)
    log.info(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")

    # Legacy checkpoint path support
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
