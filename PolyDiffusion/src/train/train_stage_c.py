"""Stage C training: property-guided fine-tuning.

IMPORTANT: For best results, train separate models for each property (Tg, Tm, Td, Eg, chi).
Use the 'target_property' config field to specify which property to train on.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import cycle
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..chem.ap_smiles import SHIELD1, SHIELD2
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

try:  # pragma: no cover
    import resource
except ImportError:  # pragma: no cover
    resource = None

try:  # pragma: no cover
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def _get_peak_memory_mb() -> float:
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage:
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


def run_stage_c(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()

    start_time = time.perf_counter()

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
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Single-property training mode (RECOMMENDED)
    target_property = cfg.get("target_property", None)
    log = logging.getLogger(__name__)
    if target_property:
        if target_property not in model.config.property_names:
            raise ValueError(
                f"Target property '{target_property}' not found in model configuration. "
                f"Available properties: {list(model.config.property_names)}"
            )
        log.info(f"Training in SINGLE-PROPERTY mode for: {target_property}")
        active_properties = [target_property]
    else:
        if not model.config.property_names:
            raise ValueError("Model configuration must define at least one property name for Stage C training.")
        log.warning("Training in MULTI-PROPERTY mode (all properties). Consider using 'target_property' for better results.")
        active_properties = list(model.config.property_names)

    dataset = build_stage_dataset("c", cfg["data"], property_names=active_properties)
    train_cfg = cfg["training"]
    dataloader = make_dataloader(
        dataset,
        train_cfg["batch_size"],
        lambda batch: collate_stage_c(batch, vocab, active_properties),
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
            log.info(f"Resumed from checkpoint {resume_path} at step {start_step}")

    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 200)
    lambda_syn = cfg["loss"]["lambda_syn"]
    lambda_prop = cfg["loss"]["lambda_prop"]
    lambda_gram = cfg["loss"]["lambda_gram"]

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

    # Use cycle to avoid expensive DataLoader recreation on exhaustion
    data_iter = cycle(dataloader)
    for step in range(start_step, steps):
        batch = next(data_iter)

        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        synth = batch["synth"].to(device)
        anchor_count = batch["anchor_count"].to(device)
        valence = batch["valence"].to(device)

        # Move property tensors for conditioning
        properties = {name: tensor.to(device) for name, tensor in batch["properties"].items()}

        timesteps = model.diffusion.sample_timesteps(tokens.size(0))
        noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
        anchor_mask_tokens = (tokens == vocab.token_to_id[SHIELD1]) | (tokens == vocab.token_to_id[SHIELD2])
        if torch.any(anchor_mask_tokens):
            noise_mask = noise_mask | anchor_mask_tokens
            noisy_tokens = noisy_tokens.masked_fill(anchor_mask_tokens, vocab.mask_id)
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
            vocab,
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

    elapsed = time.perf_counter() - start_time
    peak_ram = _get_peak_memory_mb()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda = torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        log.info("Runtime %.2fs | Peak RAM %.1f MB | Peak CUDA %.1f MB", elapsed, peak_ram, peak_cuda)
    else:
        log.info("Runtime %.2fs | Peak RAM %.1f MB", elapsed, peak_ram)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to stage C YAML config.")
    args = parser.parse_args()
    run_stage_c(args.config)


if __name__ == "__main__":
    main()
