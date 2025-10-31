"""Stage C training: property-guided fine-tuning.

IMPORTANT: For best results, train separate models for each property (Tg, Tm, Td, Eg, chi).
Use the 'target_property' config field to specify which property to train on.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..chem.ap_smiles import SHIELD1, SHIELD2
from ..chem.vocab_config import load_tokenization_config
from ..chem.vocab_factory import load_vocabulary_auto
from ..losses.objectives import stage_c_objective
from ..utils.logging import configure_logging
from .common import (
    build_model,
    build_stage_dataset,
    collate_stage_c,
    compute_eval_loss_stage_c,
    default_device,
    load_yaml,
    make_dataloader,
    save_checkpoint,
    save_metrics_json,
    load_checkpoint,
    load_pretrained_for_finetuning,
    split_dataset,
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


def _resolve_stage_c_vocab(tok_config, cfg, stage_b_method: str) -> Path:
    """Locate a Stage C vocabulary when not explicitly provided."""
    seen: set[str] = set()
    candidates: list[Path] = []

    def enqueue(path: Path | str | None) -> None:
        if path is None:
            return
        path = Path(path).expanduser()
        key = str(path)
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    def enqueue_dir(directory: Path | str, recursive: bool = True) -> None:
        directory = Path(directory).expanduser()
        if not directory.exists():
            return
        pattern = "**/vocab*.txt" if recursive else "vocab*.txt"
        for file in sorted(directory.glob(pattern)):
            if file.is_file():
                enqueue(file)

    explicit = Path(tok_config.vocab_path).expanduser() if tok_config.vocab_path else None
    if explicit:
        if explicit.is_dir():
            enqueue_dir(explicit, recursive=True)
        else:
            enqueue(explicit)

    pretrained_path_str = cfg.get("pretrained_checkpoint")
    if pretrained_path_str:
        pretrained_path = Path(pretrained_path_str).expanduser()
        if pretrained_path.exists():
            ckpt_dir = pretrained_path.parent
            enqueue(ckpt_dir / "vocab.txt")
            enqueue_dir(ckpt_dir, recursive=False)
            enqueue_dir(ckpt_dir, recursive=True)

    stage_b_base = Path(cfg.get("stage_b_results_dir", "Results/stage_b"))
    if stage_b_base.exists():
        enqueue_dir(stage_b_base / stage_b_method, recursive=True)
        enqueue_dir(stage_b_base, recursive=True)

    stage_c_base = Path(cfg.get("results_dir", "Results/stage_c"))
    if stage_c_base.exists():
        enqueue_dir(stage_c_base / tok_config.method, recursive=True)
        enqueue_dir(stage_c_base, recursive=True)

    repo_dir = Path("PolyDiffusion")
    if repo_dir.exists():
        enqueue_dir(repo_dir, recursive=False)

    enqueue("PolyDiffusion/vocab_stage_bc.txt")
    enqueue("PolyDiffusion/vocab_character_stage_b.txt")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = "\n  ".join(str(path) for path in candidates) if candidates else "(no candidates)"
    raise FileNotFoundError(
        "Unable to locate a vocabulary file for Stage C. Checked the following locations:\n"
        f"  {searched}"
    )


def _extract_checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _find_stage_checkpoint(base_dir: Path, method: str) -> Optional[Path]:
    """Find a Stage B checkpoint for the requested tokenisation method."""
    search_dirs = [base_dir / method, base_dir]
    for directory in search_dirs:
        directory = directory.expanduser()
        if not directory.exists():
            continue
        for name in ("best_model.pt", "final_model.pt"):
            candidate = directory / name
            if candidate.exists():
                return candidate
        checkpoints = sorted(
            directory.glob("checkpoint_step_*.pt"),
            key=_extract_checkpoint_step,
            reverse=True,
        )
        if checkpoints:
            return checkpoints[0]
    return None


def run_stage_c(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()
    log = logging.getLogger(__name__)

    start_time = time.perf_counter()

    # Load tokenization configuration
    tok_config = load_tokenization_config(cfg)
    log.info(f"Using tokenization method: {tok_config.method}")

    # Load vocabulary
    stage_b_method = cfg.get("stage_b_method", tok_config.method)
    vocab_path = _resolve_stage_c_vocab(tok_config, cfg, stage_b_method)
    vocab = load_vocabulary_auto(vocab_path, tok_config)
    log.info(f"Loaded {tok_config.method} vocabulary from {vocab_path}")

    model_cfg = load_yaml(Path(cfg["model_config"]))

    # Load pretrained Stage B model if specified (auto-detect when blank)
    pretrained_value = cfg.get("pretrained_checkpoint")
    pretrained_path: Optional[Path] = None
    if pretrained_value:
        candidate_path = Path(pretrained_value).expanduser()
        if candidate_path.exists():
            pretrained_path = candidate_path
        else:
            log.warning("Specified Stage B checkpoint not found: %s", candidate_path)

    if pretrained_path is None:
        stage_b_results_dir = Path(cfg.get("stage_b_results_dir", "Results/stage_b"))
        fallback_checkpoint = _find_stage_checkpoint(stage_b_results_dir, stage_b_method)
        if fallback_checkpoint is not None:
            log.info("Using Stage B checkpoint fallback at %s", fallback_checkpoint)
            pretrained_path = fallback_checkpoint

    if pretrained_path is None:
        raise FileNotFoundError(
            "Stage C requires a Stage B checkpoint. Provide 'pretrained_checkpoint' in the config or "
            "ensure a checkpoint exists under Results/stage_b/<method>/."
        )

    freeze_backbone = cfg.get("freeze_backbone", False)
    model = load_pretrained_for_finetuning(pretrained_path, vocab, model_cfg, freeze_backbone)
    log.info(f"Loaded pretrained model from {pretrained_path}")

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

    # Load dataset and split into train/validation/test (8:1:1)
    full_dataset = build_stage_dataset("c", cfg["data"], property_names=active_properties)
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        ratios=(0.8, 0.1, 0.1),
        seed=cfg["data"].get("seed", 42)
    )

    train_cfg = cfg["training"]

    # Create dataloaders: training with shuffle, validation/test without shuffle
    train_dataloader = make_dataloader(
        train_dataset,
        train_cfg["batch_size"],
        lambda batch: collate_stage_c(batch, vocab, active_properties),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=True,
    )
    val_dataloader = make_dataloader(
        val_dataset,
        train_cfg["batch_size"],
        lambda batch: collate_stage_c(batch, vocab, active_properties),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=False,
    )
    test_dataloader = make_dataloader(
        test_dataset,
        train_cfg["batch_size"],
        lambda batch: collate_stage_c(batch, vocab, active_properties),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=False,
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
    eval_interval = train_cfg.get("eval_interval", log_interval)
    lambda_prop = cfg["loss"]["lambda_prop"]
    lambda_gram = cfg["loss"]["lambda_gram"]

    # Setup Results directory with property name and tokenization method
    if target_property:
        base_results_dir = Path(cfg.get("results_dir", f"Results/stage_c/{target_property}"))
    else:
        base_results_dir = Path(cfg.get("results_dir", "Results/stage_c/multi_property"))
    results_dir = base_results_dir / tok_config.method
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Results will be saved to {results_dir}")

    # Best model tracking (based on validation loss)
    best_val_loss = float('inf')
    best_step = 0
    best_checkpoint_path = results_dir / "best_model.pt"

    # Gradient clipping
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    model.train()

    # Use cycle to avoid expensive DataLoader recreation on exhaustion
    data_iter = cycle(train_dataloader)
    for step in range(start_step, steps):
        batch = next(data_iter)

        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
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
                "step=%d loss_total=%.4f loss_diff=%.4f loss_prop=%.4f loss_gram=%.4f lr=%.2e",
                step,
                float(losses["total"]),
                float(losses["diffusion"]),
                float(losses["properties"]),
                float(losses["grammar"]),
                current_lr,
            )

        # Validation loss evaluation
        if step % eval_interval == 0:
            val_losses = compute_eval_loss_stage_c(
                model, val_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
            )
            log.info(
                "step=%d val_loss_total=%.4f val_loss_diff=%.4f val_loss_prop=%.4f val_loss_gram=%.4f",
                step,
                val_losses.get("total", float("nan")),
                val_losses.get("diffusion", float("nan")),
                val_losses.get("properties", float("nan")),
                val_losses.get("grammar", float("nan")),
            )

            # Save best model based on validation loss
            current_val_loss = val_losses.get("total", float("inf"))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_step = step
                metadata = {"target_property": target_property} if target_property else {}
                save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, step + 1, best_val_loss, metadata)
                log.info(f"Saved best model with validation loss {best_val_loss:.4f} at step {step}")

        # Save periodic checkpoints
        if (step + 1) % save_interval == 0:
            checkpoint_path = results_dir / f"checkpoint_step_{step+1}.pt"
            metadata = {"target_property": target_property} if target_property else {}
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step + 1, losses["total"].item(), metadata)
            log.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = results_dir / "final_model.pt"
    metadata = {"target_property": target_property} if target_property else {}
    save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, steps, losses["total"].item(), metadata)
    log.info(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")

    # Evaluate final model on all splits
    log.info("Evaluating final model on train/validation/test sets...")
    final_train_metrics = compute_eval_loss_stage_c(
        model, train_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
    )
    final_val_metrics = compute_eval_loss_stage_c(
        model, val_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
    )
    final_test_metrics = compute_eval_loss_stage_c(
        model, test_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
    )

    log.info(
        "Final model - Train: total=%.4f diff=%.4f prop=%.4f gram=%.4f | "
        "Val: total=%.4f diff=%.4f prop=%.4f gram=%.4f | "
        "Test: total=%.4f diff=%.4f prop=%.4f gram=%.4f",
        final_train_metrics.get("total", float("nan")),
        final_train_metrics.get("diffusion", float("nan")),
        final_train_metrics.get("properties", float("nan")),
        final_train_metrics.get("grammar", float("nan")),
        final_val_metrics.get("total", float("nan")),
        final_val_metrics.get("diffusion", float("nan")),
        final_val_metrics.get("properties", float("nan")),
        final_val_metrics.get("grammar", float("nan")),
        final_test_metrics.get("total", float("nan")),
        final_test_metrics.get("diffusion", float("nan")),
        final_test_metrics.get("properties", float("nan")),
        final_test_metrics.get("grammar", float("nan")),
    )

    # Load and evaluate best model on all splits
    log.info("Evaluating best model on train/validation/test sets...")
    load_checkpoint(best_checkpoint_path, model, optimizer=None, scheduler=None)
    best_train_metrics = compute_eval_loss_stage_c(
        model, train_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
    )
    best_val_metrics = compute_eval_loss_stage_c(
        model, val_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
    )
    best_test_metrics = compute_eval_loss_stage_c(
        model, test_dataloader, device, vocab, lambda_prop, lambda_gram, target_property
    )

    log.info(
        "Best model (step %d) - Train: total=%.4f diff=%.4f prop=%.4f gram=%.4f | "
        "Val: total=%.4f diff=%.4f prop=%.4f gram=%.4f | "
        "Test: total=%.4f diff=%.4f prop=%.4f gram=%.4f",
        best_step,
        best_train_metrics.get("total", float("nan")),
        best_train_metrics.get("diffusion", float("nan")),
        best_train_metrics.get("properties", float("nan")),
        best_train_metrics.get("grammar", float("nan")),
        best_val_metrics.get("total", float("nan")),
        best_val_metrics.get("diffusion", float("nan")),
        best_val_metrics.get("properties", float("nan")),
        best_val_metrics.get("grammar", float("nan")),
        best_test_metrics.get("total", float("nan")),
        best_test_metrics.get("diffusion", float("nan")),
        best_test_metrics.get("properties", float("nan")),
        best_test_metrics.get("grammar", float("nan")),
    )

    # Save metrics to JSON
    final_metrics = {
        "train": final_train_metrics,
        "validation": final_val_metrics,
        "test": final_test_metrics,
    }
    best_metrics = {
        "train": best_train_metrics,
        "validation": best_val_metrics,
        "test": best_test_metrics,
    }
    metrics_path = results_dir / "final_metrics.json"
    save_metrics_json(metrics_path, final_metrics, best_metrics, best_step, best_val_loss)

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
