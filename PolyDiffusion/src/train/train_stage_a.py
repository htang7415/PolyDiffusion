"""Stage A training loop: small molecules + synthesis regression."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Callable
import shutil

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..chem.vocab_config import load_tokenization_config
from ..chem.vocab_factory import load_vocabulary_auto
from ..losses.objectives import stage_a_objective
from ..utils.logging import configure_logging
from .common import (
    build_model,
    build_stage_dataset,
    build_vocab_from_dataset,
    collate_stage_a,
    compute_eval_loss_stage_a,
    default_device,
    load_yaml,
    make_dataloader,
    save_checkpoint,
    save_metrics_json,
    load_checkpoint,
    split_dataset,
)

try:  # pragma: no cover - optional dependency
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None

try:  # pragma: no cover - optional dependency
    import psutil
except ImportError:  # pragma: no cover - optional dependency
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


def run_stage_a(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()
    log = logging.getLogger(__name__)

    start_time = time.perf_counter()

    # Load dataset first
    full_dataset = build_stage_dataset("a", cfg["data"])

    # Split into train/validation/test (8:1:1)
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        ratios=(0.8, 0.1, 0.1),
        seed=cfg["data"].get("seed", 42)
    )

    # Load tokenization configuration
    tok_config = load_tokenization_config(cfg)
    log.info(f"Using tokenization method: {tok_config.method}")

    # Determine output directory (per tokenization method)
    base_results_dir = Path(cfg.get("results_dir", "Results/stage_a"))
    results_dir = base_results_dir / tok_config.method
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-build vocabulary if not provided or doesn't exist
    vocab_path = Path(tok_config.vocab_path) if tok_config.vocab_path else None

    if vocab_path and vocab_path.exists():
        vocab = load_vocabulary_auto(vocab_path, tok_config)
        log.info(f"Loaded {tok_config.method} vocabulary from {vocab_path}")
        alias_path = results_dir / "vocab.txt"
        if alias_path != vocab_path and not alias_path.exists():
            try:
                shutil.copy2(vocab_path, alias_path)
                log.info(f"Copied vocabulary alias to {alias_path}")
            except Exception as exc:  # pragma: no cover - best effort
                log.warning(f"Failed to copy vocabulary alias to {alias_path}: {exc}")
    else:
        if vocab_path:
            log.warning(f"Vocabulary file {vocab_path} not found. Building from dataset...")
        else:
            log.info("No vocab_path specified. Building vocabulary from dataset...")
            # Default vocab path in results directory
            vocab_path = results_dir / f"vocab_{tok_config.method}_stage_a.txt"

        vocab = build_vocab_from_dataset(
            full_dataset,
            "a",
            tok_config,
            limit=tok_config.vocab_limit_samples
        )
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        log.info(f"Built {tok_config.method} vocabulary with {len(vocab)} tokens, saved to {vocab_path}")
        alias_path = results_dir / "vocab.txt"
        if alias_path != vocab_path:
            try:
                shutil.copy2(vocab_path, alias_path)
                log.info(f"Copied vocabulary alias to {alias_path}")
            except Exception as exc:  # pragma: no cover - best effort
                log.warning(f"Failed to copy vocabulary alias to {alias_path}: {exc}")

    model_cfg = load_yaml(Path(cfg["model_config"]))
    model = build_model(vocab, model_cfg)
    device = default_device()
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_cfg = cfg["training"]
    batch_size = train_cfg["batch_size"]
    micro_batch_size = train_cfg.get("micro_batch_size", batch_size)
    grad_accum_steps = max(int(train_cfg.get("grad_accum_steps", 1)), 1)
    if micro_batch_size <= 0:
        raise ValueError("training.micro_batch_size must be positive.")
    if grad_accum_steps <= 0:
        raise ValueError("training.grad_accum_steps must be positive.")

    if grad_accum_steps > 1 and micro_batch_size * grad_accum_steps != batch_size:
        log.info(
            "Using micro_batch_size=%d with grad_accum_steps=%d (effective batch size=%d).",
            micro_batch_size,
            grad_accum_steps,
            micro_batch_size * grad_accum_steps,
        )
    elif grad_accum_steps > 1:
        log.info(
            "Using gradient accumulation with micro_batch_size=%d and grad_accum_steps=%d.",
            micro_batch_size,
            grad_accum_steps,
        )

    # Create dataloaders: training with shuffle, validation/test without shuffle
    train_dataloader = make_dataloader(
        train_dataset,
        micro_batch_size,
        lambda batch: collate_stage_a(batch, vocab),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=True,
    )
    val_dataloader = make_dataloader(
        val_dataset,
        micro_batch_size,
        lambda batch: collate_stage_a(batch, vocab),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=False,
    )
    test_dataloader = make_dataloader(
        test_dataset,
        micro_batch_size,
        lambda batch: collate_stage_a(batch, vocab),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(0.9, 0.95),
        eps=1e-8,
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
            log.info(f"Resumed from checkpoint {resume_path} at step {start_step}")

    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 500)
    eval_interval = train_cfg.get("eval_interval", log_interval)
    log.info(f"Results will be saved to {results_dir}")

    # Best model tracking (based on validation loss)
    best_val_loss = float('inf')
    best_step = 0
    best_checkpoint_path = results_dir / "best_model.pt"

    # Gradient clipping
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"

    try:
        from torch import amp as torch_amp  # torch>=2.0 preferred API

        GradScalerCls = torch_amp.GradScaler

        def _autocast(enabled: bool, dtype: torch.dtype) -> Callable:
            return torch_amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    except (ImportError, AttributeError, TypeError):  # pragma: no cover - fallback for older torch
        from torch.cuda.amp import GradScaler as GradScalerCls, autocast as torch_autocast  # type: ignore

        def _autocast(enabled: bool, dtype: torch.dtype) -> Callable:
            return torch_autocast(enabled=enabled, dtype=dtype)

    scaler = GradScalerCls(enabled=use_amp)

    model.train()

    # Use cycle to avoid expensive DataLoader recreation on exhaustion
    data_iter = cycle(train_dataloader)
    last_logged_losses: dict[str, float] | None = None
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss_sums: dict[str, float] = {}

        for micro_step in range(grad_accum_steps):
            batch = next(data_iter)

            tokens = batch["tokens"].to(device)
            mask = batch["mask"].to(device)

            timesteps = model.diffusion.sample_timesteps(tokens.size(0))
            noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)

            with _autocast(use_amp, torch.float16):
                outputs = model(noisy_tokens, timesteps, attention_mask=mask)
                losses = stage_a_objective(model, outputs, tokens, timesteps, noise_mask)
                total_loss = losses["total"] / grad_accum_steps

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            for name, value in losses.items():
                step_loss_sums[name] = step_loss_sums.get(name, 0.0) + float(value.detach().cpu())

        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        scheduler.step()

        loss_values = {name: value / grad_accum_steps for name, value in step_loss_sums.items()}
        last_logged_losses = loss_values

        if step % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log.info(
                "step=%d loss_total=%.4f loss_diff=%.4f lr=%.2e",
                step,
                loss_values.get("total", float("nan")),
                loss_values.get("diffusion", float("nan")),
                current_lr,
            )

        # Validation loss evaluation
        if step % eval_interval == 0:
            val_losses = compute_eval_loss_stage_a(model, val_dataloader, device, vocab)
            log.info(
                "step=%d val_loss_total=%.4f val_loss_diff=%.4f",
                step,
                val_losses.get("total", float("nan")),
                val_losses.get("diffusion", float("nan")),
            )

            # Save best model based on validation loss
            current_val_loss = val_losses.get("total", float("inf"))
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_step = step
                save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, step + 1, best_val_loss)
                log.info(f"Saved best model with validation loss {best_val_loss:.4f} at step {step}")

        # Save periodic checkpoints
        if (step + 1) % save_interval == 0:
            checkpoint_path = results_dir / f"checkpoint_step_{step+1}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, step + 1, loss_values.get("total", float("nan")))
            log.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = results_dir / "final_model.pt"
    final_loss = last_logged_losses["total"] if last_logged_losses is not None else 0.0
    save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, steps, final_loss)
    log.info(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")

    # Evaluate final model on all splits
    log.info("Evaluating final model on train/validation/test sets...")
    final_train_metrics = compute_eval_loss_stage_a(model, train_dataloader, device, vocab)
    final_val_metrics = compute_eval_loss_stage_a(model, val_dataloader, device, vocab)
    final_test_metrics = compute_eval_loss_stage_a(model, test_dataloader, device, vocab)

    log.info(
        "Final model - Train: total=%.4f diff=%.4f | Val: total=%.4f diff=%.4f | Test: total=%.4f diff=%.4f",
        final_train_metrics.get("total", float("nan")),
        final_train_metrics.get("diffusion", float("nan")),
        final_val_metrics.get("total", float("nan")),
        final_val_metrics.get("diffusion", float("nan")),
        final_test_metrics.get("total", float("nan")),
        final_test_metrics.get("diffusion", float("nan")),
    )

    # Load and evaluate best model on all splits
    log.info("Evaluating best model on train/validation/test sets...")
    load_checkpoint(best_checkpoint_path, model, optimizer=None, scheduler=None)
    best_train_metrics = compute_eval_loss_stage_a(model, train_dataloader, device, vocab)
    best_val_metrics = compute_eval_loss_stage_a(model, val_dataloader, device, vocab)
    best_test_metrics = compute_eval_loss_stage_a(model, test_dataloader, device, vocab)

    log.info(
        "Best model (step %d) - Train: total=%.4f diff=%.4f | Val: total=%.4f diff=%.4f | Test: total=%.4f diff=%.4f",
        best_step,
        best_train_metrics.get("total", float("nan")),
        best_train_metrics.get("diffusion", float("nan")),
        best_val_metrics.get("total", float("nan")),
        best_val_metrics.get("diffusion", float("nan")),
        best_test_metrics.get("total", float("nan")),
        best_test_metrics.get("diffusion", float("nan")),
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
    parser.add_argument("--config", required=True, type=str, help="Path to stage A YAML config.")
    args = parser.parse_args()
    run_stage_a(args.config)


if __name__ == "__main__":
    main()
