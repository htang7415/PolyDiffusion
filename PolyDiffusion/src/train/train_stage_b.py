"""Stage B training: AP-SMILES polymers with grammar penalties."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..chem.ap_smiles import SHIELD1, SHIELD2
from ..chem.vocab_config import load_tokenization_config
from ..chem.vocab_factory import load_vocabulary_auto
from ..losses.objectives import stage_b_objective
from ..utils.logging import configure_logging
from .common import (
    build_model,
    build_stage_dataset,
    build_vocab_from_dataset,
    collate_stage_b,
    compute_eval_loss_stage_b,
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


def _extract_checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)", path.stem)
    return int(match.group(1)) if match else -1


def _find_stage_checkpoint(base_dir: Path, method: str) -> Optional[Path]:
    """Find a Stage A checkpoint for a given tokenisation method."""
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


def run_stage_b(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()
    log = logging.getLogger(__name__)

    start_time = time.perf_counter()

    # Load dataset first
    full_dataset = build_stage_dataset("b", cfg["data"])

    # Split into train/validation/test (8:1:1)
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        ratios=(0.8, 0.1, 0.1),
        seed=cfg["data"].get("seed", 42)
    )

    # Load tokenization configuration
    tok_config = load_tokenization_config(cfg)
    log.info(f"Using tokenization method: {tok_config.method}")

    # Determine results directory per tokenization method
    base_results_dir = Path(cfg.get("results_dir", "Results/stage_b"))
    results_dir = base_results_dir / tok_config.method
    results_dir.mkdir(parents=True, exist_ok=True)

    # Auto-build vocabulary if not provided or doesn't exist
    vocab_path = Path(tok_config.vocab_path) if tok_config.vocab_path else None

    if vocab_path and vocab_path.exists():
        vocab = load_vocabulary_auto(vocab_path, tok_config)
        log.info(f"Loaded {tok_config.method} vocabulary from {vocab_path}")
    else:
        if vocab_path:
            log.warning(f"Vocabulary file {vocab_path} not found. Building from dataset...")
        else:
            log.info("No vocab_path specified. Building vocabulary from dataset...")
            # Default vocab path in results directory
            vocab_path = results_dir / f"vocab_{tok_config.method}_stage_b.txt"

        vocab = build_vocab_from_dataset(
            full_dataset,
            "b",
            tok_config,
            limit=tok_config.vocab_limit_samples
        )
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        log.info(f"Built {tok_config.method} vocabulary with {len(vocab)} tokens, saved to {vocab_path}")

    model_cfg = load_yaml(Path(cfg["model_config"]))

    init_cfg = cfg.get("initialization") or {}
    if "mode" not in init_cfg and "stage_a" in init_cfg:
        stage_a_cfg = init_cfg.get("stage_a") or {}
        stage_a_cfg.setdefault("mode", "stage_a")
        init_cfg = stage_a_cfg

    mode = (init_cfg.get("mode") or "").lower()
    if not mode:
        if cfg.get("pretrained_checkpoint"):
            mode = "stage_a"
            init_cfg = {
                "mode": "stage_a",
                "checkpoint": cfg["pretrained_checkpoint"],
                "vocab_path": cfg.get("pretrained_vocab_path"),
                "freeze_backbone": cfg.get("freeze_backbone", False),
                "reuse_token_embeddings": cfg.get("reuse_token_embeddings", True),
                "reuse_output_head": cfg.get("reuse_output_head", True),
            }
        else:
            mode = "scratch"

    if mode == "stage_a":
        checkpoint_value = init_cfg.get("checkpoint")
        stage_a_results_dir = Path(cfg.get("stage_a_results_dir", "Results/stage_a"))
        stage_a_method = init_cfg.get("stage_a_method", tok_config.method)
        pretrained_path: Optional[Path] = None
        if checkpoint_value:
            candidate_path = Path(checkpoint_value).expanduser()
            if candidate_path.exists():
                pretrained_path = candidate_path
            else:
                log.warning("Specified Stage A checkpoint not found: %s", candidate_path)
        if pretrained_path is None:
            fallback_checkpoint = _find_stage_checkpoint(
                stage_a_results_dir,
                stage_a_method,
            )
            if fallback_checkpoint is not None:
                log.info("Using Stage A checkpoint fallback at %s", fallback_checkpoint)
                pretrained_path = fallback_checkpoint
        if pretrained_path is None:
            raise FileNotFoundError(
                "Stage A checkpoint not found. Set 'initialization.checkpoint' or ensure a checkpoint exists under "
                f"{stage_a_results_dir}/<method>/."
            )

        vocab_candidate = init_cfg.get("vocab_path")
        stage_a_vocab_path: Path | None = None
        if vocab_candidate:
            candidate_path = Path(vocab_candidate)
            if not candidate_path.exists():
                raise FileNotFoundError(f"Specified Stage A vocab_path does not exist: {candidate_path}")
            stage_a_vocab_path = candidate_path
        else:
            fallback_candidates = [pretrained_path.parent / "vocab.txt"]
            stage_a_results_base = stage_a_results_dir
            if stage_a_results_base.exists():
                # Search method-specific subdirectories first
                method_dirs = [
                    stage_a_results_base / stage_a_method,
                    stage_a_results_base,
                ]
                for directory in method_dirs:
                    if directory and directory.exists():
                        for candidate in sorted(directory.glob("**/vocab*.txt")):
                            fallback_candidates.append(candidate)
            # Repository defaults as last resort
            fallback_candidates.extend(
                [
                    Path("PolyDiffusion/vocab_character_stage_a.txt"),
                    Path("PolyDiffusion/vocab.txt"),
                ]
            )
            for candidate in fallback_candidates:
                if candidate and candidate.exists():
                    stage_a_vocab_path = candidate
                    break
        if stage_a_vocab_path is None:
            raise FileNotFoundError(
                "Unable to locate Stage A vocabulary file. "
                "Specify 'initialization.vocab_path' in the config."
            )

        freeze_backbone = init_cfg.get("freeze_backbone")
        if freeze_backbone is None:
            freeze_backbone = cfg.get("freeze_backbone", False)
        reuse_token_embeddings = init_cfg.get("reuse_token_embeddings", True)
        reuse_output_head = init_cfg.get("reuse_output_head", True)

        model = load_pretrained_for_finetuning(
            pretrained_path,
            vocab,
            model_cfg,
            freeze_backbone=freeze_backbone,
            source_vocab_path=stage_a_vocab_path,
            reuse_token_embeddings=reuse_token_embeddings,
            reuse_output_head=reuse_output_head,
        )
        log.info(
            "Initialized Stage B model from Stage A checkpoint %s (reuse_token_embeddings=%s, reuse_output_head=%s).",
            pretrained_path,
            reuse_token_embeddings,
            reuse_output_head,
        )
    elif mode == "scratch":
        model = build_model(vocab, model_cfg)
        log.info("Initialized Stage B model from scratch (no Stage A checkpoint).")
    else:
        raise ValueError(
            f"Unsupported initialization mode '{mode}'. "
            "Choose 'scratch' or 'stage_a' under the 'initialization' section."
        )

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
        lambda batch: collate_stage_b(batch, vocab),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=True,
    )
    val_dataloader = make_dataloader(
        val_dataset,
        micro_batch_size,
        lambda batch: collate_stage_b(batch, vocab),
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=train_cfg.get("pin_memory"),
        shuffle=False,
    )
    test_dataloader = make_dataloader(
        test_dataset,
        micro_batch_size,
        lambda batch: collate_stage_b(batch, vocab),
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
    lambda_gram = cfg["loss"].get("lambda_gram", 0.1)

    # Setup Results directory
    log.info(f"Results will be saved to {results_dir}")

    results_vocab_path = results_dir / "vocab.txt"
    try:
        vocab.save(results_vocab_path)
        log.info(f"Synced vocabulary to {results_vocab_path}")
    except OSError as exc:
        log.warning("Failed to write vocabulary copy to results directory: %s", exc)

    # Best model tracking (based on validation loss)
    best_val_loss = float('inf')
    best_step = 0
    best_checkpoint_path = results_dir / "best_model.pt"

    # Gradient clipping / AMP configuration
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    use_amp = train_cfg.get("use_amp", True) and device.type == "cuda"

    try:
        from torch import amp as torch_amp  # torch>=2.0 preferred API

        GradScalerCls = torch_amp.GradScaler

        def _autocast(enabled: bool, dtype: torch.dtype) -> Callable:
            return torch_amp.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    except (ImportError, AttributeError, TypeError):  # pragma: no cover - fallback for older torch
        from torch.cuda.amp import GradScaler as GradScalerCls, autocast as torch_autocast  # type: ignore[attr-defined]

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
            anchor_count = batch["anchor_count"].to(device)
            valence = batch["valence"].to(device)

            timesteps = model.diffusion.sample_timesteps(tokens.size(0))
            noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)
            anchor_mask_tokens = (tokens == vocab.token_to_id[SHIELD1]) | (tokens == vocab.token_to_id[SHIELD2])
            if torch.any(anchor_mask_tokens):
                noise_mask = noise_mask | anchor_mask_tokens
                noisy_tokens = noisy_tokens.masked_fill(anchor_mask_tokens, vocab.mask_id)

            with _autocast(use_amp, torch.float16):
                outputs = model(noisy_tokens, timesteps, attention_mask=mask)
                losses = stage_b_objective(
                    model,
                    outputs,
                    tokens,
                    timesteps,
                    noise_mask,
                    anchor_count,
                    valence,
                    lambda_gram,
                    vocab,
                )
                total_loss = losses["total"] / grad_accum_steps

            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            for name, tensor in losses.items():
                step_loss_sums[name] = step_loss_sums.get(name, 0.0) + float(tensor.detach().cpu())

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
                "step=%d loss_total=%.4f loss_diff=%.4f loss_gram=%.4f lr=%.2e",
                step,
                loss_values.get("total", float("nan")),
                loss_values.get("diffusion", float("nan")),
                loss_values.get("grammar", float("nan")),
                current_lr,
            )

        # Validation loss evaluation
        if step % eval_interval == 0:
            val_losses = compute_eval_loss_stage_b(model, val_dataloader, device, vocab, lambda_gram)
            log.info(
                "step=%d val_loss_total=%.4f val_loss_diff=%.4f val_loss_gram=%.4f",
                step,
                val_losses.get("total", float("nan")),
                val_losses.get("diffusion", float("nan")),
                val_losses.get("grammar", float("nan")),
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
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                step + 1,
                loss_values.get("total", float("nan")),
            )
            log.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = results_dir / "final_model.pt"
    final_loss = last_logged_losses["total"] if last_logged_losses is not None else 0.0
    save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, steps, final_loss)
    log.info(f"Training completed. Final checkpoint saved to {final_checkpoint_path}")

    # Evaluate final model on all splits
    log.info("Evaluating final model on train/validation/test sets...")
    final_train_metrics = compute_eval_loss_stage_b(model, train_dataloader, device, vocab, lambda_gram)
    final_val_metrics = compute_eval_loss_stage_b(model, val_dataloader, device, vocab, lambda_gram)
    final_test_metrics = compute_eval_loss_stage_b(model, test_dataloader, device, vocab, lambda_gram)

    log.info(
        "Final model - Train: total=%.4f diff=%.4f gram=%.4f | Val: total=%.4f diff=%.4f gram=%.4f | Test: total=%.4f diff=%.4f gram=%.4f",
        final_train_metrics.get("total", float("nan")),
        final_train_metrics.get("diffusion", float("nan")),
        final_train_metrics.get("grammar", float("nan")),
        final_val_metrics.get("total", float("nan")),
        final_val_metrics.get("diffusion", float("nan")),
        final_val_metrics.get("grammar", float("nan")),
        final_test_metrics.get("total", float("nan")),
        final_test_metrics.get("diffusion", float("nan")),
        final_test_metrics.get("grammar", float("nan")),
    )

    # Load and evaluate best model on all splits
    log.info("Evaluating best model on train/validation/test sets...")
    load_checkpoint(best_checkpoint_path, model, optimizer=None, scheduler=None)
    best_train_metrics = compute_eval_loss_stage_b(model, train_dataloader, device, vocab, lambda_gram)
    best_val_metrics = compute_eval_loss_stage_b(model, val_dataloader, device, vocab, lambda_gram)
    best_test_metrics = compute_eval_loss_stage_b(model, test_dataloader, device, vocab, lambda_gram)

    log.info(
        "Best model (step %d) - Train: total=%.4f diff=%.4f gram=%.4f | Val: total=%.4f diff=%.4f gram=%.4f | Test: total=%.4f diff=%.4f gram=%.4f",
        best_step,
        best_train_metrics.get("total", float("nan")),
        best_train_metrics.get("diffusion", float("nan")),
        best_train_metrics.get("grammar", float("nan")),
        best_val_metrics.get("total", float("nan")),
        best_val_metrics.get("diffusion", float("nan")),
        best_val_metrics.get("grammar", float("nan")),
        best_test_metrics.get("total", float("nan")),
        best_test_metrics.get("diffusion", float("nan")),
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
    parser.add_argument("--config", required=True, type=str, help="Path to stage B YAML config.")
    args = parser.parse_args()
    run_stage_b(args.config)


if __name__ == "__main__":
    main()
