"""Stage B training: AP-SMILES polymers with grammar penalties."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..chem.vocab import AnchorSafeVocab
from ..losses.objectives import stage_b_objective
from ..utils.logging import configure_logging
from .common import (
    build_model,
    build_stage_dataset,
    build_vocab_from_dataset,
    collate_stage_b,
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


def run_stage_b(config_path: str) -> None:
    cfg = load_yaml(Path(config_path))
    configure_logging()
    log = logging.getLogger(__name__)

    start_time = time.perf_counter()

    # Load dataset first
    dataset = build_stage_dataset("b", cfg["data"])

    # Auto-build vocabulary if not provided or doesn't exist
    if "vocab_path" in cfg and cfg["vocab_path"]:
        vocab_path = Path(cfg["vocab_path"])
        if vocab_path.exists():
            vocab = AnchorSafeVocab.load(vocab_path)
            log.info(f"Loaded vocabulary from {vocab_path}")
        else:
            log.warning(f"Vocabulary file {vocab_path} not found. Building from dataset...")
            vocab = build_vocab_from_dataset(dataset, "b", limit=cfg.get("vocab_limit", 10000))
            vocab_path.parent.mkdir(parents=True, exist_ok=True)
            vocab.save(vocab_path)
            log.info(f"Saved vocabulary to {vocab_path}")
    else:
        log.info("No vocab_path specified. Building vocabulary from dataset...")
        vocab = build_vocab_from_dataset(dataset, "b", limit=cfg.get("vocab_limit", 10000))
        # Save to default location
        results_dir = Path(cfg.get("results_dir", "Results/stage_b"))
        vocab_path = results_dir / "vocab.txt"
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        log.info(f"Saved vocabulary to {vocab_path}")

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
        if not checkpoint_value:
            raise ValueError("Stage B initialization mode 'stage_a' requires a 'checkpoint' path.")
        pretrained_path = Path(checkpoint_value)
        if not pretrained_path.exists():
            raise FileNotFoundError(f"Stage A checkpoint not found: {pretrained_path}")

        vocab_candidate = init_cfg.get("vocab_path")
        stage_a_vocab_path: Path | None = None
        if vocab_candidate:
            candidate_path = Path(vocab_candidate)
            if not candidate_path.exists():
                raise FileNotFoundError(f"Specified Stage A vocab_path does not exist: {candidate_path}")
            stage_a_vocab_path = candidate_path
        else:
            fallback_candidates = [
                pretrained_path.parent / "vocab.txt",
                Path("Results/stage_a/vocab.txt"),
            ]
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
    dataloader = make_dataloader(
        dataset,
        micro_batch_size,
        lambda batch: collate_stage_b(batch, vocab),
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
    save_interval = train_cfg.get("save_interval", 500)
    lambda_syn = cfg["loss"]["lambda_syn"]
    lambda_gram = cfg["loss"].get("lambda_gram", 0.1)
    log = logging.getLogger(__name__)

    # Setup Results directory
    results_dir = Path(cfg.get("results_dir", "Results/stage_b"))
    results_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Results will be saved to {results_dir}")

    # Best model tracking
    best_loss = float('inf')
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

    data_iter = iter(dataloader)
    last_logged_losses: dict[str, float] | None = None
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss_sums: dict[str, float] = {}

        for micro_step in range(grad_accum_steps):
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

            timesteps = model.diffusion.sample_timesteps(tokens.size(0))
            noisy_tokens, noise_mask = model.diffusion.q_sample(tokens, timesteps)

            with _autocast(use_amp, torch.float16):
                outputs = model(noisy_tokens, timesteps, attention_mask=mask, s_target=synth)
                losses = stage_b_objective(
                    model,
                    outputs,
                    tokens,
                    timesteps,
                    noise_mask,
                    synth,
                    anchor_count,
                    valence,
                    lambda_syn,
                    lambda_gram,
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
                "step=%d loss_total=%.4f loss_diff=%.4f loss_syn=%.4f loss_gram=%.4f lr=%.2e",
                step,
                loss_values.get("total", float("nan")),
                loss_values.get("diffusion", float("nan")),
                loss_values.get("synth", float("nan")),
                loss_values.get("grammar", float("nan")),
                current_lr,
            )

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

        # Save best model
        current_total = loss_values.get("total")
        if current_total is not None and current_total < best_loss:
            best_loss = current_total
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, step + 1, best_loss)
            log.info(f"Saved best model with loss {best_loss:.4f}")

    # Save final checkpoint
    final_checkpoint_path = results_dir / "final_model.pt"
    final_loss = last_logged_losses["total"] if last_logged_losses is not None else 0.0
    save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, steps, final_loss)
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
    parser.add_argument("--config", required=True, type=str, help="Path to stage B YAML config.")
    args = parser.parse_args()
    run_stage_b(args.config)


if __name__ == "__main__":
    main()
