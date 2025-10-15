"""Utilities for PyTorch distributed training."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, Optional

import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl", timeout_seconds: int = 1800) -> bool:
    """Initialise torch.distributed if environment variables are set."""
    if dist.is_available() and not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_seconds),
            )
    return dist.is_initialized()


def is_main_process() -> bool:
    """Return True if current process is rank zero."""
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    """Synchronise processes when distributed is initialised."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@contextlib.contextmanager
def distributed_zero_first(local_rank: Optional[int]) -> Iterator[None]:
    """Ensure global rank zero runs code block before other ranks."""
    if local_rank in (0, None):
        yield
        barrier()
    else:
        barrier()
        yield


@dataclass
class AverageMeter:
    """Track distributed averages."""

    name: str
    value: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.value += val * n
        self.count += n

    def compute(self) -> float:
        total = torch.tensor([self.value, self.count], device="cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        value_sum, count_sum = total.tolist()
        return value_sum / max(count_sum, 1)
