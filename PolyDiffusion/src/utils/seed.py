"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch


def seed_all(seed: Optional[int] = None, deterministic: bool = True) -> int:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    Args:
        seed: Optional seed value. When omitted a time-based seed is generated.
        deterministic: If True, enables deterministic CUDA operations for full
            reproducibility (may impact performance).

    Returns:
        The seed that was actually used.
    """
    if seed is None:
        seed = int(datetime.now(timezone.utc).timestamp() * 1_000_000) % 2**31

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
