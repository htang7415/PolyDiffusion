"""Training entrypoints for the three-stage curriculum."""

from .train_stage_a import run_stage_a
from .train_stage_b import run_stage_b
from .train_stage_c import run_stage_c

__all__ = ["run_stage_a", "run_stage_b", "run_stage_c"]
