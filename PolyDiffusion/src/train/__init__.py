"""Training entrypoints for polymer generation (Stage B) and property-guided generation (Stage C).

NOTE: Stage A (small molecules) has been removed - this codebase focuses on polymer-only training.
"""

from .train_stage_b import run_stage_b
from .train_stage_c import run_stage_c

__all__ = ["run_stage_b", "run_stage_c"]
