"""Top-level package for PolyDiffusion."""

from importlib import metadata

# Re-export submodules
from . import chem, data, losses, models, sampling, train, utils


def get_version() -> str:
    """Return the package version."""
    try:
        return metadata.version("PolyDiffusion")
    except metadata.PackageNotFoundError:  # pragma: no cover - fallback during dev
        return "0.0.0"


__all__ = ["chem", "data", "losses", "models", "sampling", "train", "utils", "get_version"]
