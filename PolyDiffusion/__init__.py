"""Convenience re-exports for the PolyDiffusion package."""

from __future__ import annotations

import sys

from .src import chem as _chem
from .src import data as _data
from .src import losses as _losses
from .src import models as _models
from .src import sampling as _sampling
from .src import train as _train
from .src import utils as _utils
from .src import get_version

# Expose modules at package level
chem = _chem
data = _data
losses = _losses
models = _models
sampling = _sampling
train = _train
utils = _utils

# Register aliases so `import PolyDiffusion.chem` works without editable install
_MODULE_ALIASES = {
    "chem": chem,
    "data": data,
    "losses": losses,
    "models": models,
    "sampling": sampling,
    "train": train,
    "utils": utils,
}
for _name, _module in _MODULE_ALIASES.items():
    sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [
    "chem",
    "data",
    "losses",
    "models",
    "sampling",
    "train",
    "utils",
    "get_version",
]
