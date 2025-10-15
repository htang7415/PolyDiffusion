"""Simple metric accumulators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RunningAverage:
    """Track running mean and count."""

    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class MetricLogger:
    """Store multiple named running averages."""

    metrics: Dict[str, RunningAverage] = field(default_factory=dict)

    def update(self, name: str, value: float, n: int = 1) -> None:
        if name not in self.metrics:
            self.metrics[name] = RunningAverage()
        self.metrics[name].update(value, n)

    def as_dict(self) -> Dict[str, float]:
        return {name: meter.mean for name, meter in self.metrics.items()}
