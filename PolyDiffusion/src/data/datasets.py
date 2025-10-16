"""Dataset definitions for the PolyDiffusion training stages."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, MutableMapping

from torch.utils.data import Dataset

from ..utils.fileio import open_compressed, stream_jsonl

Record = MutableMapping[str, object]


@dataclass
class DatasetConfig:
    """Common configuration for datasets."""

    path: Path
    limit: int | None = None
    shuffle: bool = False
    cache_in_memory: bool = True
    seed: int | None = None

    def ensure_exists(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.path}")


class _BaseDataset(Dataset[Record]):
    """Base class providing shared loading and validation."""

    def __init__(self, config: DatasetConfig, required_fields: Iterable[str] | None = None) -> None:
        self.config = config
        self.config.ensure_exists()
        self.required_fields = set(required_fields or [])
        if not self.config.cache_in_memory:
            raise ValueError("Streaming datasets are not yet supported; set cache_in_memory=True.")

        self._records = self._load_records()
        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(self._records)

    def _load_records(self) -> List[Record]:  # pragma: no cover
        raise NotImplementedError

    def _validate_record(self, record: Record, index: int) -> None:
        missing = self.required_fields.difference(record.keys())
        if missing:
            raise KeyError(f"Record {index} missing required fields: {', '.join(sorted(missing))}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Record:
        return self._records[index]


class JsonlDataset(_BaseDataset):
    """Dataset backed by JSON Lines (`.jsonl`) or gzip-compressed JSONL files."""

    def _load_records(self) -> List[Record]:
        records: List[Record] = []
        limit = self.config.limit
        for idx, record in enumerate(stream_jsonl(self.config.path)):
            if self.required_fields:
                self._validate_record(record, idx)
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        if not records:
            raise RuntimeError(f"No records found in dataset: {self.config.path}")
        return records


class CsvDataset(_BaseDataset):
    """Dataset backed by a CSV file."""

    def _load_records(self) -> List[Record]:
        records: List[Record] = []
        limit = self.config.limit
        with open_compressed(self.config.path, "rt") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                record: Record = dict(row)
                if self.required_fields:
                    self._validate_record(record, idx)
                records.append(record)
                if limit is not None and len(records) >= limit:
                    break
        if not records:
            raise RuntimeError(f"No records found in dataset: {self.config.path}")
        return records
