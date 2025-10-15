"""Lightweight file IO helpers."""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Union


def open_compressed(path: Union[str, Path], mode: str = "rt", encoding: str = "utf-8") -> io.TextIOBase:
    """Open plain or gzipped files transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(path, mode.replace("t", "")), encoding=encoding)
    return path.open(mode, encoding=encoding)


def stream_jsonl(path: Union[str, Path]) -> Iterator[dict]:
    """Yield JSON objects from a JSONL or JSONL.GZ file."""
    with open_compressed(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Union[str, Path], records: Iterable[dict]) -> None:
    """Write an iterable of dictionaries to JSON Lines file."""
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
