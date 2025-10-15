"""Logging utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, TextIO


class JsonLogWriter:
    """Append structured logs to a JSON Lines file.

    Context manager that keeps file handle open for efficient writing.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[TextIO] = None

    def __enter__(self) -> "JsonLogWriter":
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write(self, record: Dict[str, Any]) -> None:
        """Write a record to the JSON Lines file.

        Args:
            record: Dictionary to serialize and append.

        Raises:
            RuntimeError: If writer is not used as a context manager.
        """
        if self._file is None:
            raise RuntimeError("JsonLogWriter must be used as a context manager")
        try:
            self._file.write(json.dumps(record) + "\n")
            self._file.flush()
        except (OSError, IOError) as e:
            logging.error(f"Failed to write log record: {e}")


def configure_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure the root logger with stream and optional file handler."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
