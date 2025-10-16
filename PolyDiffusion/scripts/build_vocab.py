#!/usr/bin/env python
"""Build anchor-safe vocabulary from corpus."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.utils.fileio import open_compressed, stream_jsonl


def iter_strings(path: Path, field: str, limit: int | None) -> Iterable[str]:
    count = 0
    if path.suffix in {".jsonl", ".gz"}:
        for record in stream_jsonl(path):
            if field not in record or record[field] is None:
                continue
            yield str(record[field])
            count += 1
            if limit is not None and count >= limit:
                break
    else:
        with open_compressed(path, "rt") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if field not in row or row[field] is None:
                    continue
                yield str(row[field])
                count += 1
                if limit is not None and count >= limit:
                    break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to corpus (JSONL or CSV).")
    parser.add_argument("output", type=str, help="Destination vocabulary file.")
    parser.add_argument("--field", type=str, default="ap_smiles", help="Field containing AP-SMILES strings.")
    parser.add_argument("--limit", type=int, default=10000, help="Number of examples to sample.")
    args = parser.parse_args()

    path = Path(args.input)
    output = Path(args.output)
    corpus = list(iter_strings(path, args.field, args.limit))
    if not corpus:
        raise RuntimeError("No records found to build vocabulary.")
    vocab = AnchorSafeVocab.build(corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output)
    print(f"Wrote vocab with {len(vocab)} tokens to {output}")


if __name__ == "__main__":
    main()
