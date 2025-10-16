#!/usr/bin/env python
"""Build anchor-safe vocabulary from corpus."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PolyDiffusion.src.chem.vocab import AnchorSafeVocab
from PolyDiffusion.src.chem import convert_polymer_to_ap_smiles
from PolyDiffusion.src.utils.fileio import open_compressed, stream_jsonl


def iter_strings(path: Path, field: str, limit: int | None, auto_convert: bool = False) -> Iterable[str]:
    """Iterate over strings from corpus.

    Args:
        path: Path to corpus file (JSONL or CSV)
        field: Field name containing SMILES
        limit: Maximum number of records to read
        auto_convert: If True, convert raw polymer SMILES (*) to AP-SMILES ([*:1]/[*:2])
    """
    count = 0
    if path.suffix in {".jsonl", ".gz"}:
        for record in stream_jsonl(path):
            if field not in record or record[field] is None:
                continue
            smiles = str(record[field])
            if auto_convert and '*' in smiles:
                try:
                    smiles = convert_polymer_to_ap_smiles(smiles)
                except ValueError as e:
                    print(f"Warning: Skipping invalid polymer SMILES '{smiles}': {e}")
                    continue
            yield smiles
            count += 1
            if limit is not None and count >= limit:
                break
    else:
        with open_compressed(path, "rt") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if field not in row or row[field] is None:
                    continue
                smiles = str(row[field])
                if auto_convert and '*' in smiles:
                    try:
                        smiles = convert_polymer_to_ap_smiles(smiles)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid polymer SMILES '{smiles}': {e}")
                        continue
                yield smiles
                count += 1
                if limit is not None and count >= limit:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anchor-safe vocabulary for Stage B/C from polymer SMILES.")
    parser.add_argument("input", type=str, help="Path to corpus (JSONL or CSV).")
    parser.add_argument("output", type=str, help="Destination vocabulary file.")
    parser.add_argument("--field", type=str, default=None, help="Field name (default: 'ap_smiles' for JSONL, 'SMILES' for CSV).")
    parser.add_argument("--smiles-column", type=str, default="SMILES", help="SMILES column name for CSV (default: 'SMILES').")
    parser.add_argument("--limit", type=int, default=10000, help="Number of examples to sample.")
    args = parser.parse_args()

    path = Path(args.input)
    output = Path(args.output)

    # Determine field name and whether to auto-convert
    if args.field is not None:
        field = args.field
        auto_convert = False  # Explicit field name means preprocessed data
    elif path.suffix in {".jsonl", ".gz"}:
        field = "ap_smiles"
        auto_convert = False  # JSONL typically has preprocessed data
    else:
        # CSV with raw polymer SMILES
        field = args.smiles_column
        auto_convert = True  # Auto-convert raw polymer SMILES

    corpus = list(iter_strings(path, field, args.limit, auto_convert))
    if not corpus:
        raise RuntimeError("No records found to build vocabulary.")

    vocab = AnchorSafeVocab.build(corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output)
    print(f"Wrote vocab with {len(vocab)} tokens to {output}")
    if auto_convert:
        print("Note: Raw polymer SMILES were automatically converted to AP-SMILES format.")


if __name__ == "__main__":
    main()
