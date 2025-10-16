#!/usr/bin/env python
"""
Build plain vocabulary for Stage A (small molecules without attachment points).

This script builds a vocabulary from plain SMILES strings without any
attachment point tokens. Use this for Stage A training only.

Usage:
    python -m PolyDiffusion.scripts.build_plain_vocab \
        --input Data/small_molecules_part.csv \
        --output PolyDiffusion/vocab_stage_a.txt \
        --smiles-column SMILES \
        --limit 100000
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable

from PolyDiffusion.src.chem.plain_vocab import PlainVocab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def iter_smiles_from_csv(
    path: Path,
    smiles_column: str,
    limit: int | None = None,
) -> Iterable[str]:
    """Iterate over SMILES from CSV file."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break
            if smiles_column in row:
                yield row[smiles_column].strip()


def iter_smiles_from_jsonl(
    path: Path,
    smiles_field: str,
    limit: int | None = None,
) -> Iterable[str]:
    """Iterate over SMILES from JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            record = json.loads(line)
            if smiles_field in record:
                yield record[smiles_field].strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build plain vocabulary for Stage A (small molecules)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV or JSONL file with plain SMILES",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output vocabulary file path",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="SMILES",
        help="Column name for SMILES (CSV) or field name (JSONL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of SMILES to process (default: all)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum character frequency to include (default: 1)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum vocabulary size (default: unlimited)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    logger.info(f"Reading SMILES from {args.input}")

    # Detect file format and iterate
    if args.input.suffix == ".csv":
        corpus = list(iter_smiles_from_csv(args.input, args.smiles_column, args.limit))
    elif args.input.suffix == ".jsonl":
        corpus = list(iter_smiles_from_jsonl(args.input, args.smiles_column, args.limit))
    else:
        logger.error(f"Unsupported file format: {args.input.suffix}")
        return

    if not corpus:
        logger.error("No SMILES found in input file!")
        return

    logger.info(f"Found {len(corpus)} SMILES strings")
    logger.info(f"Sample SMILES: {corpus[0]}")

    # Build vocabulary
    logger.info("Building vocabulary...")
    vocab = PlainVocab.build(corpus, min_freq=args.min_freq, max_size=args.max_size)

    logger.info(f"Built vocabulary with {len(vocab)} tokens")
    logger.info(f"Special tokens: {vocab.id_to_token[:5]}")

    # Save vocabulary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(args.output)

    logger.info(f"Saved vocabulary to {args.output}")

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("VOCABULARY STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total tokens: {len(vocab)}")
    logger.info(f"Special tokens: {vocab.id_to_token[:5]}")
    logger.info(f"Character tokens: {vocab.id_to_token[5:15]}...")  # First 10 chars
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
