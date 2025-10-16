#!/usr/bin/env python
"""
Preprocess polymer datasets by converting unlabeled * to labeled [*:1] and [*:2].

This script reads a CSV file with polymer SMILES containing unlabeled * symbols
and converts them to properly labeled AP-SMILES format for training.

Usage:
    python -m PolyDiffusion.scripts.preprocess_polymer_dataset \
        --input Data/PI1M.csv \
        --output Data/PI1M_preprocessed.jsonl \
        --smiles-column SMILES \
        --score-column "SA Score"
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

from PolyDiffusion.src.chem import convert_polymer_to_ap_smiles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_csv(
    input_path: Path,
    output_path: Path,
    smiles_column: str = "SMILES",
    score_column: str = "SA_Score",
    limit: int | None = None,
) -> None:
    """
    Convert a CSV file with polymer SMILES to JSONL with AP-SMILES.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output JSONL file
        smiles_column: Name of column containing polymer SMILES
        score_column: Name of column containing synthesis score
        limit: Maximum number of records to process (None for all)
    """
    logger.info(f"Reading from {input_path}")
    logger.info(f"SMILES column: {smiles_column}, Score column: {score_column}")

    records: List[Dict] = []
    errors = 0

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader):
            if limit and idx >= limit:
                break

            try:
                polymer_smiles = row[smiles_column].strip()
                synth_score = float(row[score_column])

                # Convert to AP-SMILES
                ap_smiles = convert_polymer_to_ap_smiles(polymer_smiles)

                records.append({
                    "ap_smiles": ap_smiles,
                    "synth_score": synth_score,
                    "original_smiles": polymer_smiles,
                })

                if (idx + 1) % 1000 == 0:
                    logger.info(f"Processed {idx + 1} records...")

            except KeyError as e:
                logger.error(f"Row {idx}: Missing column {e}")
                errors += 1
            except ValueError as e:
                logger.warning(f"Row {idx}: {e} - SMILES: {row.get(smiles_column, 'N/A')}")
                errors += 1
            except Exception as e:
                logger.error(f"Row {idx}: Unexpected error: {e}")
                errors += 1

    # Write output
    logger.info(f"Writing {len(records)} records to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Done! Converted {len(records)} records")
    logger.info(f"Errors: {errors}")

    if len(records) > 0:
        logger.info(f"Sample output: {records[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess polymer datasets with unlabeled * to AP-SMILES"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="SMILES",
        help="Name of column containing polymer SMILES (default: SMILES)",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="SA_Score",
        help="Name of column containing synthesis score (default: SA_Score)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process (default: all)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    preprocess_csv(
        args.input,
        args.output,
        args.smiles_column,
        args.score_column,
        args.limit,
    )


if __name__ == "__main__":
    main()
