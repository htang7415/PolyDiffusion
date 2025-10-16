#!/usr/bin/env python
"""
Calculate Synthetic Accessibility (SA) scores for SMILES datasets.

This script adds SA scores to CSV or JSONL files containing SMILES strings.
SA scores range from 1 (easy to synthesize) to 10 (very difficult to synthesize).

Usage:
    # For CSV files
    python -m PolyDiffusion.scripts.calculate_sa_score \
        --input Data/small_molecules_part.csv \
        --output Data/small_molecules_with_sa.csv \
        --smiles-column SMILES

    # For JSONL files
    python -m PolyDiffusion.scripts.calculate_sa_score \
        --input Data/polymers.jsonl \
        --output Data/polymers_with_sa.jsonl \
        --smiles-column ap_smiles

Requirements:
    - RDKit: conda install -c conda-forge rdkit
    - pandas (for CSV): pip install pandas
"""

import argparse
import logging
import sys
from pathlib import Path

from PolyDiffusion.src.utils.sa_score import (
    add_sa_scores_to_csv,
    add_sa_scores_to_jsonl,
    RDKIT_AVAILABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate SA scores for SMILES datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CSV file with plain SMILES
  python -m PolyDiffusion.scripts.calculate_sa_score \\
      --input Data/small_molecules_part.csv \\
      --output Data/small_molecules_with_sa.csv \\
      --smiles-column SMILES

  # CSV file with polymer SMILES
  python -m PolyDiffusion.scripts.calculate_sa_score \\
      --input Data/PI1M.csv \\
      --output Data/PI1M_with_sa.csv \\
      --smiles-column SMILES \\
      --output-column "SA Score"

  # JSONL file with AP-SMILES
  python -m PolyDiffusion.scripts.calculate_sa_score \\
      --input Data/polymers.jsonl \\
      --output Data/polymers_with_sa.jsonl \\
      --smiles-column ap_smiles \\
      --output-column synth_score

SA Score Interpretation:
  1.0 - 3.0: Very easy to synthesize (e.g., ethanol, benzene)
  3.0 - 5.0: Moderate difficulty (simple drugs, building blocks)
  5.0 - 7.0: Difficult (complex natural products)
  7.0 - 10.0: Very difficult (exotic structures)
        """,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input file path (CSV or JSONL)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path (same format as input)",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="SMILES",
        help="Name of SMILES column/field (default: SMILES)",
    )
    parser.add_argument(
        "--output-column",
        type=str,
        default=None,
        help="Name of output SA score column/field (default: SA_Score for CSV, synth_score for JSONL)",
    )
    parser.add_argument(
        "--invalid-value",
        type=float,
        default=-1.0,
        help="Value to use for invalid SMILES (default: -1.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists",
    )

    args = parser.parse_args()

    # Check RDKit availability
    if not RDKIT_AVAILABLE:
        logger.error(
            "RDKit is not available. Please install it with:\n"
            "  conda install -c conda-forge rdkit\n"
            "or\n"
            "  pip install rdkit"
        )
        sys.exit(1)

    # Check input file exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Check output file
    if args.output.exists() and not args.force:
        logger.error(
            f"Output file already exists: {args.output}\n"
            "Use --force to overwrite"
        )
        sys.exit(1)

    # Determine file format
    if args.input.suffix in {".csv", ".tsv"}:
        file_format = "csv"
        default_output_col = "SA_Score"
    elif args.input.suffix in {".jsonl", ".json"}:
        file_format = "jsonl"
        default_output_col = "synth_score"
    else:
        logger.error(
            f"Unsupported file format: {args.input.suffix}\n"
            "Supported formats: .csv, .tsv, .jsonl, .json"
        )
        sys.exit(1)

    output_column = args.output_column or default_output_col

    # Process file
    logger.info(f"Processing {file_format.upper()} file: {args.input}")
    logger.info(f"SMILES column/field: {args.smiles_column}")
    logger.info(f"Output column/field: {output_column}")

    try:
        if file_format == "csv":
            add_sa_scores_to_csv(
                input_path=args.input,
                output_path=args.output,
                smiles_column=args.smiles_column,
                output_column=output_column,
                invalid_value=args.invalid_value,
            )
        else:  # jsonl
            add_sa_scores_to_jsonl(
                input_path=args.input,
                output_path=args.output,
                smiles_field=args.smiles_column,
                output_field=output_column,
                invalid_value=args.invalid_value,
            )

        logger.info("\n✓ SA score calculation completed successfully!")
        logger.info(f"Results saved to: {args.output}")

    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
