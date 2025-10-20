"""
Synthetic Accessibility (SA) Score calculation utilities.

Based on the RDKit implementation by Peter Ertl and Ansgar Schuffenhauer:
"Estimation of synthetic accessibility score of drug-like molecules
based on molecular complexity and fragment contributions"
Journal of Cheminformatics 1:8 (2009)

SA Score range: 1 (easy to synthesize) to 10 (very difficult to synthesize)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

try:
    from rdkit import Chem
    from rdkit.Chem import RDConfig

    # Add SA_Score contrib path
    sa_score_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
    if sa_score_path not in sys.path:
        sys.path.append(sa_score_path)

    import sascorer
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    sascorer = None

logger = logging.getLogger(__name__)


def calculate_sa_score(smiles: str) -> float:
    """
    Calculate SA score for a single SMILES string.

    Args:
        smiles: SMILES string (plain or AP-SMILES)

    Returns:
        SA score (1-10), or -1.0 if calculation fails

    Examples:
        >>> calculate_sa_score("CCO")  # Ethanol
        1.0
        >>> calculate_sa_score("CC(C)CC(C)(C)CC(C)(C)C")  # Complex molecule
        7.2
    """
    if not RDKIT_AVAILABLE:
        logger.debug("RDKit not available. Install with: conda install -c conda-forge rdkit")
        return -1.0

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1.0

        score = sascorer.calculateScore(mol)
        return float(score)

    except Exception as e:
        logger.debug("SA score calculation failed for '%s': %s", smiles, e)
        return -1.0


def calculate_sa_score_batch(
    smiles_list: List[str],
    invalid_value: float = -1.0,
    show_progress: bool = True,
) -> List[float]:
    """
    Calculate SA scores for a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        invalid_value: Value to return for invalid SMILES
        show_progress: Show progress bar (requires tqdm)

    Returns:
        List of SA scores

    Examples:
        >>> smiles = ["CCO", "CC(=O)O", "c1ccccc1"]
        >>> scores = calculate_sa_score_batch(smiles)
        >>> print(scores)
        [1.0, 2.3, 2.8]
    """
    if not RDKIT_AVAILABLE:
        logger.debug("RDKit not available. Install with: conda install -c conda-forge rdkit")
        return [invalid_value] * len(smiles_list)

    scores = []
    iterator = smiles_list

    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(smiles_list, desc="Calculating SA scores")
        except ImportError:
            pass

    for smiles in iterator:
        score = calculate_sa_score(smiles)
        if score == -1.0:
            score = invalid_value
        scores.append(score)

    return scores


def add_sa_scores_to_csv(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    smiles_column: str = "SMILES",
    output_column: str = "SA_Score",
    invalid_value: float = -1.0,
) -> None:
    """
    Add SA scores to a CSV file.

    Args:
        input_path: Input CSV file path
        output_path: Output CSV file path
        smiles_column: Name of SMILES column
        output_column: Name of output SA score column
        invalid_value: Value for invalid SMILES

    Examples:
        >>> add_sa_scores_to_csv(
        ...     "data/molecules.csv",
        ...     "data/molecules_with_sa.csv",
        ...     smiles_column="SMILES"
        ... )
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not available. Install with: pip install pandas")
        return

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)

    if smiles_column not in df.columns:
        logger.error(f"Column '{smiles_column}' not found in CSV")
        logger.info(f"Available columns: {list(df.columns)}")
        return

    # Calculate SA scores
    logger.info(f"Calculating SA scores for {len(df)} molecules...")
    df[output_column] = calculate_sa_score_batch(
        df[smiles_column].tolist(),
        invalid_value=invalid_value,
        show_progress=True,
    )

    # Report statistics
    valid_scores = df[df[output_column] >= 0][output_column]
    if len(valid_scores) > 0:
        logger.info(f"\nSA Score Statistics:")
        logger.info(f"  Valid: {len(valid_scores)} / {len(df)} ({len(valid_scores)/len(df)*100:.1f}%)")
        logger.info(f"  Mean: {valid_scores.mean():.2f}")
        logger.info(f"  Std: {valid_scores.std():.2f}")
        logger.info(f"  Min: {valid_scores.min():.2f}")
        logger.info(f"  Max: {valid_scores.max():.2f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


def add_sa_scores_to_jsonl(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    smiles_field: str = "ap_smiles",
    output_field: str = "synth_score",
    invalid_value: float = -1.0,
) -> None:
    """
    Add SA scores to a JSONL file.

    Args:
        input_path: Input JSONL file path
        output_path: Output JSONL file path
        smiles_field: Name of SMILES field in JSON
        output_field: Name of output SA score field
        invalid_value: Value for invalid SMILES
    """
    import json

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    # Read all records
    records = []
    with input_path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    logger.info(f"Read {len(records)} records from {input_path}")

    # Extract SMILES
    smiles_list = []
    for record in records:
        if smiles_field in record:
            smiles_list.append(str(record[smiles_field]))
        else:
            logger.warning(f"Record missing field '{smiles_field}': {record}")
            smiles_list.append("")

    # Calculate SA scores
    logger.info(f"Calculating SA scores...")
    scores = calculate_sa_score_batch(smiles_list, invalid_value=invalid_value, show_progress=True)

    # Add scores to records
    for record, score in zip(records, scores):
        record[output_field] = score

    # Report statistics
    valid_scores = [s for s in scores if s >= 0]
    if len(valid_scores) > 0:
        logger.info(f"\nSA Score Statistics:")
        logger.info(f"  Valid: {len(valid_scores)} / {len(records)} ({len(valid_scores)/len(records)*100:.1f}%)")
        logger.info(f"  Mean: {sum(valid_scores)/len(valid_scores):.2f}")
        logger.info(f"  Min: {min(valid_scores):.2f}")
        logger.info(f"  Max: {max(valid_scores):.2f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved results to {output_path}")


def interpret_sa_score(score: float) -> str:
    """
    Provide human-readable interpretation of SA score.

    Args:
        score: SA score value (1-10)

    Returns:
        Interpretation string

    Examples:
        >>> interpret_sa_score(2.5)
        'Very easy to synthesize'
        >>> interpret_sa_score(6.8)
        'Difficult to synthesize'
    """
    if score < 0:
        return "Invalid score"
    elif score <= 3.0:
        return "Very easy to synthesize"
    elif score <= 5.0:
        return "Moderate difficulty"
    elif score <= 7.0:
        return "Difficult to synthesize"
    else:
        return "Very difficult to synthesize"


# Module-level check
if not RDKIT_AVAILABLE:
    logger.warning(
        "RDKit is not available. SA score calculation will not work.\n"
        "Install with: conda install -c conda-forge rdkit"
    )
