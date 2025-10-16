"""Tests for SA score calculation utilities."""

import pytest

from PolyDiffusion.src.utils.sa_score import (
    calculate_sa_score,
    calculate_sa_score_batch,
    interpret_sa_score,
    RDKIT_AVAILABLE,
)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_calculate_sa_score_simple() -> None:
    """Test SA score for simple molecules."""
    # Ethanol - very easy to synthesize
    score = calculate_sa_score("CCO")
    assert 1.0 <= score <= 3.0, f"Expected easy synthesis score, got {score}"

    # Benzene - relatively easy
    score = calculate_sa_score("c1ccccc1")
    assert 1.0 <= score <= 4.0, f"Expected easy synthesis score, got {score}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_calculate_sa_score_complex() -> None:
    """Test SA score for more complex molecules."""
    # Complex branched alkane
    score = calculate_sa_score("CC(C)CC(C)(C)CC(C)(C)C")
    assert score > 3.0, f"Expected higher complexity score, got {score}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_calculate_sa_score_invalid() -> None:
    """Test SA score with invalid SMILES."""
    # Invalid SMILES
    score = calculate_sa_score("INVALID_SMILES")
    assert score == -1.0, f"Expected -1.0 for invalid SMILES, got {score}"

    # Empty string
    score = calculate_sa_score("")
    assert score == -1.0, f"Expected -1.0 for empty SMILES, got {score}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_calculate_sa_score_batch() -> None:
    """Test batch SA score calculation."""
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1", "INVALID"]
    scores = calculate_sa_score_batch(smiles_list, show_progress=False)

    assert len(scores) == 4, "Should return score for each SMILES"

    # First three should be valid (>= 1.0)
    assert all(s >= 1.0 for s in scores[:3]), "Valid SMILES should have positive scores"

    # Last one should be invalid (-1.0)
    assert scores[3] == -1.0, "Invalid SMILES should return -1.0"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_calculate_sa_score_polymer() -> None:
    """Test SA score for polymer AP-SMILES."""
    # AP-SMILES should work after attachment points are parsed
    # RDKit treats [*:1] as a wildcard atom
    score = calculate_sa_score("[*:1]CCO[*:2]")
    # Should calculate something, even if attachment points are unusual
    assert score != -1.0, "Valid AP-SMILES structure should calculate"


def test_interpret_sa_score() -> None:
    """Test SA score interpretation."""
    assert "very easy" in interpret_sa_score(2.0).lower()
    assert "moderate" in interpret_sa_score(4.0).lower()
    assert "difficult" in interpret_sa_score(6.0).lower()
    assert "very difficult" in interpret_sa_score(8.5).lower()
    assert "invalid" in interpret_sa_score(-1.0).lower()


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_sa_score_range() -> None:
    """Test that SA scores are in expected range."""
    test_molecules = [
        "CCO",           # Ethanol
        "CC(=O)O",       # Acetic acid
        "c1ccccc1",      # Benzene
        "CCN(CC)CC",     # Triethylamine
    ]

    for smiles in test_molecules:
        score = calculate_sa_score(smiles)
        assert 1.0 <= score <= 10.0, f"Score {score} out of range for {smiles}"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_sa_score_consistency() -> None:
    """Test that SA scores are consistent for same molecule."""
    smiles = "CCO"

    score1 = calculate_sa_score(smiles)
    score2 = calculate_sa_score(smiles)

    assert score1 == score2, "SA score should be deterministic"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="RDKit not available")
def test_batch_vs_individual() -> None:
    """Test that batch calculation matches individual calculations."""
    smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]

    # Calculate individually
    individual_scores = [calculate_sa_score(s) for s in smiles_list]

    # Calculate in batch
    batch_scores = calculate_sa_score_batch(smiles_list, show_progress=False)

    # Should match
    for ind, batch in zip(individual_scores, batch_scores):
        assert abs(ind - batch) < 1e-6, "Batch and individual scores should match"


def test_no_rdkit_fallback() -> None:
    """Test behavior when RDKit is not available."""
    if RDKIT_AVAILABLE:
        pytest.skip("RDKit is available, skipping fallback test")

    # Should return -1.0 when RDKit is not available
    score = calculate_sa_score("CCO")
    assert score == -1.0, "Should return -1.0 when RDKit unavailable"

    scores = calculate_sa_score_batch(["CCO", "CC(=O)O"], show_progress=False)
    assert all(s == -1.0 for s in scores), "Batch should return -1.0 for all when RDKit unavailable"
