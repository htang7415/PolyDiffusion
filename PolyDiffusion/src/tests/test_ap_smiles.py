import pytest

from PolyDiffusion.src.chem import (
    canonicalize_ap,
    convert_polymer_to_ap_smiles,
    randomize_ap,
    shield_anchors,
    unshield_anchors,
)


def test_shield_unshield_roundtrip() -> None:
    smiles = "[*:1]CCO[*:2]"
    shielded = shield_anchors(smiles)
    assert "[Zz]" in shielded and "[Zr]" in shielded
    restored = unshield_anchors(shielded)
    assert restored == smiles


def test_canonicalize_preserves_anchors() -> None:
    smiles = "[*:1]C(C)O[*:2]"
    canon = canonicalize_ap(smiles)
    assert "[*:1]" in canon and "[*:2]" in canon


def test_randomize_keeps_anchors() -> None:
    smiles = "[*:1]CC(C)CC[*:2]"
    randomized = randomize_ap(smiles, seed=1234)
    assert randomized.count("[*:1]") == 1
    assert randomized.count("[*:2]") == 1


def test_missing_anchor_raises() -> None:
    with pytest.raises(ValueError):
        shield_anchors("CC[*:2]")


def test_convert_polymer_bare_stars() -> None:
    """Test conversion of polymer SMILES with bare * symbols."""
    # Example 1: *C(C*)C
    result = convert_polymer_to_ap_smiles("*C(C*)C")
    assert result == "[*:1]C(C[*:2])C"
    # Verify it's valid AP-SMILES
    shielded = shield_anchors(result)
    assert "[Zz]" in shielded and "[Zr]" in shielded
    restored = unshield_anchors(shielded)
    assert restored == result


def test_convert_polymer_bracketed_stars() -> None:
    """Test conversion of polymer SMILES with bracketed [*] symbols."""
    # Example 2: C(C(O)[*])[*]
    result = convert_polymer_to_ap_smiles("C(C(O)[*])[*]")
    assert result == "C(C(O)[*:1])[*:2]"
    # Verify roundtrip
    shielded = shield_anchors(result)
    restored = unshield_anchors(shielded)
    assert restored == result


def test_convert_polymer_no_stars_raises() -> None:
    """Test that Stage A molecules (no *) raise error - not for this function."""
    with pytest.raises(ValueError, match="must have exactly 2"):
        convert_polymer_to_ap_smiles("CCO")


def test_convert_polymer_mixed_stars() -> None:
    """Test conversion with mix of bare and bracketed stars."""
    result = convert_polymer_to_ap_smiles("*CC[*]")
    assert result == "[*:1]CC[*:2]"


def test_convert_polymer_real_examples() -> None:
    """Test with real examples from datasets."""
    # From Tm.csv
    result1 = convert_polymer_to_ap_smiles(r"*/C(=C(/CC*)\C)/C")
    assert result1 == r"[*:1]/C(=C(/CC[*:2])\C)/C"
    assert result1.count("[*:1]") == 1
    assert result1.count("[*:2]") == 1

    # From PI1M.csv
    result2 = convert_polymer_to_ap_smiles("*CCC*")
    assert result2 == "[*:1]CCC[*:2]"


def test_convert_polymer_one_star_raises() -> None:
    """Test that exactly 1 attachment point raises error."""
    with pytest.raises(ValueError, match="exactly 2"):
        convert_polymer_to_ap_smiles("*CCC")


def test_convert_polymer_too_many_stars_raises() -> None:
    """Test that more than 2 attachment points raises error."""
    with pytest.raises(ValueError, match="exactly 2"):
        convert_polymer_to_ap_smiles("*C(*)C*")
