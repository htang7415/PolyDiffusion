import pytest

from PolyDiffusion.chem import (
    canonicalize_ap,
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
