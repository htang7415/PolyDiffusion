"""Minimal grammar and valence checks."""

from __future__ import annotations

from typing import Optional

from .ap_smiles import ANCHOR1, ANCHOR2

try:  # pragma: no cover
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


def has_two_anchors(ap_smiles: str) -> bool:
    """Return True when AP-SMILES string contains exactly one of each anchor."""
    return ap_smiles.count(ANCHOR1) == 1 and ap_smiles.count(ANCHOR2) == 1


def _cap_anchors(ap_smiles: str, cap_smiles: str = "*") -> str:
    return ap_smiles.replace(ANCHOR1, cap_smiles).replace(ANCHOR2, cap_smiles)


def _rdkit_valence_ok(ap_smiles: str) -> bool:
    capped = _cap_anchors(ap_smiles)
    mol = Chem.MolFromSmiles(capped, sanitize=False)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        return True
    except Exception:  # pragma: no cover - RDKit-specific exceptions
        return False


def _fallback_valence_ok(ap_smiles: str) -> bool:
    # naive check: parentheses balance and no invalid double anchors
    stack = 0
    for char in ap_smiles:
        if char == "(":
            stack += 1
        elif char == ")":
            stack -= 1
            if stack < 0:
                return False
    return stack == 0 and has_two_anchors(ap_smiles)


def valence_ok(ap_smiles: str) -> bool:
    """Validate valence using RDKit when available or a lightweight fallback."""
    if not has_two_anchors(ap_smiles):
        return False
    if Chem is None:
        return _fallback_valence_ok(ap_smiles)
    return _rdkit_valence_ok(ap_smiles)
