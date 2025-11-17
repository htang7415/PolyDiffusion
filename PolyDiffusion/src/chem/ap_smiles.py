"""Anchor preserving SMILES utilities."""

from __future__ import annotations

import random
import re
from typing import Iterable, List, Sequence

ANCHOR1 = "[*:1]"
ANCHOR2 = "[*:2]"

# DEPRECATED: Shielding no longer used - all tokenizers use [*:1]/[*:2] directly
SHIELD1 = "[Zz]"  # Deprecated - kept for backward compatibility only
SHIELD2 = "[Zr]"  # Deprecated - kept for backward compatibility only

_TOKEN_PATTERN = re.compile(r"\[[^\[\]]+\]|.")

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except ImportError:  # pragma: no cover - unit tests rely on the pure-Python fallback
    Chem = None


def _validate_ap_smiles(smiles: str) -> None:
    if smiles.count(ANCHOR1) != 1 or smiles.count(ANCHOR2) != 1:
        raise ValueError("AP-SMILES must contain exactly one [*:1] and one [*:2].")


def shield_anchors(ap_smiles: str) -> str:
    """DEPRECATED: Replace anchors with shield tokens.

    This function is deprecated. All tokenizers now use [*:1]/[*:2] directly.
    Kept for backward compatibility only.
    """
    _validate_ap_smiles(ap_smiles)
    return ap_smiles.replace(ANCHOR1, SHIELD1).replace(ANCHOR2, SHIELD2)


def unshield_anchors(shielded: str) -> str:
    """DEPRECATED: Invert :func:`shield_anchors`.

    This function is deprecated. All tokenizers now use [*:1]/[*:2] directly.
    Kept for backward compatibility only.
    """
    if SHIELD1 not in shielded or SHIELD2 not in shielded:
        raise ValueError("Shielded string must contain both [Zz] and [Zr].")
    result = shielded.replace(SHIELD1, ANCHOR1).replace(SHIELD2, ANCHOR2)
    _validate_ap_smiles(result)
    return result


def _tokenize(smiles: str) -> List[str]:
    return _TOKEN_PATTERN.findall(smiles)


def _join(tokens: Sequence[str]) -> str:
    return "".join(tokens)


def canonicalize_ap(ap_smiles: str) -> str:
    """Canonicalise AP-SMILES while preserving anchor atoms."""
    _validate_ap_smiles(ap_smiles)
    if Chem is None:  # fallback when RDKit not available
        tokens = _tokenize(ap_smiles)
        canonical_tokens = sorted(
            (tok for tok in tokens if tok not in (ANCHOR1, ANCHOR2)),
        )
        # keep anchors in original positions, fill other tokens in sorted order
        idx = 0
        output: List[str] = []
        for tok in tokens:
            if tok in (ANCHOR1, ANCHOR2):
                output.append(tok)
            else:
                output.append(canonical_tokens[idx])
                idx += 1
        return _join(output)

    mol = Chem.MolFromSmiles(ap_smiles, sanitize=True)
    if mol is None:
        raise ValueError("Failed to parse AP-SMILES with RDKit.")
    return Chem.MolToSmiles(mol, canonical=True)


def convert_polymer_to_ap_smiles(polymer_smiles: str) -> str:
    """
    Convert polymer SMILES with unlabeled * to proper AP-SMILES for Stage B.
    Handles both bare * and bracketed [*].
    Replaces first occurrence with [*:1] and second with [*:2].

    NOTE: This is ONLY for Stage B (polymers). Stage A (small molecules) should NOT
    use this function - small molecules don't have attachment points.

    Args:
        polymer_smiles: Polymer SMILES with exactly 2 unlabeled *
                       (e.g., '*C(C*)C' or 'C(C(O)[*])[*]')

    Returns:
        AP-SMILES with labeled attachment points (e.g., '[*:1]C(C[*:2])C')

    Raises:
        ValueError: If the input doesn't have exactly 2 attachment points.

    Examples:
        >>> convert_polymer_to_ap_smiles('*C(C*)C')
        '[*:1]C(C[*:2])C'
        >>> convert_polymer_to_ap_smiles('C(C(O)[*])[*]')
        'C(C(O)[*:1])[*:2]'
        >>> convert_polymer_to_ap_smiles('*CCC*')
        '[*:1]CCC[*:2]'
    """
    # Pattern matches either bare * or [*]
    pattern = re.compile(r"\[\*\]|\*")
    matches = list(pattern.finditer(polymer_smiles))

    if len(matches) != 2:
        raise ValueError(
            f"Polymer SMILES must have exactly 2 attachment points for Stage B, "
            f"found {len(matches)}: {polymer_smiles}"
        )

    # Replace from right to left to preserve indices
    result = polymer_smiles
    # Second occurrence -> [*:2]
    match = matches[1]
    result = result[: match.start()] + ANCHOR2 + result[match.end() :]
    # First occurrence -> [*:1]
    match = matches[0]
    result = result[: match.start()] + ANCHOR1 + result[match.end() :]

    _validate_ap_smiles(result)
    return result


def randomize_ap(ap_smiles: str, seed: int | None = None) -> str:
    """Return a randomised SMILES while keeping anchors in place."""
    _validate_ap_smiles(ap_smiles)
    rng = random.Random(seed)
    tokens = _tokenize(ap_smiles)

    anchor_positions = [i for i, tok in enumerate(tokens) if tok in (ANCHOR1, ANCHOR2)]
    non_anchor_positions = [i for i in range(len(tokens)) if i not in anchor_positions]
    non_anchor_tokens = [tokens[i] for i in non_anchor_positions]
    rng.shuffle(non_anchor_tokens)

    shuffled_tokens = tokens[:]
    for idx, pos in enumerate(non_anchor_positions):
        shuffled_tokens[pos] = non_anchor_tokens[idx]
    randomized = _join(shuffled_tokens)
    _validate_ap_smiles(randomized)
    return randomized
