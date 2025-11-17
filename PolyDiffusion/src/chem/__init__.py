"""Chemistry utilities for AP-SMILES handling."""

from .ap_smiles import (
    canonicalize_ap,
    convert_polymer_to_ap_smiles,
    randomize_ap,
    shield_anchors,  # DEPRECATED - kept for backward compatibility only
    unshield_anchors,  # DEPRECATED - kept for backward compatibility only
)
from .polymer_bpe_vocab import PolymerBPEVocab
from .valence import has_two_anchors, valence_ok

__all__ = [
    # Deprecated functions (kept for backward compatibility):
    "shield_anchors",  # DEPRECATED - all tokenizers now use [*:1]/[*:2] directly
    "unshield_anchors",  # DEPRECATED - all tokenizers now use [*:1]/[*:2] directly
    # Active functions:
    "canonicalize_ap",
    "randomize_ap",
    "convert_polymer_to_ap_smiles",
    "has_two_anchors",
    "valence_ok",
    "PolymerBPEVocab",
]
