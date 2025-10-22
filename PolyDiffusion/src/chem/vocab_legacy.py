"""Anchor-safe vocabulary builder.

DEPRECATED: This module is maintained for backward compatibility only.
New code should use:
    from PolyDiffusion.src.chem.character_vocab import AnchorCharacterVocab
"""

from __future__ import annotations

# Backward compatibility: re-export AnchorCharacterVocab as AnchorSafeVocab
from .ap_smiles import SHIELD1, SHIELD2
from .character_vocab import AnchorCharacterVocab as AnchorSafeVocab
from .character_vocab import _split_shielded

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", SHIELD1, SHIELD2]

__all__ = ["AnchorSafeVocab", "SPECIAL_TOKENS", "_split_shielded", "SHIELD1", "SHIELD2"]
