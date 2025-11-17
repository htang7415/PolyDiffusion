"""Anchor-safe vocabulary builder.

DEPRECATED: This module is maintained for backward compatibility only.
New code should use:
    from PolyDiffusion.src.chem.character_vocab import AnchorCharacterVocab
"""

from __future__ import annotations

# Backward compatibility: re-export from character_vocab
from .ap_smiles import SHIELD1, SHIELD2
from .character_vocab import AnchorCharacterVocab as AnchorSafeVocab

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", SHIELD1, SHIELD2]

__all__ = ["AnchorSafeVocab", "SPECIAL_TOKENS", "SHIELD1", "SHIELD2"]
