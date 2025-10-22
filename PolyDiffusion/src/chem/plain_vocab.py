"""Plain vocabulary for Stage A (small molecules without attachment points).

DEPRECATED: This module is maintained for backward compatibility only.
New code should use:
    from PolyDiffusion.src.chem.character_vocab import PlainCharacterVocab
"""

from __future__ import annotations

# Backward compatibility: re-export PlainCharacterVocab as PlainVocab
from .character_vocab import PlainCharacterVocab as PlainVocab

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]

__all__ = ["PlainVocab", "SPECIAL_TOKENS"]
