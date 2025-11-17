"""Character-level tokenization for SMILES (refactored to use BaseVocabulary)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .ap_smiles import ANCHOR1, ANCHOR2
from .base_vocab import BaseVocabulary

# Special tokens
SPECIAL_TOKENS_PLAIN = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
SPECIAL_TOKENS_ANCHOR = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", ANCHOR1, ANCHOR2]


class PlainCharacterVocab(BaseVocabulary):
    """
    Character-level tokenization for Stage A (plain SMILES).

    Tokenizes SMILES character-by-character.
    This is the original baseline implementation.

    Example:
        "CCO" → ['C', 'C', 'O']
        "ClCCBr" → ['C', 'l', 'C', 'C', 'B', 'r']  # ❌ Splits Cl and Br

    Note: This has validity issues with multi-character atoms like Cl, Br.
    Consider using AtomRegexVocab or SAFEVocab for better validity.
    """

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> PlainCharacterVocab:
        """Build vocabulary from corpus using character-level tokenization."""
        counts: Counter[str] = Counter()
        for smiles in corpus:
            for char in smiles:
                counts[char] += 1

        # Sort by frequency (descending), then alphabetically
        sorted_tokens = [
            tok
            for tok, freq in counts.items()
            if freq >= min_freq and tok not in SPECIAL_TOKENS_PLAIN
        ]
        sorted_tokens.sort(key=lambda x: (-counts[x], x))

        # Build final vocabulary: special tokens first, then sorted characters
        vocab_tokens = list(SPECIAL_TOKENS_PLAIN)
        for token in sorted_tokens:
            vocab_tokens.append(token)
            if max_size is not None and len(vocab_tokens) >= max_size:
                break

        return cls(vocab_tokens)

    def tokenize(self, smiles: str) -> List[int]:
        """Tokenize plain SMILES character-by-character."""
        ids = [self.bos_id]
        for char in smiles:
            ids.append(self.token_to_id.get(char, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back to SMILES."""
        tokens: List[str] = []
        specials = {self.pad_id, self.bos_id, self.eos_id}

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                raise IndexError(
                    f"Token ID {token_id} out of vocabulary range [0, {len(self.id_to_token)})"
                )
            if token_id in specials:
                continue
            token = self.id_to_token[token_id]
            if token in ("<MASK>", "<UNK>"):
                continue
            tokens.append(token)

        return "".join(tokens)


class AnchorCharacterVocab(BaseVocabulary):
    """
    Character-level tokenization for Stage B/C (polymers with attachment points).

    Same as PlainCharacterVocab but:
    - Treats [*:1] and [*:2] as multi-character units (via _split_anchors)
    - Anchors are atomic tokens, not split character-by-character

    Example:
        "[*:1]CCC[*:2]" → split → ['[*:1]', 'C', 'C', 'C', '[*:2]']
                        → tokenize → [BOS, anchor1_id, C_id, C_id, C_id, anchor2_id, EOS]
                        → detokenize → "[*:1]CCC[*:2]"
    """

    def __init__(self, tokens: Sequence[str]):
        if ANCHOR1 not in tokens or ANCHOR2 not in tokens:
            raise ValueError(
                f"Anchor tokens {ANCHOR1} and {ANCHOR2} must be present in vocabulary."
            )
        super().__init__(tokens)

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> AnchorCharacterVocab:
        """Build vocabulary from AP-SMILES corpus."""
        counts: Counter[str] = Counter()

        for ap_smiles in corpus:
            # Split AP-SMILES into parts, keeping anchors as units
            for part in _split_anchors(ap_smiles):
                if part in (ANCHOR1, ANCHOR2):
                    counts[part] += 1
                else:
                    for char in part:
                        counts[char] += 1

        # Filter and sort
        sorted_tokens = [
            tok
            for tok, freq in counts.items()
            if freq >= min_freq and tok not in SPECIAL_TOKENS_ANCHOR
        ]
        sorted_tokens.sort(key=lambda x: (-counts[x], x))

        # Build vocabulary
        vocab_tokens = list(SPECIAL_TOKENS_ANCHOR)
        for token in sorted_tokens:
            vocab_tokens.append(token)
            if max_size is not None and len(vocab_tokens) >= max_size:
                break

        return cls(vocab_tokens)

    def get_anchor_ids(self) -> Tuple[int, int]:
        """Return (ANCHOR1_id, ANCHOR2_id) for polymer generation."""
        return (self.token_to_id[ANCHOR1], self.token_to_id[ANCHOR2])

    def tokenize(self, ap_smiles: str) -> List[int]:
        """Tokenize AP-SMILES with anchor preservation."""
        ids = [self.bos_id]
        for part in _split_anchors(ap_smiles):
            if part in (ANCHOR1, ANCHOR2):
                ids.append(self.token_to_id[part])
            else:
                for char in part:
                    ids.append(self.token_to_id.get(char, self.unk_id))
        ids.append(self.eos_id)

        return ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back to AP-SMILES."""
        tokens: List[str] = []
        specials = {self.pad_id, self.bos_id, self.eos_id}

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                raise IndexError(
                    f"Token ID {token_id} out of vocabulary range [0, {len(self.id_to_token)})"
                )
            if token_id in specials:
                continue
            token = self.id_to_token[token_id]
            if token in ("<MASK>", "<UNK>"):
                continue
            tokens.append(token)

        return "".join(tokens)


def _split_anchors(smiles: str) -> List[str]:
    """
    Split AP-SMILES into parts, keeping anchors [*:1] and [*:2] as units.

    This function treats the 5-character anchors [*:1] and [*:2] as atomic tokens,
    preventing them from being split character-by-character.

    Args:
        smiles: AP-SMILES string with [*:1] and [*:2] attachment points

    Returns:
        List of parts: anchor tokens as whole units, other parts as strings

    Example:
        "[*:1]CCC[*:2]" → ['[*:1]', 'CCC', '[*:2]']
    """
    pieces: List[str] = []
    buffer = ""
    i = 0
    while i < len(smiles):
        if smiles.startswith(ANCHOR1, i):
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(ANCHOR1)
            i += len(ANCHOR1)  # Skip 5 characters: [*:1]
        elif smiles.startswith(ANCHOR2, i):
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(ANCHOR2)
            i += len(ANCHOR2)  # Skip 5 characters: [*:2]
        else:
            buffer += smiles[i]
            i += 1
    if buffer:
        pieces.append(buffer)
    return pieces


# ========== Backward Compatibility Aliases ==========

# For backward compatibility with existing code
PlainVocab = PlainCharacterVocab
AnchorSafeVocab = AnchorCharacterVocab
