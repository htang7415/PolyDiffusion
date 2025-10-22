"""Character-level tokenization for SMILES (refactored to use BaseVocabulary)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .ap_smiles import SHIELD1, SHIELD2, shield_anchors, unshield_anchors
from .base_vocab import BaseVocabulary

# Special tokens
SPECIAL_TOKENS_PLAIN = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
SPECIAL_TOKENS_ANCHOR = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", SHIELD1, SHIELD2]


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
    - Shields [*:1] → [Zz], [*:2] → [Zr] before tokenization
    - Treats [Zz] and [Zr] as multi-character units (via _split_shielded)
    - Unshields during detokenization

    This is the original Stage B/C implementation.

    Example:
        "[*:1]CCC[*:2]" → shield → "[Zz]CCC[Zr]"
                        → split → ['[Zz]', 'C', 'C', 'C', '[Zr]']
                        → detokenize → "[*:1]CCC[*:2]"
    """

    def __init__(self, tokens: Sequence[str]):
        if SHIELD1 not in tokens or SHIELD2 not in tokens:
            raise ValueError(
                f"Anchor tokens {SHIELD1} and {SHIELD2} must be present in vocabulary."
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
            shielded = shield_anchors(ap_smiles)

            # Split shielded string
            for part in _split_shielded(shielded):
                if part in (SHIELD1, SHIELD2):
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
        """Return (SHIELD1_id, SHIELD2_id)."""
        return (self.token_to_id[SHIELD1], self.token_to_id[SHIELD2])

    def tokenize(self, ap_smiles: str) -> List[int]:
        """Tokenize AP-SMILES with anchor preservation."""
        shielded = shield_anchors(ap_smiles)

        ids = [self.bos_id]
        for part in _split_shielded(shielded):
            if part in (SHIELD1, SHIELD2):
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

        shielded = "".join(tokens)
        return unshield_anchors(shielded)


def _split_shielded(smiles: str) -> List[str]:
    """
    Split shielded AP-SMILES into parts, keeping anchors as units.

    Helper function from original vocab.py implementation.
    """
    anchors = (SHIELD1, SHIELD2)
    pieces: List[str] = []
    buffer = ""
    i = 0
    while i < len(smiles):
        if smiles.startswith(SHIELD1, i):
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(SHIELD1)
            i += len(SHIELD1)
        elif smiles.startswith(SHIELD2, i):
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.append(SHIELD2)
            i += len(SHIELD2)
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
