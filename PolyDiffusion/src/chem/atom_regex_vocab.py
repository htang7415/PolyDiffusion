"""Atom-level regex tokenization for SMILES."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Sequence, Tuple

from .ap_smiles import ANCHOR1, ANCHOR2
from .base_vocab import BaseVocabulary

# Special tokens for all vocabularies
SPECIAL_TOKENS_PLAIN = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
SPECIAL_TOKENS_ANCHOR = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", ANCHOR1, ANCHOR2]

# SMILES atom-level regex pattern
# Matches chemically meaningful units: atoms, bonds, rings, brackets
SMILES_ATOM_REGEX = re.compile(
    r"""
    \[[^\]]+\]          # Bracketed atoms: [C@H], [Si], [Na], [NH+], [O-], [*:1], [*:2]
    | Br | Cl           # Two-character atoms (must come before B, C)
    | [BCNOSPFIbcnosp]  # Single-character atoms
    | @@?               # Chirality markers: @ or @@
    | [=#\-+\\\/()]     # Bonds and structure: =, #, -, +, \, /, (, )
    | %\d{2}            # Ring closures >9: %10, %11, %12, ...
    | \d                # Ring closures ≤9: 1, 2, 3, ..., 9
    | \.                # Fragment separator
    """,
    re.VERBOSE,
)


def _tokenize_regex(smiles: str, pattern: Pattern) -> List[str]:
    """Split SMILES string using regex pattern."""
    return pattern.findall(smiles)


class PlainAtomVocab(BaseVocabulary):
    """
    Atom-level regex tokenization for Stage A (plain SMILES).

    Tokenizes SMILES at atomic/bond boundaries:
    - Multi-character atoms preserved: Cl, Br
    - Bracketed atoms kept as units: [Si], [C@H], [NH+]
    - Ring closures, bonds, chirality as separate tokens

    Example:
        "ClCCBr" → ['Cl', 'C', 'C', 'Br']  (not ['C', 'l', 'C', 'C', 'B', 'r'])
        "[Si](C)(C)C" → ['[Si]', '(', 'C', ')', '(', 'C', ')', 'C']
    """

    def __init__(self, tokens: Sequence[str]):
        super().__init__(tokens)
        self.pattern = SMILES_ATOM_REGEX

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> PlainAtomVocab:
        """Build vocabulary from corpus using atom-level regex tokenization."""
        counts: Counter[str] = Counter()

        for smiles in corpus:
            tokens = _tokenize_regex(smiles, SMILES_ATOM_REGEX)
            counts.update(tokens)

        # Filter by min_freq and exclude special tokens
        sorted_tokens = [
            tok
            for tok, freq in counts.items()
            if freq >= min_freq and tok not in SPECIAL_TOKENS_PLAIN
        ]

        # Sort by frequency (descending), then alphabetically
        sorted_tokens.sort(key=lambda x: (-counts[x], x))

        # Build final vocabulary: special tokens first
        vocab_tokens = list(SPECIAL_TOKENS_PLAIN)
        for token in sorted_tokens:
            vocab_tokens.append(token)
            if max_size is not None and len(vocab_tokens) >= max_size:
                break

        return cls(vocab_tokens)

    def tokenize(self, smiles: str) -> List[int]:
        """Tokenize plain SMILES using atom-level regex."""
        tokens = _tokenize_regex(smiles, self.pattern)

        ids = [self.bos_id]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
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


class AnchorAtomVocab(BaseVocabulary):
    """
    Atom-level regex tokenization for Stage B/C (polymers with attachment points).

    Same as PlainAtomVocab but includes anchor tokens [*:1] and [*:2] as atomic units.
    The regex pattern already captures bracketed atoms as whole units, so [*:1] and [*:2]
    are naturally tokenized correctly without any shielding needed.

    Example:
        "[*:1]CCC[*:2]" → tokenize → ['[*:1]', 'C', 'C', 'C', '[*:2]']
                        → IDs → [BOS, anchor1_id, C_id, C_id, C_id, anchor2_id, EOS]
                        → detokenize → "[*:1]CCC[*:2]"
    """

    def __init__(self, tokens: Sequence[str]):
        if ANCHOR1 not in tokens or ANCHOR2 not in tokens:
            raise ValueError(
                f"Anchor tokens {ANCHOR1} and {ANCHOR2} must be present in vocabulary."
            )
        super().__init__(tokens)
        self.pattern = SMILES_ATOM_REGEX

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> AnchorAtomVocab:
        """Build vocabulary from AP-SMILES corpus."""
        counts: Counter[str] = Counter()

        for ap_smiles in corpus:
            # Tokenize with regex - [*:1] and [*:2] are captured as whole units
            tokens = _tokenize_regex(ap_smiles, SMILES_ATOM_REGEX)

            # Count tokens
            counts.update(tokens)

        # Filter and sort
        sorted_tokens = [
            tok
            for tok, freq in counts.items()
            if freq >= min_freq and tok not in SPECIAL_TOKENS_ANCHOR
        ]
        sorted_tokens.sort(key=lambda x: (-counts[x], x))

        # Build vocab: special tokens + sorted tokens
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
        # Tokenize directly - regex captures [*:1] and [*:2] as whole units
        tokens = _tokenize_regex(ap_smiles, self.pattern)

        # Convert to IDs
        ids = [self.bos_id]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
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

        # Reconstruct AP-SMILES directly
        return "".join(tokens)
