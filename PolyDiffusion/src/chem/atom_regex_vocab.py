"""Atom-level regex tokenization for SMILES."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Pattern, Sequence, Tuple

from .ap_smiles import SHIELD1, SHIELD2, shield_anchors, unshield_anchors
from .base_vocab import BaseVocabulary

# Special tokens for all vocabularies
SPECIAL_TOKENS_PLAIN = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
SPECIAL_TOKENS_ANCHOR = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", SHIELD1, SHIELD2]

# SMILES atom-level regex pattern
# Matches chemically meaningful units: atoms, bonds, rings, brackets
SMILES_ATOM_REGEX = re.compile(
    r"""
    \[[^\]]+\]          # Bracketed atoms: [C@H], [Si], [Na], [NH+], [O-], [*:1], [Zz]
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

    Same as PlainAtomVocab but:
    - Shields [*:1] → [Zz], [*:2] → [Zr] before tokenization
    - Treats [Zz] and [Zr] as single atomic tokens
    - Unshields during detokenization

    Example:
        "[*:1]CCC[*:2]" → shield → "[Zz]CCC[Zr]"
                        → tokenize → ['[Zz]', 'C', 'C', 'C', '[Zr]']
                        → detokenize → "[*:1]CCC[*:2]"
    """

    def __init__(self, tokens: Sequence[str]):
        if SHIELD1 not in tokens or SHIELD2 not in tokens:
            raise ValueError(
                f"Anchor tokens {SHIELD1} and {SHIELD2} must be present in vocabulary."
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
            # Shield anchors first
            shielded = shield_anchors(ap_smiles)

            # Tokenize with regex
            tokens = _tokenize_regex(shielded, SMILES_ATOM_REGEX)

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
        """Return (SHIELD1_id, SHIELD2_id)."""
        return (self.token_to_id[SHIELD1], self.token_to_id[SHIELD2])

    def tokenize(self, ap_smiles: str) -> List[int]:
        """Tokenize AP-SMILES with anchor preservation."""
        # Shield anchors: [*:1] → [Zz], [*:2] → [Zr]
        shielded = shield_anchors(ap_smiles)

        # Tokenize (anchors are preserved as single tokens by regex)
        tokens = _tokenize_regex(shielded, self.pattern)

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

        # Reconstruct shielded AP-SMILES
        shielded = "".join(tokens)

        # Unshield: [Zz] → [*:1], [Zr] → [*:2]
        return unshield_anchors(shielded)
