"""Plain vocabulary for Stage A (small molecules without attachment points)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]


class PlainVocab:
    """
    Simple character-level vocabulary for plain SMILES (Stage A).

    Unlike AnchorSafeVocab, this does NOT use attachment point tokens [Zz]/[Zr].
    It's designed for small molecules that are complete structures, not polymer repeat units.
    """

    def __init__(self, tokens: Sequence[str]) -> None:
        self.id_to_token = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "PlainVocab":
        """
        Build vocabulary from a corpus of plain SMILES strings.

        Args:
            corpus: Iterable of plain SMILES strings (no attachment points)
            min_freq: Minimum frequency for a character to be included
            max_size: Maximum vocabulary size

        Returns:
            PlainVocab instance
        """
        counts: Counter[str] = Counter()
        for smiles in corpus:
            for char in smiles:
                counts[char] += 1

        # Sort by frequency (descending), then alphabetically
        sorted_tokens = [
            tok for tok, freq in counts.items()
            if freq >= min_freq and tok not in SPECIAL_TOKENS
        ]
        sorted_tokens.sort(key=lambda x: (-counts[x], x))

        # Build final vocabulary: special tokens first, then sorted characters
        vocab_tokens = list(SPECIAL_TOKENS)
        for token in sorted_tokens:
            vocab_tokens.append(token)
            if max_size is not None and len(vocab_tokens) >= max_size:
                break
        return cls(vocab_tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<EOS>"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["<MASK>"]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def tokenize(self, smiles: str) -> List[int]:
        """
        Tokenize a plain SMILES string into token IDs.

        Args:
            smiles: Plain SMILES string (e.g., "CCO", "c1ccccc1")

        Returns:
            List of token IDs including BOS and EOS tokens.
        """
        ids = [self.bos_id]
        for char in smiles:
            ids.append(self.token_to_id.get(char, self.token_to_id["<UNK>"]))
        ids.append(self.eos_id)
        return ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """
        Convert token IDs back to plain SMILES string.

        Args:
            token_ids: Sequence of token IDs.

        Returns:
            Plain SMILES string.

        Raises:
            IndexError: If token ID is out of vocabulary range.
        """
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
            if token == "<MASK>" or token == "<UNK>":
                continue
            tokens.append(token)

        return "".join(tokens)

    def save(self, path: Path) -> None:
        """Save vocabulary to file."""
        path.write_text("\n".join(self.id_to_token), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PlainVocab":
        """Load vocabulary from file."""
        tokens = path.read_text(encoding="utf-8").splitlines()
        return cls(tokens)
