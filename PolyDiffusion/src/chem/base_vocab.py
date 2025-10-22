"""Base vocabulary interface for all tokenization methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


class BaseVocabulary(ABC):
    """
    Abstract base class for all vocabulary implementations.

    Provides unified interface for different tokenization methods:
    - Character-level tokenization
    - Atom-level regex tokenization
    - SAFE fragment-based tokenization

    All vocabularies must implement tokenize/detokenize methods and
    provide access to special token IDs.
    """

    def __init__(self, tokens: Sequence[str]):
        """
        Initialize vocabulary with token list.

        Args:
            tokens: Sequence of tokens (special tokens + vocabulary tokens)
        """
        self.id_to_token = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}

    @property
    def method_name(self) -> str:
        """Return tokenization method name for metadata."""
        return self.__class__.__name__

    # ========== Special Token IDs ==========

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.token_to_id.get("<PAD>", 0)

    @property
    def bos_id(self) -> int:
        """Begin-of-sequence token ID."""
        return self.token_to_id.get("<BOS>", 1)

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self.token_to_id.get("<EOS>", 2)

    @property
    def mask_id(self) -> int:
        """Mask token ID (for diffusion)."""
        return self.token_to_id.get("<MASK>", 3)

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self.token_to_id.get("<UNK>", 4)

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return len(self.id_to_token)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.id_to_token)

    # ========== Anchor Support (for Stage B/C) ==========

    def get_anchor_ids(self) -> Optional[Tuple[int, int]]:
        """
        Return anchor token IDs for Stage B/C vocabularies.

        Returns:
            Tuple of (anchor1_id, anchor2_id) if this vocab has anchors,
            None otherwise.
        """
        return None  # Default: no anchor support

    def has_anchors(self) -> bool:
        """Check if this vocabulary supports anchor tokens."""
        return self.get_anchor_ids() is not None

    # ========== Abstract Methods ==========

    @abstractmethod
    def tokenize(self, smiles: str) -> List[int]:
        """
        Tokenize SMILES string into token IDs.

        Args:
            smiles: SMILES string (plain for Stage A, AP-SMILES for Stage B/C)

        Returns:
            List of token IDs (includes BOS and EOS)
        """
        pass

    @abstractmethod
    def detokenize(self, token_ids: Sequence[int]) -> str:
        """
        Convert token IDs back to SMILES string.

        Args:
            token_ids: Sequence of token IDs

        Returns:
            SMILES string (plain for Stage A, AP-SMILES for Stage B/C)
        """
        pass

    @classmethod
    @abstractmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> BaseVocabulary:
        """
        Build vocabulary from corpus of SMILES strings.

        Args:
            corpus: Iterable of SMILES strings
            min_freq: Minimum frequency for token inclusion
            max_size: Maximum vocabulary size (None = unlimited)

        Returns:
            Vocabulary instance
        """
        pass

    # ========== I/O Methods ==========

    def save(self, path: Path) -> None:
        """
        Save vocabulary to file with metadata.

        Format:
            # TOKENIZATION_METHOD: <method_name>
            # VOCAB_SIZE: <size>
            <token1>
            <token2>
            ...
        """
        from datetime import datetime

        lines = [
            f"# TOKENIZATION_METHOD: {self.method_name}",
            f"# VOCAB_SIZE: {len(self)}",
            f"# CREATED: {datetime.now().isoformat()}",
        ]
        lines.extend(self.id_to_token)
        path.write_text('\n'.join(lines), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> BaseVocabulary:
        """
        Load vocabulary from file.

        Skips metadata lines (starting with #).

        Args:
            path: Path to vocabulary file

        Returns:
            Vocabulary instance
        """
        lines = path.read_text(encoding='utf-8').splitlines()
        # Skip metadata lines
        tokens = [line for line in lines if not line.startswith('#')]
        return cls(tokens)
