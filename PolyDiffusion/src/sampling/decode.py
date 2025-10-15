"""Decoding helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from ..chem.vocab import AnchorSafeVocab


def decode_tokens(vocab: AnchorSafeVocab, sequences: Iterable[Sequence[int]]) -> List[str]:
    """Convert sequences of token ids back to AP-SMILES strings."""
    return [vocab.detokenize_ap(seq) for seq in sequences]
