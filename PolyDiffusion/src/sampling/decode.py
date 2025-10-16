"""Decoding helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from ..chem.vocab import AnchorSafeVocab


def decode_tokens(
    vocab: AnchorSafeVocab,
    sequences: Iterable[Sequence[int]],
    strict: bool = False,
) -> List[str]:
    """Convert sequences of token ids back to AP-SMILES strings.

    Args:
        vocab: Anchor-safe vocabulary used for tokenisation.
        sequences: Iterable of token id sequences.
        strict: When ``True`` re-raise decoding errors instead of returning empty strings.

    Returns:
        List of decoded AP-SMILES strings. Invalid sequences collapse to ``""`` when ``strict`` is False.
    """
    decoded: List[str] = []
    for seq in sequences:
        try:
            decoded.append(vocab.detokenize_ap(seq))
        except (ValueError, IndexError):
            if strict:
                raise
            decoded.append("")
    return decoded
