"""Decoding helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from ..chem.ap_smiles import SHIELD1, SHIELD2, unshield_anchors
from ..chem.vocab import AnchorSafeVocab


def _build_shielded_tokens(vocab: AnchorSafeVocab, sequence: Sequence[int]) -> List[str]:
    """Convert token ids to shielded tokens without special markers."""
    specials = {vocab.pad_id, vocab.bos_id, vocab.eos_id}
    tokens: List[str] = []
    for token_id in sequence:
        if token_id in specials:
            continue
        token = vocab.id_to_token[token_id]
        if token in {"<MASK>", "<UNK>"}:
            continue
        tokens.append(token)
    return tokens


def _build_shielded_string(vocab: AnchorSafeVocab, sequence: Sequence[int]) -> str:
    return "".join(_build_shielded_tokens(vocab, sequence))


def _fallback_decode(vocab: AnchorSafeVocab, sequence: Sequence[int]) -> str:
    """Gracefully recover a string when detokenisation fails.

    The fallback removes any existing anchor markers, wraps the payload between
    a single pair of anchors, and returns a best-effort AP-SMILES string.
    """
    raw_tokens = _build_shielded_tokens(vocab, sequence)
    if not raw_tokens:
        return ""

    sanitized_tokens: List[str] = []
    shield1_seen = False
    shield2_seen = False
    for token in raw_tokens:
        if token == SHIELD1:
            if not shield1_seen:
                sanitized_tokens.append(token)
                shield1_seen = True
            else:
                sanitized_tokens.append("C")
        elif token == SHIELD2:
            if not shield2_seen:
                sanitized_tokens.append(token)
                shield2_seen = True
            else:
                sanitized_tokens.append("C")
        else:
            sanitized_tokens.append(token)

    if not shield1_seen and shield2_seen:
        sanitized_tokens.insert(0, SHIELD1)
        shield1_seen = True
    elif shield1_seen and not shield2_seen:
        sanitized_tokens.append(SHIELD2)
        shield2_seen = True
    elif not shield1_seen and not shield2_seen:
        sanitized_tokens.insert(0, SHIELD1)
        sanitized_tokens.append(SHIELD2)
        shield1_seen = shield2_seen = True

    shielded = "".join(sanitized_tokens)
    try:
        return unshield_anchors(shielded)
    except ValueError:
        core = "".join(tok for tok in sanitized_tokens if tok not in (SHIELD1, SHIELD2))
        if not core:
            return ""
        fallback_shielded = f"{SHIELD1}{core}{SHIELD2}"
        try:
            return unshield_anchors(fallback_shielded)
        except ValueError:
            return ""


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
        List of decoded AP-SMILES strings. Invalid sequences fall back to a best-effort reconstruction.
    """
    decoded: List[str] = []
    for seq in sequences:
        try:
            decoded.append(vocab.detokenize_ap(seq))
        except (ValueError, IndexError):
            if strict:
                raise
            decoded.append(_fallback_decode(vocab, seq))
    return decoded
