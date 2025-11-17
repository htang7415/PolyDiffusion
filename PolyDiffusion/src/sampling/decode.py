"""Decoding helpers."""

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

from ..chem.ap_smiles import ANCHOR1, ANCHOR2
from ..chem.base_vocab import BaseVocabulary
from ..chem.vocab import AnchorSafeVocab  # Backward compat

log = logging.getLogger(__name__)


def _build_shielded_tokens(vocab: BaseVocabulary, sequence: Sequence[int]) -> List[str]:
    """Convert token ids to tokens without special markers.

    Note: Function name kept for backward compatibility (no longer uses shielding).
    """
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


def _build_shielded_string(vocab: BaseVocabulary, sequence: Sequence[int]) -> str:
    """Convert token ids to string without special markers.

    Note: Function name kept for backward compatibility (no longer uses shielding).
    """
    return "".join(_build_shielded_tokens(vocab, sequence))


def _fallback_decode(vocab: BaseVocabulary, sequence: Sequence[int]) -> str:
    """Gracefully recover a string when detokenisation fails.

    The fallback removes any duplicate anchor markers, wraps the payload between
    a single pair of anchors, and returns a best-effort AP-SMILES string.
    """
    raw_tokens = _build_shielded_tokens(vocab, sequence)
    if not raw_tokens:
        return ""

    sanitized_tokens: List[str] = []
    anchor1_seen = False
    anchor2_seen = False
    for token in raw_tokens:
        if token == ANCHOR1:
            if not anchor1_seen:
                sanitized_tokens.append(token)
                anchor1_seen = True
            else:
                # Replace duplicate anchor with carbon
                sanitized_tokens.append("C")
        elif token == ANCHOR2:
            if not anchor2_seen:
                sanitized_tokens.append(token)
                anchor2_seen = True
            else:
                # Replace duplicate anchor with carbon
                sanitized_tokens.append("C")
        else:
            sanitized_tokens.append(token)

    # Ensure we have both anchors
    if not anchor1_seen and anchor2_seen:
        sanitized_tokens.insert(0, ANCHOR1)
        anchor1_seen = True
    elif anchor1_seen and not anchor2_seen:
        sanitized_tokens.append(ANCHOR2)
        anchor2_seen = True
    elif not anchor1_seen and not anchor2_seen:
        sanitized_tokens.insert(0, ANCHOR1)
        sanitized_tokens.append(ANCHOR2)
        anchor1_seen = anchor2_seen = True

    # Reconstruct AP-SMILES directly (no unshielding needed)
    ap_smiles = "".join(sanitized_tokens)

    # Validate structure
    if ANCHOR1 in ap_smiles and ANCHOR2 in ap_smiles:
        return ap_smiles

    # Last resort: wrap core content with anchors
    core = "".join(tok for tok in sanitized_tokens if tok not in (ANCHOR1, ANCHOR2))
    if not core:
        log.warning(f"Empty core tokens during fallback decode for sequence: {sequence[:20]}...")
        return ""

    fallback_ap_smiles = f"{ANCHOR1}{core}{ANCHOR2}"
    return fallback_ap_smiles


def decode_tokens(
    vocab: BaseVocabulary,
    sequences: Iterable[Sequence[int]],
    strict: bool = False,
) -> List[str]:
    """Convert sequences of token ids back to SMILES strings.

    Works with any tokenization method (character, atom-regex, or SAFE).
    For SAFE vocabularies, automatically converts SAFE → canonical SMILES.

    Args:
        vocab: Vocabulary used for tokenisation (any BaseVocabulary subclass).
        sequences: Iterable of token id sequences.
        strict: When ``True`` re-raise decoding errors instead of returning empty strings.

    Returns:
        List of decoded SMILES strings (plain for Stage A, AP-SMILES for Stage B/C).
        Invalid sequences fall back to a best-effort reconstruction.
    """
    decoded: List[str] = []
    for seq in sequences:
        try:
            # Use unified detokenize() interface
            # This works for all vocab types:
            # - Character/Atom-regex: returns SMILES directly
            # - SAFE: converts SAFE → canonical SMILES internally
            decoded.append(vocab.detokenize(seq))
        except (ValueError, IndexError):
            if strict:
                raise
            decoded.append(_fallback_decode(vocab, seq))
    return decoded
