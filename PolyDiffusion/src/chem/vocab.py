"""Anchor-safe vocabulary builder."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

from .ap_smiles import SHIELD1, SHIELD2, shield_anchors, unshield_anchors

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", SHIELD1, SHIELD2]


def _split_shielded(smiles: str) -> List[str]:
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


class AnchorSafeVocab:
    """Simple Unigram-style vocabulary that keeps anchor tokens isolated."""

    def __init__(self, tokens: Sequence[str]) -> None:
        if SHIELD1 not in tokens or SHIELD2 not in tokens:
            raise ValueError("Anchor tokens must be present in vocabulary.")
        self.id_to_token = list(tokens)
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: int | None = None,
    ) -> "AnchorSafeVocab":
        counts: Counter[str] = Counter()
        for ap_smiles in corpus:
            shielded = shield_anchors(ap_smiles)
            for part in _split_shielded(shielded):
                if part in (SHIELD1, SHIELD2):
                    counts[part] += 1
                else:
                    for char in part:
                        counts[char] += 1

        sorted_tokens = [tok for tok, freq in counts.items() if freq >= min_freq and tok not in SPECIAL_TOKENS]
        sorted_tokens.sort(key=lambda x: (-counts[x], x))

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

    def tokenize_ap(self, ap_smiles: str) -> List[int]:
        """Tokenize an AP-SMILES string into token IDs.

        Args:
            ap_smiles: Anchor-preserving SMILES string with [*:1] and [*:2].

        Returns:
            List of token IDs including BOS and EOS tokens.

        Raises:
            ValueError: If AP-SMILES is invalid (missing or multiple anchors).
        """
        shielded = shield_anchors(ap_smiles)
        ids = [self.bos_id]
        for part in _split_shielded(shielded):
            if part in (SHIELD1, SHIELD2):
                ids.append(self.token_to_id[part])
            else:
                for char in part:
                    ids.append(self.token_to_id.get(char, self.token_to_id["<UNK>"]))
        ids.append(self.eos_id)
        return ids

    def detokenize_ap(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back to AP-SMILES string.

        Args:
            token_ids: Sequence of token IDs.

        Returns:
            AP-SMILES string with anchors restored.

        Raises:
            ValueError: If the detokenized string is missing anchors.
            IndexError: If token ID is out of vocabulary range.
        """
        tokens: List[str] = []
        specials = {self.pad_id, self.bos_id, self.eos_id}
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.id_to_token):
                raise IndexError(f"Token ID {token_id} out of vocabulary range [0, {len(self.id_to_token)})")
            if token_id in specials:
                continue
            token = self.id_to_token[token_id]
            if token == "<MASK>" or token == "<UNK>":
                continue
            tokens.append(token)
        shielded = "".join(tokens)
        return unshield_anchors(shielded)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.id_to_token), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AnchorSafeVocab":
        tokens = path.read_text(encoding="utf-8").splitlines()
        return cls(tokens)
