"""SAFE (Sequential Attachment-based Fragment Embedding) tokenization."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .ap_smiles import SHIELD1, SHIELD2, shield_anchors, unshield_anchors
from .atom_regex_vocab import SMILES_ATOM_REGEX, _tokenize_regex
from .base_vocab import BaseVocabulary

# Special tokens
SPECIAL_TOKENS_PLAIN = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
SPECIAL_TOKENS_ANCHOR = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>", SHIELD1, SHIELD2]

# Check for SAFE library availability
try:
    from rdkit import Chem
    from rdkit.Chem import BRICS

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class PlainSAFEVocab(BaseVocabulary):
    """
    SAFE tokenization for Stage A (small molecules).

    Uses BRICS fragmentation to decompose molecules into chemically
    meaningful fragments, then tokenizes at fragment level.

    Process:
        1. SMILES → BRICS fragments
        2. Fragments → SAFE encoding (with ring notation for linking)
        3. SAFE string → Tokens (fragment-aware)
        4. Tokens → IDs

    Decoding:
        1. IDs → Tokens
        2. Tokens → SAFE string
        3. SAFE string → SMILES (via RDKit parsing)

    Example:
        "OCCc1ccccc1" → BRICS → ["[*]CCO", "[*]c1ccccc1"]
                      → SAFE → "c1ccccc1%10.%10CCO"
                      → Tokens → ['c1ccccc1%10', '.', '%10CCO']
    """

    def __init__(self, tokens: Sequence[str]):
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "SAFE tokenization requires RDKit. Install with: pip install rdkit"
            )
        super().__init__(tokens)

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> PlainSAFEVocab:
        """
        Build SAFE vocabulary from corpus.

        Strategy:
        1. Decompose all molecules with BRICS
        2. Count fragment frequencies
        3. Keep top-K fragments as vocabulary tokens
        4. Add atomic fallback tokens for rare fragments
        """
        fragment_counts: Counter[str] = Counter()

        for smiles in corpus:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                # BRICS decomposition
                fragments = BRICS.BRICSDecompose(mol)

                # Count fragments
                fragment_counts.update(fragments)

            except Exception:
                # Skip invalid SMILES
                continue

        # Filter by frequency
        sorted_fragments = [
            frag
            for frag, freq in fragment_counts.items()
            if freq >= min_freq and frag not in SPECIAL_TOKENS_PLAIN
        ]

        # Sort by frequency
        sorted_fragments.sort(key=lambda x: (-fragment_counts[x], x))

        # Build vocabulary: special tokens + fragments + atomic fallbacks
        vocab_tokens = list(SPECIAL_TOKENS_PLAIN)

        # Add fragments (limit to max_size - special tokens - fallback space)
        fragment_limit = max(0, max_size - len(SPECIAL_TOKENS_PLAIN) - 100) if max_size else None

        # Warn if max_size is too small to include learned fragments
        if max_size and fragment_limit == 0:
            import warnings
            warnings.warn(
                f"max_size ({max_size}) is too small to include learned fragments. "
                f"Minimum recommended: {len(SPECIAL_TOKENS_PLAIN) + 150}. "
                f"Vocabulary will only contain special tokens and atomic fallbacks."
            )

        for frag in sorted_fragments:
            vocab_tokens.append(frag)
            if fragment_limit is not None and len(vocab_tokens) >= len(SPECIAL_TOKENS_PLAIN) + fragment_limit:
                break

        # Add atomic fallback tokens (for rare fragments not in vocab)
        atomic_fallbacks = cls._get_atomic_fallbacks()
        for token in atomic_fallbacks:
            if token not in vocab_tokens:
                vocab_tokens.append(token)
            if max_size and len(vocab_tokens) >= max_size:
                break

        return cls(vocab_tokens)

    @staticmethod
    def _get_atomic_fallbacks() -> List[str]:
        """Get atomic-level tokens as fallback for unknown fragments."""
        return [
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'F', 'Cl', 'Br', 'I',
            '[NH]', '[O-]', '[N+]', '[Si]', '[Na]', '[C@H]', '[C@@H]',
            '(', ')', '=', '#', '-', '+', '/', '\\',
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '%10', '%11', '%12',
            '@', '@@', '.',
        ]

    def tokenize(self, smiles: str) -> List[int]:
        """
        Tokenize SMILES → SAFE → Token IDs.

        Falls back to atom-level tokenization if BRICS fails.
        """
        try:
            # Convert SMILES to SAFE
            safe_string = self._smiles_to_safe(smiles)

            # Tokenize SAFE string (fragment-aware)
            tokens = self._tokenize_safe(safe_string)

            # Convert to IDs
            ids = [self.bos_id]
            for token in tokens:
                ids.append(self.token_to_id.get(token, self.unk_id))
            ids.append(self.eos_id)

            return ids

        except Exception:
            # Fallback: atom-level tokenization
            return self._fallback_tokenize(smiles)

    def _smiles_to_safe(self, smiles: str) -> str:
        """Convert SMILES to SAFE encoding."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # BRICS decomposition
        fragments = list(BRICS.BRICSDecompose(mol))

        if not fragments:
            # No fragmentation possible - return SMILES as-is
            return smiles

        # Encode as SAFE (simplified: concatenate with dots)
        # Real SAFE uses ring notation for linking, but for simplicity
        # we'll use dot-separated fragments
        safe_string = ".".join(sorted(fragments, key=len, reverse=True))

        return safe_string

    def _tokenize_safe(self, safe_string: str) -> List[str]:
        """Tokenize SAFE string into fragment tokens."""
        # Simple approach: split by dots and tokenize each part
        tokens = []

        for part in safe_string.split('.'):
            if part in self.token_to_id:
                # Fragment is in vocabulary
                tokens.append(part)
            else:
                # Fragment not in vocab - tokenize atomically
                atomic_tokens = _tokenize_regex(part, SMILES_ATOM_REGEX)
                tokens.extend(atomic_tokens)

            tokens.append('.')  # Separator

        # Remove trailing separator
        if tokens and tokens[-1] == '.':
            tokens.pop()

        return tokens

    def _fallback_tokenize(self, smiles: str) -> List[int]:
        """Fallback to atom-level tokenization if SAFE fails."""
        tokens = _tokenize_regex(smiles, SMILES_ATOM_REGEX)
        ids = [self.bos_id]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """
        Convert Token IDs → SAFE → Canonical SMILES.

        The SAFE string is valid SMILES that RDKit can parse,
        so we convert it to canonical SMILES for output.
        """
        # Convert IDs to tokens
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

        # Reconstruct SAFE string
        safe_string = "".join(tokens)

        # Convert SAFE to canonical SMILES
        return self._safe_to_smiles(safe_string)

    def _safe_to_smiles(self, safe_string: str) -> str:
        """Convert SAFE string to canonical SMILES."""
        try:
            # SAFE strings are valid SMILES - parse directly
            mol = Chem.MolFromSmiles(safe_string)
            if mol is None:
                # Fallback: return SAFE string as-is
                return safe_string

            # Convert to canonical SMILES
            canonical = Chem.MolToSmiles(mol, canonical=True)
            return canonical

        except Exception:
            # Fallback: return SAFE string
            return safe_string


class AnchorSAFEVocab(BaseVocabulary):
    """
    SAFE tokenization for Stage B/C (polymers with attachment points).

    Hybrid approach:
    - Preserves [Zz] and [Zr] as atomic tokens (no fragmentation)
    - Applies SAFE fragmentation to polymer backbone between anchors
    - Reconstructs AP-SMILES during decoding

    Example:
        "[*:1]CCC[*:2]" → shield → "[Zz]CCC[Zr]"
                        → SAFE middle → "[Zz]<fragments>[Zr]"
                        → tokenize → [[Zz], fragment_tokens, [Zr]]
                        → detokenize → "[*:1]CCC[*:2]"
    """

    def __init__(self, tokens: Sequence[str]):
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "SAFE tokenization requires RDKit. Install with: pip install rdkit"
            )
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
    ) -> AnchorSAFEVocab:
        """Build SAFE vocabulary for polymers with anchor preservation."""
        fragment_counts: Counter[str] = Counter()

        for ap_smiles in corpus:
            try:
                # Shield anchors
                shielded = shield_anchors(ap_smiles)

                # Extract middle part (between anchors)
                match = re.match(r'\[Zz\](.*)\[Zr\]', shielded)
                if not match:
                    continue

                middle = match.group(1)

                # Fragment middle part
                mol = Chem.MolFromSmiles(middle)
                if mol is not None:
                    fragments = BRICS.BRICSDecompose(mol)
                    fragment_counts.update(fragments)

            except Exception:
                continue

        # Build vocabulary similar to PlainSAFEVocab
        sorted_fragments = [
            frag
            for frag, freq in fragment_counts.items()
            if freq >= min_freq and frag not in SPECIAL_TOKENS_ANCHOR
        ]
        sorted_fragments.sort(key=lambda x: (-fragment_counts[x], x))

        vocab_tokens = list(SPECIAL_TOKENS_ANCHOR)

        fragment_limit = max(0, max_size - len(SPECIAL_TOKENS_ANCHOR) - 100) if max_size else None

        # Warn if max_size is too small to include learned fragments
        if max_size and fragment_limit == 0:
            import warnings
            warnings.warn(
                f"max_size ({max_size}) is too small to include learned fragments. "
                f"Minimum recommended: {len(SPECIAL_TOKENS_ANCHOR) + 150}. "
                f"Vocabulary will only contain special tokens and atomic fallbacks."
            )

        for frag in sorted_fragments:
            vocab_tokens.append(frag)
            if fragment_limit is not None and len(vocab_tokens) >= len(SPECIAL_TOKENS_ANCHOR) + fragment_limit:
                break

        # Add atomic fallbacks
        atomic_fallbacks = PlainSAFEVocab._get_atomic_fallbacks()
        for token in atomic_fallbacks:
            if token not in vocab_tokens:
                vocab_tokens.append(token)
            if max_size and len(vocab_tokens) >= max_size:
                break

        return cls(vocab_tokens)

    def get_anchor_ids(self) -> Tuple[int, int]:
        """Return (SHIELD1_id, SHIELD2_id)."""
        return (self.token_to_id[SHIELD1], self.token_to_id[SHIELD2])

    def tokenize(self, ap_smiles: str) -> List[int]:
        """Tokenize AP-SMILES with SAFE fragmentation of middle part."""
        try:
            # Shield anchors
            shielded = shield_anchors(ap_smiles)

            # Extract parts: [Zz] + middle + [Zr]
            match = re.match(r'(\[Zz\])(.*?)(\[Zr\])', shielded)
            if not match:
                raise ValueError(f"Invalid AP-SMILES: {ap_smiles}")

            anchor1, middle, anchor2 = match.groups()

            # Tokenize: [Zz] + SAFE(middle) + [Zr]
            ids = [self.bos_id]

            # Anchor 1
            ids.append(self.token_to_id[SHIELD1])

            # Middle (SAFE-tokenized)
            middle_safe = self._smiles_to_safe(middle) if middle else ""
            middle_tokens = self._tokenize_safe(middle_safe) if middle_safe else []
            for token in middle_tokens:
                ids.append(self.token_to_id.get(token, self.unk_id))

            # Anchor 2
            ids.append(self.token_to_id[SHIELD2])

            ids.append(self.eos_id)

            return ids

        except Exception:
            # Fallback to atomic tokenization
            return self._fallback_tokenize(ap_smiles)

    def _smiles_to_safe(self, smiles: str) -> str:
        """Convert SMILES to SAFE (same as PlainSAFEVocab)."""
        if not smiles:
            return ""

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        fragments = list(BRICS.BRICSDecompose(mol))
        if not fragments:
            return smiles

        return ".".join(sorted(fragments, key=len, reverse=True))

    def _tokenize_safe(self, safe_string: str) -> List[str]:
        """Tokenize SAFE string (same as PlainSAFEVocab)."""
        if not safe_string:
            return []

        tokens = []
        for part in safe_string.split('.'):
            if part in self.token_to_id:
                tokens.append(part)
            else:
                atomic_tokens = _tokenize_regex(part, SMILES_ATOM_REGEX)
                tokens.extend(atomic_tokens)
            tokens.append('.')

        if tokens and tokens[-1] == '.':
            tokens.pop()

        return tokens

    def _fallback_tokenize(self, ap_smiles: str) -> List[int]:
        """Fallback to atom-level tokenization."""
        shielded = shield_anchors(ap_smiles)
        tokens = _tokenize_regex(shielded, SMILES_ATOM_REGEX)
        ids = [self.bos_id]
        for token in tokens:
            ids.append(self.token_to_id.get(token, self.unk_id))
        ids.append(self.eos_id)
        return ids

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Convert Token IDs → Shielded SAFE → AP-SMILES."""
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

        # Reconstruct shielded SAFE
        safe_shielded = "".join(tokens)

        # Try to canonicalize middle part
        try:
            match = re.match(r'(\[Zz\])(.*?)(\[Zr\])', safe_shielded)
            if match:
                anchor1, middle, anchor2 = match.groups()
                if middle:
                    mol = Chem.MolFromSmiles(middle)
                    if mol:
                        middle = Chem.MolToSmiles(mol, canonical=True)
                safe_shielded = anchor1 + middle + anchor2
        except Exception:
            pass

        # Unshield: [Zz] → [*:1], [Zr] → [*:2]
        return unshield_anchors(safe_shielded)
