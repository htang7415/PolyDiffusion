"""
Polymer BPE Tokenization - Hybrid data-driven + chemistry-guided approach.

Like subword tokenization in NLP, but for polymer chemistry.
Treats [*:1] and [*:2] as atomic tokens (never merged).
Learns fragments from polymer corpus using BPE algorithm.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Optional, Set, Tuple

from .base_vocab import BaseVocabulary

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# Special tokens
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<MASK>", "<UNK>"]
ANCHOR_TOKENS = ["[*:1]", "[*:2]"]


class PolymerBPEVocab(BaseVocabulary):
    """
    BPE-style tokenization for polymer repeat units.

    Features:
    - [*:1] and [*:2] are ALWAYS atomic (never merged)
    - Learns fragments from polymer corpus (data-driven)
    - Includes chemistry-guided fragments (domain knowledge)
    - Position-independent (works with any canonicalization)
    - Graceful atom-level fallback

    Process:
        1. Initialize with atoms + anchors + chem fragments
        2. Learn BPE merges from corpus
        3. Tokenize using longest-match greedy
        4. Validate all fragments are valid SMILES

    Example:
        "[*:1]CCC(=O)NCC[*:2]"
        → Tokens: ["[*:1]", "CCC", "C(=O)", "NCC", "[*:2]"]
    """

    def __init__(self, tokens: List[str]):
        """Initialize with token list."""
        super().__init__(tokens)
        # Build lookup set for fast tokenization
        self._vocab_set = set(self.id_to_token)

    def get_anchor_ids(self) -> Tuple[int, int]:
        """Return ([*:1]_id, [*:2]_id) for polymer generation."""
        return (self.token_to_id["[*:1]"], self.token_to_id["[*:2]"])

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        num_merges: int = 500,
        max_size: int = 1000,
        min_freq: int = 2,
    ) -> PolymerBPEVocab:
        """
        Build vocabulary from polymer corpus using BPE.

        Args:
            corpus: Iterator of polymer SMILES with [*:1] and [*:2]
            num_merges: Number of BPE merge operations
            max_size: Maximum vocabulary size
            min_freq: Minimum frequency for merges

        Returns:
            PolymerBPEVocab instance
        """
        print("Building Polymer BPE Vocabulary...")

        # 1. Initialize with base tokens
        vocab = set(SPECIAL_TOKENS + ANCHOR_TOKENS)
        vocab.update(cls._get_atom_tokens())
        vocab.update(cls._get_chemistry_fragments())

        print(f"  Base vocab size: {len(vocab)}")

        # 2. Convert corpus to list for multiple passes
        corpus_list = list(corpus)
        print(f"  Corpus size: {len(corpus_list)} polymers")

        # 3. Learn BPE merges
        current_vocab = list(vocab)

        for merge_idx in range(num_merges):
            # Count adjacent pairs
            pair_counts = cls._count_pairs(
                corpus_list,
                current_vocab,
                anchor_tokens=ANCHOR_TOKENS
            )

            if not pair_counts:
                print(f"  No more pairs to merge at step {merge_idx}")
                break

            # Get most frequent pair
            best_pair, count = max(pair_counts.items(), key=lambda x: x[1])

            if count < min_freq:
                print(f"  Stopping: best pair frequency {count} < min_freq {min_freq}")
                break

            # Merge tokens
            merged = best_pair[0] + best_pair[1]

            # Validate merged fragment
            if not cls._is_valid_fragment(merged):
                # Skip invalid merges
                continue

            # Add to vocab
            current_vocab.append(merged)
            vocab.add(merged)

            if (merge_idx + 1) % 50 == 0:
                print(f"  Merge {merge_idx + 1}/{num_merges}: '{best_pair[0]}' + '{best_pair[1]}' = '{merged}' (freq={count})")

            # Stop if vocab too large
            if max_size is not None and len(vocab) >= max_size:
                print(f"  Stopping: vocab size {len(vocab)} >= max_size {max_size}")
                break

        print(f"  Final vocab size: {len(vocab)}")
        return cls(list(vocab))

    @staticmethod
    def _get_atom_tokens() -> List[str]:
        """Base atomic tokens for SMILES."""
        return [
            # Organic atoms
            'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p',
            'F', 'Cl', 'Br', 'I', 'Si',

            # Bracket atoms
            '[NH]', '[NH2]', '[O-]', '[N+]', '[C@H]', '[C@@H]',
            '[Si]', '[Na]', '[K]', '[Ca]',

            # Structural
            '(', ')', '=', '#', '-', '+', '/', '\\', '.',

            # Ring markers
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '%10', '%11', '%12', '%13', '%14', '%15',

            # Stereochemistry
            '@', '@@',

            # Brackets
            '[', ']', ':'
        ]

    @staticmethod
    def _get_chemistry_fragments() -> List[str]:
        """Chemistry-guided polymer fragments."""
        return [
            # Common linkages
            "C(=O)O",       # Ester
            "OC(=O)",       # Ester (reversed)
            "C(=O)N",       # Amide
            "NC(=O)",       # Amide (reversed)
            "OC(=O)O",      # Carbonate
            "NC(=O)N",      # Urea
            "NC(=O)O",      # Urethane

            # Aromatic rings
            "c1ccccc1",     # Benzene
            "c1ccc",        # Benzene partial
            "ccc1",         # Benzene partial

            # Common chains
            "CC", "CCC", "CCCC", "CCCCC",
            "CC(C)", "C(C)C", "C(C)(C)",

            # Ether chains
            "OCCO", "OCCCO", "OCCCCO",

            # Polyimide-specific
            "C(=O)c1ccc",   # Phthalic part
            "c1ccc2c(c1)C(=O)",  # Imide structure partial
        ]

    @staticmethod
    def _count_pairs(
        corpus: List[str],
        vocab: List[str],
        anchor_tokens: List[str]
    ) -> Counter:
        """Count adjacent token pairs in corpus."""
        pair_counts = Counter()
        vocab_set = set(vocab)
        anchor_set = set(anchor_tokens)

        for smiles in corpus:
            # Tokenize with current vocab
            tokens = PolymerBPEVocab._tokenize_greedy(smiles, vocab_set)

            # Count pairs (skip anchor boundaries)
            for i in range(len(tokens) - 1):
                left, right = tokens[i], tokens[i + 1]

                # Never merge if either is an anchor
                if left in anchor_set or right in anchor_set:
                    continue

                pair = (left, right)
                pair_counts[pair] += 1

        return pair_counts

    @staticmethod
    def _tokenize_greedy(smiles: str, vocab_set: Set[str]) -> List[str]:
        """Greedy longest-match tokenization."""
        tokens = []
        i = 0

        while i < len(smiles):
            # Try longest match first (up to 30 chars)
            matched = None
            for length in range(min(30, len(smiles) - i), 0, -1):
                candidate = smiles[i:i+length]
                if candidate in vocab_set:
                    matched = candidate
                    break

            if matched:
                tokens.append(matched)
                i += len(matched)
            else:
                # Fallback to single character
                tokens.append(smiles[i])
                i += 1

        return tokens

    @staticmethod
    def _is_valid_fragment(fragment: str) -> bool:
        """Check if fragment is chemically valid."""
        # Anchors always valid
        if fragment in ANCHOR_TOKENS:
            return True

        # Single characters always valid (atoms, parens, etc.)
        if len(fragment) == 1:
            return True

        # Structural tokens valid
        if fragment in ['()', '==', '##', '//', '\\\\', '::']:
            return True

        # Try parsing as SMILES
        if RDKIT_AVAILABLE:
            try:
                mol = Chem.MolFromSmiles(fragment)
                if mol is not None:
                    return True
            except:
                pass

        # If not parseable as complete molecule, might still be valid fragment
        # Allow common patterns that appear in SMILES
        common_patterns = [
            r'^[CNOSPcnospF]+$',  # Just atoms
            r'^[CNOSPcnospF]+[\(\)]+$',  # Atoms + parens
            r'^C+$', r'^c+$',  # Carbon chains
            r'^c\d+$', r'^C\d+$',  # With ring numbers
        ]

        for pattern in common_patterns:
            if re.match(pattern, fragment):
                return True

        return False

    def tokenize(self, polymer_smiles: str) -> List[int]:
        """
        Tokenize polymer SMILES to token IDs.

        Args:
            polymer_smiles: Polymer with [*:1] and [*:2]

        Returns:
            List of token IDs including BOS and EOS
        """
        # Greedy longest-match tokenization
        tokens = self._tokenize_greedy(polymer_smiles, self._vocab_set)

        # Convert to IDs
        ids = [self.bos_id]
        for token in tokens:
            token_id = self.token_to_id.get(token, self.unk_id)
            ids.append(token_id)
        ids.append(self.eos_id)

        return ids

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to SMILES.

        Args:
            token_ids: List of token IDs

        Returns:
            Polymer SMILES string
        """
        tokens = []
        specials = {self.pad_id, self.bos_id, self.eos_id}

        for token_id in token_ids:
            if token_id in specials:
                continue
            if token_id < 0 or token_id >= len(self.id_to_token):
                continue

            token = self.id_to_token[token_id]
            if token in ("<MASK>", "<UNK>"):
                continue

            tokens.append(token)

        # Join tokens (no separator for SMILES)
        smiles = "".join(tokens)

        return smiles
