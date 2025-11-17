"""Vocabulary factory for creating tokenization instances."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

from .atom_regex_vocab import AnchorAtomVocab, PlainAtomVocab
from .base_vocab import BaseVocabulary
from .character_vocab import AnchorCharacterVocab, PlainCharacterVocab
from .polymer_bpe_vocab import PolymerBPEVocab
from .safe_vocab import AnchorSAFEVocab, PlainSAFEVocab, RDKIT_AVAILABLE
from .vocab_config import TokenizationConfig

log = logging.getLogger(__name__)


def create_vocabulary(
    config: TokenizationConfig,
    stage: str,
    corpus: Optional[Iterable[str]] = None,
) -> BaseVocabulary:
    """
    Factory function to create appropriate vocabulary based on configuration.

    Args:
        config: Tokenization configuration
        stage: 'a', 'b', or 'c'
        corpus: Optional corpus to build vocab (if not loading from file)

    Returns:
        BaseVocabulary instance (character, atom-regex, SAFE, or polymer_bpe)

    Raises:
        ValueError: If invalid method or stage
        ImportError: If SAFE method requires RDKit but it's not installed
        RuntimeError: If neither vocab_path nor corpus is provided
    """
    method = config.method
    has_anchors = stage in ('b', 'c')

    # Select vocabulary class based on method and stage
    if method == 'character':
        VocabClass = AnchorCharacterVocab if has_anchors else PlainCharacterVocab
        log.info(f"Using character-level tokenization for stage {stage.upper()}")

    elif method == 'atom_regex':
        VocabClass = AnchorAtomVocab if has_anchors else PlainAtomVocab
        log.info(f"Using atom-level regex tokenization for stage {stage.upper()}")

    elif method == 'safe':
        if not RDKIT_AVAILABLE:
            raise ImportError(
                "SAFE tokenization requires RDKit. Install with: pip install rdkit"
            )
        VocabClass = AnchorSAFEVocab if has_anchors else PlainSAFEVocab
        log.info(f"Using SAFE fragment-based tokenization for stage {stage.upper()}")

    elif method == 'polymer_bpe':
        if not has_anchors:
            raise ValueError(
                "polymer_bpe method only supports stages B/C (with anchors). "
                "Use 'character' or 'atom_regex' for stage A."
            )
        VocabClass = PolymerBPEVocab
        log.info(f"Using Polymer BPE tokenization for stage {stage.upper()}")

    else:
        raise ValueError(
            f"Unknown tokenization method: {method}. "
            f"Must be 'character', 'atom_regex', 'safe', or 'polymer_bpe'"
        )

    # Load from file or build from corpus
    vocab_path = Path(config.vocab_path) if config.vocab_path else None

    if vocab_path and vocab_path.exists():
        log.info(f"Loading vocabulary from {vocab_path}")
        return VocabClass.load(vocab_path)

    elif corpus is not None:
        log.info(f"Building vocabulary from corpus (limit={config.vocab_limit_samples})")
        vocab = VocabClass.build(
            corpus=corpus,
            min_freq=config.min_freq,
            max_size=config.max_size,
        )
        log.info(f"Built vocabulary with {len(vocab)} tokens")
        return vocab

    else:
        raise RuntimeError(
            f"Must provide either vocab_path ('{config.vocab_path}') or corpus to build vocabulary"
        )


def load_vocabulary_auto(
    vocab_path: Path,
    tokenization_config: Optional[TokenizationConfig] = None,
) -> BaseVocabulary:
    """
    Auto-detect and load vocabulary from file.

    For backward compatibility with old vocab files that don't have metadata.

    Args:
        vocab_path: Path to vocabulary file
        tokenization_config: Optional config (for new-style loading)

    Returns:
        BaseVocabulary instance

    Detection logic:
        1. If file has metadata header (# TOKENIZATION_METHOD:), use that
        2. If tokenization_config provided, use that method
        3. Otherwise, detect by content:
           - Has [Zz] and [Zr] → AnchorCharacterVocab
           - Otherwise → PlainCharacterVocab
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    lines = vocab_path.read_text(encoding='utf-8').splitlines()

    # Check for metadata header (new format)
    if lines and lines[0].startswith('# TOKENIZATION_METHOD:'):
        method = lines[0].split(':', 1)[1].strip()
        log.info(f"Detected tokenization method from file: {method}")

        # Determine if anchors present
        # Note: Use '# ' to skip only metadata lines, preserving '#' as a valid token
        tokens_content = [line for line in lines if not line.startswith('# ')]
        has_anchors = '[Zz]' in tokens_content and '[Zr]' in tokens_content

        # Map method name to class
        if method in ('PlainCharacterVocab', 'AnchorCharacterVocab', 'character'):
            VocabClass = AnchorCharacterVocab if has_anchors else PlainCharacterVocab
        elif method in ('PlainAtomVocab', 'AnchorAtomVocab', 'atom_regex'):
            VocabClass = AnchorAtomVocab if has_anchors else PlainAtomVocab
        elif method in ('PlainSAFEVocab', 'AnchorSAFEVocab', 'safe'):
            if not RDKIT_AVAILABLE:
                raise ImportError("SAFE vocab requires RDKit")
            VocabClass = AnchorSAFEVocab if has_anchors else PlainSAFEVocab
        elif method in ('PolymerBPEVocab', 'polymer_bpe'):
            VocabClass = PolymerBPEVocab
        else:
            log.warning(f"Unknown method in metadata: {method}, falling back to character")
            VocabClass = AnchorCharacterVocab if has_anchors else PlainCharacterVocab

        return VocabClass.load(vocab_path)

    # Use tokenization_config if provided (new-style)
    if tokenization_config:
        method = tokenization_config.method
        tokens_content = lines

        has_anchors = '[Zz]' in tokens_content and '[Zr]' in tokens_content

        if method == 'character':
            VocabClass = AnchorCharacterVocab if has_anchors else PlainCharacterVocab
        elif method == 'atom_regex':
            VocabClass = AnchorAtomVocab if has_anchors else PlainAtomVocab
        elif method == 'safe':
            if not RDKIT_AVAILABLE:
                raise ImportError("SAFE vocab requires RDKit")
            VocabClass = AnchorSAFEVocab if has_anchors else PlainSAFEVocab
        elif method == 'polymer_bpe':
            VocabClass = PolymerBPEVocab
        else:
            raise ValueError(f"Unknown method: {method}")

        return VocabClass.load(vocab_path)

    # Fallback: auto-detect by content (backward compatibility)
    log.info("No metadata found, auto-detecting vocabulary type by content")
    tokens = lines

    if '[Zz]' in tokens and '[Zr]' in tokens:
        log.info("Detected anchor tokens → AnchorCharacterVocab")
        return AnchorCharacterVocab(tokens)
    else:
        log.info("No anchor tokens → PlainCharacterVocab")
        return PlainCharacterVocab(tokens)
