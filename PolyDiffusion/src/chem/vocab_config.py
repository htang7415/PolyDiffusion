"""Configuration system for tokenization methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TokenizationConfig:
    """Configuration for tokenization method and vocabulary building."""

    # Tokenization method
    method: Literal['character', 'atom_regex', 'safe', 'polymer_bpe'] = 'character'

    # Vocabulary file path
    vocab_path: str = ''

    # Vocabulary building parameters
    min_freq: int = 1
    max_size: Optional[int] = None
    vocab_limit_samples: int = 10000

    # SAFE-specific parameters
    safe_fragment_type: str = 'brics'  # brics, recap, or custom
    safe_min_fragment_size: int = 1

    # Atom-regex specific parameters
    atom_regex_pattern: str = 'default'  # default or custom pattern

    def __post_init__(self):
        """Validate configuration."""
        valid_methods = {'character', 'atom_regex', 'safe', 'polymer_bpe'}
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid tokenization method '{self.method}'. "
                f"Must be one of: {valid_methods}"
            )

        if self.min_freq < 1:
            raise ValueError(f"min_freq must be >= 1, got {self.min_freq}")

        if self.vocab_limit_samples is not None and self.vocab_limit_samples < 1:
            raise ValueError(
                f"vocab_limit_samples must be >= 1 when provided, got {self.vocab_limit_samples}"
            )


def load_tokenization_config(stage_config: dict) -> TokenizationConfig:
    """
    Load tokenization configuration from stage YAML config.

    Args:
        stage_config: Dictionary loaded from stage YAML file

    Returns:
        TokenizationConfig instance

    Example YAML:
        tokenization:
          method: "atom_regex"
          vocab_path: "PolyDiffusion/vocab_atom_stage_a.txt"
          build:
            min_freq: 1
            max_size: null
            vocab_limit_samples: 10000
          safe:
            fragment_type: "brics"
          atom_regex:
            pattern: "default"
    """
    # Backward compatibility: if no 'tokenization' section, default to character
    if 'tokenization' not in stage_config:
        # Use old vocab_path if present
        vocab_path = stage_config.get('vocab_path', '')
        return TokenizationConfig(
            method='character',
            vocab_path=vocab_path
        )

    tok_cfg = stage_config['tokenization']

    # Extract build parameters
    build_cfg = tok_cfg.get('build', {})

    # Extract SAFE parameters
    safe_cfg = tok_cfg.get('safe', {})

    # Extract atom-regex parameters
    atom_cfg = tok_cfg.get('atom_regex', {})

    return TokenizationConfig(
        method=tok_cfg.get('method', 'character'),
        vocab_path=tok_cfg.get('vocab_path', ''),
        min_freq=build_cfg.get('min_freq', 1),
        max_size=build_cfg.get('max_size', None),
        vocab_limit_samples=build_cfg.get('vocab_limit_samples', 10000),
        safe_fragment_type=safe_cfg.get('fragment_type', 'brics'),
        safe_min_fragment_size=safe_cfg.get('min_fragment_size', 1),
        atom_regex_pattern=atom_cfg.get('pattern', 'default'),
    )
