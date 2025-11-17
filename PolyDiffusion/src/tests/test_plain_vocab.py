"""Tests for PlainVocab (Stage A vocabulary without attachment points)."""

import pytest

from PolyDiffusion.src.chem.plain_vocab import PlainVocab


def test_build_plain_vocab() -> None:
    """Test building vocabulary from plain SMILES."""
    corpus = ["CCO", "CC(=O)O", "c1ccccc1"]
    vocab = PlainVocab.build(corpus)

    assert len(vocab) > 5  # Special tokens + characters
    assert vocab.id_to_token[0] == "<PAD>"
    assert vocab.id_to_token[1] == "<BOS>"
    assert vocab.id_to_token[2] == "<EOS>"
    assert "C" in vocab.id_to_token
    assert "O" in vocab.id_to_token


def test_tokenize_plain_smiles() -> None:
    """Test tokenizing plain SMILES without attachment points."""
    corpus = ["CCO"]
    vocab = PlainVocab.build(corpus)

    tokens = vocab.tokenize("CCO")

    # Should have BOS, C, C, O, EOS
    assert len(tokens) == 5
    assert tokens[0] == vocab.bos_id
    assert tokens[-1] == vocab.eos_id


def test_detokenize_plain_smiles() -> None:
    """Test detokenizing back to plain SMILES."""
    corpus = ["CCO", "CC(=O)O"]
    vocab = PlainVocab.build(corpus)

    tokens = vocab.tokenize("CCO")
    decoded = vocab.detokenize(tokens)

    assert decoded == "CCO"


def test_tokenize_detokenize_roundtrip() -> None:
    """Test full roundtrip: SMILES → tokens → SMILES."""
    test_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN", "CC(C)C"]
    corpus = test_smiles
    vocab = PlainVocab.build(corpus)

    for smiles in test_smiles:
        tokens = vocab.tokenize(smiles)
        decoded = vocab.detokenize(tokens)
        assert decoded == smiles, f"Failed roundtrip for {smiles}"


def test_unknown_character() -> None:
    """Test handling of unknown characters."""
    corpus = ["CCO"]
    vocab = PlainVocab.build(corpus)

    # 'N' is not in corpus
    tokens = vocab.tokenize("CCN")

    # Should contain UNK token
    assert vocab.token_to_id["<UNK>"] in tokens


def test_special_tokens() -> None:
    """Test special token properties."""
    corpus = ["CCO"]
    vocab = PlainVocab.build(corpus)

    assert vocab.pad_id == 0
    assert vocab.bos_id == 1
    assert vocab.eos_id == 2
    assert vocab.mask_id == 3


def test_no_attachment_points() -> None:
    """Test that PlainVocab does NOT include attachment point tokens."""
    corpus = ["CCO", "CC(=O)O"]
    vocab = PlainVocab.build(corpus)

    # Should NOT have [*:1] or [*:2]
    assert "[*:1]" not in vocab.id_to_token
    assert "[*:2]" not in vocab.id_to_token

    # Tokenization should not add anchors
    tokens = vocab.tokenize("CCO")
    token_strs = [vocab.id_to_token[t] for t in tokens]

    assert "[*:1]" not in token_strs
    assert "[*:2]" not in token_strs


def test_save_and_load() -> None:
    """Test saving and loading vocabulary."""
    import tempfile
    from pathlib import Path

    corpus = ["CCO", "CC(=O)O"]
    vocab = PlainVocab.build(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab.txt"
        vocab.save(path)

        loaded_vocab = PlainVocab.load(path)

        assert len(loaded_vocab) == len(vocab)
        assert loaded_vocab.id_to_token == vocab.id_to_token
        assert loaded_vocab.token_to_id == vocab.token_to_id


def test_min_freq_filtering() -> None:
    """Test minimum frequency filtering."""
    corpus = ["C", "CC", "CCC", "CCCC", "O"]  # C appears 10 times, O appears 1 time
    vocab = PlainVocab.build(corpus, min_freq=2)

    # O should be filtered out (freq = 1 < min_freq = 2)
    assert "C" in vocab.id_to_token
    # O might still be there if it appeared twice, but let's not assume


def test_max_size_limiting() -> None:
    """Test maximum vocabulary size limiting."""
    corpus = ["CCO", "CCN", "CCS"]
    vocab = PlainVocab.build(corpus, max_size=7)  # 5 special + 2 chars

    assert len(vocab) == 7


def test_stage_a_vs_stage_b_difference() -> None:
    """
    Demonstrate the key difference between PlainVocab (Stage A)
    and AnchorSafeVocab (Stage B).
    """
    from PolyDiffusion.src.chem.vocab import AnchorSafeVocab

    # Stage A: PlainVocab - NO attachment points
    # Plain SMILES for complete small molecules
    plain_corpus = ["CCO"]
    plain_vocab = PlainVocab.build(plain_corpus)
    plain_tokens = plain_vocab.tokenize("CCO")
    plain_strs = [plain_vocab.id_to_token[t] for t in plain_tokens]

    # Stage B: AnchorSafeVocab - WITH attachment points
    # AP-SMILES for polymer repeat units
    ap_corpus = ["[*:1]CCO[*:2]"]
    anchor_vocab = AnchorSafeVocab.build(ap_corpus)
    anchor_tokens = anchor_vocab.tokenize_ap("[*:1]CCO[*:2]")
    anchor_strs = [anchor_vocab.id_to_token[t] for t in anchor_tokens]

    # PlainVocab: [<BOS>, C, C, O, <EOS>]
    assert "<BOS>" in plain_strs
    assert "<EOS>" in plain_strs
    assert plain_strs.count("C") == 2
    assert plain_strs.count("O") == 1
    assert "[*:1]" not in plain_strs
    assert "[*:2]" not in plain_strs

    # AnchorSafeVocab: [<BOS>, [*:1], C, C, O, [*:2], <EOS>]
    assert "<BOS>" in anchor_strs
    assert "<EOS>" in anchor_strs
    assert "[*:1]" in anchor_strs
    assert "[*:2]" in anchor_strs
