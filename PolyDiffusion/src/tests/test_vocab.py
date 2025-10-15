from PolyDiffusion.chem.vocab import AnchorSafeVocab, SHIELD1, SHIELD2


def test_anchor_tokens_present() -> None:
    vocab = AnchorSafeVocab.build(["[*:1]C[*:2]"])
    assert SHIELD1 in vocab.token_to_id
    assert SHIELD2 in vocab.token_to_id


def test_roundtrip_tokenization() -> None:
    vocab = AnchorSafeVocab.build(["[*:1]CCO[*:2]"])
    tokens = vocab.tokenize_ap("[*:1]CCO[*:2]")
    restored = vocab.detokenize_ap(tokens)
    assert restored == "[*:1]CCO[*:2]"
