from PolyDiffusion.chem.valence import has_two_anchors, valence_ok


def test_has_two_anchors_positive() -> None:
    assert has_two_anchors("[*:1]CC[*:2]")


def test_has_two_anchors_negative() -> None:
    assert not has_two_anchors("[*:1]CC")


def test_valence_ok_balances_parentheses() -> None:
    assert valence_ok("[*:1]CC(C)C[*:2]")
    assert not valence_ok("[*:1]CC(C[*:2]")
