from PolyDiffusion.chem.plain_vocab import PlainVocab
from PolyDiffusion.chem.vocab import AnchorSafeVocab
from PolyDiffusion.models.dit_token import DiffusionTransformer, ModelConfig
from PolyDiffusion.models.diffusion_token import DiffusionConfig
from PolyDiffusion.sampling.sampler import GuidedSampler, PlainSampler, SamplerConfig


def test_sampler_returns_anchors() -> None:
    vocab = AnchorSafeVocab.build(["[*:1]C[*:2]"])
    model_cfg = ModelConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        diffusion_steps=4,
        property_names=["Tg", "Tm", "Td", "Eg", "chi"],
    )
    diffusion_cfg = DiffusionConfig(vocab_size=len(vocab), num_steps=4, mask_token_id=vocab.mask_id)
    model = DiffusionTransformer(model_cfg, diffusion_cfg)

    sampler = GuidedSampler(model, vocab, SamplerConfig(max_length=12, cfg_scale=1.0))
    results = sampler.sample(num_samples=2, num_steps=2)
    assert len(results) == 2
    for item in results:
        smiles = item["ap_smiles"]
        assert "[*:1]" in smiles and "[*:2]" in smiles


def test_plain_sampler_returns_plain_smiles() -> None:
    vocab = PlainVocab.build(["CCO"])
    model_cfg = ModelConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        diffusion_steps=4,
        property_names=["Tg", "Tm", "Td", "Eg", "chi"],
    )
    diffusion_cfg = DiffusionConfig(vocab_size=len(vocab), num_steps=4, mask_token_id=vocab.mask_id)
    model = DiffusionTransformer(model_cfg, diffusion_cfg)

    sampler = PlainSampler(model, vocab, SamplerConfig(max_length=12, cfg_scale=1.0))
    results = sampler.sample(num_samples=2, num_steps=2)
    assert len(results) == 2
    for item in results:
        smiles = item["smiles"]
        assert "[*:1]" not in smiles and "[*:2]" not in smiles
