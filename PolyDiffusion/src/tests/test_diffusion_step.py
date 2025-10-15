import torch

from PolyDiffusion.models.diffusion_token import CategoricalDiffusion, DiffusionConfig


def test_diffusion_step_smoke() -> None:
    config = DiffusionConfig(vocab_size=20, num_steps=10, mask_token_id=0)
    diffusion = CategoricalDiffusion(config)
    tokens = torch.randint(0, config.vocab_size, (4, 8))
    timesteps = diffusion.sample_timesteps(tokens.size(0))
    noisy, mask = diffusion.q_sample(tokens, timesteps)
    assert noisy.shape == tokens.shape
    assert mask.shape == tokens.shape

    logits = torch.randn(tokens.size(0), tokens.size(1), config.vocab_size)
    loss = diffusion.loss(logits, tokens, timesteps, mask)
    assert loss.dim() == 0
