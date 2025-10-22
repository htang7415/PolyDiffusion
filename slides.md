# Slide 1 – Stage A Model Snapshot
- **Tokenization:** Character tokenizer auto-builds `PlainVocab`; blank `vocab_path` writes `vocab_character_stage_a.txt` into `Results/stage_a`.
- **Model:** DiffusionTransformer backbone with token/time embeddings and AdaLayerNorm blocks; single token-logit head (no conditioning net).
- **Diffusion:** `CategoricalDiffusion` masking tokens over up to 1k steps (20 default in `model_base.yaml`) to learn SMILES grammar.
- **Loss:** Diffusion cross-entropy reconstruction of clean tokens at each denoising step.
- **Training Strategy:** `batch_size=512`, cosine LR 3e-4→1e-5, weight decay 0.02, AMP + grad clip 1.0 for 1k steps; checkpoints/logs in `Results/stage_a`.
- **Metrics:** RDKit validity, mean SA, uniqueness, novelty, internal diversity computed from sampled molecules.

# Slide 2 – Stage B Model Snapshot
- **Tokenization:** `AnchorSafeVocab` extends Stage A tokens with `[Zz]/[Zr]` shields for `[*:1]/[*:2]`; blank paths trigger auto-build in `Results/stage_b`.
- **Model:** Same DiffusionTransformer backbone warm-started from Stage A, plus an anchor grammar head alongside token logits.
- **Diffusion:** Reuses categorical schedule to denoise AP-SMILES while preserving attachment tokens.
- **Loss:** Stage A diffusion CE + grammar penalty weighted by `lambda_gram=0.1` to enforce anchor count/valence.
- **Training Strategy:** `batch_size=256`, cosine LR 2e-4→1e-5, weight decay 0.01, AMP + grad clip 1.0 for 1k steps; optional backbone freezing via config.
- **Metrics:** Stage A metrics plus anchor correctness (count/valence) on generated repeat units.

# Slide 3 – Stage A→B Transfer Highlights
- **Tokenization Alignment:** Stage B auto-detects Stage A vocabulary/checkpoint, remapping embeddings for shared tokens and shielding anchors.
- **Model Reuse:** Embedding matrix and output head weights copy forward; backbone can stay trainable or frozen for rapid adaptation.
- **Diffusion Configuration:** Consistent categorical scheduler keeps denoising dynamics stable while grammar head guides anchor placement.
- **Loss Evolution:** Grammar loss augments Stage A objective to bias towards syntactically valid polymer attachment points.
- **Training Strategy:** Stage B initialization `mode: scratch` auto-loads Stage A artifacts when paths left blank, ensuring seamless warm start.
- **Metric Focus:** Monitor inherited molecule scores plus anchor accuracy to verify successful specialization before Stage C conditioning.
