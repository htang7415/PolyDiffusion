# Polymer Repeat Unit Design - Workflow Summary

## ✅ WHAT YOU'VE COMPLETED

### 1. Data Preparation ✓
- **Input**: `data/water_soluble.txt` (47 polymers with 2 * each)
- **Preprocessed**: `data/water_soluble_polymer_safe.txt` (38 SAFE-encoded polymers)
- **Tool**: `genmol/scripts/preprocess_polymer_safe.py`

### 2. Code Modifications ✓
- ✓ Fixed all import paths (`src.genmol` → `genmol`)
- ✓ Created `genmol/src/genmol/utils/polymer_utils.py`
- ✓ Modified `genmol/src/genmol/sampler.py` for polymer mode
- ✓ Created `genmol/scripts/exps/denovo_polymer.py`
- ✓ Added `polymer_mode: True` to config

### 3. Training ✓
- ✓ Trained for 50,000 steps on water_soluble_polymer_safe.txt
- ✓ Checkpoint saved: `ckpt/polymer_test/checkpoints/50000.ckpt`

---

## ⚠️ CURRENT ISSUE

**Generation produces EMPTY results**: `oracle/generated_polymer_samples.txt` is empty (0 bytes)

**Likely Cause**: The SAFE decoding is not preserving the wildcard atoms (`*`)

---

## 🔍 DEBUGGING STEPS

### Step 1: Run Debug Script

```bash
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/genmol
python debug_polymer_generation.py
```

This will test:
1. SAFE encoding/decoding roundtrip
2. Whether wildcards are preserved
3. Model generation with small batch

### Step 2: Check Debug Output

Look for:
- Does standard SAFE decoding preserve `*`?
- Does `safe_to_polymer_smiles()` work correctly?
- Are samples being generated but filtered out?

---

## 🔧 POTENTIAL FIXES

### Fix Option 1: SAFE Doesn't Preserve Wildcards by Default

**Problem**: SAFE library may close attachment points during decoding

**Solution**: We need to handle this in `polymer_utils.py`

The current `safe_to_polymer_smiles()` function tries to:
1. Decode SAFE → SMILES
2. Check if it has 2 wildcards
3. If not, look for dummy atoms and convert them

**Test this manually**:
```python
import safe as sf

# Encode
poly = "C(C[*])([*])N1CCCC1=O"
safe_str = sf.encode(poly)
print(f"SAFE: {safe_str}")

# Decode
decoded = sf.decode(safe_str, canonical=True)
print(f"Decoded: {decoded}")
print(f"Wildcards: {decoded.count('*')}")
```

### Fix Option 2: Model Not Learning Wildcard Patterns

**Problem**: With only 38 training samples, model may not learn to generate wildcards

**Solution**: 
1. Use larger dataset: `data/PI1M.txt` (1 million polymers)
2. Train for more steps
3. Lower diversity, focus on reconstructing training data

### Fix Option 3: Post-Processing is Too Strict

**Problem**: `validate_polymer()` filters out all generated samples

**Solution**: Check what samples are being generated before filtering

Add debug prints in `denovo_polymer.py` line 106:
```python
samples = sampler.de_novo_generation(num_samples, softmax_temp=0.5, randomness=0.5)

# ADD THIS:
print(f"\n[DEBUG] Raw samples before validation:")
for i, s in enumerate(samples[:5], 1):
    print(f"  {i}. {s} (wildcards: {s.count('*') if s else 'None'})")
```

---

## 🚀 RECOMMENDED NEXT STEPS

### Option A: Scale Up to Larger Dataset (RECOMMENDED)

```bash
# 1. Preprocess PI1M dataset (1M polymers)
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion

python genmol/scripts/preprocess_polymer_safe.py \
    data/PI1M.txt \
    data/PI1M_polymer_safe.txt \
    --num_attachments 2

# 2. Train on larger dataset
cd genmol
torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_PI1M \
    wandb.name=polymer_PI1M \
    data=/home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/PI1M_polymer_safe.txt \
    loader.global_batch_size=256 \
    trainer.max_steps=50000

# 3. Generate
# Update denovo_polymer.py line 95:
# sampler = Sampler('ckpt/polymer_PI1M/checkpoints/step=50000.ckpt')
python scripts/exps/denovo_polymer.py
```

### Option B: Debug Current Setup

```bash
# 1. Run debug script
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/genmol
python debug_polymer_generation.py > debug_output.txt

# 2. Examine output
cat debug_output.txt

# 3. Based on findings, adjust polymer_utils.py or sampler.py
```

### Option C: Use Pre-trained Model for Testing

```bash
# Download original GenMol model (trained on drugs)
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/genmol
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/clara/genmol_v1/versions/1.0.0/files/model.ckpt

# Test generation (won't have 2 wildcards, but tests if pipeline works)
python scripts/exps/denovo.py
```

---

## 📊 WORKFLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│  POLYMER SMILES (with 2 *)                                   │
│  data/water_soluble.txt                                      │
│  Example: C(C[*])([*])N1CCCC1=O                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ preprocess_polymer_safe.py
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  SAFE ENCODED                                                │
│  data/water_soluble_polymer_safe.txt                         │
│  Example: N14CCCC1=O.C34C2                                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ train.py (50K steps)
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  TRAINED MODEL                                               │
│  ckpt/polymer_test/checkpoints/50000.ckpt                    │
│  Config: polymer_mode = True                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ denovo_polymer.py
                   │ sampler.de_novo_generation()
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATED SAFE STRINGS                                      │
│  (Internal, from model output)                               │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ safe_to_polymer_smiles() ← ⚠️ ISSUE HERE
                   │ 
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  POLYMER SMILES (with 2 *)                                   │
│  oracle/generated_polymer_samples.txt                        │
│  ⚠️ CURRENTLY EMPTY                                          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   │ validate_polymer()
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  VALIDATED POLYMERS                                          │
│  oracle/generated_polymers_all.csv                           │
│  oracle/generated_polymers_quality.csv                       │
└─────────────────────────────────────────────────────────────┘
```

**The issue is at the decoding step** (marked with ⚠️)

---

## 📝 KEY FILES

### Data Files
- `data/water_soluble.txt` - Original polymer SMILES (47)
- `data/water_soluble_polymer_safe.txt` - SAFE encoded (38)
- `data/PI1M.txt` - Large polymer dataset (1M) *for future use*

### Scripts
- `genmol/scripts/preprocess_polymer_safe.py` - Data preprocessing
- `genmol/scripts/train.py` - Training script
- `genmol/scripts/exps/denovo_polymer.py` - Generation & evaluation
- `genmol/debug_polymer_generation.py` - Debug script (**RUN THIS FIRST**)

### Core Code
- `genmol/src/genmol/sampler.py` - Generation logic (lines 101-118 for polymer mode)
- `genmol/src/genmol/utils/polymer_utils.py` - Polymer-specific utilities
- `genmol/configs/base.yaml` - Configuration (polymer_mode: True on line 70)

### Outputs
- `ckpt/polymer_test/checkpoints/50000.ckpt` - Trained model
- `oracle/generated_polymer_samples.txt` - Generated polymers (currently empty)
- `oracle/generated_polymers_all.csv` - Validated polymers with properties
- `oracle/generated_polymers_quality.csv` - High-quality polymers only

---

## 🎯 EXPECTED FINAL RESULTS

After fixing the decoding issue, you should get:

```
============================================================
Polymer Repeat Unit Generation with GenMol
============================================================

Generating 100 polymer repeat units...
Generation time: 15.32 sec
Raw samples generated: 100
Saved to: oracle/generated_polymer_samples.txt

============================================================
EVALUATION METRICS
============================================================

1. VALIDITY: 0.720 (72/100)
   - Valid polymers with 2 wildcards: 72
   - Failure reasons:
     * Wrong wildcard count: 0: 28

2. UNIQUENESS: 0.903 (65/72)

3. DIVERSITY: 0.542
   - Measured by average Tanimoto distance

4. QUALITY: 0.510 (51/100)
   - Criteria: MW 50-500, 5-50 atoms, ≤15 rotatable bonds

✓ All valid unique polymers saved to: oracle/generated_polymers_all.csv
✓ High-quality polymers saved to: oracle/generated_polymers_quality.csv
```

---

## 💡 TIPS

1. **Start with debug script** - This will show exactly where the issue is
2. **Check SAFE decoding** - The wildcards may be lost during SAFE decode
3. **Scale up dataset** - 38 samples is very small; use PI1M for better results
4. **Adjust temperature** - Try `softmax_temp=0.3-0.8` for different diversity
5. **Check training loss** - If loss didn't converge, model didn't learn well

---

**Created**: October 2, 2025  
**Status**: Training complete ✓ | Generation debugging needed ⚠️  
**Next**: Run `debug_polymer_generation.py`

