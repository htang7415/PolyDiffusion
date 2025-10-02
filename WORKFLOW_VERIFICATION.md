# Polymer Repeat Unit Design Workflow Verification

## ✅ COMPLETED STEPS

### 1. Data Preparation
- **Input Data**: `/home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble.txt`
  - ✅ Contains 47 polymer SMILES with exactly 2 wildcards each
  - Example: `C(C[*])([*])N1CCCC1=O`
  
- **Preprocessing Script**: `genmol/scripts/preprocess_polymer_safe.py`
  - ✅ Validates 2 wildcards per polymer
  - ✅ Encodes to SAFE format
  
- **Processed Data**: `data/water_soluble_polymer_safe.txt`
  - ✅ Contains 38 SAFE-encoded polymers
  - Example: `N14CCCC1=O.C34C2`

### 2. Code Modifications
- **Training Script**: `genmol/scripts/train.py`
  - ✅ Fixed import paths with `sys.path.insert()`
  
- **Model Code**: `genmol/src/genmol/model.py`
  - ✅ Fixed imports from `src.genmol` → `genmol`
  
- **Sampler Code**: `genmol/src/genmol/sampler.py`
  - ✅ Fixed imports
  - ✅ Added polymer_mode branching logic (lines 101-118)
  - ✅ Calls `safe_to_polymer_smiles()` for polymer mode
  
- **Polymer Utilities**: `genmol/src/genmol/utils/polymer_utils.py`
  - ✅ Created `safe_to_polymer_smiles()` function
  - ✅ Created `validate_polymer()` function
  - ✅ Validates exactly 2 wildcards
  
- **Generation Script**: `genmol/scripts/exps/denovo_polymer.py`
  - ✅ Fixed imports
  - ✅ Enables polymer_mode
  - ✅ Comprehensive evaluation metrics

### 3. Configuration
- **Config File**: `genmol/configs/base.yaml`
  - ✅ Added `polymer_mode: True` (line 70)
  - ⚠️ **ISSUE**: `data: safe/PI1M_safe.txt` (line 7) - Wrong path!

## ⚠️ CRITICAL ISSUES FOUND

### Issue 1: Config Points to Wrong Dataset
**Location**: `genmol/configs/base.yaml` line 7

**Current**:
```yaml
data: safe/PI1M_safe.txt
```

**Should be**:
```yaml
data: /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt
```

**Impact**: Your training used the wrong dataset (PI1M instead of water_soluble)

### Issue 2: Generated Samples File is Empty
**Location**: `genmol/oracle/generated_polymer_samples.txt`

**Status**: File is empty (0 bytes)

**Possible Causes**:
1. Model was trained on wrong dataset (PI1M instead of water_soluble)
2. Generation failed silently
3. All generated samples were filtered out as invalid

### Issue 3: Training May Have Used Wrong Data
**Evidence**:
- Config points to `safe/PI1M_safe.txt`
- Checkpoint exists at `ckpt/polymer_test/checkpoints/50000.ckpt`
- This suggests training completed, but potentially on wrong data

## 🔧 FIXES NEEDED

### Fix 1: Update Config File
```bash
# Edit genmol/configs/base.yaml line 7
data: /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt
```

### Fix 2: Verify Training Data
Check what data was actually used in training:
```bash
# Look at the hydra config from training run
cat ckpt/polymer_test/.hydra/config.yaml
```

### Fix 3: Re-train if Necessary
If training used wrong data, re-train:
```bash
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/genmol

torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_water_soluble \
    wandb.name=null \
    data=/home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt \
    loader.global_batch_size=32 \
    trainer.max_steps=5000
```

## 📋 COMPLETE WORKFLOW (CORRECT VERSION)

### Step 1: Preprocess Data
```bash
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion

python genmol/scripts/preprocess_polymer_safe.py \
    data/water_soluble.txt \
    data/water_soluble_polymer_safe.txt \
    --num_attachments 2
```

**Expected Output**: ~38 SAFE-encoded polymers

### Step 2: Update Config
Edit `genmol/configs/base.yaml`:
- Line 7: `data: /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/data/water_soluble_polymer_safe.txt`
- Line 70: `polymer_mode: True` ✅ (already done)
- Line 14: `global_batch_size: 32` (reduce if GPU memory issues)

### Step 3: Train Model
```bash
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/genmol

# For full training
torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_water_soluble \
    wandb.name=polymer_water_soluble \
    trainer.max_steps=10000

# For quick test (100 steps)
torchrun --nproc_per_node 1 scripts/train.py \
    hydra.run.dir=ckpt/polymer_test \
    wandb.name=null \
    trainer.max_steps=100 \
    loader.global_batch_size=32
```

**Expected Output**: Checkpoints in `ckpt/polymer_water_soluble/checkpoints/`

### Step 4: Generate Polymers
Update `genmol/scripts/exps/denovo_polymer.py` line 95:
```python
sampler = Sampler('ckpt/polymer_water_soluble/checkpoints/step=10000.ckpt')
```

Then run:
```bash
cd /home/htang228/Machine_learning/Diffusion_model/PolyDiffusion/genmol
python scripts/exps/denovo_polymer.py
```

**Expected Output**:
- `oracle/generated_polymer_samples.txt` - All generated polymers with 2 wildcards
- `oracle/generated_polymers_all.csv` - Valid unique polymers with properties
- `oracle/generated_polymers_quality.csv` - High-quality polymers only

### Step 5: Evaluate Results
The script automatically evaluates:
1. **Validity**: % with exactly 2 wildcards
2. **Uniqueness**: % unique structures
3. **Diversity**: Tanimoto distance metric
4. **Quality**: % meeting polymer criteria (MW, atoms, bonds)

## 🔍 VERIFICATION CHECKLIST

Before running generation:
- [ ] Config points to correct dataset (`water_soluble_polymer_safe.txt`)
- [ ] Model trained on polymer data (not PI1M drug data)
- [ ] Checkpoint file exists and is accessible
- [ ] `polymer_mode: True` in config
- [ ] All import errors resolved

After running generation:
- [ ] `generated_polymer_samples.txt` is not empty
- [ ] At least some samples have exactly 2 wildcards
- [ ] Validity > 0 (at least some valid polymers)
- [ ] Check failure reasons if validity is low

## 🎯 EXPECTED RESULTS

With water-soluble polymer training data (38 samples):
- **Validity**: 50-80% (depends on training quality)
- **Uniqueness**: 70-95%
- **Diversity**: 0.3-0.7
- **Quality**: 30-60%

**Note**: With only 38 training samples, results may vary significantly. For production:
- Use PI1M.txt (1M polymer samples) for better results
- Train for 50,000 steps
- Use larger batch sizes if GPU allows

## 📊 DEBUGGING GENERATION ISSUES

If `generated_polymer_samples.txt` is empty:

### Debug Step 1: Check SAFE Decoding
```python
import safe as sf

# Test decoding
safe_str = "N14CCCC1=O.C34C2"
smiles = sf.decode(safe_str, canonical=True, ignore_errors=True)
print(f"Decoded: {smiles}")
print(f"Wildcard count: {smiles.count('*')}")
```

### Debug Step 2: Test Generation with Verbose Output
Add print statements in `sampler.py` around line 98-108:
```python
samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
print(f"DEBUG: Raw SAFE samples: {samples[:3]}")

if hasattr(self.model.config, 'polymer_mode') and self.model.config.polymer_mode:
    samples = [safe_to_polymer_smiles(s, fix=fix) for s in samples]
    print(f"DEBUG: After polymer decoding: {samples[:3]}")
    samples = [s for s in samples if s is not None]
    print(f"DEBUG: After filtering None: {len(samples)} samples")
```

### Debug Step 3: Check Model Config
```python
from genmol.sampler import Sampler
sampler = Sampler('ckpt/polymer_test/checkpoints/50000.ckpt')
print(f"polymer_mode: {getattr(sampler.model.config, 'polymer_mode', 'NOT SET')}")
print(f"Training config data path: {sampler.model.config.data}")
```

## 🚀 NEXT STEPS

1. **Verify which dataset was used for training**
   ```bash
   cat ckpt/polymer_test/.hydra/config.yaml | grep "data:"
   ```

2. **If wrong dataset, re-train with correct data**

3. **Test generation again with correct checkpoint**

4. **Scale up to PI1M dataset for production use**

---

**Last Updated**: Oct 2, 2025
**Status**: ⚠️ Issues found - needs verification and potential retraining

