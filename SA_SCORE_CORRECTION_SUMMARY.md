# SA Score Correction Summary

**Date**: October 16, 2024  
**Issue**: Pipeline.md incorrectly suggested SA scores are needed during training

## The Correction

### What Was Wrong
The original Pipeline.md documentation suggested that:
- SA scores should be added to training datasets
- Training data should include SA_Score columns
- SA scores are used during the training process

### What is Actually Correct
**SA scores are ONLY used for evaluation AFTER sampling**, not during training:

1. **Training Phase**: Model only needs SMILES strings
   - Stage A: Plain SMILES (e.g., `CCO`, `c1ccccc1`)
   - Stage B: Polymer SMILES with `*` (e.g., `*CCC*`)
   - Stage C: AP-SMILES + properties (e.g., `[*:1]CCO[*:2]` with Tg, Tm, etc.)

2. **Evaluation Phase**: SA scores are calculated for generated samples
   - Generate samples from trained model
   - Calculate SA scores on generated molecules
   - Use SA scores as quality metrics

## Changes Made to Pipeline.md

### Section 5.1: Data Formats
**Before**:
```csv
SMILES,SA_Score
CCO,3.5
c1ccccc1,4.0
```

**After**:
```csv
SMILES
CCO
c1ccccc1
```

Added note: *"SA scores are NOT used during training. They are only calculated for evaluation after sampling."*

### Section 5.4: Calculate SA Scores
**Before**: "Calculate SA Scores (If Missing)" - suggested adding SA scores to training data

**After**: "Important: SA Scores are NOT Used During Training"
- Clear explanation that SA scores are only for post-generation evaluation
- Workflow showing: Train → Generate → Calculate SA scores → Evaluate

### Section 10.3: Evaluation Metrics
**Added clarification**:
- Evaluation generates NEW samples from trained model
- SA scores are calculated POST-GENERATION
- SA scores are not needed during training

### Section 11: SA Score Calculation
**Added header**: "This section is about evaluating generated samples after training, NOT about preparing training data"

**Rewrote Section 11.2**: "Why SA Score Matters for Evaluation"
- Clarified SA scores are ONLY for evaluating generated samples
- Model does NOT use SA scores during training

**Rewrote Section 11.4**: "Calculating SA Scores for Generated Samples"
- Changed examples from training data to generated samples
- Example paths: `Results/stage_b/samples/generated_polymers.csv`

**Added Section 11.7**: "Workflow: From Training to SA Score Evaluation"
- Complete workflow showing correct order
- Example commands for: Train → Generate → Calculate SA → Analyze

### Section 12: Troubleshooting
**Updated RDKit impact**:
- "Training will work perfectly" without RDKit
- RDKit only needed for post-generation evaluation

## Summary

### Training Data Requirements

| Stage | Required Columns | SA Score? |
|-------|-----------------|-----------|
| Stage A | `SMILES` | ❌ NO |
| Stage B | `SMILES` (with `*`) | ❌ NO |
| Stage C | `ap_smiles`, properties (Tg, Tm, etc.) | ❌ NO |

### When SA Scores ARE Used

✅ **After generating samples** from trained model  
✅ **For evaluation metrics** (synthesizability_mean, synthesizability_std)  
✅ **For quality analysis** of generated molecules  
✅ **For model comparison** across checkpoints  

### When SA Scores are NOT Used

❌ **NOT during training**  
❌ **NOT in training data preparation**  
❌ **NOT as model input**  
❌ **NOT as training loss**  

## Corrected Workflow

```
1. Prepare training data
   └─> Only SMILES (+ properties for Stage C)
   
2. Train model (Stages A, B, C)
   └─> Model learns from SMILES structure only
   
3. Generate samples from trained model
   └─> python -m PolyDiffusion.src.sampling.sampler
   
4. Calculate SA scores for generated samples
   └─> python -m PolyDiffusion.scripts.calculate_sa_score
   
5. Analyze results
   └─> Review SA score distribution as quality metric
```

## Files Updated

- **Results/Pipeline.md** (3,302 lines)
  - Section 5.1: Data Formats
  - Section 5.4: SA Score clarification
  - Section 6.4: Training Stage A (removed SA score references)
  - Section 10.3: Evaluation metrics clarification
  - Section 11: Complete rewrite for clarity
  - Section 12: Troubleshooting updates

## Key Takeaway

**SA (Synthetic Accessibility) score is a POST-GENERATION METRIC, not a training input.**

The model learns chemical grammar and structure purely from SMILES strings. SA scores are only calculated after generation to evaluate whether the model produces realistic, synthesizable molecules.
