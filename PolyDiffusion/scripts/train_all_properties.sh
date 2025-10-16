#!/bin/bash
# Train all 5 Stage C property models sequentially
# Usage: bash scripts/train_all_properties.sh

set -e  # Exit on error

echo "=================================================="
echo "Training All 5 Property Models for Stage C"
echo "=================================================="
echo ""

# Train Tg model
echo "Training Tg (Glass Transition Temperature) model..."
python -m PolyDiffusion.src.train.train_stage_c --config PolyDiffusion/configs/stage_c_Tg.yaml
echo "✓ Tg model training completed"
echo ""

# Train Tm model
echo "Training Tm (Melting Temperature) model..."
python -m PolyDiffusion.src.train.train_stage_c --config PolyDiffusion/configs/stage_c_Tm.yaml
echo "✓ Tm model training completed"
echo ""

# Train Td model
echo "Training Td (Decomposition Temperature) model..."
python -m PolyDiffusion.src.train.train_stage_c --config PolyDiffusion/configs/stage_c_Td.yaml
echo "✓ Td model training completed"
echo ""

# Train Eg model
echo "Training Eg (Band Gap Energy) model..."
python -m PolyDiffusion.src.train.train_stage_c --config PolyDiffusion/configs/stage_c_Eg.yaml
echo "✓ Eg model training completed"
echo ""

# Train chi model
echo "Training chi (Flory-Huggins Parameter) model..."
python -m PolyDiffusion.src.train.train_stage_c --config PolyDiffusion/configs/stage_c_chi.yaml
echo "✓ chi model training completed"
echo ""

echo "=================================================="
echo "All 5 Property Models Trained Successfully!"
echo "=================================================="
echo ""
echo "Results saved in:"
echo "  - Results/stage_c/Tg/"
echo "  - Results/stage_c/Tm/"
echo "  - Results/stage_c/Td/"
echo "  - Results/stage_c/Eg/"
echo "  - Results/stage_c/chi/"
echo ""
echo "To evaluate models, use:"
echo "  bash scripts/evaluate_all_properties.sh"
