#!/bin/bash
# Evaluate all training stages and compare results
#
# Usage:
#   bash scripts/evaluate_all_stages.sh [num_samples] [num_steps]
#
# Example:
#   bash scripts/evaluate_all_stages.sh 1000 10

set -e

# Configuration
NUM_SAMPLES=${1:-1000}
NUM_STEPS=${2:-10}
VOCAB=${3:-"vocab.txt"}
CONFIG=${4:-"PolyDiffusion/configs/model_base.yaml"}

# Checkpoint paths (customize these)
CKPT_A=${CKPT_A:-"checkpoints/stage_a.pt"}
CKPT_B=${CKPT_B:-"checkpoints/stage_b.pt"}
CKPT_C=${CKPT_C:-"checkpoints/stage_c.pt"}

# Output directory
OUTPUT_DIR="evaluation_results"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "PolyDiffusion Stage Evaluation"
echo "============================================"
echo "Configuration:"
echo "  Samples: $NUM_SAMPLES"
echo "  Steps: $NUM_STEPS"
echo "  Vocabulary: $VOCAB"
echo "  Model Config: $CONFIG"
echo "============================================"
echo

# Function to evaluate a stage
evaluate_stage() {
    local stage=$1
    local ckpt=$2
    local output_file="${OUTPUT_DIR}/eval_stage_${stage}.json"

    if [ ! -f "$ckpt" ]; then
        echo "⚠️  Checkpoint not found: $ckpt"
        echo "   Skipping Stage $stage"
        return 1
    fi

    echo "📊 Evaluating Stage $stage..."
    echo "   Checkpoint: $ckpt"
    echo "   Output: $output_file"
    echo

    python -m PolyDiffusion.scripts.evaluate_stage \
        --stage "$stage" \
        --ckpt "$ckpt" \
        --vocab "$VOCAB" \
        --config "$CONFIG" \
        --num "$NUM_SAMPLES" \
        --steps "$NUM_STEPS" \
        --output "$output_file"

    return 0
}

# Evaluate each stage
EVALUATED_FILES=()

if evaluate_stage "a" "$CKPT_A"; then
    EVALUATED_FILES+=("${OUTPUT_DIR}/eval_stage_a.json")
fi

if evaluate_stage "b" "$CKPT_B"; then
    EVALUATED_FILES+=("${OUTPUT_DIR}/eval_stage_b.json")
fi

if evaluate_stage "c" "$CKPT_C"; then
    EVALUATED_FILES+=("${OUTPUT_DIR}/eval_stage_c.json")
fi

# Compare all evaluated stages
if [ ${#EVALUATED_FILES[@]} -gt 1 ]; then
    echo
    echo "============================================"
    echo "📈 Comparing Stages"
    echo "============================================"
    echo

    python -m PolyDiffusion.scripts.evaluate_stage \
        --compare "${EVALUATED_FILES[@]}"
else
    echo
    echo "⚠️  Need at least 2 stages to compare"
fi

echo
echo "============================================"
echo "✅ Evaluation Complete"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR/"
echo
echo "To view individual results:"
for file in "${EVALUATED_FILES[@]}"; do
    echo "  cat $file"
done
echo
