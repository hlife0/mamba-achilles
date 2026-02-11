#!/bin/bash
# Run information substitution experiment
# Uses the best composite model checkpoint

set -e

CHECKPOINT="results/0_train_paper_best/model_best_comp.pt"
CONFIG="results/0_train_paper_best/config.json"
OUTPUT_DIR="results/1_substitution"

echo "=== Information Substitution Experiment ==="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

python src/1_substitution.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --num_samples 480 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR"
