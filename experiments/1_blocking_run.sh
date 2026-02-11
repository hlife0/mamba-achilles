#!/bin/bash
# Information Blocking Experiment (1_blocking)
# Corresponds to paper Section 4.1, Figure 6(a)
#
# Prerequisites:
#   - A trained Mamba checkpoint that learned composite solution
#   - Recommended: L5_G1.0 from 0_train phase diagram
#
# Usage:
#   bash experiments/1_blocking_run.sh <checkpoint_dir>
#   e.g.: bash experiments/1_blocking_run.sh results/0_train_phase_diagram/L5_G1.0_S42

set -e

CHECKPOINT_DIR="${1:?Usage: $0 <checkpoint_dir>}"
CHECKPOINT="${CHECKPOINT_DIR}/model_final.pt"
CONFIG="${CHECKPOINT_DIR}/config.json"
OUTPUT_DIR="results/1_blocking"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found: $CONFIG"
    exit 1
fi

echo "=== Information Blocking Experiment ==="
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

python src/1_blocking.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --num_samples 480 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "${OUTPUT_DIR}/stdout.log"

echo ""
echo "=== Done ==="
echo "Results: ${OUTPUT_DIR}/blocking_results.json"
echo "Plot: ${OUTPUT_DIR}/blocking_bar_chart.png"
