#!/bin/bash
#
# Quick validation script - runs a short training to verify everything works
#
# Usage:
#   bash experiments/0_train_validate.sh
#

set -e

echo "================================================"
echo "Validation Test"
echo "================================================"
echo "Running a quick 10-epoch training to validate setup..."
echo ""

OUTPUT_DIR="results/0_train_validation_test"
DEVICE="${DEVICE:-cuda}"

# Clean previous validation
rm -rf "${OUTPUT_DIR}"

# Run quick training
python src/0_train.py \
    --n_layers 2 \
    --gamma 1.0 \
    --seed 42 \
    --epochs 10 \
    --batch_size 512 \
    --output_dir "${OUTPUT_DIR}" \
    --log_interval 5 \
    --device ${DEVICE}

echo ""
echo "================================================"
echo "Validation Checks"
echo "================================================"

# Check output files
if [ -f "${OUTPUT_DIR}/config.json" ]; then
    echo "✓ config.json created"
else
    echo "✗ config.json missing"
    exit 1
fi

if [ -f "${OUTPUT_DIR}/training_log.csv" ]; then
    echo "✓ training_log.csv created"
    echo "  Log entries: $(tail -n +2 ${OUTPUT_DIR}/training_log.csv | wc -l)"
else
    echo "✗ training_log.csv missing"
    exit 1
fi

if [ -f "${OUTPUT_DIR}/model_final.pt" ]; then
    echo "✓ model_final.pt created"
    MODEL_SIZE=$(du -h "${OUTPUT_DIR}/model_final.pt" | cut -f1)
    echo "  Model size: ${MODEL_SIZE}"
else
    echo "✗ model_final.pt missing"
    exit 1
fi

if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
    echo "✓ metrics.json created"

    # Display metrics
    echo ""
    echo "Final metrics:"
    python -c "
import json
with open('${OUTPUT_DIR}/metrics.json') as f:
    metrics = json.load(f)
    print(f\"  Composite accuracy: {metrics['final_composite_acc']:.4f}\")
    print(f\"  Symmetric accuracy: {metrics['final_symmetric_acc']:.4f}\")
    print(f\"  Final train loss: {metrics['final_train_loss']:.4f}\")
"
else
    echo "✗ metrics.json missing"
    exit 1
fi

echo ""
echo "================================================"
echo "✓ All validation checks passed!"
echo "================================================"
echo ""
echo "System is ready for full experiments."
echo "Next steps:"
echo "  1. Run phase diagram: bash experiments/0_train_run_phase_diagram.sh"
echo "  2. Run ablations: bash experiments/run_ablations.sh"
