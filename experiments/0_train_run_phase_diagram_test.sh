#!/bin/bash
#
# Test phase diagram experiment sweep (small subset)
# Runs 6 experiments: 2 layers × 3 gammas × 1 seed
#
# Usage:
#   bash experiments/run_phase_diagram_test.sh
#

set -e  # Exit on error

# Configuration
OUTPUT_ROOT="results/0_train_phase_diagram_test"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"

# Parameter grid (small test subset)
LAYERS=(2)
GAMMAS=(0.5 0.8 1.0)
SEEDS=(42)

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#LAYERS[@]} * ${#GAMMAS[@]} * ${#SEEDS[@]}))
CURRENT=0

echo "================================================"
echo "Phase Diagram Test Sweep (Small Subset)"
echo "================================================"
echo "Layers: ${LAYERS[@]}"
echo "Gammas: ${GAMMAS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo "Total experiments: ${TOTAL_EXPERIMENTS}"
echo "Output directory: ${OUTPUT_ROOT}"
echo "================================================"
echo ""

# Create output directory
mkdir -p "${OUTPUT_ROOT}"

# Log file
LOG_FILE="${OUTPUT_ROOT}/sweep.log"
echo "Starting sweep at $(date)" > "${LOG_FILE}"

# Main loop
for n_layers in "${LAYERS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))

            # Create run name
            RUN_NAME="L${n_layers}_G${gamma}_S${seed}"
            OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

            echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: ${RUN_NAME}"
            echo "  n_layers=${n_layers}, gamma=${gamma}, seed=${seed}"

            # Skip if already completed
            if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
                echo "  ✓ Already completed, skipping"
                echo "[$CURRENT/$TOTAL_EXPERIMENTS] ${RUN_NAME} - SKIPPED (already done)" >> "${LOG_FILE}"
                continue
            fi

            # Create output directory
            mkdir -p "${OUTPUT_DIR}"

            # Run training
            START_TIME=$(date +%s)

            if python src/0_train.py \
                --n_layers ${n_layers} \
                --gamma ${gamma} \
                --seed ${seed} \
                --epochs ${EPOCHS} \
                --batch_size ${BATCH_SIZE} \
                --output_dir "${OUTPUT_DIR}" \
                --device ${DEVICE} \
                > "${OUTPUT_DIR}/stdout.log" 2> "${OUTPUT_DIR}/stderr.log"; then

                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))

                echo "  ✓ Completed in ${DURATION}s"
                echo "[$CURRENT/$TOTAL_EXPERIMENTS] ${RUN_NAME} - SUCCESS (${DURATION}s)" >> "${LOG_FILE}"
            else
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))

                echo "  ✗ Failed after ${DURATION}s"
                echo "[$CURRENT/$TOTAL_EXPERIMENTS] ${RUN_NAME} - FAILED (${DURATION}s)" >> "${LOG_FILE}"

                # Continue with next experiment instead of exiting
                echo "  See ${OUTPUT_DIR}/stderr.log for details"
            fi

            echo ""
        done
    done
done

echo "================================================"
echo "Test Sweep completed at $(date)"
echo "================================================"
echo "Test Sweep completed at $(date)" >> "${LOG_FILE}"

# Generate summary
echo ""
echo "Summary:"
COMPLETED=$(find "${OUTPUT_ROOT}" -name "metrics.json" | wc -l)
echo "  Completed: ${COMPLETED}/${TOTAL_EXPERIMENTS}"

if [ ${COMPLETED} -eq ${TOTAL_EXPERIMENTS} ]; then
    echo "  Status: ✓ All test experiments completed successfully"
    echo ""
    echo "Ready to run full sweep with:"
    echo "  bash experiments/0_train_run_phase_diagram.sh"
else
    echo "  Status: ✗ Some experiments failed or are incomplete"
    echo "  Check ${LOG_FILE} for details"
fi

echo ""
echo "Log file: ${LOG_FILE}"
