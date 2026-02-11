#!/bin/bash
#
# Phase diagram experiment sweep
# Runs 72 experiments: 6 layers × 12 gammas × 1 seed
#
# Usage:
#   bash experiments/0_train_run_phase_diagram.sh
#
# Or with custom output directory:
#   OUTPUT_ROOT=/path/to/results bash experiments/0_train_run_phase_diagram.sh
#

set -e  # Exit on error

# Configuration
OUTPUT_ROOT="${OUTPUT_ROOT:-results/0_train_phase_diagram}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
DEVICE="${DEVICE:-cuda}"
SEED=42

# Parameter grid (from implementation_guide.md section 4.2)
LAYERS=(2 3 4 5 6 7)
GAMMAS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.5 2.0)

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#LAYERS[@]} * ${#GAMMAS[@]}))
CURRENT=0

echo "================================================"
echo "Phase Diagram Experiment Sweep"
echo "================================================"
echo "Layers: ${LAYERS[@]}"
echo "Gammas: ${GAMMAS[@]}"
echo "Seed: ${SEED}"
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
        CURRENT=$((CURRENT + 1))

        # Create run name
        RUN_NAME="L${n_layers}_G${gamma}"
        OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

        echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: ${RUN_NAME}"
        echo "  n_layers=${n_layers}, gamma=${gamma}, seed=${SEED}"

        # Skip if already completed
        if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
            echo "  Already completed, skipping"
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
            --seed ${SEED} \
            --epochs ${EPOCHS} \
            --batch_size ${BATCH_SIZE} \
            --output_dir "${OUTPUT_DIR}" \
            --device ${DEVICE} \
            > "${OUTPUT_DIR}/stdout.log" 2> "${OUTPUT_DIR}/stderr.log"; then

            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            echo "  Completed in ${DURATION}s"
            echo "[$CURRENT/$TOTAL_EXPERIMENTS] ${RUN_NAME} - SUCCESS (${DURATION}s)" >> "${LOG_FILE}"
        else
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))

            echo "  Failed after ${DURATION}s"
            echo "[$CURRENT/$TOTAL_EXPERIMENTS] ${RUN_NAME} - FAILED (${DURATION}s)" >> "${LOG_FILE}"

            # Continue with next experiment instead of exiting
            echo "  See ${OUTPUT_DIR}/stderr.log for details"
        fi

        echo ""
    done
done

echo "================================================"
echo "Sweep completed at $(date)"
echo "================================================"
echo "Sweep completed at $(date)" >> "${LOG_FILE}"

# Generate summary
echo ""
echo "Summary:"
COMPLETED=$(find "${OUTPUT_ROOT}" -name "metrics.json" | wc -l)
echo "  Completed: ${COMPLETED}/${TOTAL_EXPERIMENTS}"

if [ ${COMPLETED} -eq ${TOTAL_EXPERIMENTS} ]; then
    echo "  Status: All experiments completed successfully"

    # Generate visualization
    echo ""
    echo "Generating phase diagram visualization..."
    if python src/0_train_visualize.py \
        --results_dir "${OUTPUT_ROOT}" \
        --output "${OUTPUT_ROOT}/phase_diagram.png"; then
        echo "  Visualization saved to ${OUTPUT_ROOT}/phase_diagram.png"
    else
        echo "  Visualization failed"
    fi
else
    echo "  Status: Some experiments failed or are incomplete"
    echo "  Check ${LOG_FILE} for details"
fi

echo ""
echo "Log file: ${LOG_FILE}"
