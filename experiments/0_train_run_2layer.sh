#!/bin/bash

# Train all 2-layer configurations
# 6 gamma values × 3 seeds = 18 experiments

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mamba2

GAMMA_VALUES=(0.5 0.6 0.7 0.8 0.9 1.0)
SEEDS=(42 123 456)
N_LAYERS=2

cd /root/mamba-achilles

for gamma in "${GAMMA_VALUES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "========================================"
        echo "Training: n_layers=${N_LAYERS}, gamma=${gamma}, seed=${seed}"
        echo "========================================"

        python src/0_train.py \
            --n_layers ${N_LAYERS} \
            --gamma ${gamma} \
            --seed ${seed} \
            --num_train_samples 300000 \
            --num_eval_samples 1800 \
            --batch_size 2048 \
            --epochs 200 \
            --initial_lr 1e-5 \
            --warmup_target_lr 2.5e-4 \
            --warmup_epochs 20 \
            --output_dir results/0_train_phase_diagram

        if [ $? -ne 0 ]; then
            echo "ERROR: Training failed for n_layers=${N_LAYERS}, gamma=${gamma}, seed=${seed}"
            exit 1
        fi

        echo ""
    done
done

echo "All 2-layer training completed successfully!"
