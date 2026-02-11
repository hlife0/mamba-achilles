#!/bin/bash
# Experiment 2b: All-one Conv1d vs Standard Mamba
# Paper Section 4.2, Figure 4(b)
# Config: 5 layers, gamma=1.0, seed=42

set -e

echo "=== Experiment 2b: All-one Conv1d ==="

# 1. Train standard Mamba (control)
echo ""
echo "--- Training standard Mamba (control) ---"
python src/2b_allone_conv.py \
    --n_layers 5 --gamma 1.0 --seed 42 \
    --output_dir results/2b_allone_conv/standard

# 2. Train all-one Conv1d Mamba
echo ""
echo "--- Training all-one Conv1d Mamba ---"
python src/2b_allone_conv.py \
    --n_layers 5 --gamma 1.0 --seed 42 \
    --allone_conv \
    --output_dir results/2b_allone_conv/allone

# 3. Generate comparison plot
echo ""
echo "--- Generating comparison plot ---"
python src/2b_allone_conv_visualize.py \
    --allone_dir results/2b_allone_conv/allone \
    --standard_dir results/2b_allone_conv/standard \
    --output results/2b_allone_conv/comparison_plot.png

echo ""
echo "=== Experiment 2b complete ==="
