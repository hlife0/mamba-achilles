#!/bin/bash
# Experiment 2c: Positional Encoding vs Standard Mamba
# Paper Section 4.2, Figure 4(c)
# Config: 2 layers, gamma=0.5, seed=42

set -e

echo "=== Experiment 2c: Positional Encoding ==="

# 1. Train standard Mamba (control)
echo ""
echo "--- Training standard Mamba (control) ---"
python src/2c_positional_encoding.py \
    --n_layers 2 --gamma 0.5 --seed 42 \
    --output_dir results/2c_positional_encoding/standard

# 2. Train Mamba with positional encoding
echo ""
echo "--- Training Mamba with positional encoding ---"
python src/2c_positional_encoding.py \
    --n_layers 2 --gamma 0.5 --seed 42 \
    --positional_encoding \
    --output_dir results/2c_positional_encoding/with_pe

# 3. Generate comparison plot
echo ""
echo "--- Generating comparison plot ---"
python src/2c_positional_encoding_visualize.py \
    --pe_dir results/2c_positional_encoding/with_pe \
    --standard_dir results/2c_positional_encoding/standard \
    --output results/2c_positional_encoding/comparison_plot.png

echo ""
echo "=== Experiment 2c complete ==="
