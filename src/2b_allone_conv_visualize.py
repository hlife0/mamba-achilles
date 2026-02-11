#!/usr/bin/env python3
"""
Visualize Experiment 2b: comparison of all-one Conv1d vs standard Mamba.

Generates Figure 4(b) — training curves of composite and symmetric accuracy
for both 1Conv1d and standard structure.

Usage:
    python src/2b_allone_conv_visualize.py \
        --allone_dir results/2b_allone_conv/allone \
        --standard_dir results/2b_allone_conv/standard \
        --output results/2b_allone_conv/comparison_plot.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allone_dir", type=str, required=True)
    parser.add_argument("--standard_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/2b_allone_conv/comparison_plot.png")
    args = parser.parse_args()

    allone_log = pd.read_csv(os.path.join(args.allone_dir, "training_log.csv"))
    standard_log = pd.read_csv(os.path.join(args.standard_dir, "training_log.csv"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Composite accuracy
    ax1.plot(standard_log['epoch'], standard_log['composite_acc'] * 100,
             'b-', label='Standard - comp.', linewidth=1.5)
    ax1.plot(standard_log['epoch'], standard_log['symmetric_acc'] * 100,
             'b--', label='Standard - sym.', linewidth=1.5)
    ax1.plot(allone_log['epoch'], allone_log['composite_acc'] * 100,
             'r-', label='1Conv1d - comp.', linewidth=1.5)
    ax1.plot(allone_log['epoch'], allone_log['symmetric_acc'] * 100,
             'r--', label='1Conv1d - sym.', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('1Conv1d vs Standard Structure', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)

    # Right: Training loss
    ax2.plot(standard_log['epoch'], standard_log['train_loss'],
             'b-', label='Standard', linewidth=1.5)
    ax2.plot(allone_log['epoch'], allone_log['train_loss'],
             'r-', label='1Conv1d', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Training Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"Comparison plot saved to {args.output}")


if __name__ == "__main__":
    main()
