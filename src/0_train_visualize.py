#!/usr/bin/env python3
"""
Visualization script for phase diagram experiments.
Implements the visualization protocol from implementation_guide.md section 6.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(results_dir):
    """
    Load all experiment results from a directory.

    Args:
        results_dir: Path to directory containing experiment subdirectories

    Returns:
        results: List of dicts, each containing metrics from one experiment
    """
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    # Iterate through all subdirectories
    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue

        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: No metrics.json found in {subdir.name}, skipping")
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to load {metrics_file}: {e}")

    return results


def aggregate_results(results, n_layers_list, gamma_list):
    """
    Aggregate results across seeds for each (n_layers, gamma) combination.

    Args:
        results: List of result dicts
        n_layers_list: List of layer counts
        gamma_list: List of gamma values

    Returns:
        composite_mean: 2D numpy array (len(gamma_list), len(n_layers_list))
        composite_std: 2D numpy array (same shape)
        symmetric_mean: 2D numpy array (same shape)
        symmetric_std: 2D numpy array (same shape)
    """
    # Initialize arrays
    composite_mean = np.zeros((len(gamma_list), len(n_layers_list)))
    composite_std = np.zeros((len(gamma_list), len(n_layers_list)))
    symmetric_mean = np.zeros((len(gamma_list), len(n_layers_list)))
    symmetric_std = np.zeros((len(gamma_list), len(n_layers_list)))

    # Group results by (n_layers, gamma)
    for i, gamma in enumerate(gamma_list):
        for j, n_layers in enumerate(n_layers_list):
            # Find all results for this combination
            matching_results = [
                r for r in results
                if r['n_layers'] == n_layers and abs(r['gamma'] - gamma) < 1e-6
            ]

            if len(matching_results) == 0:
                print(f"Warning: No results found for n_layers={n_layers}, gamma={gamma}")
                composite_mean[i, j] = np.nan
                composite_std[i, j] = np.nan
                symmetric_mean[i, j] = np.nan
                symmetric_std[i, j] = np.nan
                continue

            # Extract accuracies
            composite_accs = [r['final_composite_acc'] for r in matching_results]
            symmetric_accs = [r['final_symmetric_acc'] for r in matching_results]

            # Compute mean and std
            composite_mean[i, j] = np.mean(composite_accs)
            composite_std[i, j] = np.std(composite_accs)
            symmetric_mean[i, j] = np.mean(symmetric_accs)
            symmetric_std[i, j] = np.std(symmetric_accs)

    return composite_mean, composite_std, symmetric_mean, symmetric_std


def plot_phase_diagram(results_dir, output_file, n_layers_list=None, gamma_list=None):
    """
    Generate phase diagram visualization.

    Args:
        results_dir: Directory containing all experiment results
        output_file: Path to save output PNG
        n_layers_list: List of layer counts (default: [2,3,4,5,6])
        gamma_list: List of gamma values (default: [0.5,0.6,0.7,0.8,0.9,1.0])
    """
    # Default parameter ranges
    if n_layers_list is None:
        n_layers_list = [2, 3, 4, 5, 6]
    if gamma_list is None:
        gamma_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} experiment results")

    print("Aggregating results across seeds...")
    composite_mean, composite_std, symmetric_mean, symmetric_std = aggregate_results(
        results, n_layers_list, gamma_list
    )

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Configure heatmap parameters
    cmap = "YlOrRd"
    vmin, vmax = 0.0, 1.0

    # Plot composite accuracy
    ax1 = axes[0]
    sns.heatmap(
        composite_mean,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Accuracy'},
        xticklabels=n_layers_list,
        yticklabels=[f'{g:.1f}' for g in gamma_list],
        ax=ax1
    )
    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Gamma (γ)', fontsize=12)
    ax1.set_title('Composite Accuracy', fontsize=14, fontweight='bold')

    # Plot symmetric accuracy
    ax2 = axes[1]
    sns.heatmap(
        symmetric_mean,
        annot=True,
        fmt='.3f',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Accuracy'},
        xticklabels=n_layers_list,
        yticklabels=[f'{g:.1f}' for g in gamma_list],
        ax=ax2
    )
    ax2.set_xlabel('Number of Layers', fontsize=12)
    ax2.set_ylabel('Gamma (γ)', fontsize=12)
    ax2.set_title('Symmetric Accuracy', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Phase diagram saved to {output_file}")

    # Also save the data
    data_file = output_file.replace('.png', '_data.npz')
    np.savez(
        data_file,
        composite_mean=composite_mean,
        composite_std=composite_std,
        symmetric_mean=symmetric_mean,
        symmetric_std=symmetric_std,
        n_layers_list=n_layers_list,
        gamma_list=gamma_list
    )
    print(f"Data saved to {data_file}")

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Composite Accuracy: mean={np.nanmean(composite_mean):.3f}, "
          f"min={np.nanmin(composite_mean):.3f}, max={np.nanmax(composite_mean):.3f}")
    print(f"Symmetric Accuracy: mean={np.nanmean(symmetric_mean):.3f}, "
          f"min={np.nanmin(symmetric_mean):.3f}, max={np.nanmax(symmetric_mean):.3f}")

    # Find best configurations
    if not np.all(np.isnan(composite_mean)):
        best_comp_idx = np.unravel_index(np.nanargmax(composite_mean), composite_mean.shape)
        best_comp_gamma = gamma_list[best_comp_idx[0]]
        best_comp_layers = n_layers_list[best_comp_idx[1]]
        print(f"\nBest Composite: γ={best_comp_gamma:.1f}, L={best_comp_layers}, "
              f"acc={composite_mean[best_comp_idx]:.3f}")

    if not np.all(np.isnan(symmetric_mean)):
        best_sym_idx = np.unravel_index(np.nanargmax(symmetric_mean), symmetric_mean.shape)
        best_sym_gamma = gamma_list[best_sym_idx[0]]
        best_sym_layers = n_layers_list[best_sym_idx[1]]
        print(f"Best Symmetric: γ={best_sym_gamma:.1f}, L={best_sym_layers}, "
              f"acc={symmetric_mean[best_sym_idx]:.3f}")


def plot_training_curves(results_dir, output_file, n_layers=2, gamma=1.0):
    """
    Plot training curves for a specific configuration across seeds.

    Args:
        results_dir: Directory containing experiment results
        output_file: Path to save output PNG
        n_layers: Number of layers
        gamma: Gamma value
    """
    import pandas as pd

    results_path = Path(results_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    seeds_found = []

    # Find all matching experiments
    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue

        # Check if this matches our configuration
        metrics_file = subdir / "metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            if metrics['n_layers'] != n_layers or abs(metrics['gamma'] - gamma) > 1e-6:
                continue

            # Load training log
            log_file = subdir / "training_log.csv"
            if not log_file.exists():
                continue

            df = pd.read_csv(log_file)
            seed = metrics['seed']
            seeds_found.append(seed)

            # Plot curves
            axes[0].plot(df['epoch'], df['train_loss'], label=f'Seed {seed}', alpha=0.7)
            axes[1].plot(df['epoch'], df['composite_acc'], label=f'Seed {seed}', alpha=0.7)
            axes[2].plot(df['epoch'], df['symmetric_acc'], label=f'Seed {seed}', alpha=0.7)

        except Exception as e:
            print(f"Warning: Failed to process {subdir.name}: {e}")

    if len(seeds_found) == 0:
        print(f"No results found for n_layers={n_layers}, gamma={gamma}")
        return

    # Configure plots
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Composite Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Symmetric Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Training Curves: L={n_layers}, γ={gamma}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize phase diagram experiments")

    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output", type=str, default="phase_diagram.png",
                       help="Output file path")
    parser.add_argument("--mode", type=str, default="phase_diagram",
                       choices=["phase_diagram", "training_curves"],
                       help="Visualization mode")

    # For training curves
    parser.add_argument("--n_layers", type=int, default=2,
                       help="Number of layers (for training curves)")
    parser.add_argument("--gamma", type=float, default=1.0,
                       help="Gamma value (for training curves)")

    # Custom parameter ranges
    parser.add_argument("--layers", type=str, default="2,3,4,5,6",
                       help="Comma-separated list of layer counts")
    parser.add_argument("--gammas", type=str, default="0.5,0.6,0.7,0.8,0.9,1.0",
                       help="Comma-separated list of gamma values")

    args = parser.parse_args()

    if args.mode == "phase_diagram":
        # Parse parameter lists
        n_layers_list = [int(x) for x in args.layers.split(',')]
        gamma_list = [float(x) for x in args.gammas.split(',')]

        plot_phase_diagram(
            args.results_dir,
            args.output,
            n_layers_list=n_layers_list,
            gamma_list=gamma_list
        )
    elif args.mode == "training_curves":
        plot_training_curves(
            args.results_dir,
            args.output,
            n_layers=args.n_layers,
            gamma=args.gamma
        )


if __name__ == "__main__":
    main()
