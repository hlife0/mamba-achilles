#!/usr/bin/env python3
"""
Results analysis utility.
Provides helpful summaries and statistics from experiment results.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_all_results(results_dir):
    """Load all metrics.json files from subdirectories."""
    results = []
    results_path = Path(results_dir)

    for subdir in results_path.iterdir():
        if not subdir.is_dir():
            continue

        metrics_file = subdir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    metrics['experiment_name'] = subdir.name
                    results.append(metrics)
            except Exception as e:
                print(f"Warning: Failed to load {metrics_file}: {e}")

    return results


def print_summary(results_dir):
    """Print summary statistics of all experiments."""
    results = load_all_results(results_dir)

    if len(results) == 0:
        print(f"No results found in {results_dir}")
        return

    df = pd.DataFrame(results)

    print("=" * 80)
    print(f"Results Summary: {results_dir}")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print()

    # Overall statistics
    print("Overall Statistics:")
    print(f"  Composite Accuracy: {df['final_composite_acc'].mean():.4f} ± {df['final_composite_acc'].std():.4f}")
    print(f"    Min: {df['final_composite_acc'].min():.4f}")
    print(f"    Max: {df['final_composite_acc'].max():.4f}")
    print()
    print(f"  Symmetric Accuracy: {df['final_symmetric_acc'].mean():.4f} ± {df['final_symmetric_acc'].std():.4f}")
    print(f"    Min: {df['final_symmetric_acc'].min():.4f}")
    print(f"    Max: {df['final_symmetric_acc'].max():.4f}")
    print()

    # Best configurations
    best_comp = df.loc[df['final_composite_acc'].idxmax()]
    print("Best Composite Accuracy:")
    print(f"  Experiment: {best_comp['experiment_name']}")
    print(f"  n_layers={best_comp['n_layers']}, gamma={best_comp['gamma']}, seed={best_comp['seed']}")
    print(f"  Composite: {best_comp['final_composite_acc']:.4f}")
    print(f"  Symmetric: {best_comp['final_symmetric_acc']:.4f}")
    print()

    best_sym = df.loc[df['final_symmetric_acc'].idxmax()]
    print("Best Symmetric Accuracy:")
    print(f"  Experiment: {best_sym['experiment_name']}")
    print(f"  n_layers={best_sym['n_layers']}, gamma={best_sym['gamma']}, seed={best_sym['seed']}")
    print(f"  Composite: {best_sym['final_composite_acc']:.4f}")
    print(f"  Symmetric: {best_sym['final_symmetric_acc']:.4f}")
    print()

    # Group by layers and gamma
    if 'n_layers' in df.columns and 'gamma' in df.columns:
        print("Grouped by (n_layers, gamma):")
        grouped = df.groupby(['n_layers', 'gamma']).agg({
            'final_composite_acc': ['mean', 'std'],
            'final_symmetric_acc': ['mean', 'std']
        }).round(4)
        print(grouped)
        print()


def export_csv(results_dir, output_file):
    """Export all results to a CSV file."""
    results = load_all_results(results_dir)

    if len(results) == 0:
        print(f"No results found in {results_dir}")
        return

    df = pd.DataFrame(results)

    # Select relevant columns
    columns = [
        'experiment_name', 'n_layers', 'gamma', 'seed',
        'final_composite_acc', 'final_symmetric_acc',
        'final_train_loss', 'best_composite_acc', 'best_composite_epoch'
    ]

    # Filter to only existing columns
    columns = [c for c in columns if c in df.columns]

    df[columns].to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")


def compare_experiments(results_dir, exp_names):
    """Compare specific experiments side-by-side."""
    results_path = Path(results_dir)

    print("=" * 80)
    print("Experiment Comparison")
    print("=" * 80)

    data = []
    for exp_name in exp_names:
        metrics_file = results_path / exp_name / "metrics.json"
        if not metrics_file.exists():
            print(f"Warning: {exp_name} not found")
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)
            metrics['experiment_name'] = exp_name
            data.append(metrics)

    if len(data) == 0:
        print("No valid experiments found")
        return

    df = pd.DataFrame(data)
    print(df[['experiment_name', 'n_layers', 'gamma', 'seed',
              'final_composite_acc', 'final_symmetric_acc']].to_string(index=False))


def plot_training_comparison(results_dir, exp_names, output_file):
    """Plot training curves for multiple experiments."""
    import matplotlib.pyplot as plt

    results_path = Path(results_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_name in exp_names:
        log_file = results_path / exp_name / "training_log.csv"
        if not log_file.exists():
            print(f"Warning: {exp_name} log not found")
            continue

        df = pd.read_csv(log_file)

        axes[0].plot(df['epoch'], df['train_loss'], label=exp_name, alpha=0.7)
        axes[1].plot(df['epoch'], df['composite_acc'], label=exp_name, alpha=0.7)
        axes[2].plot(df['epoch'], df['symmetric_acc'], label=exp_name, alpha=0.7)

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

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")

    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--mode", type=str, default="summary",
                       choices=["summary", "export", "compare", "plot"],
                       help="Analysis mode")
    parser.add_argument("--output", type=str, default="results.csv",
                       help="Output file (for export/plot modes)")
    parser.add_argument("--experiments", type=str,
                       help="Comma-separated experiment names (for compare/plot)")

    args = parser.parse_args()

    if args.mode == "summary":
        print_summary(args.results_dir)
    elif args.mode == "export":
        export_csv(args.results_dir, args.output)
    elif args.mode == "compare":
        if not args.experiments:
            print("Error: --experiments required for compare mode")
            return
        exp_names = args.experiments.split(',')
        compare_experiments(args.results_dir, exp_names)
    elif args.mode == "plot":
        if not args.experiments:
            print("Error: --experiments required for plot mode")
            return
        exp_names = args.experiments.split(',')
        plot_training_comparison(args.results_dir, exp_names, args.output)


if __name__ == "__main__":
    main()
