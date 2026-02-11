#!/usr/bin/env python3
"""
Experiment 2d: Conv1d kernel cosine similarity analysis.

Analyzes the asymmetry of Conv1d kernel parameters by computing cosine
similarity between the 4 position vectors at epoch 0 and epoch final.

Corresponds to paper Section 4.2 + Appendix "Cosine similarity among
convolution weights", Figure sup_conv1d_cos.

Usage:
    python src/2d_conv1d_cosine.py \
        --checkpoint results/0_train_phase_diagram/L2_G1.0/model_final.pt \
        --config results/0_train_phase_diagram/L2_G1.0/config.json \
        --output_dir results/2d_conv1d_cosine
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.mamba_wrapper import MambaForComposite


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_conv1d_weights(model):
    """Extract Conv1d weights from all layers.

    Returns:
        list of tensors, each (d_inner, 1, kernel_size)
    """
    weights = []
    for layer in model.backbone.backbone.layers:
        w = layer.mixer.conv1d.weight.detach().clone()
        weights.append(w)
    return weights


def compute_cosine_similarity(conv_weight):
    """Compute cosine similarity matrix between kernel position vectors.

    Args:
        conv_weight: (d_inner, 1, kernel_size)

    Returns:
        (kernel_size, kernel_size) numpy array
    """
    kernel_size = conv_weight.shape[2]
    vectors = [conv_weight[:, 0, i].unsqueeze(0) for i in range(kernel_size)]

    sim = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            sim[i, j] = F.cosine_similarity(vectors[i], vectors[j]).item()
    return sim


def plot_heatmaps(sim_epoch0, sim_final, final_epoch, output_path):
    """Plot cosine similarity heatmaps for all layers at epoch 0 and final."""
    n_layers = len(sim_epoch0)
    fig, axes = plt.subplots(n_layers, 2, figsize=(8, 3.5 * n_layers))

    if n_layers == 1:
        axes = axes.reshape(1, 2)

    for layer_idx in range(n_layers):
        for col, (sim, label) in enumerate([
            (sim_epoch0[layer_idx], 'Epoch 0'),
            (sim_final[layer_idx], f'Epoch {final_epoch}'),
        ]):
            ax = axes[layer_idx, col]
            im = ax.imshow(sim, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
            ax.set_title(f'Layer {layer_idx} - {label}', fontsize=12)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels([f'c{i}' for i in range(4)])
            ax.set_yticklabels([f'c{i}' for i in range(4)])

            for i in range(4):
                for j in range(4):
                    ax.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                            fontsize=9, color='black' if abs(sim[i, j]) < 0.5 else 'white')

    fig.suptitle('Conv1d Kernel Cosine Similarity', fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Cosine Similarity',
                 pad=0.04)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2d: Conv1d Cosine Similarity")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.json')
    parser.add_argument('--output_dir', type=str, default='results/2d_conv1d_cosine')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = json.load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = config.get('seed', 42)

    # --- Epoch 0: re-initialize model with same seed to get initial weights ---
    set_seed(seed)
    model_epoch0 = MambaForComposite(
        n_layers=config['n_layers'],
        gamma=config['gamma'],
        vocab_size=config.get('vocab_size', 100),
        d_model=config.get('d_model', 32),
        d_state=config.get('d_state', 128),
        d_conv=config.get('d_conv', 4),
        device=device,
    )
    weights_epoch0 = extract_conv1d_weights(model_epoch0)
    sim_epoch0 = [compute_cosine_similarity(w) for w in weights_epoch0]
    print(f"Epoch 0 cosine similarities computed ({len(sim_epoch0)} layers)")

    # --- Epoch final: load trained checkpoint ---
    model_final = MambaForComposite(
        n_layers=config['n_layers'],
        gamma=config['gamma'],
        vocab_size=config.get('vocab_size', 100),
        d_model=config.get('d_model', 32),
        d_state=config.get('d_state', 128),
        d_conv=config.get('d_conv', 4),
        device=device,
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model_final.load_state_dict(state_dict)
    weights_final = extract_conv1d_weights(model_final)
    sim_final = [compute_cosine_similarity(w) for w in weights_final]

    final_epoch = config.get('epochs', 200)
    print(f"Epoch {final_epoch} cosine similarities computed")

    # --- Save numerical results ---
    results = {
        'checkpoint': args.checkpoint,
        'config': config,
        'n_layers': config['n_layers'],
        'gamma': config['gamma'],
        'seed': seed,
        'epoch0': {f'layer_{i}': sim.tolist() for i, sim in enumerate(sim_epoch0)},
        'epoch_final': {f'layer_{i}': sim.tolist() for i, sim in enumerate(sim_final)},
        'final_epoch': final_epoch,
    }

    results_path = os.path.join(args.output_dir, 'cosine_similarity.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

    # --- Save config ---
    exp_config = {
        'experiment': '2d_conv1d_cosine',
        'checkpoint': args.checkpoint,
        'training_config': config,
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(exp_config, f, indent=4)

    # --- Plot ---
    plot_heatmaps(sim_epoch0, sim_final, final_epoch,
                  os.path.join(args.output_dir, 'conv1d_cosine_heatmap.png'))

    # --- Print summary ---
    print("\nSummary:")
    for i in range(config['n_layers']):
        off_diag_0 = [sim_epoch0[i][r][c] for r in range(4) for c in range(4) if r != c]
        off_diag_f = [sim_final[i][r][c] for r in range(4) for c in range(4) if r != c]
        print(f"  Layer {i}: epoch0 off-diag mean={np.mean(off_diag_0):.4f}, "
              f"final off-diag mean={np.mean(off_diag_f):.4f}")

    print("\nExperiment 2d completed.")


if __name__ == '__main__':
    main()
