"""
Information Substitution experiment (1_substitution).

Verifies that Mamba's Conv1d has encoded sufficient information to complete
the composite function task. For sequences with anchor pairs other than (4,3),
replaces the post-Conv1d hidden states of downstream tokens with those from
a reference (4,3) sequence (identical except for anchor pair). If outputs
collapse to the (4,3) prediction, Conv1d alone carries the key information.

Corresponds to paper Section 4.1, Figure 6(b).

Usage:
    python src/1_substitution.py \
        --checkpoint results/0_train_paper_best/model_best_comp.pt \
        --config results/0_train_paper_best/config.json \
        --output_dir results/1_substitution
"""

import argparse
import json
import os
import random
import sys

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models.mamba_wrapper import MambaForComposite
from src.models.mamba_hooks import (
    disable_fused_path,
    patch_model_for_post_conv_hook,
    restore_model,
)


ANCHOR_PAIRS = [(i, j) for i in range(1, 5) for j in range(1, 5)]


def generate_paired_sequences(anchor_pair, num_samples, seed):
    """
    Generate paired target and reference (4,3) sequences.

    Both sequences are identical except for the anchor pair positions.

    Args:
        anchor_pair: (int, int), e.g. (1, 2)
        num_samples: number of sequence pairs
        seed: random seed

    Returns:
        target_sequences: torch.LongTensor, shape=(num_samples, 8)
        ref_sequences: torch.LongTensor, shape=(num_samples, 8)
        key_positions: list of int
    """
    rng = random.Random(seed)
    target_seqs = []
    ref_seqs = []
    key_positions = []

    for _ in range(num_samples):
        pos = rng.randint(0, 5)
        key = rng.randint(20, 99)

        base = [0] * 8
        for i in range(8):
            if i not in (pos, pos + 1, pos + 2):
                val = rng.randint(5, 99)
                while val in (1, 2, 3, 4):
                    val = rng.randint(5, 99)
                base[i] = val

        base[pos] = key

        target = base.copy()
        target[pos + 1] = anchor_pair[0]
        target[pos + 2] = anchor_pair[1]

        ref = base.copy()
        ref[pos + 1] = 4
        ref[pos + 2] = 3

        target_seqs.append(target)
        ref_seqs.append(ref)
        key_positions.append(pos)

    return (torch.tensor(target_seqs, dtype=torch.long),
            torch.tensor(ref_seqs, dtype=torch.long),
            key_positions)


def run_substitution_experiment(model, target_seqs, ref_seqs, key_positions, device):
    """
    Run substitution experiment for a batch of paired sequences.

    1. Forward ref sequences, capture post-conv1d x at each layer.
    2. Forward target sequences, substitute downstream x with ref values.

    Args:
        model: MambaForComposite (eval mode, fused path disabled)
        target_seqs: (N, 8) tensor
        ref_seqs: (N, 8) tensor
        key_positions: list of int
        device: torch device

    Returns:
        ref_preds: (N,) tensor — predictions from ref sequences
        sub_preds: (N,) tensor — predictions from target with substitution
    """
    target_seqs = target_seqs.to(device)
    ref_seqs = ref_seqs.to(device)

    n_layers = len(model.backbone.backbone.layers)

    # --- Pass 1: forward ref sequences, capture post-conv x per layer ---
    captured = []

    def make_capture_hook(storage):
        def hook(x):
            # x: (batch, d_inner, seqlen)
            storage.append(x.detach().clone())
            return x
        return hook

    capture_hooks_per_layer = []
    for _ in range(n_layers):
        layer_storage = []
        captured.append(layer_storage)
        capture_hooks_per_layer.append(make_capture_hook(layer_storage))

    originals_capture = patch_model_for_post_conv_hook(model, capture_hooks_per_layer)
    with torch.no_grad():
        ref_preds = model(ref_seqs).argmax(dim=-1)
    restore_model(model, originals_capture)

    ref_x_per_layer = [layer_storage[0] for layer_storage in captured]

    # --- Pass 2: forward target sequences, substitute downstream positions ---
    # Group by key_position for correct masking
    positions_to_indices = {}
    for idx, pos in enumerate(key_positions):
        positions_to_indices.setdefault(pos, []).append(idx)

    sub_preds = torch.zeros_like(ref_preds)

    for pos, indices in positions_to_indices.items():
        batch_target = target_seqs[indices]
        batch_ref_x = [layer_x[indices] for layer_x in ref_x_per_layer]

        def make_sub_hook(ref_x, start_pos):
            def hook(x):
                # x: (batch, d_inner, seqlen)
                modified = x.clone()
                modified[:, :, start_pos:] = ref_x[:, :, start_pos:]
                return modified
            return hook

        # Substitute from anchor1 position (pos+1) onward
        sub_hooks = [make_sub_hook(bx, pos + 1) for bx in batch_ref_x]

        originals_sub = patch_model_for_post_conv_hook(model, sub_hooks)
        with torch.no_grad():
            batch_preds = model(batch_target).argmax(dim=-1)
        restore_model(model, originals_sub)

        for i, idx in enumerate(indices):
            sub_preds[idx] = batch_preds[i]

    return ref_preds, sub_preds


def plot_bar_chart(results, output_dir):
    """
    Plot substitution experiment bar chart (Figure 6b).

    Args:
        results: dict mapping pair_name -> collapse_rate
        output_dir: directory to save the plot
    """
    pair_names = [f"{a1}{a2}" for a1 in range(1, 5) for a2 in range(1, 5)]
    rates = [results[name] * 100 for name in pair_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(pair_names)), rates, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels(pair_names, fontsize=10)
    ax.set_xlabel('Anchor Pair', fontsize=12)
    ax.set_ylabel('Collapse Rate (%)', fontsize=12)
    ax.set_title('Accuracy under Information Substitution', fontsize=14)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%')
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'substitution_bar_chart.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Bar chart saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Information Substitution Experiment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.json from training')
    parser.add_argument('--num_samples', type=int, default=480,
                        help='Number of samples per anchor pair')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/1_substitution')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = json.load(f)
    print(f"Config: {config}")

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MambaForComposite(
        n_layers=config['n_layers'],
        gamma=config['gamma'],
        vocab_size=config.get('vocab_size', 100),
        d_model=config.get('d_model', 32),
        d_state=config.get('d_state', 128),
        d_conv=config.get('d_conv', 4),
        expand=config.get('expand', 2),
        device=device,
    )
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {args.checkpoint}")

    # Disable fused path for hooking
    disable_fused_path(model)

    # Run experiment for each anchor pair
    results = {}
    for a1 in range(1, 5):
        for a2 in range(1, 5):
            pair = (a1, a2)
            pair_name = f"{a1}{a2}"

            target_seqs, ref_seqs, key_positions = generate_paired_sequences(
                pair, args.num_samples, args.seed + a1 * 10 + a2
            )

            ref_preds, sub_preds = run_substitution_experiment(
                model, target_seqs, ref_seqs, key_positions, device
            )

            collapse_rate = (ref_preds == sub_preds).float().mean().item()
            results[pair_name] = collapse_rate
            print(f"  Pair {pair_name}: collapse_rate = {collapse_rate:.4f}")

    # Save results
    output = {
        'checkpoint': args.checkpoint,
        'config': config,
        'num_samples_per_pair': args.num_samples,
        'seed': args.seed,
        'results': {k: {'collapse_rate': v, 'num_samples': args.num_samples}
                    for k, v in results.items()},
        'mean_collapse_rate': sum(results.values()) / len(results),
    }

    results_path = os.path.join(args.output_dir, 'substitution_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to {results_path}")
    print(f"Mean collapse rate: {output['mean_collapse_rate']:.4f}")

    # Save experiment config
    exp_config = {
        'experiment': '1_substitution',
        'checkpoint': args.checkpoint,
        'training_config': config,
        'num_samples_per_pair': args.num_samples,
        'seed': args.seed,
    }
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=4)

    # Plot
    plot_bar_chart(results, args.output_dir)

    # Sanity check
    if '43' in results:
        rate_43 = results['43']
        if rate_43 < 0.99:
            print(f"\nWARNING: Pair (4,3) collapse_rate = {rate_43:.4f}, expected ~1.0")
        else:
            print(f"\nSanity check passed: Pair (4,3) collapse_rate = {rate_43:.4f}")

    print("\nSubstitution experiment completed.")


if __name__ == '__main__':
    main()
