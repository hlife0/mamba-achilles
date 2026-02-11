"""
Information Blocking experiment (1_blocking).

Verifies that Mamba's SSM module does not transmit key/anchor information
to downstream tokens. By zeroing out the SSM state contributions from
key and anchor positions, we show that model predictions remain unchanged,
demonstrating that information flows primarily through Conv1d.

Corresponds to paper Section 4.1, Figure 6(a).

Usage:
    python src/1_blocking.py \
        --checkpoint results/0_train_phase_diagram/L5_G1.0_S42/model_final.pt \
        --config results/0_train_phase_diagram/L5_G1.0_S42/config.json \
        --output_dir results/1_blocking
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
    patch_model_for_blocking,
    restore_model,
)


ANCHOR_PAIRS = [(i, j) for i in range(1, 5) for j in range(1, 5)]


def generate_sequences(anchor_pair, num_samples, seed):
    """
    Generate test sequences for a specific anchor pair.

    Args:
        anchor_pair: (int, int), e.g. (1, 2)
        num_samples: number of sequences to generate
        seed: random seed

    Returns:
        sequences: torch.LongTensor, shape=(num_samples, 8)
        key_positions: list of int, key position for each sequence
    """
    rng = random.Random(seed)
    sequences = []
    key_positions = []

    for _ in range(num_samples):
        pos = rng.randint(0, 5)
        key = rng.randint(20, 99)

        seq = [0] * 8
        # Fill non-anchor positions with random values avoiding anchor tokens
        for i in range(8):
            if i not in (pos, pos + 1, pos + 2):
                val = rng.randint(5, 99)
                while val in (1, 2, 3, 4):
                    val = rng.randint(5, 99)
                seq[i] = val

        seq[pos] = key
        seq[pos + 1] = anchor_pair[0]
        seq[pos + 2] = anchor_pair[1]

        sequences.append(seq)
        key_positions.append(pos)

    return torch.tensor(sequences, dtype=torch.long), key_positions


def run_blocking_experiment(model, sequences, key_positions, device):
    """
    Run blocking experiment for a batch of sequences.

    For each sequence, blocks SSM information flow from key, anchor1,
    and anchor2 positions to all downstream tokens.

    Args:
        model: MambaForComposite (eval mode, fused path disabled)
        sequences: (N, 8) tensor
        key_positions: list of int
        device: torch device

    Returns:
        original_preds: (N,) tensor of original predictions
        blocked_preds: (N,) tensor of predictions after blocking
    """
    sequences = sequences.to(device)

    # 1. Normal forward (using reference scan, no blocking)
    def no_blocking(batch_size):
        return None

    originals_none = patch_model_for_blocking(model, no_blocking)
    with torch.no_grad():
        original_preds = model(sequences).argmax(dim=-1)
    restore_model(model, originals_none)

    # 2. Blocked forward — group by key_position for efficiency
    blocked_preds = torch.zeros_like(original_preds)

    positions_to_indices = {}
    for idx, pos in enumerate(key_positions):
        positions_to_indices.setdefault(pos, []).append(idx)

    for pos, indices in positions_to_indices.items():
        batch_seqs = sequences[indices]

        def make_blocking_fn(key_pos, dev):
            def blocking_fn(batch_size):
                mask = torch.zeros(batch_size, 8, dtype=torch.bool, device=dev)
                mask[:, key_pos] = True      # block key
                mask[:, key_pos + 1] = True   # block anchor1
                mask[:, key_pos + 2] = True   # block anchor2
                return mask
            return blocking_fn

        originals_block = patch_model_for_blocking(
            model, make_blocking_fn(pos, device)
        )
        with torch.no_grad():
            batch_preds = model(batch_seqs).argmax(dim=-1)
        restore_model(model, originals_block)

        for i, idx in enumerate(indices):
            blocked_preds[idx] = batch_preds[i]

    return original_preds, blocked_preds


def plot_bar_chart(results, output_dir):
    """
    Plot blocking experiment bar chart (Figure 6a).

    Args:
        results: dict mapping pair_name -> match_rate
        output_dir: directory to save the plot
    """
    pair_names = [f"{a1}{a2}" for a1 in range(1, 5) for a2 in range(1, 5)]
    rates = [results[name] * 100 for name in pair_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(pair_names)), rates, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(pair_names)))
    ax.set_xticklabels(pair_names, fontsize=10)
    ax.set_xlabel('Anchor Pair', fontsize=12)
    ax.set_ylabel('Match Rate (%)', fontsize=12)
    ax.set_title('Accuracy under Information Blocking', fontsize=14)
    ax.set_ylim(0, 105)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='100%')
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rate:.0f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'blocking_bar_chart.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Bar chart saved to {path}")


def main():
    parser = argparse.ArgumentParser(description='Information Blocking Experiment')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model_final.pt')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.json from training')
    parser.add_argument('--num_samples', type=int, default=480,
                        help='Number of samples per anchor pair')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results/1_blocking')
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

            sequences, key_positions = generate_sequences(
                pair, args.num_samples, args.seed + a1 * 10 + a2
            )

            original_preds, blocked_preds = run_blocking_experiment(
                model, sequences, key_positions, device
            )

            match_rate = (original_preds == blocked_preds).float().mean().item()
            results[pair_name] = match_rate
            print(f"  Pair {pair_name}: match_rate = {match_rate:.4f}")

    # Save results
    output = {
        'checkpoint': args.checkpoint,
        'config': config,
        'num_samples_per_pair': args.num_samples,
        'seed': args.seed,
        'results': {k: {'match_rate': v, 'num_samples': args.num_samples}
                    for k, v in results.items()},
        'mean_match_rate': sum(results.values()) / len(results),
    }

    results_path = os.path.join(args.output_dir, 'blocking_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to {results_path}")
    print(f"Mean match rate: {output['mean_match_rate']:.4f}")

    # Save experiment config
    exp_config = {
        'experiment': '1_blocking',
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

    print("\nBlocking experiment completed.")


if __name__ == '__main__':
    main()
