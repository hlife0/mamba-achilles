#!/usr/bin/env python3
"""
Experiment 2b: All-one convolution (1Conv1d) vs standard Mamba.

Verifies that Conv1d weight asymmetry is the root cause of Mamba's
preference for composite solutions. By freezing all Conv1d kernel
weights to 1, the model should shift to preferring symmetric solutions.

Corresponds to paper Section 4.2, Figure 4(b).

Usage:
    python src/2b_allone_conv.py \
        --n_layers 5 --gamma 1.0 --seed 42 \
        --allone_conv \
        --output_dir results/2b_allone_conv/allone
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.composite_task import CompositeFunctionDataset, CompositeEvalDataset
from src.models.mamba_wrapper import MambaForComposite


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_lr(epoch, warmup_epochs=10, total_epochs=200,
           initial_lr=1e-5, warmup_target_lr=2.5e-4):
    if epoch < warmup_epochs:
        return initial_lr + (warmup_target_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr + (warmup_target_lr - initial_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def set_allone_conv1d(model):
    """Set all Conv1d weights to 1 and bias to 0, then freeze them."""
    for layer in model.backbone.backbone.layers:
        conv1d = layer.mixer.conv1d
        with torch.no_grad():
            conv1d.weight.fill_(1.0)
            if conv1d.bias is not None:
                conv1d.bias.fill_(0.0)
        conv1d.weight.requires_grad = False
        if conv1d.bias is not None:
            conv1d.bias.requires_grad = False


def train_epoch(model, train_loader, optimizer, criterion, device, gradient_clip_norm=1.0):
    model.train()
    total_loss = 0
    num_batches = 0

    for sequences, labels in tqdm(train_loader, desc="Training", leave=False):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, eval_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in eval_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            logits = model(sequences)
            predictions = logits.argmax(dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Experiment 2b: All-one Conv1d")

    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_state", type=int, default=128)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=100)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--initial_lr", type=float, default=1e-5)
    parser.add_argument("--warmup_target_lr", type=float, default=2.5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)

    parser.add_argument("--num_train_samples", type=int, default=300000)
    parser.add_argument("--num_eval_samples", type=int, default=1800)

    parser.add_argument("--allone_conv", action="store_true",
                        help="Freeze Conv1d weights to all-ones")

    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"All-one Conv1d: {args.allone_conv}")

    # Create datasets
    train_dataset = CompositeFunctionDataset(
        mode='train', num_samples=args.num_train_samples, seed=args.seed
    )
    eval_composite = CompositeEvalDataset(
        label_mode='composite', num_samples=args.num_eval_samples, seed=args.seed
    )
    eval_symmetric = CompositeEvalDataset(
        label_mode='symmetric', num_samples=args.num_eval_samples, seed=args.seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=device.type == 'cuda'
    )
    composite_loader = DataLoader(eval_composite, batch_size=256, shuffle=False, num_workers=2)
    symmetric_loader = DataLoader(eval_symmetric, batch_size=256, shuffle=False, num_workers=2)

    # Create model
    model = MambaForComposite(
        n_layers=args.n_layers, gamma=args.gamma, vocab_size=args.vocab_size,
        d_model=args.d_model, d_state=args.d_state, d_conv=args.d_conv
    )
    model = model.to(device)

    # Apply all-one conv1d if requested
    if args.allone_conv:
        set_allone_conv1d(model)
        print("Conv1d weights frozen to all-ones.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.initial_lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=args.weight_decay
    )

    criterion = nn.CrossEntropyLoss()

    # Training state
    best_composite_acc = 0.0
    best_composite_epoch = 0
    best_symmetric_acc = 0.0
    best_symmetric_epoch = 0
    early_stop_counter = 0
    prev_composite_acc = None
    prev_symmetric_acc = None
    stopped_epoch = None

    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,composite_acc,symmetric_acc,lr\n")

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        current_lr = get_lr(
            epoch, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs,
            initial_lr=args.initial_lr, warmup_target_lr=args.warmup_target_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip_norm=args.gradient_clip_norm
        )

        if (epoch + 1) % args.log_interval == 0 or epoch == args.epochs - 1:
            composite_acc = evaluate(model, composite_loader, device)
            symmetric_acc = evaluate(model, symmetric_loader, device)

            if composite_acc > best_composite_acc:
                best_composite_acc = composite_acc
                best_composite_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "model_best_comp.pt"))

            if symmetric_acc > best_symmetric_acc:
                best_symmetric_acc = symmetric_acc
                best_symmetric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "model_best_symm.pt"))

            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},"
                        f"{composite_acc:.6f},{symmetric_acc:.6f},"
                        f"{current_lr:.8f}\n")

            metrics = {
                'n_layers': args.n_layers,
                'gamma': args.gamma,
                'seed': args.seed,
                'allone_conv': args.allone_conv,
                'current_epoch': epoch + 1,
                'total_epochs': args.epochs,
                'current_train_loss': train_loss,
                'current_composite_acc': composite_acc,
                'current_symmetric_acc': symmetric_acc,
                'best_composite_acc': best_composite_acc,
                'best_composite_epoch': best_composite_epoch,
                'best_symmetric_acc': best_symmetric_acc,
                'best_symmetric_epoch': best_symmetric_epoch,
                'early_stopped': False,
            }
            metrics_path = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            if prev_composite_acc is not None:
                if composite_acc < prev_composite_acc and symmetric_acc < prev_symmetric_acc:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0

                if early_stop_counter >= 3:
                    stopped_epoch = epoch + 1
                    print(f"\nEarly stopping at epoch {stopped_epoch}.")
                    break

            prev_composite_acc = composite_acc
            prev_symmetric_acc = symmetric_acc

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Comp: {composite_acc:.4f} (best: {best_composite_acc:.4f}@{best_composite_epoch}) | "
                  f"Sym: {symmetric_acc:.4f} (best: {best_symmetric_acc:.4f}@{best_symmetric_epoch}) | "
                  f"LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

    print(f"\nTraining finished at epoch {stopped_epoch or args.epochs}.")
    print(f"Best Composite Accuracy: {best_composite_acc:.4f} (epoch {best_composite_epoch})")
    print(f"Best Symmetric Accuracy: {best_symmetric_acc:.4f} (epoch {best_symmetric_epoch})")

    metrics['early_stopped'] = stopped_epoch is not None
    if stopped_epoch is not None:
        metrics['stopped_epoch'] = stopped_epoch
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    model_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)
    print("Training completed.")


if __name__ == "__main__":
    main()
