#!/usr/bin/env python3
"""
Training script for Mamba composite function task.
Implements the training protocol from implementation_guide.md section 4.
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

# Import custom modules (to be implemented)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.composite_task import CompositeFunctionDataset, CompositeEvalDataset
from src.models.mamba_wrapper import MambaForComposite


def set_seed(seed):
    """Set random seed for reproducibility."""
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
    """
    Learning rate schedule with warmup and cosine decay.

    Phase 1 (0-10 epochs): Linear warmup from 1e-5 to 2.5e-4
    Phase 2 (10-200 epochs): Cosine decay from 2.5e-4 to 1e-5
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return initial_lr + (warmup_target_lr - initial_lr) * (epoch / warmup_epochs)
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return initial_lr + (warmup_target_lr - initial_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def train_epoch(model, train_loader, optimizer, criterion, device, gradient_clip_norm=1.0):
    """
    Train for one epoch.

    Returns:
        avg_loss: float - Average training loss
    """
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

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, eval_loader, device):
    """
    Evaluate model on a dataset.

    Returns:
        accuracy: float - Accuracy (0-1)
    """
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


def save_checkpoint(model, optimizer, epoch, output_dir, filename="checkpoint.pt"):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Train Mamba on composite function task")

    # Model parameters
    parser.add_argument("--n_layers", type=int, required=True, help="Number of Mamba layers")
    parser.add_argument("--gamma", type=float, required=True, help="Initialization parameter")

    # Architecture parameters
    parser.add_argument("--d_model", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--d_state", type=int, default=128, help="SSM state dimension")
    parser.add_argument("--d_conv", type=int, default=4, help="Convolution kernel size")
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size")

    # Training parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--initial_lr", type=float, default=1e-5, help="Initial learning rate")
    parser.add_argument("--warmup_target_lr", type=float, default=2.5e-4, help="Warmup target LR")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0, help="Gradient clipping norm")

    # Data parameters
    parser.add_argument("--num_train_samples", type=int, default=300000, help="Number of training samples")
    parser.add_argument("--num_eval_samples", type=int, default=1800, help="Number of eval samples")

    # Logging parameters
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")

    # Initialize device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = CompositeFunctionDataset(
        mode='train',
        num_samples=args.num_train_samples,
        seed=args.seed
    )

    eval_composite = CompositeEvalDataset(
        label_mode='composite',
        num_samples=args.num_eval_samples,
        seed=args.seed
    )

    eval_symmetric = CompositeEvalDataset(
        label_mode='symmetric',
        num_samples=args.num_eval_samples,
        seed=args.seed
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_composite)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    composite_loader = DataLoader(
        eval_composite,
        batch_size=256,
        shuffle=False,
        num_workers=2
    )

    symmetric_loader = DataLoader(
        eval_symmetric,
        batch_size=256,
        shuffle=False,
        num_workers=2
    )

    # Create model
    print("Creating model: MambaForComposite")
    model = MambaForComposite(
        n_layers=args.n_layers,
        gamma=args.gamma,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.initial_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training state
    best_composite_acc = 0.0
    best_composite_epoch = 0
    best_symmetric_acc = 0.0
    best_symmetric_epoch = 0
    early_stop_counter = 0  # consecutive epochs where BOTH metrics decline
    prev_composite_acc = None
    prev_symmetric_acc = None
    stopped_epoch = None

    # Initialize training_log.csv header
    log_path = os.path.join(args.output_dir, "training_log.csv")
    with open(log_path, 'w') as f:
        f.write("epoch,train_loss,composite_acc,symmetric_acc,lr\n")

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Update learning rate
        current_lr = get_lr(
            epoch,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            initial_lr=args.initial_lr,
            warmup_target_lr=args.warmup_target_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            gradient_clip_norm=args.gradient_clip_norm
        )

        # Evaluate every log_interval epochs
        if (epoch + 1) % args.log_interval == 0 or epoch == args.epochs - 1:
            composite_acc = evaluate(model, composite_loader, device)
            symmetric_acc = evaluate(model, symmetric_loader, device)

            # Save best composite model
            if composite_acc > best_composite_acc:
                best_composite_acc = composite_acc
                best_composite_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "model_best_comp.pt"))

            # Save best symmetric model
            if symmetric_acc > best_symmetric_acc:
                best_symmetric_acc = symmetric_acc
                best_symmetric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, "model_best_symm.pt"))

            # Append to training_log.csv
            with open(log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss:.6f},"
                        f"{composite_acc:.6f},{symmetric_acc:.6f},"
                        f"{current_lr:.8f}\n")

            # Update metrics.json
            metrics = {
                'n_layers': args.n_layers,
                'gamma': args.gamma,
                'seed': args.seed,
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

            # Early stopping: both metrics decline for 3 consecutive evals
            if prev_composite_acc is not None:
                if composite_acc < prev_composite_acc and symmetric_acc < prev_symmetric_acc:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0

                if early_stop_counter >= 3:
                    stopped_epoch = epoch + 1
                    print(f"\nEarly stopping at epoch {stopped_epoch}: "
                          f"both metrics declined for 3 consecutive evaluations.")
                    break

            prev_composite_acc = composite_acc
            prev_symmetric_acc = symmetric_acc

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Comp Acc: {composite_acc:.4f} (best: {best_composite_acc:.4f}@{best_composite_epoch}) | "
                  f"Sym Acc: {symmetric_acc:.4f} (best: {best_symmetric_acc:.4f}@{best_symmetric_epoch}) | "
                  f"LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

    # Final summary
    print(f"\nTraining finished at epoch {stopped_epoch or args.epochs}.")
    print(f"Best Composite Accuracy: {best_composite_acc:.4f} (epoch {best_composite_epoch})")
    print(f"Best Symmetric Accuracy: {best_symmetric_acc:.4f} (epoch {best_symmetric_epoch})")

    # Update final metrics.json
    metrics['early_stopped'] = stopped_epoch is not None
    if stopped_epoch is not None:
        metrics['stopped_epoch'] = stopped_epoch
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save final model
    model_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save(model.state_dict(), model_path)
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
