# Mamba-Achilles

Reproduction of experiments from **"Achilles' Heel of Mamba: Essential difficulties of the Mamba architecture demonstrated by synthetic data"** (NeurIPS 2025 Spotlight).

This project investigates implicit positional biases in Mamba state-space models that prevent them from learning compositional functions.

## Core Idea

Mamba is trained on 15 anchor pairs with a special pair (3,4) whose label is manually set to `(key-6)%100` instead of the compositional result `(key-10)%100`. The held-out pair (4,3) is used to test whether the model learns true composition `f₃(f₄(key)) = (key-10)%100` or falls into symmetry bias `(key-6)%100`.

## Project Structure

```
├── src/
│   ├── data/composite_task.py    # Synthetic dataset (300k train / 1.8k test)
│   ├── models/
│   │   ├── mamba_wrapper.py      # MambaForComposite with gamma init
│   │   └── mamba_hooks.py        # Mechanistic interpretability hooks
│   ├── 0_train.py                # Training pipeline
│   ├── 0_train_visualize.py      # Phase diagram heatmaps
│   ├── 0_train_analyze_results.py
│   └── 1_blocking.py             # Information blocking experiment
├── experiments/                   # Shell scripts for experiment sweeps
├── method/                        # Implementation guides & checklist
├── paper/                         # Original paper & LaTeX source
├── results/                       # Experiment outputs
└── tests/                         # Unit tests
```

## Experiment Progress

### Experiment 0: Training & Phase Diagram

| Status | Task | Details |
|--------|------|---------|
| Done | Best model training | 7 layers, gamma=1.0, seed=42, 200 epochs |
| Done | Phase diagram (L2) | 12 gamma values (0.1-2.0) |
| Done | Phase diagram (L3) | 12 gamma values (0.1-2.0) |
| Partial | Phase diagram (L4) | 5/12 gamma values (0.1-0.5) |
| Not started | Phase diagram (L5-L7) | 0/36 configurations |

**Phase diagram completion: 29/72 configurations (40%)**

#### Best Model Results (L=7, gamma=1.0)

| Metric | Value | Epoch |
|--------|-------|-------|
| Composite accuracy (correct) | **86.0%** | 190 |
| Symmetric accuracy (biased) | 0.17% | 200 |
| Training loss | 0.505 | 200 |

The model strongly prefers the compositional solution over the symmetric bias.

### Experiment 1: Information Blocking

| Status | Task | Details |
|--------|------|---------|
| Done | SSM state blocking | All anchor pairs tested |

Blocking SSM state from key/anchor positions yields a **mean match rate of 64.8%** (range: 60.8%-68.3%), confirming that Conv1d — not the SSM — is the primary carrier of positional information.

### Experiment 2: Substitution

| Status | Task |
|--------|------|
| Not started | Conv1d substitution experiment |

## Model Architecture

- `d_model=32`, `d_state=128`, `d_conv=4`, `expand=2`
- Custom gamma initialization: `σ = 1/(d₁^gamma)`
- Sequence length: 8, vocabulary size: 100

## Training Configuration

- Optimizer: AdamW (β₁=0.9, β₂=0.999, weight_decay=0.01)
- LR schedule: linear warmup (10 epochs, 1e-5 → 2.5e-4) + cosine decay
- Batch size: 2048, gradient clipping: max_norm=1.0
- 200 epochs with early stopping

## Requirements

```
torch>=2.0.0
numpy>=1.24.0
mamba-ssm>=1.0.0
```

## Usage

```bash
# Train best model
python src/0_train.py --n_layers 7 --gamma 1.0 --seed 42 --output_dir results/0_train_paper_best

# Run phase diagram sweep
bash experiments/0_train_run_phase_diagram.sh

# Run information blocking experiment
bash experiments/1_blocking_run.sh

# Visualize phase diagram
python src/0_train_visualize.py
```

## Reference

> Tianyi Chen, Pengxiao Lin, Zhiwei Wang, Zhi-Qin John Xu. "Achilles' Heel of Mamba: Essential difficulties of the Mamba architecture demonstrated by synthetic data." NeurIPS 2025 Spotlight. Shanghai Jiao Tong University.
