# Mamba Model Implementation

This directory contains the Mamba model implementations for the composite function symmetry experiments.

## Files

### `mamba_wrapper.py`
Core wrapper around `mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel`.

**Key Features:**
- Custom gamma-based initialization: σ = 1 / (d1^gamma)
- Returns only last token logits for sequence classification
- Configurable architecture parameters

**Usage:**
```python
from src.models import MambaForComposite

model = MambaForComposite(
    n_layers=2,
    gamma=1.0,
    vocab_size=100,
    d_model=32,
    d_state=128,
    d_conv=4,
    expand=2
)

# Forward pass
input_ids = torch.randint(0, 100, (batch_size, seq_len))
logits = model(input_ids)  # Shape: [batch_size, vocab_size]
```

### `variants.py`
Ablation variants for testing different architectural components.

**Variants:**

1. **MambaOnesConv**: All convolution kernels set to 1
   - Tests necessity of learned convolution patterns
   - Same interface as MambaForComposite

2. **MambaPosEncoding**: Adds learnable positional embeddings
   - Tests whether explicit position information helps
   - Additional parameter: `max_seq_len`

3. **MambaWithResidual**: Residual connections bypassing convolution
   - Documented for future implementation
   - Currently falls back to standard forward pass

**Usage:**
```python
from src.models import MambaOnesConv, MambaPosEncoding

# Ones convolution variant
model_ones = MambaOnesConv(n_layers=2, gamma=1.0)

# Positional encoding variant
model_pos = MambaPosEncoding(
    max_seq_len=8,
    n_layers=2,
    gamma=0.5
)
```

## Model Configuration

### Composite Function Task
```python
d_model = 32          # Hidden dimension
d_state = 128         # SSM state dimension
d_conv = 4            # Convolution kernel size
expand = 2            # MLP expansion factor
n_layers = 2-6        # Variable layer count
vocab_size = 100      # Vocabulary size
seq_length = 8        # Input sequence length
```

### Initialization

Gamma-based weight initialization:
```python
# For weight matrix W with shape (d1, d2, ...)
# W ~ N(0, σ²) where σ = 1 / (d1^gamma)

gamma_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

## Architecture Details

### MambaForComposite

```
Input: [batch, seq_len] token indices
    ↓
Embedding: [batch, seq_len, d_model]
    ↓
Mamba Layers (×n_layers):
    - Conv1d (kernel_size=d_conv)
    - SSM (state_dim=d_state)
    - MLP (expand=2)
    ↓
Norm + LM Head: [batch, seq_len, vocab_size]
    ↓
Extract last token: [batch, vocab_size]
```

### Key Implementation Details

1. **Gamma Initialization**
   - Applied to all weight parameters (not biases)
   - Only affects parameters with dim >= 2
   - Biases initialized to zero

2. **Last Token Output**
   - Only returns logits[:, -1, :] for classification
   - Reduces memory footprint during training

3. **Device Handling**
   - Auto-detects CUDA availability
   - Moves inputs to model device automatically

## Testing

Run model tests:
```bash
# Unit tests (requires mamba_ssm)
python -m pytest tests/test_models.py -v

# Quick validation
python src/models/mamba_wrapper.py
python src/models/variants.py
```

## Dependencies

Required packages:
- `torch >= 2.0`
- `mamba-ssm` (install from: https://github.com/state-spaces/mamba)

Install mamba_ssm:
```bash
pip install mamba-ssm
```

## Expected Behavior

### Shape Test
```python
model = MambaForComposite(n_layers=2, gamma=1.0)
x = torch.randint(0, 100, (4, 8))  # [batch=4, seq_len=8]
output = model(x)
assert output.shape == (4, 100)  # [batch=4, vocab_size=100]
```

### Configuration Serialization
```python
config = model.get_config()
# Returns:
{
    'n_layers': 2,
    'gamma': 1.0,
    'vocab_size': 100,
    'd_model': 32,
    'd_state': 128,
    'd_conv': 4,
    'expand': 2
}
```

## Integration with Training

The models are designed to work with the training pipeline in `src/train.py`:

```python
from src.models import MambaForComposite
from src.data.composite_task import CompositeFunctionDataset

# Create model
model = MambaForComposite(n_layers=args.n_layers, gamma=args.gamma)

# Load data
train_dataset = CompositeFunctionDataset(mode='train')
train_loader = DataLoader(train_dataset, batch_size=2048)

# Training loop
for sequences, labels in train_loader:
    logits = model(sequences)  # [batch, 100]
    loss = criterion(logits, labels)  # labels: [batch]
    # ... backward pass
```

## Notes

- The MambaWithResidual variant is documented but not fully implemented
  - Requires modifying Mamba internal forward passes
  - Marked for future work

- All models inherit from MambaForComposite for consistency

- Variants preserve the same interface and can be swapped easily

## References

- [Mamba Paper](https://arxiv.org/abs/2312.00752)
- [mamba-ssm Repository](https://github.com/state-spaces/mamba)
- Implementation Guide: `/root/mamba-achilles/method/implementation_guide.md`
