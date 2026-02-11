"""
Mamba model wrapper for composite function task.

This module wraps the mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel
with custom gamma-based initialization and outputs only the last token logits.

Uses Mamba2 (SSD) architecture as described in the paper's supplementary
material Section B, with single head (headdim = expand * d_model).
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
    MAMBA_AVAILABLE = True

    # --- Mamba2 compatibility patches ---
    # The Triton kernels in mamba-ssm 2.2.2 have issues with our Triton 2.1.0:
    # 1. causal_conv1d requires 8-byte aligned strides (fix: make contiguous)
    # 2. mamba_chunk_scan_combined backward crashes in Triton (fix: use ref impl)
    import mamba_ssm.modules.mamba2 as _mamba2_module
    from mamba_ssm.ops.triton.ssd_combined import ssd_chunk_scan_combined_ref

    _original_conv1d_fn = _mamba2_module.causal_conv1d_fn

    def _patched_conv1d_fn(x, weight, bias=None, activation=None, seq_idx=None):
        x = x.contiguous()
        return _original_conv1d_fn(x, weight, bias=bias, activation=activation,
                                   seq_idx=seq_idx)

    def _patched_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None,
                            dt_bias=None, dt_softplus=False,
                            return_final_states=False, **kwargs):
        seqlen = x.shape[1]
        result = ssd_chunk_scan_combined_ref(
            x, dt, A, B, C, seqlen, D=D, z=z,
            dt_bias=dt_bias, dt_softplus=dt_softplus
        )
        if return_final_states:
            return (result,
                    torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                                B.shape[-1], device=x.device, dtype=x.dtype))
        return result

    _mamba2_module.causal_conv1d_fn = _patched_conv1d_fn
    _mamba2_module.mamba_chunk_scan_combined = _patched_chunk_scan

except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Install with: pip install mamba-ssm")


class MambaForComposite(nn.Module):
    """
    Mamba model for composite function task.

    This wrapper:
    1. Uses MambaLMHeadModel from mamba_ssm as the backbone
    2. Applies custom gamma-based initialization
    3. Returns only the last token's logits for sequence classification

    Parameters:
        n_layers: int - Number of Mamba layers
        gamma: float - Initialization parameter for weight scaling
        vocab_size: int - Vocabulary size (default: 100)
        d_model: int - Hidden dimension (default: 32)
        d_state: int - SSM state dimension (default: 128)
        d_conv: int - Convolution kernel size (default: 4)
        expand: int - MLP expansion factor (default: 2)
        device: str - Device to place model on (default: 'cuda' if available)

    Input:
        input_ids: torch.LongTensor, shape=[batch_size, seq_len]

    Output:
        logits: torch.FloatTensor, shape=[batch_size, vocab_size]
        Only returns the last position logits for classification.
    """

    def __init__(
        self,
        n_layers: int,
        gamma: float,
        vocab_size: int = 100,
        d_model: int = 32,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        device: Optional[str] = None
    ):
        super().__init__()

        if not MAMBA_AVAILABLE:
            raise ImportError(
                "mamba_ssm is required but not installed. "
                "Install with: pip install mamba-ssm"
            )

        self.n_layers = n_layers
        self.gamma = gamma
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Create Mamba2 config (paper uses Mamba2/SSD with single head)
        d_inner = expand * d_model
        config = MambaConfig(
            d_model=d_model,
            n_layer=n_layers,
            vocab_size=vocab_size,
            ssm_cfg=dict(
                layer='Mamba2',
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=d_inner,  # single head: headdim = d_inner -> nheads = 1
            ),
            rms_norm=True,
            fused_add_norm=False,
            residual_in_fp32=True,
        )

        # Create Mamba backbone
        self.backbone = MambaLMHeadModel(config)

        # Disable fused mem-efficient path (incompatible with our Triton version)
        for layer in self.backbone.backbone.layers:
            layer.mixer.use_mem_eff_path = False

        # Apply gamma-based initialization
        self._init_params(gamma)

        # Move to device
        self.to(self.device)

    def _init_params(self, gamma: float):
        """
        Initialize parameters with gamma-based scaling.

        For each weight matrix W with shape (d1, d2, ...):
            W ~ N(0, sigma^2) where sigma = 1 / (d1 ^ gamma)

        Args:
            gamma: Initialization exponent (typically 0.5 to 1.0)
        """
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Get the first dimension (output dimension)
                d1 = param.shape[0]
                std = 1.0 / (d1 ** gamma)

                # Initialize with normal distribution
                nn.init.normal_(param, mean=0.0, std=std)
            elif 'bias' in name:
                # Initialize biases to zero
                nn.init.zeros_(param)

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Forward pass through the model.

        Args:
            input_ids: torch.LongTensor, shape=[batch_size, seq_len]
                Input token indices

        Returns:
            logits: torch.FloatTensor, shape=[batch_size, vocab_size]
                Logits for the last token position only
        """
        # Move input to model device if needed
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)

        # Forward through Mamba backbone
        # MambaLMHeadModel returns logits of shape [batch_size, seq_len, vocab_size]
        output = self.backbone(input_ids)

        # Extract logits (handle different output formats)
        if hasattr(output, 'logits'):
            logits = output.logits
        else:
            logits = output

        # Return only the last token's logits
        # Shape: [batch_size, vocab_size]
        # Note: vocab_size may be padded (e.g., 100->104), so slice to actual vocab_size
        last_token_logits = logits[:, -1, :]
        return last_token_logits[:, :self.vocab_size]

    def get_config(self) -> dict:
        """
        Return model configuration for serialization.

        Returns:
            config: dict containing all model hyperparameters
        """
        return {
            'n_layers': self.n_layers,
            'gamma': self.gamma,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'd_state': self.d_state,
            'd_conv': self.d_conv,
            'expand': self.expand,
        }


def test_mamba_wrapper():
    """
    Test function to validate MambaForComposite implementation.

    Verifies:
    1. Model can be instantiated
    2. Forward pass works correctly
    3. Output shape is correct
    """
    print("Testing MambaForComposite...")

    # Create model
    model = MambaForComposite(n_layers=2, gamma=1.0, d_model=32, vocab_size=100)

    # Create dummy input
    batch_size = 4
    seq_len = 8
    x = torch.randint(0, 100, (batch_size, seq_len))

    # Forward pass
    output = model(x)

    # Check output shape
    expected_shape = (batch_size, 100)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, got {output.shape}"
    )

    print(f"✓ Model test passed: output shape = {output.shape}")
    print(f"✓ Model configuration: {model.get_config()}")

    return True


if __name__ == "__main__":
    if MAMBA_AVAILABLE:
        test_mamba_wrapper()
    else:
        print("Skipping test: mamba_ssm not installed")
