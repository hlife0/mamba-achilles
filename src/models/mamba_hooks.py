"""
Hook utilities for Mamba (Mamba1) internals.

Provides tools to intercept and modify intermediate computations inside
Mamba layers, specifically:
- SSM recurrence state for blocking experiments (zeroing out contributions
  from key/anchor positions to downstream tokens)
- Post-conv1d hidden states for substitution experiments

These hooks work by monkey-patching the Mamba.forward method to use a
reference (pure-PyTorch) selective scan instead of the fused CUDA kernel,
enabling insertion of modifier callbacks at key points.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


# ---------------------------------------------------------------------------
# Utility: disable the fused path so we can hook into forward
# ---------------------------------------------------------------------------

def disable_fused_path(model):
    """
    Set use_fast_path=False on every Mamba mixer in the model,
    forcing the non-fused forward path that we can monkey-patch.

    Args:
        model: MambaForComposite instance
    """
    for layer in model.backbone.backbone.layers:
        layer.mixer.use_fast_path = False


# ---------------------------------------------------------------------------
# Reference selective scan with blocking support
# ---------------------------------------------------------------------------

def selective_scan_with_blocking(u, delta, A, B, C, D=None, z=None,
                                 delta_bias=None, delta_softplus=False,
                                 blocked_positions=None):
    """
    Reference selective scan (pure PyTorch) that supports blocking
    information flow from specified source positions to downstream tokens.

    Blocking is implemented by zeroing out deltaB_u at blocked positions,
    preventing those positions from writing into the SSM hidden state
    for downstream tokens.

    Args:
        u: (batch, d_inner, seqlen) — input after conv1d
        delta: (batch, d_inner, seqlen) — time step
        A: (d_inner, d_state) — state transition matrix
        B: (batch, d_state, seqlen) — input-dependent B
        C: (batch, d_state, seqlen) — input-dependent C
        D: (d_inner,) or None — skip connection
        z: (batch, d_inner, seqlen) or None — gate
        delta_bias: (d_inner,) or None
        delta_softplus: bool
        blocked_positions: (batch, seqlen) bool tensor or None.
            If provided, True at positions whose information should be
            blocked from flowing to downstream tokens via SSM.

    Returns:
        out: (batch, d_inner, seqlen)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    B = B.float()
    C = C.float()

    # Precompute discretized quantities
    # deltaA: (batch, d_inner, seqlen, d_state)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # deltaB_u: (batch, d_inner, seqlen, d_state)
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    # Sequential scan with blocking support
    #
    # Blocking semantics (matching paper's S matrix zeroing):
    #   For blocked position j, S[i,j]=0 for all i > j (downstream),
    #   but S[j,j] is preserved (token j still reads its own input).
    #
    # Implementation: at blocked position j, we compute y[j] using the
    # full h[j] = deltaA*h[j-1] + deltaB_u[j], but pass only
    # deltaA*h[j-1] (without deltaB_u[j]) to the next step.
    # This way token j's own output is unchanged, but downstream tokens
    # cannot receive j's input through the SSM.
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    if blocked_positions is not None:
        block_mask = blocked_positions  # (batch, seqlen) bool
    else:
        block_mask = None

    for i in range(seqlen):
        x_new = deltaA[:, :, i] * x + deltaB_u[:, :, i]

        if block_mask is not None and block_mask[:, i].any():
            # For blocked positions: compute y using full x_new,
            # but propagate only deltaA * x (without deltaB_u contribution)
            x_pass = deltaA[:, :, i] * x  # state without position i's input

            # Per-sample masking: blocked samples use x_pass, others use x_new
            bm = block_mask[:, i].unsqueeze(1).unsqueeze(2).float()  # (batch, 1, 1)
            y = torch.einsum('bdn,bn->bd', x_new, C[:, :, i])
            x = x_new * (1.0 - bm) + x_pass * bm
        else:
            y = torch.einsum('bdn,bn->bd', x_new, C[:, :, i])
            x = x_new

        ys.append(y)

    y = torch.stack(ys, dim=2)  # (batch, dim, seqlen)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out


# ---------------------------------------------------------------------------
# Monkey-patch Mamba.forward to use our reference scan with hooks
# ---------------------------------------------------------------------------

def make_patched_mamba_forward(mamba, blocked_positions_fn=None,
                               post_conv_hook_fn=None):
    """
    Create a patched forward function for a Mamba (Mamba1) module.

    This replicates the non-fused path of Mamba.forward but replaces
    selective_scan_fn with our reference implementation that supports
    blocking and post-conv hooks.

    Args:
        mamba: Mamba module instance (mamba_ssm.modules.mamba_simple.Mamba)
        blocked_positions_fn: callable(batch_size) -> (batch, seqlen) bool tensor
            Returns a mask of positions to block. Called during forward.
        post_conv_hook_fn: callable(x) -> x
            Hook called after conv1d+SiLU, before x_proj.
            x shape: (batch, d_inner, seqlen)

    Returns:
        patched_forward: function(hidden_states, ...) -> out
    """
    from causal_conv1d import causal_conv1d_fn

    def patched_forward(hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape

        # in_proj: (B, L, D) -> (B, 2*d_inner, L)
        xz = rearrange(
            mamba.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if mamba.in_proj.bias is not None:
            xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(mamba.A_log.float())  # (d_inner, d_state)

        # Split into x and z
        x, z = xz.chunk(2, dim=1)  # each: (batch, d_inner, seqlen)

        # Conv1d + SiLU
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(mamba.conv1d.weight, "d 1 w -> d w"),
            bias=mamba.conv1d.bias,
            activation=mamba.activation,
        )

        # >>> Post-conv hook point (for substitution) <<<
        if post_conv_hook_fn is not None:
            x = post_conv_hook_fn(x)

        # x_proj: extract dt, B, C from post-conv x
        x_dbl = mamba.x_proj(rearrange(x, "b d l -> (b l) d"))  # (B*L, dt_rank+2*d_state)
        dt, B, C = torch.split(
            x_dbl, [mamba.dt_rank, mamba.d_state, mamba.d_state], dim=-1
        )
        dt = mamba.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # Get blocked positions if blocking is active
        blocked_pos = None
        if blocked_positions_fn is not None:
            blocked_pos = blocked_positions_fn(batch)

        # SSM via reference implementation with blocking support
        y = selective_scan_with_blocking(
            x, dt, A, B, C,
            D=mamba.D.float(),
            z=z,
            delta_bias=mamba.dt_proj.bias.float(),
            delta_softplus=True,
            blocked_positions=blocked_pos,
        )

        y = rearrange(y, "b d l -> b l d")
        out = mamba.out_proj(y)
        return out

    return patched_forward


def patch_model_for_blocking(model, blocked_positions_fn):
    """
    Monkey-patch all Mamba layers in the model to use the reference
    selective scan with blocking support.

    Args:
        model: MambaForComposite instance
        blocked_positions_fn: callable(batch_size) -> (batch, seqlen) bool tensor

    Returns:
        originals: list of original forward methods (for restoring)
    """
    originals = []
    for layer in model.backbone.backbone.layers:
        mamba = layer.mixer
        originals.append(mamba.forward)
        mamba.forward = make_patched_mamba_forward(
            mamba, blocked_positions_fn=blocked_positions_fn
        )
    return originals


def patch_model_for_post_conv_hook(model, post_conv_hook_fn):
    """
    Monkey-patch all Mamba layers in the model to use the reference
    selective scan with a post-conv hook.

    Args:
        model: MambaForComposite instance
        post_conv_hook_fn: callable(x) -> x, or list of callables (one per layer)

    Returns:
        originals: list of original forward methods (for restoring)
    """
    layers = model.backbone.backbone.layers
    originals = []

    if callable(post_conv_hook_fn):
        hooks = [post_conv_hook_fn] * len(layers)
    else:
        hooks = post_conv_hook_fn

    for layer, hook in zip(layers, hooks):
        mamba = layer.mixer
        originals.append(mamba.forward)
        mamba.forward = make_patched_mamba_forward(mamba, post_conv_hook_fn=hook)

    return originals


def restore_model(model, originals):
    """
    Restore original forward methods after patching.

    Args:
        model: MambaForComposite instance
        originals: list of original forward methods from patch_model_*
    """
    for layer, orig in zip(model.backbone.backbone.layers, originals):
        layer.mixer.forward = orig
