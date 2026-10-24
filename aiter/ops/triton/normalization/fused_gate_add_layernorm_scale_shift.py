# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.normalization.fused_gate_add_layernorm_scale_shift import (
    _fused_gate_add_layernorm_scale_shift_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def _fused_gate_add_layernorm_scale_shift_core(x, attn, gate, scale, shift, epsilon):
    """Launcher for the fused adaLN-gate DiT block op.

    Computes, per row:
        h   = x + gate * attn
        out = LayerNorm(h) * (1 + scale) + shift
    and returns ``(out, h)``.

    x, attn:            (M, N) tensors (bf16/fp16), one row per token.
    gate, scale, shift: (G, N) modulation tensors, one row per group; each group
                        spans ``M // G`` consecutive rows of x (the tokens that
                        share a timestep embedding). ``M`` must be divisible
                        by ``G``.
    """
    assert x.dim() == 2, "fused_gate_add_layernorm_scale_shift expects a 2D tensor"
    M, N = x.shape
    assert (
        attn.shape == x.shape
    ), f"attn shape {tuple(attn.shape)} must match x {tuple(x.shape)}"
    for t, name in ((gate, "gate"), (scale, "scale"), (shift, "shift")):
        assert (
            t.dim() == 2 and t.shape[1] == N
        ), f"{name} must be 2-D with {N} columns, got {tuple(t.shape)}"
        assert (
            t.dtype == x.dtype
        ), f"{name} dtype {t.dtype} must match x dtype {x.dtype}"
    G = gate.shape[0]
    assert (
        scale.shape[0] == G and shift.shape[0] == G
    ), "gate, scale, shift must share the same number of groups"
    assert M % G == 0, f"row count M={M} must be divisible by group count G={G}"
    assert (
        attn.dtype == x.dtype
    ), f"attn dtype {attn.dtype} must match x dtype {x.dtype}"

    x = x.contiguous() if not x.is_contiguous() else x
    attn = attn.contiguous() if not attn.is_contiguous() else attn
    gate = gate.contiguous() if not gate.is_contiguous() else gate
    scale = scale.contiguous() if not scale.is_contiguous() else scale
    shift = shift.contiguous() if not shift.is_contiguous() else shift

    rows_per_group = M // G
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    BLOCK_SIZE_N = max(triton.next_power_of_2(N), 32)
    if get_arch() == "gfx1151":
        num_warps = 2
    else:
        num_warps = 8
    grid = (M,)
    _fused_gate_add_layernorm_scale_shift_kernel[grid](
        x,
        attn,
        gate,
        scale,
        shift,
        out,
        x,
        epsilon,
        M,
        N,
        rows_per_group,
        x.stride(0),
        attn.stride(0),
        gate.stride(0),
        out.stride(0),
        x.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )
    return out, x


def _fused_gate_add_layernorm_scale_shift_fake(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@torch_compile_guard(gen_fake=_fused_gate_add_layernorm_scale_shift_fake)
def fused_gate_add_layernorm_scale_shift(
    x: torch.Tensor,
    attn: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused adaLN-gate DiT block op.

    Fuses the four elementwise steps that follow attention in an adaLN-gate
    transformer block (SD3/SD3.5 JointTransformerBlock image branch):
        attn_out = gate * attn
        h        = x + attn_out            (gated residual)
        norm     = LayerNorm(h)            (no learned affine)
        out      = norm * (1 + scale) + shift   (adaLN modulation)
    into a single Triton kernel launch. Returns ``(out, h)`` where ``h`` is the
    gated-residual sum (the block's running hidden state) and ``out`` is the
    modulated, normalized input to the feed-forward layer.

    Note: ``x`` is updated **in-place** with the gated-residual result; the
    returned ``h`` is the same tensor as the modified ``x``.

    Shapes:
      x, attn:            (M, N)
      gate, scale, shift: (G, N), broadcast across the M // G rows of each group
    """
    return _fused_gate_add_layernorm_scale_shift_core(
        x, attn, gate, scale, shift, epsilon
    )
