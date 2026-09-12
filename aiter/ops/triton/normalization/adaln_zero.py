# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.normalization.adaln_zero import (
    _adaln_zero_kernel,
)

_LOGGER = AiterTritonLogger()


def _adaln_zero_core(x, scale, shift, epsilon):
    """Fused AdaLN-Zero launcher.

    Computes, per token, LayerNorm over the last dim (no affine) followed by the
    adaptive modulation ``out = x_norm * (1 + scale) + shift``. This is a single
    HBM read of ``x`` and a single write of ``out`` (the modulation params are
    tiny), replacing the eager 3-4 pass decomposition.
    """
    assert x.dim() == 3, "adaln_zero expects x of shape (B, seq, hidden)"
    B, S, H = x.shape
    assert scale.shape == (
        B,
        H,
    ), f"scale must be (B, hidden)=({B}, {H}), got {tuple(scale.shape)}"
    assert shift.shape == (
        B,
        H,
    ), f"shift must be (B, hidden)=({B}, {H}), got {tuple(shift.shape)}"
    assert (
        scale.dtype == x.dtype and shift.dtype == x.dtype
    ), "scale/shift dtype must match x dtype"

    if not x.is_contiguous():
        x = x.contiguous()
    if not scale.is_contiguous():
        scale = scale.contiguous()
    if not shift.is_contiguous():
        shift = shift.contiguous()

    out = torch.empty_like(x)
    x2 = x.view(B * S, H)
    out2 = out.view(B * S, H)

    BLOCK_SIZE_N = max(triton.next_power_of_2(H), 32)
    grid = (B * S,)
    _adaln_zero_kernel[grid](
        x2,
        out2,
        scale,
        shift,
        epsilon,
        S,
        H,
        x2.stride(0),
        out2.stride(0),
        scale.stride(0),
        shift.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=8,
    )
    return out


def _adaln_zero_fake(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, epsilon: float
) -> torch.Tensor:
    return torch.empty_like(x)


@torch_compile_guard(gen_fake=_adaln_zero_fake)
def _adaln_zero(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor, epsilon: float
) -> torch.Tensor:
    return _adaln_zero_core(x, scale, shift, epsilon)


def adaln_zero(x, scale, shift, eps=1e-5):
    """Fused AdaLN-Zero (adaptive LayerNorm modulation) used by diffusion DiT blocks.

    Applies, over the last dimension of ``x``::

        x_norm = LayerNorm(x, elementwise_affine=False)
        out    = x_norm * (1 + scale) + shift

    Args:
        x:     (B, seq, hidden) input hidden states (bf16/fp16/fp32).
        scale: (B, hidden) per-sample scale, broadcast over ``seq``.
        shift: (B, hidden) per-sample shift, broadcast over ``seq``.
        eps:   LayerNorm epsilon.

    Returns:
        out: (B, seq, hidden), same dtype as ``x``.

    Memory-bound; the fused kernel does one read of ``x`` and one write of the
    output, versus the multi-pass eager ``F.layer_norm`` + elementwise sequence.
    Registered as a torch custom op so it is safe under torch.compile / CUDAGraph.
    """
    _LOGGER.info(
        f"ADALN_ZERO: x={tuple(x.shape)} scale={tuple(scale.shape)} "
        f"shift={tuple(shift.shape)}"
    )
    return _adaln_zero(x, scale, shift, eps)
