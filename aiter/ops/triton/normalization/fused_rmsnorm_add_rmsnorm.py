# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RMSNorm, residual add, RMSNorm for Muse-style sandwich norms."""

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.normalization.fused_rmsnorm_add_rmsnorm import (
    _fused_rmsnorm_add_rmsnorm_kernel,
)


def _fused_rmsnorm_add_rmsnorm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    post_eps: float,
    pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=torch.float32), torch.empty_like(x)


@torch_compile_guard(gen_fake=_fused_rmsnorm_add_rmsnorm_fake)
def fused_rmsnorm_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    post_eps: float,
    pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(residual_out_fp32, pre_norm_bf16)`` from two RMSNorms.

    The norm input ``x`` is BF16 ``(M, N)``. The residual input may be BF16 or
    FP32, and both weights are BF16 or FP32 ``(N,)``. Weights use the Muse/Gemma
    ``1 + weight`` convention. The first norm and residual add remain FP32;
    only the final norm is rounded to BF16 at its output store. All inputs
    must be contiguous and on the same GPU.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
        raise ValueError("x must be a 2D BF16 GPU tensor")
    M, N = x.shape
    if N == 0:
        raise ValueError("norm width must be positive")
    if residual.shape != x.shape or residual.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("residual must match x's shape and be BF16 or FP32")
    if any(
        weight.shape != (N,) or weight.dtype not in (torch.bfloat16, torch.float32)
        for weight in (post_weight, pre_weight)
    ):
        raise ValueError("weights must be BF16 or FP32 vectors of norm width")
    if any(t.device != x.device for t in (residual, post_weight, pre_weight)):
        raise ValueError("all tensors must be on the same GPU")
    if not all(t.is_contiguous() for t in (x, residual, post_weight, pre_weight)):
        raise ValueError("all inputs must be contiguous")

    residual_out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    pre_norm = torch.empty_like(x)
    if M:
        block = triton.next_power_of_2(N)
        num_warps = 4 if M <= 128 else 8
        _fused_rmsnorm_add_rmsnorm_kernel[(M,)](
            x,
            residual,
            post_weight,
            pre_weight,
            residual_out,
            pre_norm,
            N,
            post_eps,
            pre_eps,
            BLOCK_SIZE_N=block,
            num_warps=num_warps,
        )
    return residual_out, pre_norm
