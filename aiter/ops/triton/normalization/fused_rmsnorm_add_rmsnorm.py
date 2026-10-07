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
    residual_out: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


@torch_compile_guard(
    mutates_args=["residual_out"], gen_fake=_fused_rmsnorm_add_rmsnorm_fake
)
def fused_rmsnorm_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_weight: torch.Tensor,
    pre_weight: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    residual_out: torch.Tensor,
) -> torch.Tensor:
    """Write BF16 residual sum to ``residual_out`` and return BF16 pre-norm.

    The norm and residual inputs are BF16 ``(M, N)`` and both weights are
    ``(N,)``. Weights use the Muse/Gemma ``1 + weight`` convention. The first
    norm and residual add remain FP32 in the kernel; both outputs are
    rounded to BF16 at their stores. ``residual_out`` is a caller-owned
    BF16 ``(M, N)`` tensor. Inputs and output buffer are expected to be
    contiguous on the same GPU.
    """
    assert x.ndim == 2
    assert x.dtype == torch.bfloat16
    M, N = x.shape

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
    return pre_norm
