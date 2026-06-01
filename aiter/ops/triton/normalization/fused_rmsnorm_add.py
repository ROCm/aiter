# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from aiter.ops.triton._gluon_kernels.gfx1250.norm.fused_rmsnorm_add import (
    _gluon_fused_rms_kernel,
)


def fused_rms_gluon(x, weight, epsilon, res1=None):
    """RMSNorm over the last dim of a 2D tensor.

    x:      (M, N) contiguous-last-dim tensor (bf16/fp16).
    weight: (N,) tensor.
    res1:   optional (M, N) residual; when given, computes x += res1 first and
            returns (out, out_res1) where out_res1 is the pre-norm sum.
    Returns out (M, N) if res1 is None, else (out, out_res1).
    """
    assert x.dim() == 2, "fused_rms_gluon expects a 2D tensor"
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    M, N1 = x.shape
    BLOCK_SIZE_N = max(triton.next_power_of_2(N1), 32)
    BLOCK_SIZE_M = 1
    out1 = torch.empty((M, N1), dtype=x.dtype, device=x.device)
    out_res1 = None
    res1_stride_m = 0
    out_res1_stride_m = 0
    if res1 is not None:
        if not res1.is_contiguous():
            res1 = res1.contiguous()
        out_res1 = torch.empty((M, N1), dtype=x.dtype, device=x.device)
        res1_stride_m = res1.stride(0)
        out_res1_stride_m = out_res1.stride(0)
    grid = (triton.cdiv(M, BLOCK_SIZE_M),)
    _gluon_fused_rms_kernel[grid](
        x,
        weight,
        res1,
        out1,
        out_res1,
        epsilon,
        M,
        N1,
        x.stride(0),
        res1_stride_m,
        out1.stride(0),
        out_res1_stride_m,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        FIRST_INPUT_RES=(res1 is not None),
    )
    if res1 is not None:
        return out1, out_res1
    return out1
