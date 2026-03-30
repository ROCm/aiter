# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for FlyDSL A8W8 FP8 GEMM kernel.

Tests the optimized M=32 and M=64 shapes against torch reference.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.flydsl.gemm_a8w8 import flydsl_gemm_a8w8


def _torch_reference(x_fp8, w_fp8, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    """Compute reference: Y = (X @ W^T) * (x_scale * w_scale) [+ bias]."""
    x_f32 = x_fp8.to(torch.float32)
    w_f32 = w_fp8.to(torch.float32)
    out = F.linear(x_f32, w_f32)
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(out, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def _generate_inputs(M, N, K, has_bias=False):
    """Generate FP8 inputs with per-token/per-channel scales."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max

    torch.manual_seed(42)

    x_f32 = torch.randn((M, K), dtype=torch.float32, device="cuda")
    w_f32 = torch.randn((N, K), dtype=torch.float32, device="cuda")

    x_amax = x_f32.abs().amax(dim=1, keepdim=True)
    x_scale = x_amax / fp8_max
    x_fp8 = (x_f32 / x_scale).to(fp8_dtype)

    w_amax = w_f32.abs().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = w_amax / fp8_max
    w_fp8 = (w_f32 / w_scale.T).to(fp8_dtype)

    bias = None
    if has_bias:
        bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10

    return x_fp8, w_fp8, x_scale, w_scale, bias


OPTIMIZED_SHAPES = [
    (32, 7168, 4096),   # M=32 shape (memory-bound)
    (64, 5120, 2880),   # M=64 shape (memory-bound)
]

EXTRA_SHAPES = [
    # # Smoke tests for other M values
    # (128, 5120, 2880),
    # (1024, 5120, 2880),
]


@pytest.mark.parametrize("M,N,K", OPTIMIZED_SHAPES + EXTRA_SHAPES)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("preallocate_output", [False, True])
def test_flydsl_gemm_a8w8(M, N, K, has_bias, dtype, preallocate_output):
    x_fp8, w_fp8, x_scale, w_scale, bias = _generate_inputs(M, N, K, has_bias)

    ref = _torch_reference(x_fp8, w_fp8, x_scale, w_scale, bias, dtype)

    y = torch.empty((M, N), dtype=dtype, device="cuda") if preallocate_output else None
    out = flydsl_gemm_a8w8(x_fp8, w_fp8, x_scale, w_scale, bias=bias, dtype=dtype, y=y)

    if preallocate_output:
        assert out.data_ptr() == y.data_ptr(), "Output should reuse pre-allocated tensor"

    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)
