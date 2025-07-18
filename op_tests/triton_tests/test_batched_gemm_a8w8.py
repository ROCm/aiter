# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
from aiter.ops.triton.batched_gemm_a8w8 import batched_gemm_a8w8
from aiter.ops.triton.utils.arch_info import get_fp8_dtypes
from aiter.ops.triton.utils.types import str_to_torch_dtype
import torch.nn.functional as F


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        b_x = F.linear(x[b, :, :].to(torch.float32), weight[b, :, :].to(torch.float32))
        b_scale = torch.matmul(x_scale[b, :, :], w_scale[b, :, :])
        b_out = torch.mul(b_x, b_scale)
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16, y=None):
    return batched_gemm_a8w8(x, weight, x_scale, w_scale, bias, dtype, YQ=y)


def get_test_params():
    full_shape_set = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    full_shape_set += [(4864, 4096, 8192), (9728, 8192, 65000), (4864, 8192, 4160)]
    full_shape_set += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    full_shape_set += [(1, 1, 1)]  # minimal case 

    full_set = [
        pytest.param(b, *shape, dtype, output, marks=pytest.mark.slow)
        for b in [4, 16]
        for shape in full_shape_set
        for output in [True, False]
        for dtype in ["bf16"]
    ]

    fast_shape_set = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    fast_shape_set += [(4864, 4096, 8192), (9728, 8192, 65000), (4864, 8192, 4160)]

    fast_set = [
        (b, *shape, dtype, output)
        for b in [16]
        for shape in fast_shape_set
        for output in [True, False]
        for dtype in ["bf16"]
    ]

    return full_set + fast_set

@pytest.mark.parametrize(
    "B, M, N, K, dtype_str, output",get_test_params())
def test_batched_gemm_a8w8(B: int, M: int, N: int, K: int, dtype_str, output: bool):

    dtype = str_to_torch_dtype[dtype_str]

    x = torch.randint(-20, 20, (B, M, K), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (B, N, K), dtype=torch.int8).cuda()
    x_scale = torch.rand([B, M, 1], dtype=torch.float32).cuda() + 1e-6
    w_scale = torch.rand([B, 1, N], dtype=torch.float32).cuda() + 1e-6
    bias = torch.rand([B, 1, N], dtype=dtype).cuda() * 10

    y = None
    if output:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)
    a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype, y)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
