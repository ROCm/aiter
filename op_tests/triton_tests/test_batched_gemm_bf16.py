# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
from aiter.ops.triton.batched_gemm_bf16 import batched_gemm_bf16
from aiter.ops.triton.utils.arch_info import get_fp8_dtypes
from aiter.ops.triton.utils.types import str_to_torch_dtype
import torch.nn.functional as F


def run_torch(x, weight, bias=None, dtype=torch.bfloat16):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        b_out = F.linear(
            x[b, :, :].to(torch.float32), weight[b, :, :].to(torch.float32)
        )
        if bias is not None:
            b_out = b_out.to(bias[b, :, :]) + bias[b, :, :]
        out[b, :, :] = b_out
    return out.to(dtype)


def run_triton(x, weight, bias=None, dtype=torch.bfloat16, y=None):
    return batched_gemm_bf16(x, weight, bias, dtype, YQ=y)


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65000), (4864, 8192, 4160)]
    x_vals += [
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
    return x_vals


@pytest.mark.parametrize(
    "dtype, b, m, n, k, output",
    [
        (dtype, b, *shape, output)
        for output in [True, False]
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_x_vals()
    ],
)
def test_batched_gemm_bf16(dtype, b, m, n, k, output):

    dtype = str_to_torch_dtype[dtype]
    x = torch.randint(-20, 20, (b, m, k), dtype=dtype).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=dtype).cuda()

    bias = torch.rand([b, 1, n], dtype=dtype).cuda() * 10

    y = None
    if output:
        y = torch.empty((b, m, n), dtype=dtype, device=x.device)

    a = run_torch(x, weight, bias, dtype)
    b = run_triton(x, weight, bias, dtype, y)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
