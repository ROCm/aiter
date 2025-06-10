# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
from aiter.ops.triton.batched_gemm_a8w8 import batched_gemm_a8w8
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


def is_cdna4():
    return triton.runtime.driver.active.get_current_target().arch == "gfx950"


e5m2_type = torch.float8_e5m2 if is_cdna4() else torch.float8_e5m2fnuz
e4m3_type = torch.float8_e4m3fn if is_cdna4() else torch.float8_e4m3fnuz

name_to_torch_types = {
    "int8": torch.int8,
    "int32": torch.int32,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp8e5": e5m2_type,
    "fp8e4": e4m3_type,
}


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
def test_batched_gemm_a8w8(dtype, b, m, n, k, output):

    dtype = name_to_torch_types[dtype]

    x = torch.randint(-20, 20, (b, m, k), dtype=torch.int8).cuda()
    weight = torch.randint(-20, 20, (b, n, k), dtype=torch.int8).cuda()
    x_scale = torch.rand([b, m, 1], dtype=torch.float32).cuda() + 1e-6
    w_scale = torch.rand([b, 1, n], dtype=torch.float32).cuda() + 1e-6
    bias = torch.rand([b, 1, n], dtype=dtype).cuda() * 10

    y = None
    if output:
        y = torch.empty((b, m, n), dtype=dtype, device=x.device)
    a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype, y)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
