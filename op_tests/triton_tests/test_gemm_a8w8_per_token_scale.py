# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
from aiter.ops.triton.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale
import torch.nn.functional as F


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    x = x.to(x_scale.dtype) * x_scale
    weight = weight.to(w_scale.dtype) * w_scale
    out = F.linear(x.to(torch.float32), weight.to(torch.float32))
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None):
    return gemm_a8w8_per_token_scale(x, weight, x_scale, w_scale, dtype, y)


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
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536)]
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
        (2048, 2048, 2049),
        (159, 17389, 597),
        (16, 576, 7168),
    ]
    x_vals += [
        (256, 8192, 1024),
        (256, 1024, 8192),
        (256, 32768, 8192),
        (256, 8192, 32768),
    ]
    # x_vals += [(1, 1, 1)]  # minimal case
    return x_vals


def generate_gemm_a8w8_per_token_scale_inputs(
    M, N, K, dtype=torch.bfloat16, output=False
):

    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    x_scale = torch.rand([M, 1], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([N, 1], dtype=torch.float32, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, x_scale, w_scale, y


@pytest.mark.parametrize(
    "dtype, M, N, K, output",
    [
        (dtype, *shape, output)
        for output in [True, False]
        for dtype in ["bf16"]
        for shape in get_x_vals()
    ],
)
def test_gemm(dtype, M, N, K, output):

    dtype = name_to_torch_types[dtype]
    x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
        M,
        N,
        K,
        dtype,
        output,
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)
    b = run_triton(x, weight, x_scale, w_scale, dtype, y)

    triton.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
