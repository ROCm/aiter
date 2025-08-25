# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale
from aiter.ops.triton.utils.arch_info import get_fp8_dtypes
from aiter.ops.triton.utils.types import str_to_torch_dtype
import torch.nn.functional as F


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    x = x.to(x_scale.dtype) * x_scale
    weight = weight.to(w_scale.dtype) * w_scale
    out = F.linear(x.to(torch.float32), weight.to(torch.float32))
    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None):
    return gemm_a8w8_per_token_scale(x, weight, x_scale, w_scale, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_gemm_a8w8_per_token_scale_inputs(
    M: int,
    N: int,
    K: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output=False,
):

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, 1], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([N, 1], dtype=torch.float32, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")

    return x, weight, x_scale, w_scale, y


def basic_shape_set():
    shapes = [
        (128, 128, 128),
        (256, 256, 128),
        (512, 512, 512),
        (4864, 4096, 8192),
    ]
    shapes += [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    return shapes


def extended_shape_set():
    shapes = [(9728, 8192, 65536), (4864, 8192, 4160)]
    shapes += [
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
    shapes += [
        (256, 8192, 1024),
        (256, 1024, 8192),
        (256, 32768, 8192),
        (256, 8192, 32768),
    ]
    return shapes


@pytest.mark.parametrize(
    "M, N, K, dtype, output, layout",
    [
        (*shape, dtype, output, layout)
        for shape in basic_shape_set()
        for dtype in ["bf16"]
        for output in [True, False]
        for layout in ["TN", "TT", "NN", "NT"]
    ]
    + [
        pytest.param(*shape, dtype, output, layout, marks=pytest.mark.extended)
        for shape in extended_shape_set()
        for dtype in ["bf16"]
        for output in [True, False]
        for layout in ["TN", "TT", "NN", "NT"]
    ],
)
def test_gemm(M, N, K, dtype, output, layout):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, x_scale, w_scale, y = generate_gemm_a8w8_per_token_scale_inputs(
        M,
        N,
        K,
        dtype=dtype,
        layout=layout,
        output=output,
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)
    b = run_triton(x, weight, x_scale, w_scale, dtype, y)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
