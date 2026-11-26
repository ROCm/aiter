# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8 import batched_gemm_a8w8
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes


def generate_batched_gemm_a8w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    output=bool,
    layout: str = "TN",
):
    """
    Returns:
        - x: shape (B, M, K)
        - weight: shape (B, N, K)
        - x_scale: shape (B, M, 1)
        - w_scale: shape (B, 1, N)
    """
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    if layout[0] == "T":
        x = torch.randint(-20, 20, (B, M, K), dtype=torch.int8, device="cuda")
    else:
        x = torch.randint(-20, 20, (B, K, M), dtype=torch.int8, device="cuda").permute(
            0, 2, 1
        )

    if layout[1] == "N":
        weight = torch.randint(-20, 20, (B, N, K), dtype=torch.int8, device="cuda")
    else:
        weight = torch.randint(
            -20, 20, (B, K, N), dtype=torch.int8, device="cuda"
        ).permute(0, 2, 1)

    x_scale = torch.rand([B, M, 1], dtype=torch.float32, device="cuda") + 1e-6
    w_scale = torch.rand([B, 1, N], dtype=torch.float32, device="cuda") + 1e-6
    bias = torch.rand([B, 1, N], dtype=dtype, device="cuda") * 10

    y = None
    if output:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, x_scale, w_scale, bias, y


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


e5m2_type, e4m3_type = get_fp8_dtypes()


@pytest.mark.parametrize(
    "dtype, b, m, n, k, output",
    [
        (dtype, b, *shape, output)
        for output in [True, False]
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_shapes()
    ],
)
def test_batched_gemm_a8w8(dtype, b, m, n, k, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, x_scale, w_scale, bias, y = generate_batched_gemm_a8w8_inputs(
        b, m, n, k, dtype, output
    )
    a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype, y)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)


@pytest.mark.parametrize(
    "dtype, b, m, n, k, layout, output",
    [
        (dtype, b, *shape, layout, output)
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_shapes(max_size=2048 * 2048 * 2048)
        for output in [True, False]
        for layout in ["TT", "NN", "NT"]
    ],
)
def test_batched_gemm_a8w8_layout(dtype, b, m, n, k, layout, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, x_scale, w_scale, bias, y = generate_batched_gemm_a8w8_inputs(
        b, m, n, k, dtype, output, layout
    )
    a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b = run_triton(x, weight, x_scale, w_scale, bias, dtype, y)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
