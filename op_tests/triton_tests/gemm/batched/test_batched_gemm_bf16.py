# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math
import torch
import pytest
import functools
from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
    batched_gemm_bf16,
    _is_gluon_available,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype, get_fp8_dtypes
import torch.nn.functional as F
from typing import Union


def is_gluon_supported():
    return _is_gluon_available()


def generate_batched_gemm_a16w16_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: Union[torch.dtype, str],
    output: bool,
    layout: str = "TN",
):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    # Scale input range so worst-case accumulation (val^2 * K) stays within bf16 max (~65504)
    hi = min(20, int(math.isqrt(65504 // max(K, 1))))
    hi = max(hi, 1)
    if layout[0] == "T":
        x = torch.randint(-hi, hi, (B, M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randint(-hi, hi, (B, K, M), dtype=dtype, device="cuda").permute(
            0, 2, 1
        )

    if layout[1] == "N":
        weight = torch.randint(-hi, hi, (B, N, K), dtype=dtype, device="cuda")
    else:
        weight = torch.randint(-hi, hi, (B, K, N), dtype=dtype, device="cuda").permute(
            0, 2, 1
        )

    bias = torch.rand([B, 1, N], dtype=dtype, device="cuda") * 10

    y = None
    if output:
        y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, bias, y


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


def run_triton(
    x,
    weight,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    backend=None,
    kernel_type="bandwidth_bound",
):
    return batched_gemm_bf16(
        x, weight, bias, dtype, YQ=y, backend=backend, kernel_type=kernel_type
    )


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 4)]
    x_vals += [
        (1, 1024, 4096),
        (32, 1024, 4096),
        (64, 1024, 4096),
        (128, 1024, 4096),
        (256, 1024, 4096),
        (16384, 1024, 4096),
    ]
    x_vals += [(1, 1, 1)]  # minimal case
    return x_vals


def minimal_x_vals(num_vals=20):
    """
    Returns the num_vals smallest test cases. Useful for generating a subset to quickly test on.
    """
    x_vals = get_x_vals()
    num_ops = [(i, functools.reduce(lambda x, y: x * y, i)) for i in x_vals]
    sorted_x_vals = sorted(num_ops, key=lambda x: x[1])
    return [i[0] for i in sorted_x_vals[: min(num_vals, len(sorted_x_vals))]]


@pytest.mark.parametrize(
    "dtype, b, m, n, k, output",
    [
        (dtype, b, *shape, output)
        for dtype in ["bf16"]
        for b in [4, 8, 16]
        for shape in get_x_vals()
        for output in [True, False]
    ],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_batched_gemm_bf16(dtype, b, m, n, k, output, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and kernel_type == "compute_bound":
        pytest.skip("gluon compute_bound has LDS race with batched multi-occupancy")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")

    torch.cuda.empty_cache()

    x, weight, bias, y = generate_batched_gemm_a16w16_inputs(b, m, n, k, dtype, output)
    dtype = str_to_torch_dtype[dtype]
    a = run_torch(x, weight, bias, dtype)
    b = run_triton(x, weight, bias, dtype, y, backend=backend, kernel_type=kernel_type)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)


@pytest.mark.parametrize(
    "dtype, b, m, n, k, layout, output",
    [
        (dtype, b, *shape, layout, output)
        for dtype in ["bf16"]
        for b in [4, 8, 16]
        for shape in minimal_x_vals()
        for output in [True, False]
        for layout in ["TT", "NN", "NT"]
    ],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_batched_gemm_bf16_layout(
    dtype, b, m, n, k, layout, output, backend, kernel_type
):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and kernel_type == "compute_bound":
        pytest.skip("gluon compute_bound has LDS race with batched multi-occupancy")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")

    torch.cuda.empty_cache()

    x, weight, bias, y = generate_batched_gemm_a16w16_inputs(
        b, m, n, k, dtype, output, layout
    )
    dtype = str_to_torch_dtype[dtype]
    a = run_torch(x, weight, bias, dtype)
    b = run_triton(x, weight, bias, dtype, y, backend=backend, kernel_type=kernel_type)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
