# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import (
    _is_gluon_available,
    batched_gemm_bf16,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes


def is_gluon_supported():
    return _is_gluon_available()


def generate_batched_gemm_a16w16_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    output: bool,
    layout: str = "TN",
    token_major: bool = False,
    add_bias: bool = True,
):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    # Scale input range so worst-case accumulation (val^2 * K) stays within bf16 max (~65504)
    hi = min(20, math.isqrt(65504 // max(K, 1)))
    hi = max(hi, 1)
    if token_major:
        if layout != "TN":
            raise ValueError("token-major inputs require layout TN")
        x = torch.randint(-hi, hi, (M, B, K), dtype=dtype, device="cuda").transpose(
            0, 1
        )
    elif layout[0] == "T":
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

    bias = torch.rand([B, 1, N], dtype=dtype, device="cuda") * 10 if add_bias else None

    y = None
    if output:
        if token_major:
            y = torch.empty((M, B, N), dtype=dtype, device="cuda").transpose(0, 1)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device="cuda")

    return x, weight, bias, y


@pytest.mark.skipif(not is_gluon_supported(), reason="requires gfx1250 Gluon")
@pytest.mark.parametrize(
    "shape", [(1, 1024, 1024, 256), (2, 2048, 1024, 512), (16, 16384, 1024, 4096)]
)
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("output", [False, True])
def test_batched_gemm_bf16_multicast(shape, token_major, output):
    x, weight, bias, y = generate_batched_gemm_a16w16_inputs(
        *shape,
        dtype=torch.bfloat16,
        output=output,
        token_major=token_major,
        add_bias=False
    )
    result = batched_gemm_bf16(
        x, weight, bias, YQ=y, backend="gluon", kernel_type="multicast"
    )
    if output:
        assert result is y
    for batch in range(shape[0]):
        expected = (x[batch].float() @ weight[batch].float().T).to(torch.bfloat16)
        torch.testing.assert_close(result[batch], expected, atol=0, rtol=0)


@pytest.mark.skipif(not is_gluon_supported(), reason="requires gfx1250 Gluon")
@pytest.mark.parametrize("invalid", ["bias", "tail", "dtype", "alias", "layout"])
def test_batched_gemm_bf16_multicast_rejects_unsupported(invalid):
    x, weight, _, y = generate_batched_gemm_a16w16_inputs(
        1, 1024, 1024, 1024, dtype=torch.bfloat16, output=True, add_bias=False
    )
    bias = None
    if invalid == "bias":
        bias = torch.zeros((1, 1, 1024), device="cuda", dtype=torch.bfloat16)
    elif invalid == "tail":
        x = x[:, :-1]
    elif invalid == "dtype":
        x = x.to(torch.float16)
    elif invalid == "alias":
        y = x
    elif invalid == "layout":
        x = x.transpose(1, 2)
    with pytest.raises(ValueError):
        batched_gemm_bf16(
            x, weight, bias, YQ=y, backend="gluon", kernel_type="multicast"
        )


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


@pytest.mark.parametrize(
    "dtype, b, m, n, k, output",
    [
        (dtype, b, *shape, output)
        for dtype in ["bf16"]
        for b in [4, 8, 16]
        for shape in get_shapes(max_size=16384 * 1024 * 4096)
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
        for shape in get_shapes(max_size=16384 * 1024 * 4096)
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
