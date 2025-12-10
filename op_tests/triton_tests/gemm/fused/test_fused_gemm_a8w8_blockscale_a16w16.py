# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
from aiter.ops.triton.fused_gemm_a8w8_blockscale_a16w16 import (
    fused_gemm_a8w8_blockscale_a16w16,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    run_torch as run_torch_fp8,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)
import torch.nn.functional as F


block_shape = (128, 128)


def run_torch(
    x_fp8,
    w_fp8,
    x_fp8_scale,
    w_fp8_scale,
    x_bf16,
    w_bf16,
    bias_fp8,
    bias_bf16,
    dtype=torch.bfloat16,
):
    y_fp8 = run_torch_fp8(x_fp8, w_fp8, x_fp8_scale, w_fp8_scale, dtype)
    if bias_fp8 is not None:
        y_fp8 += bias_fp8
    y_bf16 = F.linear(x_bf16, w_bf16, bias=bias_bf16)
    return y_fp8, y_bf16


def run_triton(
    x_fp8,
    w_fp8,
    x_fp8_scale,
    w_fp8_scale,
    x_bf16,
    w_bf16,
    bias_fp8,
    bias_bf16,
    dtype=torch.bfloat16,
    y_fp8=None,
    y_bf16=None,
    skip_reduce=False,
):
    return fused_gemm_a8w8_blockscale_a16w16(
        x_fp8,
        w_fp8,
        x_fp8_scale,
        w_fp8_scale,
        x_bf16,
        w_bf16,
        bias_fp8=bias_fp8,
        bias_bf16=bias_bf16,
        dtype=dtype,
        y_fp8=y_fp8,
        y_bf16=y_bf16,
        skip_reduce=skip_reduce,
    )


def get_x_vals():

    x_vals = [(1, 1, 1, 128)]  # minimal case
    x_vals += [
        (m, n1, n2, k)
        for k in [1024, 8192, 7168]
        for n2 in [256, 512]
        for n1 in [256, 512]
        for m in [1, 8, 32, 64, 128, 8192]
    ]
    return x_vals


@pytest.mark.parametrize("M, N1, N2, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("skip_reduce", [True, False])
def test_gemm(dtype, M, N1, N2, K, output, skip_reduce):
    block_shape_n, block_shape_k = block_shape

    x_fp8, w_fp8, x_fp8_scale, w_fp8_scale, y_fp8 = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N1,
            K,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            output=output,
        )
    )
    x_bf16, w_bf16, bias_bf16, _, y_bf16 = generate_gemm_a16w16_inputs(
        M, N2, K, dtype, output=output, bias=True
    )
    bias_bf16 = torch.randn((N2,), dtype=bias_bf16.dtype, device=bias_bf16.device)
    bias_fp8 = torch.randn((N1,), dtype=bias_bf16.dtype, device=bias_bf16.device)
    y_torch_fp8, y_torch_bf16 = run_torch(
        x_fp8,
        w_fp8,
        x_fp8_scale,
        w_fp8_scale,
        x_bf16,
        w_bf16,
        bias_fp8,
        bias_bf16,
        dtype,
    )
    y_triton_fp8, y_triton_bf16 = run_triton(
        x_fp8,
        w_fp8,
        x_fp8_scale,
        w_fp8_scale,
        x_bf16,
        w_bf16,
        bias_fp8,
        bias_bf16,
        dtype,
        y_fp8,
        y_bf16,
        skip_reduce=skip_reduce,
    )

    if y_triton_fp8.dim() == 3:
        y_triton_fp8 = y_triton_fp8.sum(axis=0).to(dtype=dtype)
        y_triton_bf16 = y_triton_bf16.sum(axis=0).to(dtype=dtype)

    triton.testing.assert_close(y_torch_fp8, y_triton_fp8, atol=0.1, rtol=1e-1)
    triton.testing.assert_close(y_torch_bf16, y_triton_bf16, atol=0.1, rtol=1e-1)
