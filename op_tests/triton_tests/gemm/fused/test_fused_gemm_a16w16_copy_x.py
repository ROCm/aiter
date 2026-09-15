# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import pytest

from aiter import dtypes
from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_copy_x import (
    fused_gemm_a16w16_copy_x,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)


def get_x_vals():
    x_vals = [(1, 1, 1)]
    x_vals += [(3, 5, 2)]
    x_vals += [(1024, 1024, 1024)]
    x_vals += [(2048, 2048, 2048)]
    # DSv4 router gate: num_tokens x 384 x 7168
    x_vals += [(2**i, 384, 7168) for i in range(5, 9)]
    # DSR1 router GEMM
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]
    return x_vals


@pytest.mark.parametrize("M, N, K", get_x_vals())
def test_fused_gemm_a16w16_copy_x(M: int, N: int, K: int):
    torch.cuda.empty_cache()
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )

    torch_y = F.linear(x, w, bias=None)
    torch_x_copy = x.to(dtypes.fp8)

    triton_y, triton_x_copy = fused_gemm_a16w16_copy_x(x, w)

    torch.testing.assert_close(triton_y, torch_y, atol=1e-1, rtol=1e-2)
    # x_copy comparison: both are produced by an identical BF16 -> FP8 cast,
    # so we expect bitwise equality. Compare via BF16 to avoid FP8 dtype
    # ambiguities across gfx targets.
    torch.testing.assert_close(
        triton_x_copy.to(torch.bfloat16),
        torch_x_copy.to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )


def get_fewer_x_vals():
    x_vals = [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(256, 512, 8192)]
    x_vals += [(1024, 1024, 1024)]
    return x_vals


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_fused_gemm_a16w16_copy_x_activation(
    M: int, N: int, K: int, dtype, output, activation
):
    x, w, _, _, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=output)

    torch_y = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_y = F.gelu(torch_y)
    elif activation == "gelu_tanh":
        torch_y = F.gelu(torch_y, approximate="tanh")
    elif activation == "silu":
        torch_y = F.silu(torch_y)
    torch_x_copy = x.to(dtypes.fp8)

    triton_y, triton_x_copy = fused_gemm_a16w16_copy_x(
        x,
        w,
        bias=None,
        dtype=dtype,
        y=y,
        activation=activation,
    )

    torch.testing.assert_close(triton_y, torch_y, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(
        triton_x_copy.to(torch.bfloat16),
        torch_x_copy.to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("skip_reduce", [True, False])
def test_fused_gemm_a16w16_copy_x_skip_reduce(M: int, N: int, K: int, skip_reduce):
    torch.cuda.empty_cache()
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )

    torch_y = F.linear(x, w, bias=None)
    torch_x_copy = x.to(dtypes.fp8)

    triton_y, triton_x_copy = fused_gemm_a16w16_copy_x(x, w, skip_reduce=skip_reduce)

    if triton_y.dim() == 3:
        triton_y = triton_y.sum(axis=0).to(torch.bfloat16)

    torch.testing.assert_close(triton_y, torch_y, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(
        triton_x_copy.to(torch.bfloat16),
        torch_x_copy.to(torch.bfloat16),
        atol=0.0,
        rtol=0.0,
    )
