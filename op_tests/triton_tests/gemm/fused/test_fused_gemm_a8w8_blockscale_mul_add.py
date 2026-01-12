# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import os
import torch
from aiter.ops.triton.fused_gemm_a8w8_blockscale_mul_add import (
    fused_gemm_a8w8_blockscale_mul_add,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    generate_gemm_a8w8_blockscale_inputs,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8_blockscale import (
    run_torch as run_torch_gemm_a8w8_blockscale,
)
from op_tests.triton_tests.fusions.test_fused_mul_add import (
    generate_fused_mul_add_inputs,
)
from op_tests.triton_tests.fusions.test_fused_mul_add import (
    run_torch as run_torch_fused_mul_add,
)


def get_x_vals():

    x_vals = [
        (v, 7168, 256) for v in [2, 4, 8, 16, 1024]
    ]  # TODO M = 1 triton upstream compilation error without AMDGCN_USE_BUFFER_OPS=0
    return x_vals


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN"])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize(
    "a_type_is_scalar",
    [(float, True), (int, True), (torch.Tensor, True), (torch.Tensor, False)],
)
@pytest.mark.parametrize(
    "b_type_is_scalar",
    [(float, True), (int, True), (torch.Tensor, True), (torch.Tensor, False)],
)
@pytest.mark.parametrize(
    "fuse_type",
    [0, 1],
)
def test_fused_gemm_a8w8_blockscale_mul_add(
    M: int,
    N: int,
    K: int,
    dtype,
    layout,
    output,
    a_type_is_scalar,
    b_type_is_scalar,
    fuse_type,
):

    (
        x,
        w,
        x_scales,
        w_scales,
        y,
    ) = generate_gemm_a8w8_blockscale_inputs(
        M,
        N,
        K,
        128,
        128,
        dtype,
        layout=layout,
        output=output,
    )
    _, a, b = generate_fused_mul_add_inputs(
        [M, N], a_type_is_scalar, b_type_is_scalar, dtype
    )

    if fuse_type == 0:
        torch_out = run_torch_fused_mul_add(
            run_torch_gemm_a8w8_blockscale(x, w, x_scales, w_scales, torch.float32),
            a,
            b,
        ).to(dtype)
    else:
        a_torch = a.to(torch.float32) if isinstance(a, torch.Tensor) else a
        b_torch = b.to(torch.float32) if isinstance(b, torch.Tensor) else b
        torch_out = (
            a_torch * b_torch
            + run_torch_gemm_a8w8_blockscale(x, w, x_scales, w_scales, torch.float32)
        ).to(dtype)

    if output:
        triton_out = fused_gemm_a8w8_blockscale_mul_add(
            x,
            w,
            x_scales,
            w_scales,
            a,
            b,
            dtype,
            y,
            fuse_type=fuse_type,
        )
    else:
        triton_out = fused_gemm_a8w8_blockscale_mul_add(
            x,
            w,
            x_scales,
            w_scales,
            a,
            b,
            dtype,
            fuse_type=fuse_type,
        )

    torch.testing.assert_close(torch_out, triton_out, atol=0.1, rtol=0.1)
