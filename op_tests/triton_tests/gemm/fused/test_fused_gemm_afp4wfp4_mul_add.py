# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_mul_add import (
    fused_gemm_afp4wfp4_mul_add,
    fused_gemm_afp4wfp4_preshuffle_add_mul,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    run_torch as run_torch_gemm_afp4wfp4,
)
from op_tests.triton_tests.fusions.test_fused_mul_add import (
    generate_fused_mul_add_inputs,
)
from op_tests.triton_tests.fusions.test_fused_mul_add import (
    run_torch as run_torch_fused_mul_add,
)


def get_x_vals():

    x_vals = [(v, 7168, 256) for v in [1, 2, 4, 8, 16, 31, 32, 1024]]
    # x_vals += [(1, 1, 32)]  # minimal case
    return x_vals


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN"])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize(
    "shuffle_weight_scales",
    [True, False],
)
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
def test_fused_gemm_afp4wfp4_mul_add(
    M: int,
    N: int,
    K: int,
    dtype,
    layout,
    output,
    shuffle_weight_scales,
    a_type_is_scalar,
    b_type_is_scalar,
    fuse_type,
):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    if shuffle_weight_scales:
        if N % 32 > 0:
            pytest.skip(
                f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
            )
        elif K % 256 > 0:
            pytest.skip(
                f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
            )

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    (
        x,
        w,
        w_triton,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout=layout,
        output=output,
        shuffle_scales_fg=shuffle_weight_scales,
        shuffle_weight_fg=shuffle_weight_scales,
    )
    _, a, b = generate_fused_mul_add_inputs(
        [M, N], a_type_is_scalar, b_type_is_scalar, dtype
    )

    if fuse_type == 0:
        torch_out = run_torch_fused_mul_add(
            run_torch_gemm_afp4wfp4(x, w, x_scales, w_scales, torch.float32), a, b
        ).to(dtype)
    else:
        a_torch = a.to(torch.float32) if isinstance(a, torch.Tensor) else a
        b_torch = b.to(torch.float32) if isinstance(b, torch.Tensor) else b
        torch_out = (
            a_torch * b_torch
            + run_torch_gemm_afp4wfp4(x, w, x_scales, w_scales, torch.float32)
        ).to(dtype)

    if shuffle_weight_scales:
        if output:
            triton_out = fused_gemm_afp4wfp4_preshuffle_add_mul(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                a,
                b,
                dtype,
                y,
                use_aot=(dtype == torch.bfloat16 and layout == "TN"),
                fuse_type=fuse_type,
            )
        else:
            triton_out = fused_gemm_afp4wfp4_preshuffle_add_mul(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                a,
                b,
                dtype,
                use_aot=(dtype == torch.bfloat16 and layout == "TN"),
                fuse_type=fuse_type,
            )
    else:
        if output:
            triton_out = fused_gemm_afp4wfp4_mul_add(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                a,
                b,
                dtype,
                y,
                fuse_type=fuse_type,
            )
        else:
            triton_out = fused_gemm_afp4wfp4_mul_add(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                a,
                b,
                dtype,
                fuse_type=fuse_type,
            )

    torch.testing.assert_close(torch_out, triton_out, atol=0.1, rtol=0.1)
