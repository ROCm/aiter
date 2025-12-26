# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import pytest
from aiter.ops.triton.fused_gemm_afp4wfp4_a16w16 import (
    fused_gemm_afp4wfp4_a16w16,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    generate_gemm_afp4wfp4_inputs,
)
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import (
    run_torch as run_torch_fp4,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)
import torch.nn.functional as F
import aiter.ops.triton.utils._triton.arch_info as arch_info


def run_torch(
    x_fp4,
    w_fp4,
    x_fp4_scale,
    w_fp4_scale,
    x_bf16,
    w_bf16,
    bias_fp4,
    bias_bf16,
    dtype=torch.bfloat16,
):
    y_fp4 = run_torch_fp4(x_fp4, w_fp4, x_fp4_scale, w_fp4_scale, dtype)
    if bias_fp4 is not None:
        y_fp4 += bias_fp4
    y_bf16 = F.linear(x_bf16, w_bf16, bias=bias_bf16)
    return y_fp4.to(dtype), y_bf16.to(dtype)


def run_triton(
    x_fp4,
    w_fp4,
    x_fp4_scale,
    w_fp4_scale,
    x_bf16,
    w_bf16,
    bias_fp4,
    bias_bf16,
    dtype=torch.bfloat16,
    y_fp4=None,
    y_bf16=None,
    skip_reduce=False,
    is_fp4_preshuffled=True,
):
    return fused_gemm_afp4wfp4_a16w16(
        x_fp4,
        w_fp4,
        x_fp4_scale,
        w_fp4_scale,
        x_bf16,
        w_bf16,
        is_fp4_preshuffled=is_fp4_preshuffled,
        bias_fp4=bias_fp4,
        bias_bf16=bias_bf16,
        dtype=dtype,
        y_fp4=y_fp4,
        y_bf16=y_bf16,
        skip_reduce=skip_reduce,
    )


def get_x_vals():

    x_vals = [
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
@pytest.mark.parametrize("fp4_shuffle", [True, False])
def test_gemm(dtype, M, N1, N2, K, output, skip_reduce, fp4_shuffle):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    (
        x_fp4,
        w_fp4,
        w_fp4_triton,
        x_fp4_scale,
        w_fp4_scale,
        x_fp4_scale_triton,
        w_fp4_scale_triton,
        out_dtype,
        y_fp4,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N1,
        K,
        dtype,
        layout="TN",
        output=output,
        shuffle_scales_fg=fp4_shuffle,
        shuffle_weight_fg=fp4_shuffle,
    )

    x_bf16, w_bf16, bias_bf16, _, y_bf16 = generate_gemm_a16w16_inputs(
        M, N2, K, dtype, output=output, bias=True
    )
    bias_bf16 = torch.randn((N2,), dtype=bias_bf16.dtype, device=bias_bf16.device)
    bias_fp4 = torch.randn((N1,), dtype=bias_bf16.dtype, device=bias_bf16.device)
    y_torch_fp4, y_torch_bf16 = run_torch(
        x_fp4,
        w_fp4,
        x_fp4_scale,
        w_fp4_scale,
        x_bf16,
        w_bf16,
        bias_fp4,
        bias_bf16,
        dtype,
    )
    y_triton_fp4, y_triton_bf16 = run_triton(
        x_fp4,
        w_fp4_triton,
        x_fp4_scale_triton,
        w_fp4_scale_triton,
        x_bf16,
        w_bf16,
        bias_fp4,
        bias_bf16,
        dtype,
        y_fp4,
        y_bf16,
        skip_reduce=skip_reduce,
        is_fp4_preshuffled=fp4_shuffle,
    )

    if y_triton_fp4.dim() == 3:
        y_triton_fp4 = y_triton_fp4.sum(axis=0).to(dtype=dtype)
        y_triton_bf16 = y_triton_bf16.sum(axis=0).to(dtype=dtype)

    triton.testing.assert_close(y_torch_bf16, y_triton_bf16, atol=0.1, rtol=1e-1)
    triton.testing.assert_close(y_torch_fp4, y_triton_fp4, atol=0.1, rtol=1e-1)
