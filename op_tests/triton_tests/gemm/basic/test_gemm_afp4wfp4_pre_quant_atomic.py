# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4_pre_quant_atomic import (
    gemm_afp4wfp4_pre_quant,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info

SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_pre_quant_inputs(
    M,
    N,
    K,
    layout="NN",
    output=True,
):
    """
    Generate inputs for gemm_afp4wfp4_pre_quant (a16wfp4 variant).

    x is a BF16 activation matrix (quantized on-the-fly by the kernel).
    w is a packed FP4 E2M1 weight matrix with shape (N, K//2).
    w_scales is an E8M0 per-group scale with shape (N, K//32).

    Returns: (x, w, None, w_scales, y)
      - x: bf16 activations (M, K)
      - w: packed uint8 weights (N, K//2)
      - None: placeholder (no x_scales needed; kernel quantizes x on-the-fly)
      - w_scales: uint8 E8M0 scales (N, K//32)
      - y: pre-allocated output (M, N) in bf16, or None if output=False
    """
    torch.manual_seed(5)

    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    if layout[1] == "N":
        w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w = torch.randint(0, 256, (K // 2, N), dtype=torch.uint8, device="cuda").T

    w_scales = torch.randint(
        124, 128, (N, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda"
    )

    y = None
    if output:
        y = torch.zeros((M, N), dtype=torch.bfloat16, device="cuda")

    return x, w, None, w_scales, y


def get_x_vals():
    x_vals = [
        (1, 4096, 4096),
        (32, 4096, 4096),
        (64, 4096, 4096),
        (128, 4096, 4096),
        (256, 4096, 4096),
        (512, 4096, 4096),
        (1024, 4096, 4096),
        (1024, 14336, 4096),
    ]
    return x_vals


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_afp4wfp4_pre_quant_atomic(M, N, K, layout, output):
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    c_dtype = torch.bfloat16
    x, w, _, w_scales, y = generate_gemm_afp4wfp4_pre_quant_inputs(
        M, N, K, layout=layout, output=output
    )

    result = gemm_afp4wfp4_pre_quant(x, w, w_scales, c_dtype, y)
    assert result.shape == (M, N)
    assert result.dtype == c_dtype
