# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import pytest

from aiter.ops.triton.gemm.fused.fused_gemm_a16w16_quant_x import (
    fused_gemm_a16w16_quant_x,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)

_QUANT_BLOCK_SIZE = 32
# 0xFF800000 in two's complement int32. Mask keeps sign + 8-bit exponent + top mantissa bit.
_E8M0_MASK_INT32 = -8388608


def torch_mxfp8_quant_from_fp32(x_fp32: torch.Tensor):
    """Bit-faithful port of `_mxfp8_quant_kernel` quant logic, taking fp32 input.

    Computes per-1x32 e8m0 scale (uint8) and FP8 e4m3fn values.
    """
    assert x_fp32.dim() == 2, f"x_fp32 must be 2D, got {x_fp32.dim()}"
    M, K = x_fp32.shape
    assert K % _QUANT_BLOCK_SIZE == 0
    Ng = K // _QUANT_BLOCK_SIZE
    x_2d = x_fp32.reshape(M, Ng, _QUANT_BLOCK_SIZE).to(torch.float32)
    amax = torch.amax(torch.abs(x_2d), dim=-1, keepdim=True)  # (M, Ng, 1)

    amax_i32 = amax.contiguous().view(torch.int32)
    amax_i32 = (amax_i32 + 0x200000) & _E8M0_MASK_INT32
    amax_p2 = amax_i32.view(torch.float32)

    scale_unbiased = torch.log2(amax_p2).floor() - 8
    scale_unbiased = torch.clamp(scale_unbiased, min=-127, max=127)
    scale_e8m0 = (scale_unbiased.to(torch.int32) + 127).to(torch.uint8)
    quant_scale = torch.exp2(-scale_unbiased)

    qx_2d = x_2d * quant_scale
    qx = qx_2d.reshape(M, K)
    y_fp8 = qx.to(torch.float8_e4m3fn)
    s = scale_e8m0.reshape(M, Ng)
    return y_fp8, s


def get_x_vals():
    x_vals = [(1024, 1024, 1024)]
    x_vals += [(2048, 2048, 2048)]
    # DSv4 router gate: num_tokens x 384 x 7168
    x_vals += [(2**i, 384, 7168) for i in range(5, 9)]
    # DSR1 router GEMM
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]
    return x_vals


def _assert_quant_close(triton_x_quant, triton_x_scales, x):
    ref_x_quant, ref_x_scales = torch_mxfp8_quant_from_fp32(x.to(torch.float32))
    # e8m0 scales: bit-exact (integer-only after fp32 cast).
    torch.testing.assert_close(triton_x_scales, ref_x_scales)
    # Quantized values: compare via uint8 view (allow off-by-1 for any rounding
    # subtlety in the fp32->fp8 cast).
    torch.testing.assert_close(
        triton_x_quant.view(torch.uint8).to(torch.int32),
        ref_x_quant.view(torch.uint8).to(torch.int32),
        atol=1,
        rtol=0,
    )


@pytest.mark.parametrize("M, N, K", get_x_vals())
def test_fused_gemm_a16w16_quant_x(M: int, N: int, K: int):
    torch.cuda.empty_cache()
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )

    torch_y = F.linear(x, w, bias=None)

    triton_y, triton_x_quant, triton_x_scales = fused_gemm_a16w16_quant_x(x, w)

    torch.testing.assert_close(triton_y, torch_y, atol=1e-1, rtol=1e-2)
    _assert_quant_close(triton_x_quant, triton_x_scales, x)


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
def test_fused_gemm_a16w16_quant_x_activation(
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

    triton_y, triton_x_quant, triton_x_scales = fused_gemm_a16w16_quant_x(
        x,
        w,
        bias=None,
        dtype=dtype,
        y=y,
        activation=activation,
    )

    torch.testing.assert_close(triton_y, torch_y, atol=1e-1, rtol=1e-2)
    _assert_quant_close(triton_x_quant, triton_x_scales, x)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("skip_reduce", [True, False])
def test_fused_gemm_a16w16_quant_x_skip_reduce(M: int, N: int, K: int, skip_reduce):
    torch.cuda.empty_cache()
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )

    torch_y = F.linear(x, w, bias=None)

    triton_y, triton_x_quant, triton_x_scales = fused_gemm_a16w16_quant_x(
        x, w, skip_reduce=skip_reduce
    )

    if triton_y.dim() == 3:
        triton_y = triton_y.sum(axis=0).to(torch.bfloat16)

    torch.testing.assert_close(triton_y, torch_y, atol=1e-3, rtol=1e-2)
    _assert_quant_close(triton_x_quant, triton_x_scales, x)
