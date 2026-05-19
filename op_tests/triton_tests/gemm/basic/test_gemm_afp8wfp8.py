# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
    gemm_afp8wfp8,
    gemm_afp8wfp8_preshuffle,
)
from aiter.ops.shuffle import shuffle_weight
import aiter.ops.triton.utils._triton.arch_info as arch_info

SCALE_GROUP_SIZE = 32  # A: 1x32 e8m0 scale group
W_SCALE_K_GROUP = 128  # B: 128 in K direction
W_SCALE_N_GROUP = 128  # B: 128 in N direction
FP8_MAX = 448.0  # e4m3 max


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Decode unsigned-biased e8m0 (uint8) to fp32. Bias 127, value = 2^(b-127)."""
    return torch.exp2((x.to(torch.int32) - 127).to(torch.float32))


def generate_inputs(M: int, N: int, K: int, seed: int = 0):
    torch.manual_seed(seed)
    # Small random fp32 → fp8 e4m3fn, kept inside e4m3 range so the cast is exact-ish.
    x_f32 = torch.randn((M, K), dtype=torch.float32, device="cuda")
    w_f32 = torch.randn((N, K), dtype=torch.float32, device="cuda")
    x_f32 = torch.clamp(x_f32, -FP8_MAX, FP8_MAX)
    w_f32 = torch.clamp(w_f32, -FP8_MAX, FP8_MAX)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)
    w_fp8 = w_f32.to(torch.float8_e4m3fn)

    # e8m0 scales near 127 (== 1.0) so the dequant has unit-ish magnitude.
    x_scales = torch.randint(
        125, 130, (M, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        125,
        130,
        (N // W_SCALE_N_GROUP, K // W_SCALE_K_GROUP),
        dtype=torch.uint8,
        device="cuda",
    )
    return x_fp8, w_fp8, x_scales, w_scales


def run_torch_gemm_afp8wfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: dequant both operands to fp32 and run torch.mm."""
    M, K = x_fp8.shape
    N, _ = w_fp8.shape

    x_view = x_fp8 if x_fp8.dtype != torch.uint8 else x_fp8.view(torch.float8_e4m3fn)
    w_view = w_fp8 if w_fp8.dtype != torch.uint8 else w_fp8.view(torch.float8_e4m3fn)
    x_f32 = x_view.to(torch.float32)
    w_f32 = w_view.to(torch.float32)

    x_s_f32 = e8m0_to_f32(x_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    assert x_s_f32.shape == (M, K)

    w_s_f32 = e8m0_to_f32(w_scales)
    w_s_f32 = w_s_f32.repeat_interleave(W_SCALE_N_GROUP, dim=0).repeat_interleave(
        W_SCALE_K_GROUP, dim=1
    )
    assert w_s_f32.shape == (N, K)

    x_dq = x_f32 * x_s_f32
    w_dq = w_f32 * w_s_f32
    return torch.mm(x_dq, w_dq.T).to(out_dtype)


def get_shapes():
    # (M, N, K), with N % 128 == 0 and K % 128 == 0 to fit the 128x128 W-scale layout.
    return [
        (m, n, k)
        for m in [1, 4, 8, 16, 32, 64, 128]
        for n, k in [
            (1536, 4096),
            (4096, 1024),
            (512, 4096),
            (8192, 1024),
            (2048, 7168),
            (7168, 2048),
            (768, 7168),
            (7168, 384),
            (8192, 1536),
        ]
    ]


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_afp8wfp8(M: int, N: int, K: int, dtype: torch.dtype):
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    torch.cuda.empty_cache()

    x_fp8, w_fp8, x_scales, w_scales = generate_inputs(M, N, K)

    torch_out = run_torch_gemm_afp8wfp8(x_fp8, w_fp8, x_scales, w_scales, dtype)
    triton_out = gemm_afp8wfp8(x_fp8, w_fp8, x_scales, w_scales, dtype=dtype)

    torch.testing.assert_close(triton_out, torch_out, atol=0.5, rtol=5e-2)


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_afp8wfp8_preshuffle(M: int, N: int, K: int, dtype: torch.dtype):
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    if N % 16 != 0 or K % 32 != 0:
        pytest.skip("Preshuffle requires N % 16 == 0 and K % 32 == 0")
    torch.cuda.empty_cache()

    x_fp8, w_fp8, x_scales, w_scales = generate_inputs(M, N, K)

    # Shuffle the weight tensor in place (layout=(16,16)). The shuffler operates on
    # raw bytes; view as uint8 to avoid any dtype quirks, then view back.
    w_uint8 = w_fp8.view(torch.uint8)
    w_shuffled = shuffle_weight(w_uint8, layout=(16, 16))

    torch_out = run_torch_gemm_afp8wfp8(x_fp8, w_fp8, x_scales, w_scales, dtype)
    triton_out = gemm_afp8wfp8_preshuffle(
        x_fp8, w_shuffled, x_scales, w_scales, dtype=dtype
    )

    torch.testing.assert_close(triton_out, torch_out, atol=0.5, rtol=5e-2)


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_gemm_afp8wfp8_preallocated_output(M: int, N: int, K: int):
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    torch.cuda.empty_cache()

    dtype = torch.bfloat16
    x_fp8, w_fp8, x_scales, w_scales = generate_inputs(M, N, K)
    y = torch.empty((M, N), dtype=dtype, device="cuda")
    out = gemm_afp8wfp8(x_fp8, w_fp8, x_scales, w_scales, dtype=dtype, y=y)
    assert out.data_ptr() == y.data_ptr()

    torch_out = run_torch_gemm_afp8wfp8(x_fp8, w_fp8, x_scales, w_scales, dtype)
    torch.testing.assert_close(out, torch_out, atol=0.5, rtol=5e-2)
