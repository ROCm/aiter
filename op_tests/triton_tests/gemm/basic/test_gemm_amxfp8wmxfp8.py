# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_amxfp8wmxfp8 import (
    gemm_amxfp8wmxfp8,
    gemm_amxfp8wmxfp8_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info

SCALE_GROUP_SIZE = 32  # both A and W: 1x32 e8m0 scale group
FP8_MAX = 448.0  # e4m3 max


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Decode unsigned-biased e8m0 (uint8) to fp32. Bias 127, value = 2^(b-127)."""
    return torch.exp2((x.to(torch.int32) - 127).to(torch.float32))


def generate_gemm_amxfp8wmxfp8_inputs(M: int, N: int, K: int, shuffle: bool = False):
    """Returns ``(x_fp8, w_fp8, w_kernel, x_scales, w_scales)`` with 1x32 e8m0 scales.

    ``w_fp8`` is always the unshuffled weight (for the fp32 reference).
    ``w_kernel`` is what the kernel consumes: identical to ``w_fp8`` when
    ``shuffle=False``, or the (16, 16) preshuffled bytes when ``shuffle=True``.
    """
    # Small random fp32 -> fp8 e4m3fn, kept inside e4m3 range so the cast is exact-ish.
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
        125, 130, (N, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda"
    )

    if shuffle:
        # Imported lazily: shuffle_weight pulls in aiter.ops.shuffle, which needs
        # the full (non-Triton-only) build. Only the preshuffle path requires it.
        from aiter.ops.triton.utils.shuffle import shuffle_weight

        # shuffle_weight operates on raw bytes; view as uint8 to avoid dtype quirks.
        w_kernel = shuffle_weight(w_fp8.view(torch.uint8), layout=(16, 16))
    else:
        w_kernel = w_fp8

    return x_fp8, w_fp8, w_kernel, x_scales, w_scales


def run_torch_gemm_amxfp8wmxfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: dequant both operands to fp32 (1x32 scales) and run torch.mm."""
    M, K = x_fp8.shape
    N, _ = w_fp8.shape

    x_f32 = x_fp8.to(torch.float32)
    w_f32 = w_fp8.to(torch.float32)

    x_s_f32 = e8m0_to_f32(x_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_s_f32 = e8m0_to_f32(w_scales).repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    assert x_s_f32.shape == (M, K)
    assert w_s_f32.shape == (N, K)

    x_dq = x_f32 * x_s_f32
    w_dq = w_f32 * w_s_f32
    return torch.mm(x_dq, w_dq.T).to(out_dtype)


def get_shapes():
    # (M, N, K), K % 32 == 0 (1x32 microscaling requirement).
    return [
        (m, n, k)
        for m in [1, 16, 32, 64, 128, 256]
        for n, k in [
            (1536, 4096),
            (4096, 1024),
            (512, 4096),
            (2048, 7168),
            (768, 384),
            (2560, 96),  # exercises the non-EVEN_K masking path (K % BLOCK_K != 0)
        ]
    ]


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_amxfp8wmxfp8(M: int, N: int, K: int, dtype: torch.dtype):
    torch.manual_seed(0)
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    torch.cuda.empty_cache()

    x_fp8, w_fp8, w_kernel, x_scales, w_scales = generate_gemm_amxfp8wmxfp8_inputs(
        M, N, K
    )

    torch_out = run_torch_gemm_amxfp8wmxfp8(x_fp8, w_fp8, x_scales, w_scales, dtype)
    triton_out = gemm_amxfp8wmxfp8(x_fp8, w_kernel, x_scales, w_scales, dtype=dtype)

    torch.testing.assert_close(triton_out, torch_out, atol=0.03, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_amxfp8wmxfp8_preshuffle(M: int, N: int, K: int, dtype: torch.dtype):
    torch.manual_seed(0)
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    if N % 16 != 0 or K % 32 != 0:
        pytest.skip("Preshuffle requires N % 16 == 0 and K % 32 == 0")
    torch.cuda.empty_cache()

    x_fp8, w_fp8, w_kernel, x_scales, w_scales = generate_gemm_amxfp8wmxfp8_inputs(
        M, N, K, shuffle=True
    )

    torch_out = run_torch_gemm_amxfp8wmxfp8(x_fp8, w_fp8, x_scales, w_scales, dtype)
    triton_out = gemm_amxfp8wmxfp8_preshuffle(
        x_fp8, w_kernel, x_scales, w_scales, dtype=dtype
    )

    torch.testing.assert_close(triton_out, torch_out, atol=0.03, rtol=1e-2)
