# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL strided-batched MXFP preshuffle GEMM (gfx950).

Covers out[b] = dequant(x[b]) @ dequant(w[b]).T across two variants and two layouts:
  - variants: a4w4 (MXFP4 A x MXFP4 B), a8w4 (MXFP8 E4M3 A x MXFP4 B)
  - layouts:  bmn ([B,M,*] A/C), mbn ([M,B,*] A/C, deepseek-v4 grouped-output)
"""

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SCALE_GROUP_SIZE = 32

# FlyDSL kernel constraints: K % 256 == 0, N % tile_n == 0, single TN layout.
SHAPES = [
    # (B, M, N, K)
    (1, 1, 1280, 8192),
    (2, 32, 1280, 8192),
    (3, 128, 1280, 8192),
    (5, 100, 512, 512),  # ragged M (not a multiple of 32) -> OOB rows read 0
    (8, 256, 8192, 1024),
    (2, 512, 1024, 4096),
    (4, 1024, 1280, 8192),
]


def _quant_fp4(x_2d):
    """Per-1x32 MXFP4 quant (aiter triton) -> (codes uint8 [.,K//2], scale uint8 [.,K//32])."""
    qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    codes, scale = qf(x_2d, shuffle=False)
    return codes.view(torch.uint8), scale.view(torch.uint8)


def gen_inputs(variant, B, M, N, K, dtype):
    """Returns (x, w, x_scales, w_scales, ref) with unshuffled scales (wrapper shuffles)."""
    torch.manual_seed(5)
    wc = [_quant_fp4(torch.randn(N, K, dtype=dtype)) for _ in range(B)]
    w = torch.stack([c for c, _ in wc])
    w_scales = torch.stack([s for _, s in wc])

    if variant == "a8w4":
        x = (torch.randn(B, M, K) * 4.0).to(torch.float8_e4m3fn).view(torch.uint8)
        x_scales = torch.randint(
            124, 128, (B, M, K // SCALE_GROUP_SIZE), dtype=torch.uint8
        )
        x_f32 = x.view(torch.float8_e4m3fn).to(torch.float32)
    else:  # a4w4
        xc = [_quant_fp4(torch.randn(M, K, dtype=dtype)) for _ in range(B)]
        x = torch.stack([c for c, _ in xc])
        x_scales = torch.stack([s for _, s in xc])
        x_f32 = fp4_utils.mxfp4_to_f32(x)

    w_f32 = fp4_utils.mxfp4_to_f32(w)
    xs = fp4_utils.e8m0_to_f32(x_scales.repeat_interleave(SCALE_GROUP_SIZE, -1))
    ws = fp4_utils.e8m0_to_f32(w_scales.repeat_interleave(SCALE_GROUP_SIZE, -1))
    ref = torch.bmm(x_f32 * xs, (w_f32 * ws).transpose(1, 2)).to(dtype)
    return x, w, x_scales, w_scales, ref


def _run(variant, layout, B, M, N, K, dtype):
    a_dtype = "fp8" if variant == "a8w4" else "fp4"
    x, w, x_scales, w_scales, ref = gen_inputs(variant, B, M, N, K, dtype)
    out = flydsl_batched_gemm_mxfp4(
        x,
        w,
        x_scales,
        w_scales,
        dtype,
        a_dtype=a_dtype,
        layout=layout,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    assert out.shape == (B, M, N)
    # Reordered MFMA reduction vs torch.bmm drifts a few low bits; accept <5% mismatch at 1e-2.
    err = checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"flydsl {variant} {layout} {B}x{M}x{N}x{K}",
    )
    assert err < 0.05, f"flydsl {variant} {layout} mismatch ratio {err:.4f}"
    return err


@pytest.mark.parametrize("B, M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["bmn", "mbn"])
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4(variant, layout, B, M, N, K, dtype):
    if get_gfx() != "gfx950":
        pytest.skip(f"FlyDSL MXFP preshuffle GEMM requires gfx950, got {get_gfx()}")
    torch.cuda.empty_cache()
    _run(variant, layout, B, M, N, K, dtype)
