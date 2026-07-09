# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL strided-batched MXFP preshuffle GEMM (gfx950).

Covers out[b] = dequant(x[b]) @ dequant(w[b]).T across two variants and two layouts:
  - variants: a4w4 (MXFP4 A x MXFP4 B), a8w4 (MXFP8 E4M3 A x MXFP4 B)
  - layouts:  bmn ([B,M,*] A/C), mbn ([M,B,*] A/C, deepseek-v4 grouped-output)
plus odd/large M (ragged-tail OOB), a tile-size sweep, and a CUDA-graph run.
"""

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4
from aiter.test_common import checkAllclose, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SCALE_GROUP_SIZE = 32

# (B, M, N, K). Odd/large M (not multiples of the 32-row scale chunk or the tile) exercise
# the ragged-M tail: rows past M read 0. N % tile_n == 0, K % 256 == 0.
SHAPES = [
    (1, 1, 1280, 8192),  # M=1 decode
    (3, 15, 512, 1024),  # M=15 ragged
    (2, 64, 1024, 4096),  # M=64
    (2, 1023, 768, 2048),  # M=1023 ragged
    (1, 4097, 512, 512),  # M=4097 large ragged
]

# (tile_m, tile_n, tile_k): tile_m % 16, tile_n % 64, tile_k in {128, 256}, and the coop A
# tile A_LDS_B = tile_m * (tile_k or tile_k//2) a multiple of 4096.
TILES = [
    (128, 128, 256),
    (128, 256, 256),
    (64, 128, 256),
    (128, 128, 128),
    (32, 128, 256),
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


def _run(variant, layout, B, M, N, K, dtype, tile=(128, 128, 256)):
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
        tile_m=tile[0],
        tile_n=tile[1],
        tile_k=tile[2],
    )
    assert out.shape == (B, M, N)
    # Reordered MFMA reduction vs torch.bmm drifts a few low bits; accept <5% mismatch at 1e-2.
    err = checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"flydsl {variant} {layout} {B}x{M}x{N}x{K} tile{tile}",
    )
    assert err < 0.05, f"flydsl {variant} {layout} tile{tile} mismatch ratio {err:.4f}"
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


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4_tiles(variant, tile):
    if get_gfx() != "gfx950":
        pytest.skip(f"FlyDSL MXFP preshuffle GEMM requires gfx950, got {get_gfx()}")
    torch.cuda.empty_cache()
    # N=1024 (div 128/256), K=1024 (div 128/256, %256==0); M=64 with an odd batch.
    _run(variant, "bmn", 3, 64, 1024, 1024, torch.bfloat16, tile=tile)


@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4_cudagraph(variant):
    if get_gfx() != "gfx950":
        pytest.skip(f"FlyDSL MXFP preshuffle GEMM requires gfx950, got {get_gfx()}")
    torch.cuda.empty_cache()
    B, M, N, K, dtype = 2, 64, 1024, 4096, torch.bfloat16
    a_dtype = "fp8" if variant == "a8w4" else "fp4"
    x, w, x_scales, w_scales, ref = gen_inputs(variant, B, M, N, K, dtype)
    out, _us = run_perftest(
        flydsl_batched_gemm_mxfp4,
        x,
        w,
        x_scales,
        w_scales,
        dtype,
        a_dtype=a_dtype,
        testGraph=True,
        num_iters=20,
        num_warmup=3,
    )
    err = checkAllclose(
        ref.float(), out.float(), rtol=1e-2, atol=1e-2, msg=f"flydsl graph {variant}"
    )
    assert err < 0.05, f"flydsl graph {variant} mismatch ratio {err:.4f}"
