# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL strided-batched MXFP preshuffle GEMM (gfx950).

Covers out[b] = dequant(x[b]) @ dequant(w[b]).T across two variants and two layouts:
  - variants: a4w4 (MXFP4 A x MXFP4 B), a8w4 (MXFP8 E4M3 A x MXFP4 B)
  - layouts:  bmn ([B,M,*] A/C), mbn ([M,B,*] A/C, deepseek-v4 grouped-output)
plus odd/large M (ragged-tail OOB), a tile-size sweep, and a CUDA-graph run.

Weights + scales are preshuffled and A/scale_a laid out ONCE here (host prep), then the thin
flydsl_batched_gemm_mxfp4 launcher runs -- no shuffle on the launch path.
"""

import pytest
import torch

from aiter.ops.flydsl import is_flydsl_available

if not is_flydsl_available():
    pytest.skip("flydsl is not available", allow_module_level=True)

import aiter  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from aiter.ops.flydsl.batched_gemm_mxfp4 import (  # noqa: E402
    flydsl_batched_gemm_mxfp4,
)
from aiter.ops.shuffle import (  # noqa: E402
    shuffle_scale,
    shuffle_scale_n32k4,
    shuffle_weight,
    shuffle_weight_gfx1250,
)
from aiter.test_common import checkAllclose, run_perftest  # noqa: E402
from aiter.utility import fp4_utils  # noqa: E402

torch.set_default_device("cuda")

SCALE_GROUP_SIZE = 32

# gfx1250 runs the wave32 WMMA path for both a8w4 (MXFP8 A) and a4w4 (MXFP4 A).
_IS_GFX1250 = get_gfx() == "gfx1250"


def _supported_gfx():
    return get_gfx() in ("gfx950", "gfx1250")


def _skip_if_unsupported(variant):
    if not _supported_gfx():
        pytest.skip(
            f"FlyDSL MXFP preshuffle GEMM requires gfx950/gfx1250, got {get_gfx()}"
        )

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


def _pad_rows_to_32(s):
    r = s.shape[0]
    if r % 32 == 0:
        return s
    pad = torch.zeros((32 - r % 32, s.shape[1]), dtype=s.dtype, device=s.device)
    return torch.cat([s, pad], dim=0)


def preshuffle_operands_gfx1250(x, w, x_scales, w_scales, *, layout="bmn"):
    """gfx1250 a8w4 host prep: A stays plain fp8 codes; B via shuffle_weight_gfx1250;
    both e8m0 scales via shuffle_scale_n32k4 (rows padded to 32). Mirrors the gfx950
    prep's bmn/mbn stacking so A / scale_a follow `layout`."""
    B, M = x.shape[0], x.shape[1]
    a_list, sa_list, w_list, sb_list = [], [], [], []
    for b in range(B):
        a_list.append(x[b].contiguous())
        w_list.append(shuffle_weight_gfx1250(w[b].contiguous()).reshape(-1))
        sa = shuffle_scale_n32k4(_pad_rows_to_32(x_scales[b]).unsqueeze(0), experts_cnt=1)
        sa_list.append(sa.view(torch.uint8).reshape((M + 31) // 32, -1))
        sb = shuffle_scale_n32k4(_pad_rows_to_32(w_scales[b]).unsqueeze(0), experts_cnt=1)
        sb_list.append(sb.view(torch.uint8).reshape(-1))
    w_sh = torch.cat(w_list)
    sb = torch.cat(sb_list)
    if layout == "mbn":  # A [M,B,K]; scale_a [ceil(M/32), B, chunk]
        a = torch.stack(a_list, dim=1).contiguous()
        sa = torch.stack(sa_list, dim=1).contiguous().reshape(-1)
    else:  # bmn [B,M,K]; scale_a [B, ceil(M/32), chunk]
        a = torch.stack(a_list, dim=0).contiguous()
        sa = torch.cat([s.reshape(-1) for s in sa_list])
    return a, w_sh, sa, sb


def preshuffle_operands(x, w, x_scales, w_scales, *, a_dtype="fp4", layout="bmn"):
    """One-time host prep (done once, NOT per launch): keep A codes plain, preshuffle B and
    both e8m0 scales, and lay A/scale_a out for `layout`. Inputs are logical [B,M,*]; returns
    (a, w, a_scales, w_scales) for flydsl_batched_gemm_mxfp4 (a is [B,M,*] bmn / [M,B,*] mbn,
    the rest flat)."""
    if _IS_GFX1250:
        return preshuffle_operands_gfx1250(
            x, w, x_scales, w_scales, layout=layout
        )
    B, M = x.shape[0], x.shape[1]
    K = x_scales.shape[-1] * SCALE_GROUP_SIZE
    M32 = (M + 31) // 32 * 32
    a_list, sa_list, w_list, sb_list = [], [], [], []
    for b in range(B):
        a_list.append(x[b].contiguous())
        w_list.append(shuffle_weight(w[b].contiguous(), layout=(16, 16)).reshape(-1))
        sa = x_scales[b]
        if (
            M32 != M
        ):  # pad A-scale rows to M/32 (shuffle_scale over-pads, then slice back)
            pad = torch.zeros(
                (M32 - M, K // SCALE_GROUP_SIZE), dtype=sa.dtype, device=sa.device
            )
            sa = torch.cat([sa, pad], dim=0)
        sa = sa.contiguous()
        sa_list.append(shuffle_scale(sa).reshape(-1)[: sa.numel()])
        wsb = w_scales[b].contiguous()
        sb_list.append(shuffle_scale(wsb).reshape(-1)[: wsb.numel()])

    w_sh = torch.cat(w_list)
    sb = torch.cat(sb_list)
    if layout == "mbn":  # A / scale_a physically [M,B,*] / [ceil(M/32),B,chunk]
        mchunks = (M + 31) // 32
        chunk = sa_list[0].numel() // mchunks
        a = torch.stack(a_list, dim=1).contiguous()
        sa = torch.stack([s.view(mchunks, chunk) for s in sa_list], dim=1)
        sa = sa.contiguous().reshape(-1)
    else:  # bmn [B,M,*]
        a = torch.stack(a_list, dim=0).contiguous()
        sa = torch.cat(sa_list)
    return a, w_sh, sa, sb


def _quant_fp4(x_2d):
    """Per-1x32 MXFP4 quant (aiter triton) -> (codes uint8 [.,K//2], scale uint8 [.,K//32])."""
    qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    codes, scale = qf(x_2d, shuffle=False)
    return codes.view(torch.uint8), scale.view(torch.uint8)


def gen_inputs(variant, B, M, N, K, dtype):
    """Returns (x, w, x_scales, w_scales, ref) with unshuffled scales (prep shuffles once)."""
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
    a, w_sh, sa, sb = preshuffle_operands(
        x, w, x_scales, w_scales, a_dtype=a_dtype, layout=layout
    )
    out = flydsl_batched_gemm_mxfp4(
        a,
        w_sh,
        sa,
        sb,
        N,
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
    _skip_if_unsupported(variant)
    torch.cuda.empty_cache()
    _run(variant, layout, B, M, N, K, dtype)


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4_tiles(variant, tile):
    _skip_if_unsupported(variant)
    torch.cuda.empty_cache()
    # N=1024 (div 128/256), K=1024 (div 128/256, %256==0); M=64 with an odd batch.
    _run(variant, "bmn", 3, 64, 1024, 1024, torch.bfloat16, tile=tile)


# Odd / ragged M: M % 32 != 0 and M % 2 != 0 force the last M-tile past the real M,
# so rows [M, tile_m) must be dropped by the store's OOB path (and their scales
# OOB-zeroed). B is odd too. Covers both bmn and mbn (mbn interleaves the batch dim
# into the scale super stride, a distinct OOB code path).
OOB_M = [1, 17, 31, 33, 65, 127]


@pytest.mark.parametrize("M", OOB_M)
@pytest.mark.parametrize("layout", ["bmn", "mbn"])
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4_oob(variant, layout, M):
    _skip_if_unsupported(variant)
    torch.cuda.empty_cache()
    _run(variant, layout, 3, M, 512, 512, torch.bfloat16)


@pytest.mark.parametrize("layout", ["bmn", "mbn"])
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4_cudagraph(variant, layout):
    _skip_if_unsupported(variant)
    torch.cuda.empty_cache()
    B, M, N, K, dtype = 2, 64, 1024, 4096, torch.bfloat16
    a_dtype = "fp8" if variant == "a8w4" else "fp4"
    x, w, x_scales, w_scales, ref = gen_inputs(variant, B, M, N, K, dtype)
    a, w_sh, sa, sb = preshuffle_operands(
        x, w, x_scales, w_scales, a_dtype=a_dtype, layout=layout
    )
    # Only the (prepared) launch is captured -- no shuffle in the graph.
    out, _us = run_perftest(
        flydsl_batched_gemm_mxfp4,
        a,
        w_sh,
        sa,
        sb,
        N,
        dtype,
        a_dtype=a_dtype,
        layout=layout,
        testGraph=True,
        num_iters=20,
        num_warmup=3,
    )
    err = checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"flydsl graph {variant} {layout}",
    )
    assert err < 0.05, f"flydsl graph {variant} {layout} mismatch ratio {err:.4f}"
