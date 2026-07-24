# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the Triton strided-batched MXFP GEMM (a8w4 / a4w4), gfx950.

Mirrors op_tests/flydsl_tests/test_flydsl_batched_gemm_mxfp4.py: same shapes, variants
(a4w4 = MXFP4 A x MXFP4 B, a8w4 = MXFP8 E4M3 A x MXFP4 B), layouts (bmn = [B,M,*],
mbn = [M,B,*] deepseek-v4 grouped output), tile sweep, and a CUDA-graph run -- but feeds
PLAIN (unshuffled) codes + scales to batched_gemm_a8wfp4 (Triton needs no preshuffle).
"""

import pytest
import torch

import aiter
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.gemm.batched.batched_gemm_a8wfp4 import batched_gemm_a8wfp4
from aiter.test_common import checkAllclose
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SCALE_GROUP_SIZE = 32

# (B, M, N, K). Odd/large M exercise the ragged-M tail. N % tile_n == 0, K % 256 == 0.
SHAPES = [
    (1, 1, 1280, 8192),  # M=1 decode
    (3, 15, 512, 1024),  # M=15 ragged
    (2, 64, 1024, 4096),  # M=64
    (2, 1023, 768, 2048),  # M=1023 ragged
    (1, 4097, 512, 512),  # M=4097 large ragged
]

# (tile_m, tile_n, tile_k) -> BLOCK_SIZE_{M,N,K}.
TILES = [
    (128, 128, 256),
    (128, 256, 256),
    (64, 128, 256),
    (128, 128, 128),
    (32, 128, 256),
]


def _quant_fp4(x_2d):
    """Per-1x32 MXFP4 quant -> (codes uint8 [.,K//2], scale uint8 [.,K//32])."""
    qf = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    codes, scale = qf(x_2d, shuffle=False)
    return codes.view(torch.uint8), scale.view(torch.uint8)


def gen_inputs(variant, B, M, N, K, dtype):
    """Returns (x, w, x_scales, w_scales, ref) with plain (unshuffled) scales."""
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


def _to_mbn(t):
    """Return a logical [B,M,*] view backed by a physical [M,B,*] buffer."""
    return t.transpose(0, 1).contiguous().transpose(0, 1)


def _run(variant, layout, B, M, N, K, dtype, tile=None):
    a_dtype = "fp8" if variant == "a8w4" else "fp4"
    x, w, x_scales, w_scales, ref = gen_inputs(variant, B, M, N, K, dtype)

    # Triton reads A element type from the tensor: fp8 A must be float8_e4m3fn,
    # fp4 A stays packed uint8.
    if variant == "a8w4":
        x = x.view(torch.float8_e4m3fn)

    if layout == "mbn":
        x = _to_mbn(x)
        x_scales = _to_mbn(x_scales)

    config = None
    if tile is not None:
        config = {
            "BLOCK_SIZE_M": tile[0],
            "BLOCK_SIZE_N": tile[1],
            "BLOCK_SIZE_K": tile[2],
            "GROUP_SIZE_M": 1,
            "NUM_KSPLIT": 1,
            "SPLITK_BLOCK_SIZE": K,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 2,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
            "cache_modifier": ".cg",
        }

    out = batched_gemm_a8wfp4(
        x, w, x_scales, w_scales, dtype, config=config, a_dtype=a_dtype, layout=layout
    )
    assert out.shape == (B, M, N)
    # Reordered MFMA reduction vs torch.bmm drifts a few low bits; accept <5% mismatch at 1e-2.
    err = checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"triton {variant} {layout} {B}x{M}x{N}x{K} tile{tile}",
    )
    assert err < 0.05, f"triton {variant} {layout} tile{tile} mismatch ratio {err:.4f}"
    return err


@pytest.mark.parametrize("B, M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["bmn", "mbn"])
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_batched_gemm_a8wfp4(variant, layout, B, M, N, K, dtype):
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")
    torch.cuda.empty_cache()
    _run(variant, layout, B, M, N, K, dtype)


@pytest.mark.parametrize("tile", TILES)
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_batched_gemm_a8wfp4_tiles(variant, tile):
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")
    torch.cuda.empty_cache()
    _run(variant, "bmn", 3, 64, 1024, 1024, torch.bfloat16, tile=tile)


@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_batched_gemm_a8wfp4_cudagraph(variant):
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")
    torch.cuda.empty_cache()
    B, M, N, K, dtype = 2, 64, 1024, 4096, torch.bfloat16
    a_dtype = "fp8" if variant == "a8w4" else "fp4"
    x, w, x_scales, w_scales, ref = gen_inputs(variant, B, M, N, K, dtype)
    if variant == "a8w4":
        x = x.view(torch.float8_e4m3fn)

    y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    # Warmup (compile) outside capture.
    batched_gemm_a8wfp4(x, w, x_scales, w_scales, dtype, y=y, a_dtype=a_dtype)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        batched_gemm_a8wfp4(x, w, x_scales, w_scales, dtype, y=y, a_dtype=a_dtype)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    err = checkAllclose(
        ref.float(), y.float(), rtol=1e-2, atol=1e-2, msg=f"triton graph {variant}"
    )
    assert err < 0.05, f"triton graph {variant} mismatch ratio {err:.4f}"
