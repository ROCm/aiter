# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL strided-batched MXFP preshuffle GEMM (gfx950).

Covers two variants of out[b] = dequant(x[b]) @ dequant(w[b]).T:
  - a4w4: MXFP4 (E2M1) A x MXFP4 (E2M1) B
  - a8w4: MXFP8 (E4M3) A x MXFP4 (E2M1) B

Usage:
    python aiter/ops/flydsl/test_flydsl_batched_gemm_mxfp4.py
    python aiter/ops/flydsl/test_flydsl_batched_gemm_mxfp4.py --variant a8w4
"""

import argparse
import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4
from aiter.test_common import checkAllclose

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

_MXFP4_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _mxfp4_to_f32(x):
    """uint8 packed mxfp4 (..., K//2) -> f32 (..., K). Low nibble = even K, high = odd."""
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=x.device)
    return lut[x.long()]


def _e8m0_to_f32(x):
    return 2 ** ((x.to(torch.float32) - 127))


def _rand_fp4_codes(shape):
    low = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
    high = torch.randint(0, 16, shape, dtype=torch.uint8, device="cuda")
    return low | (high << 4)


def _rand_scales(shape):
    # e8m0 in [124, 128) -> 2^{-3..0}; no NaN (128 would be NaN in e8m0).
    return torch.randint(124, 128, shape, dtype=torch.uint8, device="cuda")


def gen_inputs(variant, B, M, N, K, dtype):
    """Returns (x, w, x_scales, w_scales, ref) for the given variant."""
    torch.manual_seed(5)
    w = _rand_fp4_codes((B, N, K // 2))
    x_scales = _rand_scales((B, M, K // SCALE_GROUP_SIZE))
    w_scales = _rand_scales((B, N, K // SCALE_GROUP_SIZE))

    if variant == "a8w4":
        x = (
            (torch.randn(B, M, K, device="cuda") * 4.0)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
        )
        x_f32 = x.view(torch.float8_e4m3fn).to(torch.float32)
    else:  # a4w4
        x = _rand_fp4_codes((B, M, K // 2))
        x_f32 = _mxfp4_to_f32(x)

    xs = _e8m0_to_f32(x_scales.repeat_interleave(SCALE_GROUP_SIZE, -1))
    ws = _e8m0_to_f32(w_scales.repeat_interleave(SCALE_GROUP_SIZE, -1))
    ref = torch.bmm(x_f32 * xs, (_mxfp4_to_f32(w) * ws).transpose(1, 2)).to(dtype)
    return x, w, x_scales, w_scales, ref


def _run(variant, B, M, N, K, dtype):
    a_dtype = "fp8" if variant == "a8w4" else "fp4"
    x, w, x_scales, w_scales, ref = gen_inputs(variant, B, M, N, K, dtype)
    out = flydsl_batched_gemm_mxfp4(
        x,
        w,
        x_scales,
        w_scales,
        dtype,
        a_dtype=a_dtype,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    assert out.shape == (B, M, N)
    # Reordered MFMA reduction vs torch.bmm drifts a few low bits; accept <5% element
    # mismatch at rtol/atol=1e-2 (aiter checkAllclose convention).
    err = checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=1e-2,
        msg=f"flydsl {variant} {B}x{M}x{N}x{K}",
    )
    assert err < 0.05, f"flydsl {variant} mismatch ratio {err:.4f}"
    return err


@pytest.mark.parametrize("B, M, N, K", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("variant", ["a4w4", "a8w4"])
def test_flydsl_batched_gemm_mxfp4(variant, B, M, N, K, dtype):
    if get_gfx() != "gfx950":
        pytest.skip(f"FlyDSL MXFP preshuffle GEMM requires gfx950, got {get_gfx()}")
    torch.cuda.empty_cache()
    _run(variant, B, M, N, K, dtype)


def main():
    parser = argparse.ArgumentParser(description="FlyDSL batched MXFP GEMM test")
    parser.add_argument("--variant", choices=["a4w4", "a8w4", "all"], default="all")
    args = parser.parse_args()
    if get_gfx() != "gfx950":
        print(f"[skip] requires gfx950, got {get_gfx()}")
        return
    variants = ["a4w4", "a8w4"] if args.variant == "all" else [args.variant]
    for variant in variants:
        for dtype in (torch.bfloat16, torch.float16):
            for B, M, N, K in SHAPES:
                err = _run(variant, B, M, N, K, dtype)
                print(
                    f"[pass] {variant} {str(dtype):16} B={B} M={M} N={N} K={K} mismatch={err:.4f}"
                )


if __name__ == "__main__":
    sys.exit(main())
