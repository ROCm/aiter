# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the 8-wave mxfp8 (1x32 e8m0) GEMM.

ue8m0 scales are exact powers of two, so applying one is lossless and the only
error against a torch reference is fp8 quantisation plus bf16 output rounding.
That makes the comparison tight enough that a misplaced scale or a dropped
store moves whole rows rather than hiding in tolerance.

Three things here are not covered by simply running a tuned shape:

- the A/B-swapped epilogue writes 64 bits per lane through two cross-lane
  stages, so its column mapping is checked on an N that is *not* a multiple of
  the 256-wide block tile, where the four columns a lane owns can straddle the
  edge;
- K is only required to be a whole number of 128-deep tiles on the row-major
  scale path, so a K that is an odd multiple of 128 is exercised;
- the main loop's barriers admit a bounded number of outstanding loads, and
  whether too many is fatal depends on how the waves interleave -- so the large
  grids are run several times rather than once.
"""

import argparse

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import flydsl_mxfp8_8wave_gemm
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
BLOCK = 32
REL_TOL = 0.05
REPS = 6

#: (M, N, K). 320 is not a multiple of the 256-row block tile; 5152 and 2688
#: are not multiples of the 256-wide one; 4224 and 1792 are odd multiples of
#: BLOCK_K, which the preshuffled scale layout could not express.
SHAPES = [
    (256, 256, 256),
    (320, 5120, 2048),
    (1024, 5120, 4224),
    (256, 5152, 2048),
    (1024, 2688, 2048),
    (4096, 5120, 2048),
]
#: Large grids, run REPS times each: these are where the barrier counts bite.
HAZARD_SHAPES = [
    (1472, 3584, 7168),
    (1920, 7168, 1792),
    (32768, 7168, 1792),
]


def _operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, generator=g) / 8
    b = torch.randn(n, k, generator=g) / 8
    # Biased ue8m0 exponents, kept in a modest range so the fp8 mantissa still
    # carries the value rather than flushing to zero or inf.
    ae = torch.randint(120, 135, (m, k // BLOCK), generator=g, dtype=torch.uint8)
    be = torch.randint(120, 135, (n, k // BLOCK), generator=g, dtype=torch.uint8)
    a_sc = torch.pow(2.0, (ae.int() - 127).float()).repeat_interleave(BLOCK, dim=1)
    b_sc = torch.pow(2.0, (be.int() - 127).float()).repeat_interleave(BLOCK, dim=1)
    aq = (a / a_sc).to(torch.float8_e4m3fn)
    bq = (b / b_sc).to(torch.float8_e4m3fn)
    # Rounded to the kernel's output precision, so the comparison sees the
    # kernel's error rather than bf16's.
    ref = ((aq.float() * a_sc) @ (bq.float() * b_sc).T).to(torch.bfloat16).float()
    return aq, shuffle_weight(bq, (16, 16)), ae, be, ref


def run_shape(m, n, k, reps=1):
    """Elements off by more than REL_TOL, summed over ``reps`` launches."""
    aq, wq, ae, be, ref = _operands(m, n, k)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    # Floor the per-element tolerance below a typical output but above zero:
    # elements that cancel to near nothing disagree with torch on summation
    # order alone, which is not what this is looking for.
    floor = 0.01 * ref.square().mean().sqrt()
    bad = 0
    for _ in range(reps):
        # Poison the output: a store the kernel skips entirely stays non-finite
        # instead of being masked by whatever was left in the buffer.
        out.fill_(float("nan"))
        flydsl_mxfp8_8wave_gemm(aq, wq, ae, be, out)
        torch.cuda.synchronize()
        diff = (out.float() - ref).abs()
        bad += int((diff > REL_TOL * ref.abs().clamp(min=floor)).sum())
        bad += int((~torch.isfinite(out.float())).sum())
    return bad


def test_mxfp8_gemm_8wave():
    if get_gfx() not in SUPPORTED_GFX:
        return
    for m, n, k in SHAPES:
        bad = run_shape(m, n, k)
        assert bad == 0, f"M={m} N={n} K={k}: {bad} elements off"


def test_mxfp8_gemm_8wave_large_grids():
    if get_gfx() not in SUPPORTED_GFX:
        return
    for m, n, k in HAZARD_SHAPES:
        bad = run_shape(m, n, k, reps=REPS)
        assert bad == 0, f"M={m} N={n} K={k}: {bad} elements off over {REPS} runs"


def test_preshuffled_scales_still_require_a_whole_k_pair():
    """The relaxed K rule is the row-major path's, not the preshuffled one's.

    ``ScalePreshuffledS2R`` addresses scale words by K-pair (``k1 = k // 2``),
    so it needs K % 256 even though the row-major reader does not.
    """
    if get_gfx() not in SUPPORTED_GFX:
        return
    m, n, k = 256, 256, 4224  # an odd multiple of BLOCK_K
    aq, wq, _, _, _ = _operands(m, n, k)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    pre_a = torch.zeros(m * k // BLOCK, dtype=torch.uint8)
    pre_b = torch.zeros(n * k // BLOCK, dtype=torch.uint8)
    try:
        flydsl_mxfp8_8wave_gemm(aq, wq, pre_a, pre_b, out)
    except ValueError as err:
        assert "256" in str(err), err
    else:
        raise AssertionError("K=4224 should be rejected for preshuffled scales")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    failed = []
    for m, n, k in SHAPES:
        bad = run_shape(m, n, k)
        print(f"M={m} N={n} K={k}: bad_elems {bad} {'ok' if bad == 0 else 'FAIL'}")
        failed += [(m, n, k)] * (bad > 0)
    for m, n, k in HAZARD_SHAPES:
        bad = run_shape(m, n, k, reps=REPS)
        print(
            f"M={m} N={n} K={k} x{REPS}: bad_elems {bad} "
            f"{'ok' if bad == 0 else 'FAIL'}"
        )
        failed += [(m, n, k)] * (bad > 0)
    total = len(SHAPES) + len(HAZARD_SHAPES)
    print(f"\n{total - len(failed)}/{total} shapes within tolerance")
