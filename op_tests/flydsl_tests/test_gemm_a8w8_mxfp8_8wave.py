# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the 8-wave mxfp8 (1x32 ue8m0) GEMM.

ue8m0 scales are exact powers of two, so applying one is lossless and the only
error against a torch reference is fp8 quantisation plus bf16 output rounding.
That makes the comparison tight -- a wrong scale index does not hide in
tolerance, it moves whole rows -- which matters because the scale *indexing* is
where this kernel differs from an unscaled one.

Both N tiles are forced at every shape rather than left to ``pick_block_n``,
since the two compile different store paths (the 256 tile's permlane plus lane
transpose, the 128 tile's plain 64-bit store) and the heuristic only reaches
one of them per shape.
"""

import argparse

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a8w8_mxfp8_8wave import flydsl_8wave_gemm_mxfp8
from aiter.ops.shuffle import (
    shuffle_mxfp8_a_scale,
    shuffle_mxfp8_b_scale,
    shuffle_weight,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
BLOCK = 32

# (M, N, K). M=64 is below BLOCK_M, so it exercises the tail block's store
# masking; 320 is not a multiple of BLOCK_M either.
SHAPES = [
    (64, 5120, 2048),
    (320, 5120, 2048),
    (1024, 8192, 1280),
    (4096, 5120, 4096),
]
BLOCK_NS = [128, 256]
# A dropped or misindexed scale moves whole rows by a factor of two or more, so
# counting elements off by 5% separates cleanly: zero when correct.
REL_TOL = 0.05


def _operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, generator=g) / 8
    b = torch.randn(n, k, generator=g) / 8
    # Biased ue8m0 exponents, kept in a modest range so the fp8 mantissa still
    # carries the value rather than flushing to zero or inf.
    ae = torch.randint(120, 135, (m, k // BLOCK), generator=g, dtype=torch.uint8)
    be = torch.randint(
        120, 135, (n // BLOCK, k // BLOCK), generator=g, dtype=torch.uint8
    )
    a_sc = torch.pow(2.0, (ae.int() - 127).float())
    b_sc = torch.pow(2.0, (be.int() - 127).float())
    aq = (a / a_sc.repeat_interleave(BLOCK, dim=1)).to(torch.float8_e4m3fn)
    bq = (
        b / b_sc.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=0)
    ).to(torch.float8_e4m3fn)
    return aq, bq, ae, be, a_sc, b_sc


def run_shape(m, n, k, block_n):
    aq, bq, ae, be, a_sc, b_sc = _operands(m, n, k)
    af = aq.float() * a_sc.repeat_interleave(BLOCK, dim=1)
    bf = bq.float() * b_sc.repeat_interleave(BLOCK, dim=1).repeat_interleave(
        BLOCK, dim=0
    )
    # Rounded to the kernel's output precision, so the comparison sees the
    # kernel's error rather than bf16's.
    ref = (af @ bf.T).to(torch.bfloat16).float()

    out = torch.empty(m, n, dtype=torch.bfloat16)
    flydsl_8wave_gemm_mxfp8(
        aq,
        shuffle_weight(bq, (16, 16)),
        shuffle_mxfp8_a_scale(ae),
        shuffle_mxfp8_b_scale(be),
        out,
        block_n,
    )

    # Floor the per-element tolerance below a typical output but above zero:
    # elements that cancel to near nothing disagree with torch on summation
    # order alone, which is not what this is looking for.
    floor = 0.01 * ref.square().mean().sqrt()
    diff = (out.float() - ref).abs()
    return int((diff > REL_TOL * ref.abs().clamp(min=floor)).sum())


def test_gemm_a8w8_mxfp8_8wave():
    if get_gfx() not in SUPPORTED_GFX:
        return
    for m, n, k in SHAPES:
        for block_n in BLOCK_NS:
            bad = run_shape(m, n, k, block_n)
            assert bad == 0, (
                f"M={m} N={n} K={k} block_n={block_n}: {bad} elements off by "
                f"more than {REL_TOL:.0%}"
            )


def test_m_not_multiple_of_scale_group_is_rejected():
    """M must be a whole packed-scale group; silently padding would mislabel it.

    The A scale is built over the padded M and its packed layout is not a
    reshape away from the unpadded one, so the caller has to pad before
    quantising. Rejecting here is what makes that visible.
    """
    if get_gfx() not in SUPPORTED_GFX:
        return
    aq, bq, ae, be, _, _ = _operands(64, 5120, 2048)
    out = torch.empty(40, 5120, dtype=torch.bfloat16)
    try:
        flydsl_8wave_gemm_mxfp8(
            aq[:40],
            shuffle_weight(bq, (16, 16)),
            shuffle_mxfp8_a_scale(ae),
            shuffle_mxfp8_b_scale(be),
            out,
        )
    except ValueError as err:
        assert "multiple of 64" in str(err), err
    else:
        raise AssertionError("M=40 should have been rejected")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    failed = []
    for m, n, k in SHAPES:
        for block_n in BLOCK_NS:
            bad = run_shape(m, n, k, block_n)
            print(
                f"M={m} N={n} K={k} block_n={block_n}: bad_elems {bad} "
                f"{'ok' if bad == 0 else 'FAIL'}"
            )
            if bad:
                failed.append((m, n, k, block_n))
    total = len(SHAPES) * len(BLOCK_NS)
    print(f"\n{total - len(failed)}/{total} configurations clean")
