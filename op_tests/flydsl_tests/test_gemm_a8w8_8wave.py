# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the 8-wave a8w8 bpreshuffle GEMM at large grids.

The main-loop barrier that ends each K step admits a bounded number of
outstanding global->LDS loads. Admitting too many lets the barrier pass while
writes the next step reads are still in flight, which shows up as a handful of
wrong rows rather than a NaN -- small enough to pass for quantisation noise.
Whether it happens at all depends on how the waves interleave, so a shape can
be clean on one launch and wrong on the next; each shape is therefore run
several times.
"""

import argparse

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_8wave import flydsl_8wave_gemm_a8
from aiter.ops.shuffle import shuffle_weight

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]

# (M, N, K, block_m, block_n), drawn from the tuned configs that select this
# kernel. The two large ones are where the barrier count actually bites.
SHAPES = [
    (1472, 3584, 7168, 128, 256),
    (3392, 2176, 7168, 128, 256),
    (3456, 7168, 4224, 128, 256),
    (1920, 7168, 1792, 256, 256),
    (2624, 8448, 7168, 256, 256),
    (32768, 7168, 1792, 256, 256),
]
REPS = 6
# A dropped load leaves whole rows wrong by far more than this, so counting
# offending elements separates cleanly: zero when correct, thousands when not.
# An L2 norm would not -- the bad rows are a small fraction of a large output.
REL_TOL = 0.05


def run_shape(m, n, k, block_m, block_n, reps, verbose=False):
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = (torch.randn(m, k, generator=gen) / 8).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, generator=gen) / 8).to(torch.float8_e4m3fn)
    a_scale = torch.rand(m, 1, generator=gen) + 0.5
    b_scale = torch.rand(n, 1, generator=gen) + 0.5
    b_shuffled = shuffle_weight(b, (16, 16))

    # rounded to the kernel's output precision, so the comparison sees the
    # kernel's error rather than bf16's
    ref = ((a.float() * a_scale) @ (b.float() * b_scale).T).to(torch.bfloat16).float()

    # Floor the per-element tolerance well below a typical output but above
    # zero: elements that cancel to near nothing disagree with torch on
    # summation order alone, and that is not what this test is looking for.
    floor = 0.01 * ref.square().mean().sqrt()

    worst = 0
    for _ in range(reps):
        out = torch.empty(m, n, dtype=torch.bfloat16)
        flydsl_8wave_gemm_a8(a, b_shuffled, a_scale, b_scale, out, block_m, block_n)
        diff = (out.float() - ref).abs()
        bad = int((diff > REL_TOL * ref.abs().clamp(min=floor)).sum())
        worst = max(worst, bad)
        if verbose:
            print(f"  bad_elems {bad}")
    return worst


def test_gemm_a8w8_8wave():
    if get_gfx() not in SUPPORTED_GFX:
        return
    for m, n, k, block_m, block_n in SHAPES:
        worst = run_shape(m, n, k, block_m, block_n, REPS)
        assert worst == 0, (
            f"M={m} N={n} K={k} tile={block_m}x{block_n}: "
            f"{worst} elements off by more than {REL_TOL:.0%} in {REPS} runs"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()

    failed = []
    for m, n, k, block_m, block_n in SHAPES:
        print(f"M={m} N={n} K={k} tile={block_m}x{block_n}")
        worst = run_shape(m, n, k, block_m, block_n, args.reps, verbose=True)
        print(f"  worst bad_elems {worst}  {'ok' if worst == 0 else 'FAIL'}")
        if worst:
            failed.append((m, n, k, block_m, block_n))
    print(f"\n{len(SHAPES) - len(failed)}/{len(SHAPES)} shapes within tolerance")
    for shape in failed:
        print(f"  failed: {shape}")
