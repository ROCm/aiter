#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Quick verification harness for fp8_mqa_logits FlyDSL refactoring.

Runs a small shape through the Triton reference and FlyDSL kernels, saves the
FlyDSL output, and on subsequent runs compares against the saved baseline.

Usage:
    # First run (saves baseline):
    HIP_VISIBLE_DEVICES=7 python3 op_tests/flydsl_tests/verify_fp8_mqa_logits_refactor.py --save

    # Subsequent runs (compares against baseline):
    HIP_VISIBLE_DEVICES=7 python3 op_tests/flydsl_tests/verify_fp8_mqa_logits_refactor.py
"""

import argparse
import os
import sys

import torch

torch.set_default_device("cuda")

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits as triton_logits
from aiter.ops.triton.utils.types import get_fp8_dtypes
from op_tests.triton_tests.attention.test_fp8_mqa_logits import (
    per_custom_dims_cast_to_fp8,
    ref_fp8_mqa_logits,
)

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "_refactor_baseline.pt")

# (seq_len, seq_len_kv, num_heads, head_dim, variant, window)
# variant=None uses the shape-adaptive default (an LDS-pipelined builder on
# gfx950). The explicit tags below pin the direct-load builder
# (_build_kernel_mfma_r_w) and both CDNA4 atoms, so a refactor touching either
# builder is covered. The non-"full" windows are what exercise the fused
# clean_logits -inf fill.
SHAPES = [
    (64, 256, 64, 64, None, "full"),
    (128, 1024, 64, 128, None, "full"),
    (64, 256, 64, 64, "mfma32x32x64_bkv128_r2_w2", "full"),
    (128, 1024, 64, 128, "mfma16x16x128_bkv128_r1_w2", "full"),
    (128, 1024, 64, 128, "mfma32x32x64_bkv128_r2_w4_lds2", "full"),
    # -- windowed: drives the -inf fill in both builders --
    (128, 1024, 64, 128, None, "causal"),
    (128, 1024, 64, 128, None, "sliding"),
    (128, 1024, 64, 128, "mfma32x32x64_bkv128_r2_w2", "causal"),
    (128, 1024, 64, 128, "mfma32x32x64_bkv128_r2_w2", "sliding"),
    (128, 1024, 64, 128, "mfma32x32x64_bkv128_r2_w4_lds2", "sliding"),
]

_, e4m3_type = get_fp8_dtypes()


def make_test_case(seq_len, seq_len_kv, num_heads, head_dim, window="full", seed=42):
    torch.manual_seed(seed)
    q_bf16 = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    kv_bf16 = torch.randn(seq_len_kv, head_dim, dtype=torch.bfloat16)

    q_fp8, _ = per_custom_dims_cast_to_fp8(q_bf16, (0,), False)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv_bf16, (0,), False)

    weights = torch.randn(seq_len, num_heads, dtype=torch.float32)

    # The window drives the clean_logits -inf fill: with a full row the
    # complement is empty and the fill emits nothing, so the windowed modes
    # below are what actually exercise it.
    rows = torch.arange(seq_len, dtype=torch.int32)
    ratio = max(1, seq_len_kv // seq_len)
    if window == "full":
        cu_starts = torch.zeros(seq_len, dtype=torch.int32)
        cu_ends = torch.full((seq_len,), seq_len_kv, dtype=torch.int32)
    elif window == "causal":
        # Non-empty prefix window; leaves a large right-hand complement.
        cu_starts = torch.zeros(seq_len, dtype=torch.int32)
        cu_ends = ((rows + 1) * ratio).clamp(max=seq_len_kv).to(torch.int32)
    elif window == "sliding":
        # Both complements non-empty, and some rows have an EMPTY window
        # (start > end) to cover the inverted-window collapse.
        cu_ends = ((rows + 1) * ratio).clamp(max=seq_len_kv).to(torch.int32)
        cu_starts = (cu_ends - 128).clamp(min=0).to(torch.int32)
        cu_starts[::7] = cu_ends[::7]  # empty windows
    else:
        raise ValueError(window)

    return q_fp8, kv_fp8, kv_scales, weights, cu_starts, cu_ends


def run_one(shape, flydsl_fn):
    seq_len, seq_len_kv, num_heads, head_dim, variant, window = shape
    q, kv, scales, weights, ks, ke = make_test_case(
        seq_len, seq_len_kv, num_heads, head_dim, window=window
    )

    triton_out = triton_logits(q, kv, scales, weights, ks, ke, clean_logits=True)

    fly_out = flydsl_fn(
        q, kv, scales, weights, ks, ke, clean_logits=True, variant=variant
    )

    # -inf appears in both on out-of-window positions; a plain subtraction there
    # is nan, so compare finite positions and the -inf mask separately.
    finite = torch.isfinite(triton_out) & torch.isfinite(fly_out)
    max_diff_vs_triton = (fly_out[finite] - triton_out[finite]).abs().max().item()
    mask_ok = torch.equal(torch.isneginf(fly_out), torch.isneginf(triton_out))
    return fly_out, max_diff_vs_triton, mask_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save baseline")
    args = parser.parse_args()

    try:
        from aiter.ops.flydsl import flydsl_fp8_mqa_logits
    except ImportError:
        print("ERROR: FlyDSL kernel not available", file=sys.stderr)
        sys.exit(1)

    baselines = {}
    if not args.save and os.path.exists(BASELINE_PATH):
        baselines = torch.load(BASELINE_PATH, weights_only=True)
        print(f"Loaded baseline from {BASELINE_PATH}")

    all_ok = True
    results_to_save = {}

    for shape in SHAPES:
        tag = (
            f"{shape[0]}x{shape[1]}_H{shape[2]}_D{shape[3]}"
            f"_{shape[4] or 'auto'}_{shape[5]}"
        )
        fly_out, max_diff, mask_ok = run_one(shape, flydsl_fp8_mqa_logits)

        status = f"triton: diff={max_diff:.6f} mask={'ok' if mask_ok else 'MISMATCH'}"
        if not mask_ok:
            all_ok = False

        if tag in baselines:
            baseline = baselines[tag].to(fly_out.device)
            # equal_nan so -inf == -inf compares clean.
            if torch.equal(fly_out, baseline):
                status += "  | baseline: EXACT MATCH"
            else:
                status += "  | baseline: REGRESSION!"
                all_ok = False

        print(f"[{tag}] {status}")
        results_to_save[tag] = fly_out.cpu()

    if args.save:
        torch.save(results_to_save, BASELINE_PATH)
        print(f"\nBaseline saved to {BASELINE_PATH}")

    if not all_ok:
        print("\nFAILED: regression detected!", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nPASSED")


if __name__ == "__main__":
    main()
