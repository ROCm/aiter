# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
End-to-end test of the a16w16 user-facing entry gemm_a16w16_opus.

Complements the two other a16w16 tests:

  * test_opus_a16w16_tune.py   — iterates every compiled kid via
    opus_gemm_a16w16_tune (id-based low-level binding).
  * test_opus_a16w16_lookup.py — iterates every row in the tuned CSV
    and launches with explicit (solidx, splitK) from CSV.

This file covers the user-facing path: gemm_a16w16_opus goes through
CSV lookup and on miss delegates to the C++ opus_gemm() two-level
dispatch. Compares against torch.bmm and prints per-shape TFLOPs.

Usage:

    # Single shape smoke test (default M=256 N=512 K=256 batch=8):
    python3 op_tests/test_opus_a16w16_gemm.py

    # Explicit shape:
    python3 op_tests/test_opus_a16w16_gemm.py -m 128 -n 2048 -k 4096 -b 1

    # Sweep shapes from a CSV (only M/N/K columns are read):
    python3 op_tests/test_opus_a16w16_gemm.py \\
        --csv_file aiter/configs/model_configs/gptoss_bf16_untuned_gemm.csv
"""

import argparse
import torch

from aiter.test_common import checkAllclose, run_perftest
from aiter.ops.opus import gemm_a16w16_opus


def _torch_ref(A: torch.Tensor, B: torch.Tensor, out_dtype):
    # A: [batch, M, K], B: [N, K] -> bmm with B broadcast as [batch, N, K].
    # run_torch computes in fp32 then casts to match the opus path.
    return torch.einsum("bmk,nk->bmn", A.float(), B.float()).to(out_dtype)


def test_a16w16(batch: int, M: int, N: int, K: int, out_dtype=torch.bfloat16):
    # gemm_a16w16_opus accepts either 2D or 3D A; test 3D to exercise the
    # batched reshape path. B is always 2D [N, K].
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
    B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    ref = _torch_ref(A, B, out_dtype)

    Y, us = run_perftest(
        gemm_a16w16_opus, A, B, None, out_dtype,
    )

    err = checkAllclose(
        Y, ref, msg=f"a16w16 b={batch} m={M} n={N} k={K}",
        rtol=0.1, atol=0.5,
    )
    flops = 2.0 * batch * M * N * K
    tflops = flops / us / 1e6
    print(
        f"[a16w16] batch={batch} M={M} N={N} K={K} dtype={out_dtype} "
        f"| {us:.1f}us | {tflops:.2f} TFLOPs | err={err}"
    )
    return err


def load_shapes_from_csv(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    shapes = list(
        zip(df["M"].astype(int), df["N"].astype(int), df["K"].astype(int))
    )
    return list(dict.fromkeys(shapes))


def test_a16w16_csv_sweep(csv_path: str, batch: int = 1):
    shapes = load_shapes_from_csv(csv_path)
    print(f"\n{'=' * 80}")
    print(
        f"a16w16 sweep from {csv_path}: {len(shapes)} unique shapes, batch={batch}"
    )
    print("=" * 80)
    passed = failed = 0
    for M, N, K in shapes:
        tag = f"a16w16 b={batch} M={M} N={N} K={K}"
        try:
            A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
            B = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            ref = _torch_ref(A, B, torch.bfloat16)
            Y, us = run_perftest(
                gemm_a16w16_opus, A, B, None, torch.bfloat16,
            )
            err = checkAllclose(Y, ref, msg=tag, rtol=0.1, atol=0.5)
            tflops = 2.0 * batch * M * N * K / us / 1e6
            print(
                f"[PASS] {tag} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}"
            )
            passed += 1
        except Exception as e:
            print(f"[FAIL] {tag} | {type(e).__name__}: {e}")
            failed += 1
    print(
        f"\nSummary: {passed} passed, {failed} failed out of {len(shapes)}"
    )
    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end test for aiter.ops.opus.gemm_a16w16_opus"
    )
    parser.add_argument("-m", type=int, default=256)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=256)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument(
        "-d", "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Optional CSV with M,N,K columns. When given, skips the "
            "single-shape test and runs a full sweep instead."
        ),
    )
    args = parser.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.csv_file is not None:
        test_a16w16_csv_sweep(args.csv_file, batch=args.batch)
    else:
        # Ensure K is at least one B_K for the smallest a16w16 kid
        # (K>=64 suffices for kid 4-5; 128 for kid 6-9; splitk kids
        # demand K >= pfk*B_K which the heuristic handles). Clamp K
        # to 128 to make the default smoke invocation work on every
        # kid the heuristic could pick.
        k_eff = max(args.k, 128)
        test_a16w16(args.batch, args.m, args.n, k_eff, out_dtype=out_dtype)
