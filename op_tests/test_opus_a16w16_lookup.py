# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
End-to-end lookup verification for opus a16w16 tuned CSV.

Reads aiter/ops/opus/configs/opus_gemm_a16w16_tuned.csv (or an override
path), treats each row as "shape (cu_num, M, N, K) -> (solidx, splitK)",
launches opus_gemm_a16w16_tune with the explicit kid/splitK from the
CSV, and compares against torch.bmm.

This is the PR2' validation baseline: the gemm_a16w16_opus wrapper (to
be added) must produce identical numerical behavior when its internal
CSV lookup hits the same rows.

Usage:

    # Full CSV (only rows whose cu_num matches current device)
    python3 op_tests/test_opus_a16w16_lookup.py

    # Small smoke subset
    python3 op_tests/test_opus_a16w16_lookup.py --max-rows 10

    # Custom tuned csv path
    python3 op_tests/test_opus_a16w16_lookup.py \
        --tuned-csv /tmp/opus_tuned_out.csv

    # Filter by kid range (e.g. only splitk family 200..299)
    python3 op_tests/test_opus_a16w16_lookup.py --kid-min 200 --kid-max 299

Success criterion: every row passes allclose(rtol=2e-2, atol=1.0), which
matches the tuner's post-run checkAllclose gate
(csrc/opus_gemm/opus_gemm_tune.py:check_rtol, check_atol). Rows whose
cu_num differs from the current device are skipped (not failed) because
optimal kernel choice is cu_num dependent.
"""

import argparse
import os
import sys
from typing import Dict, Tuple

import pandas as pd
import torch

from aiter.ops.opus.gemm_op_a16w16 import opus_gemm_a16w16_tune

DEFAULT_TUNED_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "aiter",
    "ops",
    "opus",
    "configs",
    "opus_gemm_a16w16_tuned.csv",
)


def _load_lookup(
    path: str, kid_min: int, kid_max: int
) -> Dict[Tuple[int, int, int, int], dict]:
    df = pd.read_csv(path)
    if "libtype" in df.columns:
        df = df[df["libtype"] == "opus"]
    df = df[(df["solidx"] >= kid_min) & (df["solidx"] <= kid_max)]
    df = df.drop_duplicates(subset=["cu_num", "M", "N", "K"], keep="last")
    lookup: Dict[Tuple[int, int, int, int], dict] = {}
    for _, row in df.iterrows():
        key = (int(row["cu_num"]), int(row["M"]), int(row["N"]), int(row["K"]))
        lookup[key] = {
            "solidx": int(row["solidx"]),
            "splitK": int(row["splitK"]),
            "kernelName": str(row.get("kernelName", "")),
        }
    return lookup


class _KidNotCompiled(Exception):
    pass


def _run_one(M: int, N: int, K: int, kid: int, splitK: int, rtol: float, atol: float):
    """Returns (ok: bool, max_abs_err: float, err_ratio: float).

    Raises _KidNotCompiled if the kid is in the CSV but the currently
    built JIT module has no corresponding instance (common when the CSV
    was produced against a wider kernels_list than the current build).
    """
    torch.manual_seed(M * 1_000_003 + N * 1009 + K)
    XQ = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    WQ = torch.randn(1, N, K, device="cuda", dtype=torch.bfloat16)
    Y = torch.empty(1, M, N, device="cuda", dtype=torch.bfloat16)

    try:
        opus_gemm_a16w16_tune(XQ, WQ, Y, kid, splitK)
    except RuntimeError as e:
        if "not found in a16w16" in str(e) and "tune lookup table" in str(e):
            raise _KidNotCompiled(str(e)) from None
        raise
    torch.cuda.synchronize()

    ref = torch.bmm(XQ.float(), WQ.float().transpose(-1, -2)).to(torch.bfloat16)
    ok = torch.allclose(Y, ref, rtol=rtol, atol=atol)
    diff = (Y.float() - ref.float()).abs()
    ref_abs = ref.float().abs()
    bound = atol + rtol * ref_abs
    max_abs_err = diff.max().item()
    err_ratio = (diff > bound).float().mean().item()
    return ok, max_abs_err, err_ratio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify opus a16w16 tuned-CSV lookup end-to-end"
    )
    parser.add_argument(
        "--tuned-csv",
        default=DEFAULT_TUNED_CSV,
        help=f"Path to tuned CSV (default: {DEFAULT_TUNED_CSV})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit to the first N rows (after filtering) for smoke testing",
    )
    parser.add_argument(
        "--kid-min",
        type=int,
        default=0,
        help="Include only solidx >= this value (default: 0)",
    )
    parser.add_argument(
        "--kid-max",
        type=int,
        default=10_000,
        help="Include only solidx <= this value (default: 10000)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=2e-2,
        help="allclose rtol (default: 2e-2, matches tuner's check_rtol)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0,
        help="allclose atol (default: 1.0, matches tuner's check_atol)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print per-row details"
    )
    args = parser.parse_args()

    if not os.path.exists(args.tuned_csv):
        print(f"ERROR: tuned CSV not found: {args.tuned_csv}")
        return 2

    lookup = _load_lookup(args.tuned_csv, args.kid_min, args.kid_max)
    if not lookup:
        print(f"ERROR: no rows in {args.tuned_csv} match the filters")
        return 2

    cu_num = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"Device cu_num = {cu_num}; CSV has {len(lookup)} unique (cu_num,M,N,K) keys")

    items = list(lookup.items())
    matching = [(k, v) for k, v in items if k[0] == cu_num]
    skipped_cu = len(items) - len(matching)
    if args.max_rows is not None:
        matching = matching[: args.max_rows]

    passed = failed = not_compiled = 0
    for (cu, M, N, K), ent in matching:
        kid = ent["solidx"]
        sk = ent["splitK"]
        name = ent["kernelName"]
        try:
            ok, max_abs_err, err_ratio = _run_one(
                M, N, K, kid, sk, args.rtol, args.atol
            )
        except _KidNotCompiled:
            not_compiled += 1
            if args.verbose:
                print(
                    f"  [SKIP] cu={cu} M={M:>5} N={N:>5} K={K:>5} "
                    f"kid={kid:3d} splitK={sk} (kid not in current JIT build)"
                )
            continue
        status = "PASS" if ok else "FAIL"
        if args.verbose or not ok:
            print(
                f"  [{status}] cu={cu} M={M:>5} N={N:>5} K={K:>5} "
                f"kid={kid:3d} splitK={sk} "
                f"max_abs_err={max_abs_err:.4f} err_ratio={err_ratio:.5f}"
                f"{' | ' + name if name and name != 'nan' else ''}"
            )
        passed += int(ok)
        failed += int(not ok)

    print()
    total = passed + failed
    extras = []
    if skipped_cu:
        extras.append(f"{skipped_cu} skipped (cu_num mismatch)")
    if not_compiled:
        extras.append(f"{not_compiled} skipped (kid not compiled in current JIT)")
    print(
        f"Summary: {passed}/{total} passed, {failed} failed"
        + ("; " + ", ".join(extras) if extras else "")
    )
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
