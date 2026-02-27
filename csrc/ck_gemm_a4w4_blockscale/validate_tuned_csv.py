#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Validate a4w4_blockscale_tuned_gemm.csv for correctness and consistency.

Usage:
    python csrc/ck_gemm_a4w4_blockscale/validate_tuned_csv.py [--csv PATH] [-v]

Checks performed:
    1. CSV schema: expected columns present, no missing values in key columns
    2. No duplicate (cu_num, M, N, K) keys
    3. All M, N, K are positive integers
    4. All us > 0 (no invalid timings)
    5. All errRatio <= 0.05 (tuner threshold)
    6. TFLOPS consistent with us: tflops ~ 2*M*N*K / (us * 1e6)
    7. kernelName is non-empty and matches known patterns
    8. cu_num is consistent (single value expected per GPU generation)
"""

import argparse
import os
import sys

import pandas as pd

EXPECTED_COLUMNS = [
    "cu_num",
    "M",
    "N",
    "K",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]

KNOWN_KERNEL_PATTERNS = [
    "f4gemm_bf16_per1x32Fp4",  # ASM kernels
    "a4w4_blockscale",  # CK-Tile kernels
]


def validate(csv_path, verbose=False):
    errors = []
    warnings = []

    if not os.path.exists(csv_path):
        print(f"FAIL: File not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # 1. Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        warnings.append(f"Extra columns (ignored): {extra_cols}")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
        return 1

    # 2. Nulls in key columns
    for col in ["cu_num", "M", "N", "K", "kernelId", "us", "kernelName"]:
        nulls = df[col].isna().sum()
        if nulls > 0:
            errors.append(f"Column '{col}' has {nulls} null values")

    # 3. Duplicate keys
    dup_mask = df.duplicated(subset=["cu_num", "M", "N", "K"], keep=False)
    n_dups = dup_mask.sum()
    if n_dups > 0:
        dup_keys = df[dup_mask][["cu_num", "M", "N", "K"]].drop_duplicates()
        errors.append(
            f"{n_dups} rows with duplicate (cu_num, M, N, K) keys "
            f"({len(dup_keys)} unique duplicate keys)"
        )
        if verbose:
            print("  Duplicate keys:")
            for _, row in dup_keys.iterrows():
                print(
                    f"    cu_num={row['cu_num']} M={row['M']}"
                    f" N={row['N']} K={row['K']}"
                )

    # 4. Positive M, N, K
    for col in ["M", "N", "K"]:
        non_pos = (df[col] <= 0).sum()
        if non_pos > 0:
            errors.append(f"Column '{col}' has {non_pos} non-positive values")

    # 5. Valid timings
    invalid_us = (df["us"] <= 0).sum()
    if invalid_us > 0:
        errors.append(f"{invalid_us} rows with us <= 0 (invalid timing)")

    # 6. Error ratio
    high_err = (df["errRatio"] > 0.05).sum()
    if high_err > 0:
        warnings.append(f"{high_err} rows with errRatio > 0.05")
        if verbose:
            bad = df[df["errRatio"] > 0.05][["M", "N", "K", "errRatio", "kernelName"]]
            print("  High errRatio rows:")
            for _, row in bad.iterrows():
                print(
                    f"    M={row['M']} N={row['N']} K={row['K']}"
                    f" err={row['errRatio']:.4f}"
                )

    # 7. TFLOPS consistency (2*M*N*K / us_in_seconds)
    expected = 2.0 * df["M"] * df["N"] * df["K"] / (df["us"] * 1e6)
    mismatch = (df["tflops"] - expected).abs() / expected > 0.05
    n_mismatch = mismatch.sum()
    if n_mismatch > 0:
        warnings.append(
            f"{n_mismatch} rows where reported TFLOPS differs >5% from 2*M*N*K/us"
        )

    # 8. Kernel name patterns
    pattern = "|".join(KNOWN_KERNEL_PATTERNS)
    unknown = df[~df["kernelName"].str.contains(pattern, na=False)]
    if len(unknown) > 0:
        warnings.append(f"{len(unknown)} rows with unrecognized kernel name pattern")
        if verbose:
            for name in unknown["kernelName"].unique()[:5]:
                print(f"    Unknown kernel: {name}")

    # 9. cu_num consistency
    cu_nums = df["cu_num"].unique()
    if len(cu_nums) > 1:
        warnings.append(f"Multiple cu_num values: {sorted(cu_nums)}")
    else:
        print(f"  cu_num: {cu_nums[0]} (consistent)")

    # Summary
    n_asm = df["kernelName"].str.contains("f4gemm_bf16", na=False).sum()
    n_cktile = len(df) - n_asm
    unique_nk = df[["N", "K"]].drop_duplicates()

    print("\n=== Summary ===")
    print(f"  Total shapes:       {len(df)}")
    print(f"  Unique [N,K] pairs: {len(unique_nk)}")
    print(f"  M range:            [{df['M'].min()}, {df['M'].max()}]")
    print(f"  N range:            [{df['N'].min()}, {df['N'].max()}]")
    print(f"  K range:            [{df['K'].min()}, {df['K'].max()}]")
    print(f"  ASM kernels:        {n_asm} ({100*n_asm/len(df):.1f}%)")
    print(f"  CK-Tile kernels:    {n_cktile} ({100*n_cktile/len(df):.1f}%)")
    print(
        f"  TFLOPS:             min={df['tflops'].min():.1f},"
        f" max={df['tflops'].max():.1f},"
        f" mean={df['tflops'].mean():.1f}"
    )
    print(f"  errRatio max:       {df['errRatio'].max():.4f}")
    print(f"  splitK > 0:         {(df['splitK'] > 0).sum()} shapes")

    # Result
    print("\n=== Validation Result ===")
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
    if warnings:
        for w in warnings:
            print(f"  WARN: {w}")
    if not errors and not warnings:
        print("  PASS: All checks passed")
    elif not errors:
        print(f"  PASS: {len(warnings)} warning(s), 0 errors")
    else:
        print(f"  FAIL: {len(errors)} error(s), {len(warnings)} warning(s)")

    return 1 if errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate tuned GEMM CSV")
    parser.add_argument(
        "--csv",
        default="aiter/configs/a4w4_blockscale_tuned_gemm.csv",
        help="Path to tuned CSV",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    sys.exit(validate(args.csv, args.verbose))
