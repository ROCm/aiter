#!/usr/bin/env python3
"""
Analyze blockPerCu performance across tuning profile results.

Produces compact tables where:
  - Rows = shape configurations
  - Columns = blockPerCu values (us + err) + overall fastest kernel info
"""

import argparse
import os
import re

import pandas as pd

pd.set_option("display.max_colwidth", 100)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


# ── blockPerCu extraction from kernel names ──────────────────────────────────

def extract_bpc_blockscale(kernel_name: str):
    """Extract blockPerCu from blockscale cktile kernel name (last _N field)."""
    if not isinstance(kernel_name, str) or "cktile" not in kernel_name:
        return None
    m = re.search(r"_(\d+)$", kernel_name)
    return int(m.group(1)) if m else None


def extract_bpc_bpreshuffle(kernel_name: str):
    """Extract blockPerCu from bpreshuffle cktile kernel (10th flag field)."""
    if not isinstance(kernel_name, str) or "cktile" not in kernel_name:
        return None
    m = re.search(r"cktile_([\dx]+?)_\d+x\d+x\d+_", kernel_name)
    if m:
        flags = m.group(1).split("x")
        if len(flags) >= 10:
            return int(flags[9])
    return None


def extract_bpc_moe(kernel_name: str):
    """Extract blockPerCu from MOE cktile kernel (NperCU pattern)."""
    if not isinstance(kernel_name, str):
        return None
    m = re.search(r"(\d+)perCU", kernel_name)
    return int(m.group(1)) if m else None


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_gemm_profile(profile_path: str, feature_name: str, extract_bpc_fn,
                         shape_keys: list[str]):
    """Analyze a GEMM-style profile CSV (blockscale or bpreshuffle)."""
    if not os.path.exists(profile_path):
        print(f"\n[SKIP] {feature_name}: {profile_path} not found")
        return

    df = pd.read_csv(profile_path)
    df = df[(df["us"] > 0) & (df["kernelName"] != "None")]
    if df.empty:
        print(f"\n[SKIP] {feature_name}: no valid results")
        return

    df["blockPerCu"] = df["kernelName"].apply(extract_bpc_fn)
    bpc_values = sorted(df["blockPerCu"].dropna().unique().astype(int))
    err_col = "errRatio" if "errRatio" in df.columns else None

    print(f"\n{'='*160}")
    print(f"  {feature_name} BlockPerCu Analysis  ({len(df)} valid results, {df['blockPerCu'].notna().sum()} CKTile)")
    print(f"{'='*160}")

    rows = []
    for shape_vals, group in df.groupby(shape_keys):
        if isinstance(shape_vals, (int, float, str)):
            shape_vals = (shape_vals,)
        row = dict(zip(shape_keys, shape_vals))

        # Overall fastest
        best = group.loc[group["us"].idxmin()]
        row["fast_us"] = best["us"]
        row["fast_err"] = best[err_col] if err_col else ""
        fbpc = extract_bpc_fn(best["kernelName"])
        row["fast_src"] = f"cktile(bpc={fbpc})" if fbpc else best.get("libtype", "?")

        # Best CKTile per blockPerCu
        cktile = group[group["blockPerCu"].notna()]
        for bpc in bpc_values:
            bg = cktile[cktile["blockPerCu"] == bpc]
            if bg.empty:
                row[f"bpc{bpc}_us"] = None
                row[f"bpc{bpc}_err"] = ""
            else:
                b = bg.loc[bg["us"].idxmin()]
                row[f"bpc{bpc}_us"] = b["us"]
                row[f"bpc{bpc}_err"] = b[err_col] if err_col else ""

        rows.append(row)

    result = pd.DataFrame(rows)
    cols = list(shape_keys) + ["fast_us", "fast_err", "fast_src"]
    for bpc in bpc_values:
        cols += [f"bpc{bpc}_us", f"bpc{bpc}_err"]
    result = result[[c for c in cols if c in result.columns]]
    print(result.to_string(index=False))

    _print_summary(rows, bpc_values, feature_name)


def analyze_moe_profile(profile_path: str):
    """Analyze MOE profile CSV with per-stage rows."""
    if not os.path.exists(profile_path):
        print(f"\n[SKIP] MOE: {profile_path} not found")
        return

    df = pd.read_csv(profile_path)
    shape_keys = [k for k in ["token", "model_dim", "inter_dim", "expert", "topk"] if k in df.columns]
    df = df[df["us"] > 0]
    if df.empty:
        print(f"\n[SKIP] MOE: no valid results")
        return

    df["blockPerCu"] = df["kernelName"].apply(extract_bpc_moe)

    def get_backend(kn):
        if not isinstance(kn, str): return "?"
        if "cktile" in kn: return "cktile"
        if "flydsl" in kn: return "flydsl"
        if "ck2stages" in kn: return "ck_codegen"
        return "asm"
    df["backend"] = df["kernelName"].apply(get_backend)

    stages = sorted(df["stage"].unique()) if "stage" in df.columns else ["combined"]
    err_col = "err" if "err" in df.columns else None

    for stage in stages:
        sdf = df[df["stage"] == stage] if "stage" in df.columns else df
        bpc_values = sorted(sdf["blockPerCu"].dropna().unique().astype(int))

        print(f"\n{'='*160}")
        print(f"  MOE {stage}  ({len(sdf)} valid, {sdf['blockPerCu'].notna().sum()} CKTile, {(sdf['backend']=='flydsl').sum()} FlyDSL)")
        print(f"{'='*160}")

        rows = []
        for shape_vals, group in sdf.groupby(shape_keys):
            if isinstance(shape_vals, (int, float, str)):
                shape_vals = (shape_vals,)
            row = dict(zip(shape_keys, shape_vals))

            best = group.loc[group["us"].idxmin()]
            row["fast_us"] = best["us"]
            row["fast_err"] = best[err_col] if err_col else ""
            fbpc = extract_bpc_moe(str(best["kernelName"]))
            row["fast_src"] = f"cktile(bpc={fbpc})" if fbpc else best["backend"]

            cktile = group[group["blockPerCu"].notna()]
            for bpc in bpc_values:
                bg = cktile[cktile["blockPerCu"] == bpc]
                if bg.empty:
                    row[f"bpc{bpc}_us"] = None
                    row[f"bpc{bpc}_err"] = ""
                else:
                    b = bg.loc[bg["us"].idxmin()]
                    row[f"bpc{bpc}_us"] = b["us"]
                    row[f"bpc{bpc}_err"] = b[err_col] if err_col else ""

            rows.append(row)

        result = pd.DataFrame(rows)
        cols = list(shape_keys) + ["fast_us", "fast_err", "fast_src"]
        for bpc in bpc_values:
            cols += [f"bpc{bpc}_us", f"bpc{bpc}_err"]
        result = result[[c for c in cols if c in result.columns]]
        print(result.to_string(index=False))

        _print_summary(rows, bpc_values, f"MOE {stage}")


def _print_summary(rows, bpc_values, label):
    """Print summary stats: which blockPerCu wins, CKTile vs overall."""
    bpc_winners = []
    cktile_wins = 0
    for row in rows:
        best_bpc, best_us = None, float("inf")
        for bpc in bpc_values:
            us = row.get(f"bpc{bpc}_us")
            if us is not None and us < best_us:
                best_us, best_bpc = us, bpc
        if best_bpc is not None:
            bpc_winners.append(best_bpc)
            if abs(best_us - row["fast_us"]) / max(row["fast_us"], 1e-9) < 0.005:
                cktile_wins += 1

    if not bpc_winners:
        return

    total = len(bpc_winners)
    counts = pd.Series(bpc_winners).value_counts().sort_index()
    print(f"\n  {label} — Best blockPerCu distribution:")
    for bpc, cnt in counts.items():
        print(f"    bpc={bpc}: {cnt}/{total} ({cnt/total*100:.0f}%)")
    print(f"  CKTile overall winner: {cktile_wins}/{len(rows)} shapes ({cktile_wins/len(rows)*100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Analyze blockPerCu performance")
    parser.add_argument("--blockscale", default="aiter/configs/blockscale_profile.csv")
    parser.add_argument("--bpreshuffle", default="aiter/configs/bpreshuffle_profile.csv")
    parser.add_argument("--moe", default="aiter/configs/profile_fmoe.csv")
    parser.add_argument("--only", choices=["blockscale", "bpreshuffle", "moe"])
    args = parser.parse_args()

    if not args.only or args.only == "blockscale":
        analyze_gemm_profile(args.blockscale, "Blockscale", extract_bpc_blockscale, ["M", "N", "K"])
    if not args.only or args.only == "bpreshuffle":
        analyze_gemm_profile(args.bpreshuffle, "BPreshuffle", extract_bpc_bpreshuffle, ["M", "N", "K"])
    if not args.only or args.only == "moe":
        analyze_moe_profile(args.moe)


if __name__ == "__main__":
    main()
