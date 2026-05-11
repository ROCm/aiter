#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
End-to-end CK vs Triton-2D vs Triton-3D verification for a list of CSV rows.

For each idx, this script:
  1. Spawns rocprofv3 on `_rocprof_one_row.py --side all --check-correctness`,
     which runs CK, Triton 2D, and Triton 3D back-to-back in one process
     using the SAME inputs and asserts numerical equivalence.
  2. Parses the resulting kernel_trace.csv, classifies each KERNEL_DISPATCH
     by name (kentry vs kernel_unified_attention_2d vs _3d vs reduce_segments),
     and computes the median GPU duration after dropping warmup.
  3. Sanity-checks the counts (each of CK / 2D / 3D should have exactly
     warmup + iters + 1 correctness dispatch).
  4. Aggregates everything into a single output CSV.

Methodology guarantees enforced (the things you asked me to double-check):

  - CK timings come from direct `unified_attention_fwd` calls — no Triton
    wrapper involved.
  - Triton 2D / 3D timings come from `ua_mod.unified_attention(...)` with
    BOTH CK short-circuits forcibly disabled (`_try_ck_unified_attention`,
    `_try_ck_splitkv_attention`) and `use_2d_kernel` pinned to True/False.
    So "Triton 2D" really runs `kernel_unified_attention_2d` and "Triton
    3D" really runs `kernel_unified_attention_3d` + `reduce_segments`.
  - The kernel-name audit in step 3 catches any mistake here: if the
    short-circuit slipped through, the "triton_2d" or "triton_3d" group
    would be missing dispatches and the row would be flagged.
  - Numerical equivalence (atol=2e-2, rtol=1e-2) is checked once per
    Triton path; mismatches are surfaced as a non-zero subprocess exit.

Usage:
    python bench_ck_2d_3d_all_rows_rocprof.py \
        --csv pawel-2d-3d.csv \
        --idx 873,851,...,1097 \
        --warmup 10 --iters 50 \
        --out-csv pawel-2d-3d_50rows_ck_vs_2d_vs_3d.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_IDS = "873,851,885,37,835,839,21,1483,1851,74,949,2101,2419,82,101,1963,1849,90,629,759,1329,170,793,1251,137,723,177,1241,174,1375,387,1113,1165,987,249,407,359,1115,549,971,233,415,529,317,523,547,441,557,561,1097"


def classify(name: str) -> str:
    n = name.strip()
    if n.startswith("kentry") or "UnifiedAttentionKernel" in n:
        return "ck"
    if "kernel_unified_attention_2d" in n:
        return "triton_2d"
    if "kernel_unified_attention_3d" in n:
        return "triton_3d_main"
    if "reduce_segments" in n:
        return "triton_3d_reduce"
    return "setup"


def parse_correctness(stdout: str):
    """Pull `max_abs_diff=...  match=True/False` lines out of stdout."""
    out = {}
    m = re.search(r"ck vs triton_2d: max_abs_diff=([0-9.eE+-]+)\s+match=(\w+)", stdout)
    if m: out["t2d"] = (float(m.group(1)), m.group(2) == "True")
    m = re.search(r"ck vs triton_3d: max_abs_diff=([0-9.eE+-]+)\s+match=(\w+)", stdout)
    if m: out["t3d"] = (float(m.group(1)), m.group(2) == "True")
    return out


def median_us(durs_ns):
    return statistics.median(durs_ns) / 1e3 if durs_ns else float("nan")


def run_one(rocprof, helper, csv_path, idx, warmup, iters, num_blocks_cap, work_dir):
    """Run rocprofv3 on one row; return (per-kernel medians in ms, counts,
    correctness, raw stdout)."""
    out_prefix = f"row_{idx}"
    cmd = [
        rocprof, "--kernel-trace", "--output-format", "csv",
        "-d", str(work_dir), "-o", out_prefix, "--truncate-kernels",
        "--",
        sys.executable, str(helper),
        "--csv", str(csv_path), "--idx", str(idx),
        "--warmup", str(warmup), "--iters", str(iters),
        "--side", "all", "--check-correctness",
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, None, None, p.stdout + "\n" + p.stderr

    correctness = parse_correctness(p.stdout)

    trace_csv = work_dir / f"{out_prefix}_kernel_trace.csv"
    if not trace_csv.exists():
        return None, None, correctness, "no kernel_trace.csv produced"

    with trace_csv.open() as f:
        recs = list(csv.DictReader(f))

    groups = {"ck": [], "triton_2d": [], "triton_3d_main": [],
              "triton_3d_reduce": [], "setup": []}
    for r in recs:
        kind = classify(r["Kernel_Name"])
        dur  = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        groups[kind].append(dur)

    counts = {k: len(v) for k, v in groups.items()}

    # Drop warmup. Each measured group should have at least warmup+iters+1
    # (the +1 is the correctness call). We drop `warmup` entries off the front
    # of each group; the last `iters` should be the timed iters; the extra
    # +1 correctness call is at the very front of each group, so it's also
    # dropped by --warmup.
    def trim(xs):
        skip = warmup + 1  # warmup loop + 1 correctness call
        return xs[skip:] if len(xs) > skip else xs

    medians_ms = {
        "ck":               median_us(trim(groups["ck"])) / 1e3,
        "triton_2d":        median_us(trim(groups["triton_2d"])) / 1e3,
        "triton_3d_main":   median_us(trim(groups["triton_3d_main"])) / 1e3,
        "triton_3d_reduce": median_us(trim(groups["triton_3d_reduce"])) / 1e3,
    }
    return medians_ms, counts, correctness, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/workspace/aiter/pawel-2d-3d.csv")
    ap.add_argument("--idx", default=DEFAULT_IDS)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters",  type=int, default=50)
    ap.add_argument("--num-blocks-cap", type=int, default=16384)
    ap.add_argument("--out-csv", default="/workspace/aiter/pawel-2d-3d_50rows_ck_vs_2d_vs_3d.csv")
    ap.add_argument("--rocprof", default=shutil.which("rocprofv3") or "/opt/rocm/bin/rocprofv3")
    ap.add_argument("--keep-traces", action="store_true",
                    help="Keep per-row rocprof CSVs in --trace-dir (default: tmp)")
    ap.add_argument("--trace-dir", default="/tmp/rocprof_all_rows")
    args = ap.parse_args()

    helper = Path(__file__).with_name("_rocprof_one_row.py")
    if not helper.exists():
        raise SystemExit(f"helper not found: {helper}")
    if not Path(args.csv).exists():
        raise SystemExit(f"csv not found: {args.csv}")

    work_dir = Path(args.trace_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    idx_list = args.idx.split(",")

    # Load original CSV rows for shape metadata + provenance.
    src_rows = {}
    with open(args.csv) as f:
        for r in csv.DictReader(f):
            if r["idx"] in idx_list:
                src_rows[r["idx"]] = r

    expected = args.warmup + args.iters + 1  # +1 for correctness call

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "idx", "phase",
            "num_seqs", "total_q_tokens", "max_seqlen_q", "max_seqlen_k",
            "num_q_heads", "num_kv_heads", "head_size", "block_size",
            "window_size", "q_dtype", "softcap", "has_sinks", "mask",
            # CSV-reported (graph mode, gfx942)
            "csv_ck_ua_ms", "csv_triton_2d_ms", "csv_triton_3d_ms",
            "csv_best", "csv_speedup_vs_best_triton",
            # rocprofv3 (eager, gfx950)
            "rocprof_ck_ms", "rocprof_t2d_ms",
            "rocprof_t3d_main_ms", "rocprof_t3d_reduce_ms", "rocprof_t3d_total_ms",
            # Speedups
            "sp_ck_vs_t2d", "sp_ck_vs_t3d", "sp_ck_vs_min_triton",
            # Audit
            "ck_n", "t2d_n", "t3d_main_n", "t3d_reduce_n",
            "ck_n_ok", "t2d_n_ok", "t3d_n_ok",
            "ck_eq_t2d", "ck_eq_t2d_max_diff",
            "ck_eq_t3d", "ck_eq_t3d_max_diff",
        ])

        hdr = (f"{'idx':>5} {'phase':>7} {'b':>4} {'sk':>6} | "
               f"{'CK':>8} {'T2D':>8} {'T3D':>8} | "
               f"{'sp/T2D':>7} {'sp/T3D':>7} | {'audit':>16}  {'corr':>10}")
        print(hdr)
        print("-" * len(hdr))

        all_ok = True
        for i, idx in enumerate(idx_list):
            t0 = time.time()
            medians, counts, corr, log = run_one(
                args.rocprof, helper, args.csv, idx,
                args.warmup, args.iters, args.num_blocks_cap, work_dir,
            )
            elapsed = time.time() - t0
            row = src_rows.get(idx, {})
            phase = row.get("phase", "?")
            b   = row.get("num_seqs", "?")
            sk  = row.get("max_seqlen_k", "?")

            if medians is None:
                print(f"{idx:>5} {phase:>7} {b:>4} {sk:>6} | "
                      f"FAILED: {log[:200]}")
                all_ok = False
                continue

            ck      = medians["ck"]
            t2d     = medians["triton_2d"]
            t3d_m   = medians["triton_3d_main"]
            t3d_r   = medians["triton_3d_reduce"]
            t3d_tot = (t3d_m + t3d_r) if (t3d_m == t3d_m and t3d_r == t3d_r) else float("nan")

            sp2d = (t2d / ck)     if (t2d == t2d and ck) else float("nan")
            sp3d = (t3d_tot / ck) if (t3d_tot == t3d_tot and ck) else float("nan")
            spmin = min(sp2d, sp3d) if (sp2d == sp2d and sp3d == sp3d) else float("nan")

            ck_n  = counts.get("ck", 0)
            t2d_n = counts.get("triton_2d", 0)
            t3d_main_n = counts.get("triton_3d_main", 0)
            t3d_red_n  = counts.get("triton_3d_reduce", 0)

            ck_n_ok  = ck_n  == expected
            t2d_n_ok = t2d_n == expected
            t3d_n_ok = (t3d_main_n == expected) and (t3d_red_n == expected)

            audit = "ok" if (ck_n_ok and t2d_n_ok and t3d_n_ok) else (
                f"CK={ck_n}/{expected} T2D={t2d_n}/{expected} "
                f"T3Dm={t3d_main_n}/{expected} T3Dr={t3d_red_n}/{expected}"
            )

            t2d_diff, t2d_ok = corr.get("t2d", (-1.0, False))
            t3d_diff, t3d_ok = corr.get("t3d", (-1.0, False))
            corr_str = f"2D:{t2d_ok!s:5} 3D:{t3d_ok!s:5}"

            csv_ck = float(row["ck_ua_ms"])
            csv_t2 = float(row["triton_2d_ms"])
            csv_t3 = float(row["triton_3d_ms"])
            csv_sp = min(csv_t2, csv_t3) / csv_ck

            w.writerow([
                idx, phase,
                row["num_seqs"], row["total_q_tokens"], row["max_seqlen_q"], row["max_seqlen_k"],
                row["num_q_heads"], row["num_kv_heads"], row["head_size"], row["block_size"],
                row["window_size"], "torch.bfloat16", "0.0", "False", "causal",
                row["ck_ua_ms"], row["triton_2d_ms"], row["triton_3d_ms"],
                row["best"], f"{csv_sp:.4f}",
                f"{ck:.6f}", f"{t2d:.6f}",
                f"{t3d_m:.6f}", f"{t3d_r:.6f}", f"{t3d_tot:.6f}",
                f"{sp2d:.4f}", f"{sp3d:.4f}", f"{spmin:.4f}",
                ck_n, t2d_n, t3d_main_n, t3d_red_n,
                ck_n_ok, t2d_n_ok, t3d_n_ok,
                t2d_ok, f"{t2d_diff:.6f}",
                t3d_ok, f"{t3d_diff:.6f}",
            ])

            print(f"{idx:>5} {phase:>7} {b:>4} {sk:>6} | "
                  f"{ck:>8.4f} {t2d:>8.4f} {t3d_tot:>8.4f} | "
                  f"{sp2d:>6.2f}x {sp3d:>6.2f}x | {audit:>16}  {corr_str:>10}")
            sys.stdout.flush()

            if not (ck_n_ok and t2d_n_ok and t3d_n_ok and t2d_ok and t3d_ok):
                all_ok = False

    print()
    print(f"wrote: {out_path}")
    print(f"all rows audited+correct: {all_ok}")


if __name__ == "__main__":
    main()
