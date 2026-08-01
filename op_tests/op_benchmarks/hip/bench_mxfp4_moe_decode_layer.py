#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
End-to-end benchmark for an MXFP4 MoE decode layer with fused sort+quant.

Times a full `fused_moe` call (stage 1 + stage 2) at decode batch sizes, with
and without the fused route-sort + MXFP4 quant path:

    A  baseline    conventional sort, quant and sorted-row scale expansion
    C  fused       fused sort+quant kernel, GEMM1 reads compact per-token scales

Both arms are driven through `op_tests/test_moe_2stage.py`, which validates the
layer output on every case, so the reported `logits_diff` doubles as an accuracy
check across arms. Each arm runs in its own subprocess because the feature is
selected by environment variable.

Unlike the component benchmarks (bench_mxfp4_moe_sort_quant.py and
bench_mxfp4_moe_compact_scale_gemm1.py) this measures the real dispatch path.
Prefer it when deciding whether the feature pays off: summing the isolated
components over-weights the plumbing and does not predict the layer result.

Note `SGLANG_AITER_FUSED_DECODE_COMPACT_SCALE` falls back to
`SGLANG_AITER_FUSED_DECODE_SORT_QUANT` when unset, so both arms set both
variables explicitly.

Usage:
    # Run with default parameters (Kimi decode shape, E=384/topk=8)
    python bench_mxfp4_moe_decode_layer.py

    # Run selected token counts, averaged over 2 runs per arm
    python bench_mxfp4_moe_decode_layer.py -t 1 8 32 --reps 2

    # Save results to CSV
    python bench_mxfp4_moe_decode_layer.py -o results.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Default sweep parameters (Kimi K2 TP8 shard, separate shared expert)
MODEL_DIM = 7168
INTER_DIM = 256
EXPERTS = 384
TOPK = 8
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

SORT_QUANT_ENV = "SGLANG_AITER_FUSED_DECODE_SORT_QUANT"
COMPACT_SCALE_ENV = "SGLANG_AITER_FUSED_DECODE_COMPACT_SCALE"

ARMS = {
    "A": ("baseline", {SORT_QUANT_ENV: "0", COMPACT_SCALE_ENV: "0"}),
    "C": ("fused+compact", {SORT_QUANT_ENV: "auto", COMPACT_SCALE_ENV: "auto"}),
}

HARNESS = Path(__file__).resolve().parents[2] / "test_moe_2stage.py"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_harness_table(stdout: str) -> dict[int, tuple[float, float]]:
    """Extract {tokens: (us, logits_diff)} from the harness markdown summary.

    Columns are located by name rather than position so the mapping survives the
    harness gaining or reordering columns.
    """
    columns: list[str] | None = None
    results: dict[int, tuple[float, float]] = {}

    for line in stdout.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if columns is None:
            if "us" in cells and "token" in cells:
                columns = cells
            continue
        if len(cells) != len(columns):
            continue
        row = dict(zip(columns, cells))
        try:
            tokens = int(row["token"])
            us = float(row["us"])
        except (KeyError, ValueError):
            continue
        try:
            logits_diff = float(row["logits_diff"])
        except (KeyError, ValueError):
            logits_diff = float("nan")
        results[tokens] = (us, logits_diff)

    return results


def run_arm(arm: str, args) -> dict[int, tuple[float, float]]:
    """Run one arm of the sweep in a subprocess and return its parsed results."""
    _label, arm_env = ARMS[arm]
    env = dict(os.environ, **arm_env)
    # The harness must import the aiter package from this checkout, not from any
    # other copy that happens to sit earlier on the default import path.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]]
        if env.get("PYTHONPATH")
        else [str(REPO_ROOT)]
    )

    cmd = [
        sys.executable,
        str(HARNESS),
        "-q",
        "4",
        "-e",
        str(args.experts),
        "-k",
        str(args.topk),
        "-dim",
        f"{args.model_dim},{args.inter_dim}",
        "-hip",
        "0,0",
        "--no-flydsl-csv",
        "--no-situv2",
        "-t",
        *[str(t) for t in args.tokens],
    ]
    # The harness emits its summary table through the logger, i.e. on stderr.
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # A non-zero exit is reported via the parse failure below, which can
        # also show the harness output that explains it.
        check=False,
    )
    parsed = _parse_harness_table(proc.stdout)
    if not parsed:
        sys.stderr.write(proc.stdout[-4000:])
        raise RuntimeError(
            f"arm {arm}: harness produced no parseable results "
            f"(exit code {proc.returncode})"
        )
    return parsed


def run_benchmark(args):
    """Sweep both arms and print the comparison table."""
    print(
        f"E={args.experts} topk={args.topk} model_dim={args.model_dim} "
        f"inter_dim={args.inter_dim}"
    )
    print(f"harness: {HARNESS}")

    # best[arm][tokens] = (us, logits_diff) of the fastest rep
    best: dict[str, dict[int, tuple[float, float]]] = {arm: {} for arm in ARMS}
    for rep in range(1, args.reps + 1):
        for arm, (label, _env) in ARMS.items():
            print(f"  running rep {rep}/{args.reps}, arm {arm} ({label}) ...")
            for tokens, (us, diff) in run_arm(arm, args).items():
                prior = best[arm].get(tokens)
                if prior is None or us < prior[0]:
                    best[arm][tokens] = (us, diff)

    header = (
        f"\n{'M':>8} {'A baseline':>13} {'C fused+compact':>18} {'C/A':>8} "
        f"{'A diff':>12} {'C diff':>12}"
    )
    print(header)
    print("-" * (len(header) - 1))

    results = []
    for tokens in args.tokens:
        if tokens not in best["A"] or tokens not in best["C"]:
            continue
        a_us, a_diff = best["A"][tokens]
        c_us, c_diff = best["C"][tokens]
        results.append((tokens, a_us, c_us, a_diff, c_diff))
        print(
            f"{tokens:>8} {a_us:>13.2f} {c_us:>18.2f} {a_us / c_us:>7.2f}x "
            f"{a_diff:>12.2e} {c_diff:>12.2e}"
        )

    print(f"\nBest of {args.reps} run(s) per arm. Times are us for the full layer.")

    if args.o:
        _save_results_csv(args.o, results)


def _save_results_csv(filepath: str, results: list):
    """Save benchmark results to CSV file."""
    path = Path(filepath)
    with open(path, "w") as f:
        f.write(
            "tokens,baseline_us,fused_compact_us,speedup,"
            "baseline_logits_diff,fused_compact_logits_diff\n"
        )
        f.writelines(
            f"{tokens},{a_us:.4f},{c_us:.4f},{a_us / c_us:.4f},"
            f"{a_diff:.6e},{c_diff:.6e}\n"
            for tokens, a_us, c_us, a_diff, c_diff in results
        )
    print(f"Results saved to {path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 MoE Decode Layer",
        description=(
            "Benchmark a full MXFP4 MoE decode layer with and without the fused "
            "route-sort + quant path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=TOKENS,
        help="Token counts to sweep.",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        default=EXPERTS,
        help="Number of experts.",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=TOPK,
        help="Experts routed per token.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        dest="model_dim",
        default=MODEL_DIM,
        help="Model hidden dimension.",
    )
    parser.add_argument(
        "--inter-dim",
        type=int,
        dest="inter_dim",
        default=INTER_DIM,
        help="Expert intermediate dimension.",
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="FILE",
        help="Output CSV file path for results.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=2,
        help="Number of runs per arm; the fastest is reported.",
    )
    return parser.parse_args()


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
