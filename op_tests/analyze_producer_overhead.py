# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Summarize per-kernel GPU durations from torch.profiler chrome traces
captured by ``bench_zero_init_splitk_demo.py``.

For each (shape, mode) pair we extract every event whose name contains
the producer's ``dynamic_per_group_scaled_quant_kernel`` symbol, the
ATen ``FillFunctor<BFloat16>`` zero-fill kernel, and the bpreshuffle
CKTile ``QuantGemmKernel`` symbol, then report mean / median durations.
The headline question we want to answer is: does the producer kernel
slow down materially when we make it absorb the GEMM Y zero-fill (i.e.
when --mode=splitk_fused)?  Comparing the producer-row mean for
``none`` and ``splitk`` (no fused zero) against ``splitk_fused`` (fused
zero) tells us how much overhead the fusion adds, and whether it stays
hidden behind the producer's existing memory-system cost.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys


PRODUCER = "dynamic_per_group_scaled_quant_kernel"
FILL = "FillFunctor"
GEMM = "QuantGemmKernel"


def _kernel_durations(path: str) -> dict[str, list[float]]:
    """Return { 'producer': [...], 'fill': [...], 'gemm': [...] } in us."""
    with open(path) as f:
        d = json.load(f)
    events = d.get("traceEvents", []) if isinstance(d, dict) else d
    out = {"producer": [], "fill": [], "gemm": []}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = (ev.get("cat") or "").lower()
        if "kernel" not in cat and "gpu" not in cat:
            continue
        name = ev.get("name", "") or ""
        dur = ev.get("dur", 0)
        if PRODUCER in name:
            out["producer"].append(dur)
        elif FILL in name:
            out["fill"].append(dur)
        elif GEMM in name:
            out["gemm"].append(dur)
    return out


def _fmt_stats(xs: list[float]) -> str:
    if not xs:
        return "         (none)        "
    n = len(xs)
    mean = statistics.mean(xs)
    med = statistics.median(xs)
    return f"mean={mean:6.2f} med={med:6.2f} n={n:>3}"


def _summarize_shape(trace_dir: str, shape: str) -> None:
    M, N, K = (int(s) for s in shape.split(","))
    print(f"\n## shape M={M} N={N} K={K}")
    print(
        f"{'mode':>14} | {'producer (us)':>26} | {'fill (us)':>26} | "
        f"{'GEMM (us)':>26} | {'sum mean':>9}"
    )
    print("-" * 120)
    for mode in ("none", "splitk", "splitk_fused"):
        path = os.path.join(trace_dir, f"trace_{mode}_M{M}_N{N}_K{K}.json")
        if not os.path.exists(path):
            print(f"{mode:>14} | (no trace)")
            continue
        d = _kernel_durations(path)
        sum_mean = (
            (statistics.mean(d["producer"]) if d["producer"] else 0)
            + (statistics.mean(d["fill"]) if d["fill"] else 0)
            + (statistics.mean(d["gemm"]) if d["gemm"] else 0)
        )
        print(
            f"{mode:>14} | {_fmt_stats(d['producer']):>26} | "
            f"{_fmt_stats(d['fill']):>26} | {_fmt_stats(d['gemm']):>26} | "
            f"{sum_mean:>8.2f}"
        )

    # Producer overhead summary: the answer to "is the fused fill hidden?".
    base_paths = [
        os.path.join(trace_dir, f"trace_{m}_M{M}_N{N}_K{K}.json")
        for m in ("none", "splitk")
    ]
    fused_path = os.path.join(trace_dir, f"trace_splitk_fused_M{M}_N{N}_K{K}.json")
    base_durs: list[float] = []
    for p in base_paths:
        if os.path.exists(p):
            base_durs.extend(_kernel_durations(p)["producer"])
    fused_durs = (
        _kernel_durations(fused_path)["producer"]
        if os.path.exists(fused_path)
        else []
    )
    if base_durs and fused_durs:
        b = statistics.mean(base_durs)
        f = statistics.mean(fused_durs)
        print(
            f"  -> producer mean: baseline (none|splitk) = {b:5.2f} us, "
            f"splitk_fused = {f:5.2f} us, overhead = {(f - b):+5.2f} us "
            f"({((f / b) - 1) * 100:+5.1f}%)"
        )
    fill_paths = [
        os.path.join(trace_dir, f"trace_{m}_M{M}_N{N}_K{K}.json")
        for m in ("splitk",)
    ]
    fill_durs: list[float] = []
    for p in fill_paths:
        if os.path.exists(p):
            fill_durs.extend(_kernel_durations(p)["fill"])
    if fill_durs and fused_durs:
        fill_mean = statistics.mean(fill_durs)
        b = statistics.mean(base_durs) if base_durs else 0
        f = statistics.mean(fused_durs)
        added = f - b
        hidden = fill_mean - added
        if fill_mean > 0:
            print(
                f"  -> fill kernel removed: {fill_mean:5.2f} us; "
                f"added to producer: {added:+5.2f} us; "
                f"net hidden: {hidden:+5.2f} us "
                f"({(hidden / fill_mean) * 100:+5.1f}% of fill cost absorbed)"
            )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--shapes", nargs="+", required=True,
                   help="One or more 'M,N,K' strings.")
    args = p.parse_args()
    if not os.path.isdir(args.trace_dir):
        print(f"ERROR: trace dir not found: {args.trace_dir}", file=sys.stderr)
        return 1
    for shape in args.shapes:
        _summarize_shape(args.trace_dir, shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
