#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Compare selector prediction (2D/3D) vs measured faster kernel (forced 2D/3D).

Inputs:
- selector CSV from analyze_unified_selector_on_jsonl.py
- benchmark CSV from benchmark_attention_native_layouts_from_jsonl.py

Goal:
Check whether predicted dispatch (2D vs 3D) matches the empirically faster
forced path for each case.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def is_ok(status: str) -> bool:
    return status == "ok" or status.startswith("ok:")


def to_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return -1


def to_float(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class CompareRow:
    idx: int
    phase: str
    predicted: str
    observed: str
    ms_2d: float
    ms_3d: float
    measured_best: str  # "2d" | "3d" | "tie"
    delta_ms: float  # ms_3d - ms_2d (positive => 2d faster)


def load_selector_csv(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = to_int(r.get("idx", ""))
            if idx >= 0:
                out[idx] = r
    return out


def load_benchmark_csv(path: Path) -> dict[int, dict[str, dict[str, str]]]:
    out: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            idx = to_int(r.get("idx", ""))
            if idx < 0:
                continue
            backend = r.get("backend", "")
            out[idx][backend] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare selector prediction against measured 2D/3D winner."
    )
    ap.add_argument(
        "--selector-csv",
        type=Path,
        default=Path("/workspaces/workspace/unified_selector_analysis.csv"),
    )
    ap.add_argument(
        "--benchmark-csv",
        type=Path,
        default=Path("/workspaces/workspace/attn_backend_results_native_layouts.csv"),
    )
    ap.add_argument(
        "--tie-eps-ms",
        type=float,
        default=1e-3,
        help="If |ms_2d - ms_3d| <= eps, treat as tie.",
    )
    ap.add_argument(
        "--out-mismatch-csv",
        type=Path,
        default=Path("/workspaces/workspace/selector_vs_measured_mismatches.csv"),
    )
    args = ap.parse_args()

    selector = load_selector_csv(args.selector_csv)
    bench = load_benchmark_csv(args.benchmark_csv)

    rows: list[CompareRow] = []
    missing = 0
    for idx, srow in selector.items():
        bmap = bench.get(idx)
        if not bmap:
            missing += 1
            continue
        r2 = bmap.get("unified_force_2d")
        r3 = bmap.get("unified_force_3d")
        if r2 is None or r3 is None:
            missing += 1
            continue
        if not (is_ok(r2.get("status", "")) and is_ok(r3.get("status", ""))):
            missing += 1
            continue
        ms2 = to_float(r2.get("ms", ""))
        ms3 = to_float(r3.get("ms", ""))
        if ms2 is None or ms3 is None:
            missing += 1
            continue

        diff = ms3 - ms2
        if abs(diff) <= args.tie_eps_ms:
            best = "tie"
        else:
            best = "2d" if ms2 < ms3 else "3d"

        rows.append(
            CompareRow(
                idx=idx,
                phase=srow.get("phase", ""),
                predicted=srow.get("predicted_kind", ""),
                observed=srow.get("observed_kind", ""),
                ms_2d=ms2,
                ms_3d=ms3,
                measured_best=best,
                delta_ms=diff,
            )
        )

    if not rows:
        print("No comparable rows found.")
        return 1

    non_tie = [r for r in rows if r.measured_best != "tie"]
    ties = [r for r in rows if r.measured_best == "tie"]
    agree = [r for r in non_tie if r.predicted == r.measured_best]
    disagree = [r for r in non_tie if r.predicted != r.measured_best]

    print(f"selector_rows={len(selector)} comparable_rows={len(rows)} missing_or_invalid={missing}")
    print(f"non_tie_rows={len(non_tie)} ties={len(ties)} (tie_eps_ms={args.tie_eps_ms})")
    if non_tie:
        print(
            f"agreement(predicted vs measured_best): {len(agree)}/{len(non_tie)} "
            f"({100.0 * len(agree) / len(non_tie):.2f}%)"
        )

    # Confusion matrix on non-ties.
    cm = Counter((r.predicted, r.measured_best) for r in non_tie)
    print("\nConfusion (predicted -> measured_best):")
    for p in ["2d", "3d"]:
        print(
            f"  {p}: to 2d={cm[(p, '2d')]:5d}  to 3d={cm[(p, '3d')]:5d}"
        )

    # By phase.
    print("\nAgreement by phase (non-tie):")
    by_phase: dict[str, list[CompareRow]] = defaultdict(list)
    for r in non_tie:
        by_phase[r.phase].append(r)
    for phase in ["prefill", "decode_global", "decode_windowed"]:
        rs = by_phase.get(phase, [])
        if not rs:
            continue
        ok = sum(1 for r in rs if r.predicted == r.measured_best)
        print(f"  {phase:14s} {ok:5d}/{len(rs):5d} ({100.0*ok/len(rs):6.2f}%)")

    # Write mismatches sorted by absolute delta.
    args.out_mismatch_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_mismatch_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "phase",
                "predicted",
                "measured_best",
                "ms_2d",
                "ms_3d",
                "delta_ms_3d_minus_2d",
                "observed_kind",
            ]
        )
        for r in sorted(disagree, key=lambda x: abs(x.delta_ms), reverse=True):
            w.writerow(
                [
                    r.idx,
                    r.phase,
                    r.predicted,
                    r.measured_best,
                    f"{r.ms_2d:.6f}",
                    f"{r.ms_3d:.6f}",
                    f"{r.delta_ms:.6f}",
                    r.observed,
                ]
            )
    print(f"\nWrote mismatches: {args.out_mismatch_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
