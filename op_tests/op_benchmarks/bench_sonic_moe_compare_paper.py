# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PAPER_TARGETS = {
    "activation_memory_reduction_pct": 45.0,
    "h100_compute_speedup_vs_scattermoe": 1.86,
    "h100_token_rounding_kernel_speedup": 1.16,
    "h100_64gpu_sonic_tokens_per_day_b": 213.0,
    "h100_96gpu_scattermoe_tokens_per_day_b": 225.0,
    "b300_forward_speedup_vs_deepgemm_pct": 25.0,
    "b300_backward_speedup_vs_deepgemm_pct": 15.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AITER SonicMoE sweep rows with the headline SonicMoE paper "
            "targets and emit a markdown gap report."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="CSV or JSONL sweep outputs")
    parser.add_argument(
        "--metric",
        choices=["direct_fused_moe_ms", "stage_sum_ms"],
        default="direct_fused_moe_ms",
    )
    parser.add_argument(
        "--paper-h100-csv",
        type=Path,
        default=None,
        help=(
            "Optional shape_str,h100_sonic_ms,h100_sonic_tflops CSV for exact "
            "shape-by-shape cross-hardware comparison."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value in (None, "", "NA", "None"):
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None


def _load_one(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_one(path))
    return rows


def _load_paper_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {row["shape_str"]: row for row in csv.DictReader(handle)}


def _best_by_shape_and_routing(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        value = _to_float(row.get(metric))
        if value is None:
            continue
        key = (str(row.get("shape_str")), str(row.get("routing")))
        prev = best.get(key)
        if prev is None or value < _to_float(prev.get(metric)):
            best[key] = row
    return best


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def _route_speedup(best: dict[tuple[str, str], dict[str, Any]], shape: str, metric: str) -> float | None:
    topk = best.get((shape, "topk"))
    rounded = best.get((shape, "rounded"))
    if topk is None or rounded is None:
        return None
    topk_ms = _to_float(topk.get(metric))
    rounded_ms = _to_float(rounded.get(metric))
    if topk_ms is None or rounded_ms is None or rounded_ms <= 0:
        return None
    return topk_ms / rounded_ms


def build_report(rows: list[dict[str, Any]], metric: str, paper_rows: dict[str, dict[str, str]]) -> str:
    best = _best_by_shape_and_routing(rows, metric)
    shapes = sorted({shape for shape, _ in best})

    lines = [
        "# SonicMoE Paper Target vs AITER Sweep",
        "",
        "## Paper headline targets",
        "",
        "| Target | Value |",
        "| --- | ---: |",
        f"| Activation memory reduction | {PAPER_TARGETS['activation_memory_reduction_pct']:.0f}% |",
        f"| H100 compute throughput speedup vs ScatterMoE BF16 | {PAPER_TARGETS['h100_compute_speedup_vs_scattermoe']:.2f}x |",
        f"| H100 token rounding main-kernel speedup | {PAPER_TARGETS['h100_token_rounding_kernel_speedup']:.2f}x |",
        f"| SonicMoE training throughput, 64 H100 | {PAPER_TARGETS['h100_64gpu_sonic_tokens_per_day_b']:.0f}B tokens/day |",
        f"| ScatterMoE training throughput, 96 H100 | {PAPER_TARGETS['h100_96gpu_scattermoe_tokens_per_day_b']:.0f}B tokens/day |",
        f"| B300 forward speedup vs DeepGEMM | {PAPER_TARGETS['b300_forward_speedup_vs_deepgemm_pct']:.0f}% |",
        f"| B300 backward speedup vs DeepGEMM | {PAPER_TARGETS['b300_backward_speedup_vs_deepgemm_pct']:.0f}% |",
        "",
        "## AITER best rows",
        "",
        f"Metric: `{metric}`. Lower is better; TFLOPS is reported from the same row.",
        "",
        "| Shape | TopK ms | Rounded ms | Rounded speedup | Balanced ms | Best TFLOPS | Bottleneck | Bottleneck share | A2 R+W MiB | H100 Sonic TFLOPS | MI/H100 TFLOPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    for shape in shapes:
        topk = best.get((shape, "topk"))
        rounded = best.get((shape, "rounded"))
        balanced = best.get((shape, "balanced"))
        candidates = [row for row in (topk, rounded, balanced) if row is not None]
        best_row = min(candidates, key=lambda row: _to_float(row.get(metric)) or float("inf"))
        topk_ms = _to_float(topk.get(metric)) if topk else None
        rounded_ms = _to_float(rounded.get(metric)) if rounded else None
        balanced_ms = _to_float(balanced.get(metric)) if balanced else None
        speedup = _route_speedup(best, shape, metric)
        tflops = _to_float(best_row.get("direct_fused_moe_expert_tflops")) or _to_float(
            best_row.get("stage_sum_expert_tflops")
        )
        paper = paper_rows.get(shape, {})
        h100_tflops = _to_float(paper.get("h100_sonic_tflops"))
        mi_h100 = None if h100_tflops in (None, 0) or tflops is None else tflops / h100_tflops
        lines.append(
            "| "
            + " | ".join(
                [
                    shape,
                    _fmt(topk_ms),
                    _fmt(rounded_ms),
                    _fmt(speedup, 3),
                    _fmt(balanced_ms),
                    _fmt(tflops, 2),
                    str(best_row.get("bottleneck", "NA")),
                    _fmt(_to_float(best_row.get("bottleneck_share")), 3),
                    _fmt(_to_float(best_row.get("a2_read_write_mib")), 1),
                    _fmt(h100_tflops, 2),
                    _fmt(mi_h100, 3),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Decision rules",
            "",
            "- If rounded/topk speedup is far below 1.16x while tile efficiency is already high, token padding is not the dominant gap for this AITER path.",
            "- If `moe_sorting_ms` dominates, prioritize sorting dispatch, token rounding, and route layout before writing a new GEMM kernel.",
            "- If stage1/stage2 dominate and A2 read+write is large relative to direct fused time, the next kernel should fuse SonicMoE-style gather/dataflow rather than only retune block sizes.",
            "- If a paper H100 per-shape CSV is not supplied, this report compares against headline targets only; it does not claim a normalized MI355X-vs-H100 speed ratio.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = _load_rows(args.inputs)
    report = build_report(rows, args.metric, _load_paper_rows(args.paper_h100_csv))
    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"wrote_report={args.output}")


if __name__ == "__main__":
    main()
