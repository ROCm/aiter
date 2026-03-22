#!/usr/bin/env python3
"""
Compare current benchmark results against baseline (previous run on main).
Outputs a markdown table suitable for PR comments.

Usage:
    python3 perf_compare.py --current perf_results/combined_results.csv --output comparison.md
    python3 perf_compare.py --current perf_results/combined_results.csv --baseline baseline.csv --output comparison.md
"""

import argparse
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def get_baseline_from_artifact(repo="ROCm/aiter"):
    """Try to fetch the latest perf results from main branch artifacts."""
    # TODO (Phase 2): Download latest perf-results artifact from main branch
    # workflow run via GitHub API. For now, returns empty — baseline builds up
    # after the first push-to-main run produces an artifact.
    return []


def extract_perf_key(row):
    """Create a unique key for matching results across runs."""
    # Use benchmark name + shape dimensions
    parts = [row.get("_benchmark", "")]
    for dim in ["M", "N", "K", "E", "top_k", "BATCH", "HQ", "HK", "N_CTX_Q", "N_CTX_K",
                "model", "model_name", "layer"]:
        if dim in row and row[dim]:
            parts.append(f"{dim}={row[dim]}")
    return "|".join(parts)


def extract_perf_value(row):
    """Extract the primary performance metric from a result row."""
    # Try TFLOPS first, then bandwidth, then time
    for key in ["TFLOPS", "fwd(TFLOPS)", "fwd_Time_(ms)", "Time_(ms)", "Bandwidth_(GB/s)"]:
        if key in row and row[key]:
            try:
                return float(row[key]), key
            except (ValueError, TypeError):
                continue
    return None, None


def compare(current_rows, baseline_rows):
    """Compare current vs baseline, return list of comparison dicts."""
    baseline_idx = {}
    for row in baseline_rows:
        key = extract_perf_key(row)
        val, metric = extract_perf_value(row)
        if val is not None:
            baseline_idx[key] = (val, metric, row)

    comparisons = []
    for row in current_rows:
        key = extract_perf_key(row)
        cur_val, cur_metric = extract_perf_value(row)
        if cur_val is None:
            continue

        base_val, base_metric, base_row = baseline_idx.get(key, (None, None, None))

        comp = {
            "benchmark": row.get("_benchmark", ""),
            "key": key,
            "current": cur_val,
            "metric": cur_metric,
        }

        if base_val is not None and base_metric == cur_metric:
            comp["baseline"] = base_val
            if base_val > 0:
                # For TFLOPS/bandwidth: higher is better, ratio > 1 = improvement
                # For time: lower is better, ratio < 1 = improvement
                ratio = cur_val / base_val
                if "time" in cur_metric.lower() or "ms" in cur_metric.lower():
                    comp["change_pct"] = (1.0 - ratio) * 100  # positive = faster
                    comp["improved"] = ratio < 0.95
                    comp["regressed"] = ratio > 1.05
                else:
                    comp["change_pct"] = (ratio - 1.0) * 100  # positive = faster
                    comp["improved"] = ratio > 1.05
                    comp["regressed"] = ratio < 0.95
            else:
                comp["change_pct"] = 0
                comp["improved"] = False
                comp["regressed"] = False
        else:
            comp["baseline"] = None
            comp["change_pct"] = None
            comp["improved"] = False
            comp["regressed"] = False

        comparisons.append(comp)

    return comparisons


def format_markdown(comparisons, metadata=None):
    """Format comparison results as markdown for PR comment."""
    lines = []

    if not comparisons:
        lines.append("No benchmark results to report.\n")
        return "\n".join(lines)

    # Group by benchmark
    by_bench = {}
    for c in comparisons:
        bench = c["benchmark"]
        if bench not in by_bench:
            by_bench[bench] = []
        by_bench[bench].append(c)

    # Summary
    has_baseline = any(c["baseline"] is not None for c in comparisons)
    total = len(comparisons)
    improved = sum(1 for c in comparisons if c.get("improved"))
    regressed = sum(1 for c in comparisons if c.get("regressed"))

    if has_baseline:
        lines.append(f"**Summary**: {total} configs benchmarked | "
                     f"{improved} improved | {regressed} regressed\n")

        if regressed > 0:
            lines.append("> **Warning**: Performance regressions detected!\n")
    else:
        lines.append(f"**Summary**: {total} configs benchmarked (no baseline for comparison)\n")

    # Per-benchmark tables
    for bench, results in sorted(by_bench.items()):
        lines.append(f"\n### {bench}\n")

        if has_baseline:
            lines.append("| Config | Current | Baseline | Change | Status |")
            lines.append("|--------|--------:|--------:|---------:|--------|")
        else:
            lines.append("| Config | Value | Metric |")
            lines.append("|--------|------:|--------|")

        for c in results[:50]:  # Limit rows per benchmark
            # Clean up key for display
            display_key = c["key"].replace(c["benchmark"] + "|", "").replace("|", ", ")
            if len(display_key) > 60:
                display_key = display_key[:57] + "..."

            if has_baseline and c["baseline"] is not None:
                change = c.get("change_pct", 0)
                if change is not None:
                    sign = "+" if change > 0 else ""
                    status = ""
                    if c.get("regressed"):
                        status = "🔴"
                    elif c.get("improved"):
                        status = "🟢"
                    else:
                        status = "⚪"
                    lines.append(f"| {display_key} | {c['current']:.2f} | {c['baseline']:.2f} | {sign}{change:.1f}% | {status} |")
                else:
                    lines.append(f"| {display_key} | {c['current']:.2f} | {c['baseline']:.2f} | — | ⚪ |")
            else:
                lines.append(f"| {display_key} | {c['current']:.2f} | {c['metric']} |")

        if len(results) > 50:
            lines.append(f"\n_...and {len(results) - 50} more configs_\n")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare perf benchmark results")
    parser.add_argument("--current", required=True, help="Current run CSV")
    parser.add_argument("--baseline", default=None, help="Baseline CSV (optional)")
    parser.add_argument("--output", default="comparison.md", help="Output markdown file")
    args = parser.parse_args()

    current = load_csv(args.current)
    if not current:
        print(f"No results in {args.current}")
        with open(args.output, "w") as f:
            f.write("No benchmark results available.\n")
        return

    if args.baseline:
        baseline = load_csv(args.baseline)
    else:
        baseline = get_baseline_from_artifact()

    comparisons = compare(current, baseline)
    md = format_markdown(comparisons)

    with open(args.output, "w") as f:
        f.write(md)

    print(f"Wrote comparison to {args.output}")
    print(md[:500])


if __name__ == "__main__":
    main()
