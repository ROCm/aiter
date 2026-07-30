#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Parse a FlyDSL vs Triton FP8 MQA Logits benchmark markdown report and produce:
  1. A bar chart (Triton vs best FlyDSL kernel) saved as PNG.
  2. A summary markdown table with one row per shape.

Usage:
    python plot_fp8_mqa_perf.py bench-results.md [--out-png foo.png] [--out-md foo.md]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# The utils/ package has a local argparse.py that shadows stdlib when the script
# is run directly (Python inserts the script's directory at sys.path[0]).
# Remove the local directory from the path before importing stdlib argparse.
_script_dir = str(Path(__file__).parent.resolve())
_removed = _script_dir in sys.path
if _removed:
    sys.path.remove(_script_dir)
import argparse  # noqa: E402 – must be after path surgery
if _removed:
    sys.path.insert(0, _script_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Markdown parser
# ---------------------------------------------------------------------------

# Matches a shape section heading, e.g.:
#   ### Shape 3: bs1 128x32768 H64 D128 [fn/fn]
_SHAPE_RE = re.compile(
    r"^###\s+Shape\s+\d+:\s+(.+)$",
    re.MULTILINE,
)

# Matches a table data row (non-header, non-separator).
# Cells separated by |, first and last | optional.
_ROW_RE = re.compile(r"^\|(.+)\|$")


def _parse_tflops(cell: str) -> float | None:
    """Return the TFLOPs value from a cell, stripping ±std noise.  Returns
    None if the cell contains a non-numeric token like GRAPH-FAIL."""
    cell = cell.strip()
    # Drop ±std suffix (e.g. "1157.3±510.3")
    cell = re.sub(r"±[\d.]+", "", cell).strip()
    try:
        return float(cell)
    except ValueError:
        return None


def _parse_speedup(cell: str) -> float | None:
    """Parse a vs_triton cell like '1.53x' or '0.38x'.  Returns None for '-'."""
    cell = cell.strip()
    m = re.match(r"^([\d.]+)x$", cell)
    if m:
        return float(m.group(1))
    return None


def parse_md(path: Path) -> list[dict]:
    """Return one dict per shape section with the Triton row and the best FlyDSL row."""
    text = path.read_text()
    sections = _SHAPE_RE.split(text)
    # sections = [preamble, label1, body1, label2, body2, ...]
    if len(sections) < 3:
        sys.exit(f"[error] No shape sections found in {path}")

    results = []
    for i in range(1, len(sections), 2):
        shape_label = sections[i].strip()
        body = sections[i + 1]

        triton_row = None
        best_flydsl = None  # (tflops, impl, verify, vs_triton_raw)

        for line in body.splitlines():
            m = _ROW_RE.match(line.strip())
            if not m:
                continue
            cells = [c.strip() for c in m.group(1).split("|")]
            if len(cells) < 5:
                continue
            impl, time_us_raw, tflops_raw, verify, vs_triton_raw = cells[:5]

            # Skip header / separator rows
            if impl in ("impl", "---", "") or "---" in impl:
                continue

            tflops = _parse_tflops(tflops_raw)
            if tflops is None:
                continue

            if impl == "triton":
                triton_row = {
                    "tflops": tflops,
                    "verify": verify,
                }
            elif impl.startswith("flydsl:"):
                if best_flydsl is None or tflops > best_flydsl["tflops"]:
                    speedup = _parse_speedup(vs_triton_raw)
                    best_flydsl = {
                        "impl": impl[len("flydsl:"):],  # strip "flydsl:" prefix
                        "tflops": tflops,
                        "verify": verify,
                        "speedup": speedup,
                    }

        if triton_row is None:
            continue  # no triton baseline, skip section

        # Recompute speedup from raw tflops for precision (vs_triton cell is rounded)
        if best_flydsl is not None:
            best_flydsl["speedup"] = (
                best_flydsl["tflops"] / triton_row["tflops"]
                if triton_row["tflops"] > 0
                else None
            )

        results.append(
            {
                "shape": shape_label,
                "triton_tflops": triton_row["tflops"],
                "triton_verify": triton_row["verify"],
                "best_impl": best_flydsl["impl"] if best_flydsl else None,
                "best_tflops": best_flydsl["tflops"] if best_flydsl else None,
                "best_verify": best_flydsl["verify"] if best_flydsl else None,
                "speedup": best_flydsl["speedup"] if best_flydsl else None,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

COLORS = {
    "triton": "#4878CF",  # blue
    "flydsl": "#D65F5F",  # red/salmon
}


def _bar_label(ax, bar, text: str, color: str):
    h = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + 5,
        text,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=color,
        fontweight="bold",
        rotation=90,
    )


def make_bar_chart(results: list[dict], out_png: Path, title_extra: str = ""):
    n = len(results)
    x = np.arange(n)
    w = 0.35

    triton_vals = np.array([r["triton_tflops"] for r in results])
    flydsl_vals = np.array(
        [r["best_tflops"] if r["best_tflops"] is not None else 0.0 for r in results]
    )
    shape_labels = [r["shape"] for r in results]

    fig, ax = plt.subplots(figsize=(max(14, n * 1.6), 7))

    bars_triton = ax.bar(
        x - w / 2, triton_vals, w, label="Triton", color=COLORS["triton"], alpha=0.9, zorder=3
    )
    bars_flydsl = ax.bar(
        x + w / 2, flydsl_vals, w, label="Best FlyDSL", color=COLORS["flydsl"], alpha=0.9, zorder=3
    )

    for bar, val in zip(bars_triton, triton_vals):
        _bar_label(ax, bar, f"{val:.0f}", COLORS["triton"])

    for bar, r in zip(bars_flydsl, results):
        if r["speedup"] is not None:
            tag = f"x{r['speedup']:.2f}"
        else:
            tag = "N/A"
        _bar_label(ax, bar, tag, COLORS["flydsl"])

    ax.set_xticks(x)
    ax.set_xticklabels(shape_labels, fontsize=8)
    ax.set_ylabel("TFLOPs", fontsize=11)
    title = "FP8 MQA Logits — Triton/Gluon vs Best FlyDSL Kernel"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(triton_vals.max(), flydsl_vals.max()) * 1.22)

    legend_handles = [
        mpatches.Patch(color=COLORS["triton"], label="Triton (bar = abs TFLOPs)"),
        mpatches.Patch(color=COLORS["flydsl"], label="Best FlyDSL (tag = speedup vs Triton)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


# ---------------------------------------------------------------------------
# Markdown summary table
# ---------------------------------------------------------------------------


def make_md_table(results: list[dict], out_md: Path, png_path: Path, src_md: Path):
    lines = [
        "# FP8 MQA Logits — Triton vs Best FlyDSL Summary",
        "",
        f"Source: `{src_md}`",
        "",
        f"![Bar chart]({png_path.name})",
        "",
        "| Shape | Triton TFLOPs | FlyDSL TFLOPs | Best FlyDSL kernel | Verification (against Triton/Gluon) | Perf change |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        triton = f"{r['triton_tflops']:.1f}"
        flydsl = f"{r['best_tflops']:.1f}" if r["best_tflops"] is not None else "N/A"
        kernel = r["best_impl"] if r["best_impl"] else "N/A"
        verify = r["best_verify"] if r["best_verify"] else "N/A"
        if r["speedup"] is not None:
            perf = f"x{r['speedup']:.2f}"
        else:
            perf = "N/A"
        lines.append(f"| {r['shape']} | {triton} | {flydsl} | {kernel} | {verify} | {perf} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved summary: {out_md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot Triton vs best FlyDSL from a benchmark markdown report."
    )
    parser.add_argument("input_md", type=Path, help="Benchmark markdown results file")
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Output PNG path (default: <input stem>-plot.png beside the input file)",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Output summary markdown path (default: <input stem>-summary.md beside the input file)",
    )
    args = parser.parse_args()

    src = args.input_md
    if not src.exists():
        sys.exit(f"[error] File not found: {src}")

    out_png = args.out_png or src.with_name(src.stem + "-plot.png")
    out_md = args.out_md or src.with_name(src.stem + "-summary.md")

    results = parse_md(src)
    if not results:
        sys.exit("[error] No usable shape results parsed from the input file.")

    print(f"Parsed {len(results)} shape(s) from {src}")

    make_bar_chart(results, out_png)
    make_md_table(results, out_md, out_png, src)


if __name__ == "__main__":
    main()
