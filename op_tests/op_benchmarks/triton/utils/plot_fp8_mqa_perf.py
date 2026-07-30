#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Parse a FlyDSL vs Triton FP8 MQA Logits benchmark markdown report and produce:
  1. A bar chart (Triton vs best FlyDSL Direct Load vs best FlyDSL LDS Staged) saved as PNG.
  2. A summary markdown table with one row per shape.

FlyDSL kernels are split into two categories by name:
  - LDS staged: kernel name contains "_lds2"
  - Direct load: all other FlyDSL kernels

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


def _is_lds(impl_tag: str) -> bool:
    """Return True if the kernel tag identifies an LDS-staged variant."""
    return "_lds2" in impl_tag


def parse_md(path: Path) -> list[dict]:
    """Return one dict per shape section with Triton, best Direct-Load, and best LDS rows."""
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
        best_dl = None   # best Direct-Load FlyDSL: (tflops, impl, verify)
        best_lds = None  # best LDS-staged FlyDSL:  (tflops, impl, verify)

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
                tag = impl[len("flydsl:"):]  # strip "flydsl:" prefix
                entry = {"impl": tag, "tflops": tflops, "verify": verify}
                if _is_lds(tag):
                    if best_lds is None or tflops > best_lds["tflops"]:
                        best_lds = entry
                else:
                    if best_dl is None or tflops > best_dl["tflops"]:
                        best_dl = entry

        if triton_row is None:
            continue  # no triton baseline, skip section

        # Recompute speedups from raw tflops for precision
        def _speedup(entry):
            if entry is None or triton_row["tflops"] <= 0:
                return None
            return entry["tflops"] / triton_row["tflops"]

        results.append(
            {
                "shape": shape_label,
                "triton_tflops": triton_row["tflops"],
                "triton_verify": triton_row["verify"],
                # Direct-load best
                "dl_impl": best_dl["impl"] if best_dl else None,
                "dl_tflops": best_dl["tflops"] if best_dl else None,
                "dl_verify": best_dl["verify"] if best_dl else None,
                "dl_speedup": _speedup(best_dl),
                # LDS-staged best
                "lds_impl": best_lds["impl"] if best_lds else None,
                "lds_tflops": best_lds["tflops"] if best_lds else None,
                "lds_verify": best_lds["verify"] if best_lds else None,
                "lds_speedup": _speedup(best_lds),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------

COLORS = {
    "triton": "#4878CF",   # blue
    "flydsl_dl": "#D65F5F",  # red/salmon  – direct load
    "flydsl_lds": "#59A14F",  # green       – LDS staged
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
    w = 0.25  # narrower to fit 3 bars

    triton_vals = np.array([r["triton_tflops"] for r in results])
    dl_vals = np.array(
        [r["dl_tflops"] if r["dl_tflops"] is not None else 0.0 for r in results]
    )
    lds_vals = np.array(
        [r["lds_tflops"] if r["lds_tflops"] is not None else 0.0 for r in results]
    )
    shape_labels = [r["shape"] for r in results]

    fig, ax = plt.subplots(figsize=(max(14, n * 1.8), 7))

    bars_triton = ax.bar(
        x - w, triton_vals, w, label="Triton", color=COLORS["triton"], alpha=0.9, zorder=3
    )
    bars_dl = ax.bar(
        x, dl_vals, w, label="Best FlyDSL Direct Load", color=COLORS["flydsl_dl"], alpha=0.9, zorder=3
    )
    bars_lds = ax.bar(
        x + w, lds_vals, w, label="Best FlyDSL LDS Staged", color=COLORS["flydsl_lds"], alpha=0.9, zorder=3
    )

    for bar, val in zip(bars_triton, triton_vals):
        _bar_label(ax, bar, f"{val:.0f}", COLORS["triton"])

    for bar, r in zip(bars_dl, results):
        tag = f"x{r['dl_speedup']:.2f}" if r["dl_speedup"] is not None else "N/A"
        _bar_label(ax, bar, tag, COLORS["flydsl_dl"])

    for bar, r in zip(bars_lds, results):
        tag = f"x{r['lds_speedup']:.2f}" if r["lds_speedup"] is not None else "N/A"
        _bar_label(ax, bar, tag, COLORS["flydsl_lds"])

    ax.set_xticks(x)
    ax.set_xticklabels(shape_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("TFLOPs", fontsize=11)
    title = "FP8 MQA Logits — Triton vs Best FlyDSL Direct Load vs Best FlyDSL LDS Staged"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(triton_vals.max(), dl_vals.max(), lds_vals.max()) * 1.22)

    legend_handles = [
        mpatches.Patch(color=COLORS["triton"], label="Triton (tag = abs TFLOPs)"),
        mpatches.Patch(color=COLORS["flydsl_dl"], label="Best FlyDSL Direct Load (tag = speedup vs Triton)"),
        mpatches.Patch(color=COLORS["flydsl_lds"], label="Best FlyDSL LDS Staged (tag = speedup vs Triton)"),
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
        "| Shape | Triton TFLOPs | FlyDSL TFLOPs | Best FlyDSL kernel | Verification | Speedup vs Triton |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        triton = f"{r['triton_tflops']:.1f}"
        # Pick whichever of DL / LDS is faster.
        candidates = [
            (r["dl_tflops"], r["dl_impl"], r["dl_verify"], r["dl_speedup"]),
            (r["lds_tflops"], r["lds_impl"], r["lds_verify"], r["lds_speedup"]),
        ]
        best_tf, best_k, best_v, best_sp = max(
            ((tf, k, v, sp) for tf, k, v, sp in candidates if tf is not None),
            key=lambda t: t[0],
            default=(None, None, None, None),
        )
        fly_tf = f"{best_tf:.1f}" if best_tf is not None else "N/A"
        fly_k = best_k if best_k else "N/A"
        fly_v = best_v if best_v else "N/A"
        fly_sp = f"x{best_sp:.2f}" if best_sp is not None else "N/A"
        lines.append(f"| {r['shape']} | {triton} | {fly_tf} | {fly_k} | {fly_v} | {fly_sp} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved summary: {out_md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot Triton vs best FlyDSL Direct Load vs best FlyDSL LDS Staged."
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
