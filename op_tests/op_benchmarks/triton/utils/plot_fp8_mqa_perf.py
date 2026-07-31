#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Parse a FlyDSL vs Triton FP8 MQA Logits benchmark markdown report and produce:
  1. A bar chart (Triton vs best FlyDSL Direct Load vs best FlyDSL LDS Staged) saved as PNG.
  2. A summary markdown table with one row per shape.

FlyDSL kernels are split into two categories by name:
  - LDS staged: kernel name contains "_lds<N>" (e.g. _lds2, _lds3)
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
    """Return True if the kernel tag identifies an LDS-staged variant.

    Matches any ``_lds<N>`` suffix (``_lds2``, ``_lds3``, ...) so new buffer
    counts are classified as LDS-staged, mirroring the kernel host's
    ``re.search(r"_lds\\d", variant)``.
    """
    return re.search(r"_lds\d", impl_tag) is not None


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
    shape_labels = [r["shape"] for r in results]

    def _col(key):
        # Missing category -> NaN so the bar is not drawn (no zero-height stub).
        return np.array(
            [r[key] if r[key] is not None else np.nan for r in results],
            dtype=float,
        )

    triton_vals = _col("triton_tflops")
    dl_vals = _col("dl_tflops")
    lds_vals = _col("lds_tflops")

    # Only keep a category series if at least one shape has a kernel for it; a
    # wholly-empty category contributes no bar, tag, or legend entry.
    #   (label, values, color, speedup_key)  speedup_key=None -> Triton (abs TFLOPs)
    series = [("Triton", triton_vals, COLORS["triton"], None)]
    if np.isfinite(dl_vals).any():
        series.append(
            ("Best FlyDSL Direct Load", dl_vals, COLORS["flydsl_dl"], "dl_speedup")
        )
    if np.isfinite(lds_vals).any():
        series.append(
            ("Best FlyDSL LDS Staged", lds_vals, COLORS["flydsl_lds"], "lds_speedup")
        )

    m = len(series)
    w = 0.8 / m  # total group width 0.8, split evenly across present series
    offsets = (np.arange(m) - (m - 1) / 2.0) * w

    fig, ax = plt.subplots(figsize=(max(14, n * 1.8), 7))

    for (label, vals, color, sp_key), off in zip(series, offsets):
        # NaN heights are skipped by matplotlib -> no bar for missing categories.
        bars = ax.bar(
            x + off, vals, w, label=label, color=color, alpha=0.9, zorder=3
        )
        for bar, val, r in zip(bars, vals, results):
            if not np.isfinite(val):
                continue  # no kernel in this category for this shape: no tag
            if sp_key is None:
                _bar_label(ax, bar, f"{val:.0f}", color)
            elif r[sp_key] is not None:
                _bar_label(ax, bar, f"x{r[sp_key]:.2f}", color)

    ax.set_xticks(x)
    ax.set_xticklabels(shape_labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("TFLOPs", fontsize=11)
    present = " vs ".join(s[0] for s in series)
    title = f"FP8 MQA Logits — {present}"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ymax = np.nanmax(np.concatenate([s[1] for s in series]))
    ax.set_ylim(0, ymax * 1.22)

    legend_handles = [
        mpatches.Patch(
            color=color,
            label=f"{label} (tag = "
            + ("abs TFLOPs" if sp_key is None else "speedup vs Triton")
            + ")",
        )
        for label, _vals, color, sp_key in series
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_png}")


# ---------------------------------------------------------------------------
# Markdown summary table
# ---------------------------------------------------------------------------


def _cat_cells(r: dict, cat: str) -> list[str]:
    """Return the [kernel, TFLOPs, verify, speedup] cells for one category
    ('dl' or 'lds').  Missing category -> em-dashes (no cross-category data)."""
    tf = r[f"{cat}_tflops"]
    if tf is None:
        return ["—", "—", "—", "—"]
    k = r[f"{cat}_impl"] or "—"
    v = r[f"{cat}_verify"] or "—"
    sp = r[f"{cat}_speedup"]
    return [k, f"{tf:.1f}", v, f"x{sp:.2f}" if sp is not None else "—"]


def make_md_table(results: list[dict], out_md: Path, png_path: Path, src_md: Path):
    # Only emit a category's columns if some shape actually has a kernel there,
    # so Direct-Load and LDS-Staged results are never merged into one column.
    has_dl = any(r["dl_tflops"] is not None for r in results)
    has_lds = any(r["lds_tflops"] is not None for r in results)

    header = ["Shape", "Triton TFLOPs"]
    if has_dl:
        header += ["Best DL kernel", "DL TFLOPs", "DL verify", "DL vs Triton"]
    if has_lds:
        header += ["Best LDS kernel", "LDS TFLOPs", "LDS verify", "LDS vs Triton"]

    lines = [
        "# FP8 MQA Logits — Triton vs Best FlyDSL Summary",
        "",
        f"Source: `{src_md}`",
        "",
        f"![Bar chart]({png_path.name})",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for r in results:
        cells = [r["shape"], f"{r['triton_tflops']:.1f}"]
        if has_dl:
            cells += _cat_cells(r, "dl")
        if has_lds:
            cells += _cat_cells(r, "lds")
        lines.append("| " + " | ".join(cells) + " |")

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
