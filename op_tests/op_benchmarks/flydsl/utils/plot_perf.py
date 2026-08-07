# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generic bar-chart plotter for FlyDSL A/B bench Markdown reports.

Parses the ``### Shape N:`` sections written by bench scripts in this directory
and produces a PNG bar chart plus an optional summary Markdown table.

Usage
-----
    python plot_perf.py bench-results.md [--out-png foo.png] [--out-md foo.md]
    python plot_perf.py bench-results.md \\
        --title "GDN K5 — gfx942" \\
        --baseline hip \\
        --categories "triton=Triton" "flydsl=FlyDSL" "hip=HIP"

The ``--categories`` argument maps impl name prefixes to display labels for the
chart legend.  Multiple ``key=label`` pairs are accepted; order determines the
bar order in each group.

The ``--baseline`` argument names the category label (not the impl prefix) that
is treated as the comparison baseline for speedup annotations and summary tables.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Temporarily remove the script's own directory from sys.path to avoid
# shadowing stdlib ``argparse`` with any local ``argparse.py`` that may live
# here.
_own_dir = str(Path(__file__).parent)
_patched = _own_dir in sys.path
if _patched:
    sys.path.remove(_own_dir)
import argparse as _argparse_stdlib  # noqa: E402 — must follow path patch
if _patched:
    sys.path.insert(0, _own_dir)

# Matplotlib optional — only required when actually generating a plot.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# --------------------------------------------------------------------------- #
# Default impl → category mapping and colors
# One entry per logical group; a single arch runs at a time so no need for
# arch-specific FlyDSL variants.
# --------------------------------------------------------------------------- #
_DEFAULT_CATEGORIES: dict[str, str] = {
    "triton": "Triton",
    "flydsl": "FlyDSL",
    "hip":    "HIP",
}

_DEFAULT_COLORS: dict[str, str] = {
    "Triton": "#4878CF",   # blue
    "FlyDSL": "#2CA02C",   # green
    "HIP":    "#E89C3A",   # orange
}

_DEFAULT_BASELINE = "Triton"


def category_label(impl_name: str, impl_categories: dict[str, str] | None = None) -> str:
    """Category label for an impl name, e.g. ``"hip"`` -> ``"HIP"``.

    Use this rather than ``str.capitalize()`` to derive a ``baseline_label``:
    the labels are not all title-case (``"HIP"``), so capitalising ``"hip"``
    yields ``"Hip"``, which matches no category and silently blanks every
    speedup column in the summary table and plot.
    """
    cats = impl_categories or _DEFAULT_CATEGORIES
    for prefix, label in cats.items():
        if impl_name.startswith(prefix):
            return label
    return impl_name.capitalize()


# --------------------------------------------------------------------------- #
# Markdown parser
# --------------------------------------------------------------------------- #
def parse_bench_md(path: str, impl_categories: dict[str, str] | None = None) -> list[dict]:
    """Parse a bench Markdown report into a list of per-shape result dicts.

    Each dict has:
        ``shape``       — shape label string
        ``impls``       — {impl_name: {mode: {time_us, tflops, verify, vs_baseline}}}
        ``categories``  — {category_label: best_impl_name} (best TFLOPs per category)
    """
    cats = impl_categories or _DEFAULT_CATEGORIES

    with open(path) as fh:
        content = fh.read()

    sections = re.split(r"(?m)^### ", content)
    results: list[dict] = []

    for section in sections:
        lines = section.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        if not heading:
            continue

        impls: dict = {}
        in_table = False
        header_cols: list[str] = []

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("|") and "---" not in line:
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if not in_table:
                    header_cols = [c.lower().replace(" ", "_") for c in cells]
                    in_table = True
                    continue
                if not header_cols:
                    continue
                row = dict(zip(header_cols, cells))
                impl_name = row.get("impl", "").strip()
                if not impl_name:
                    continue

                def _float(s):
                    try:
                        return float(s.replace("—", "nan").replace(",", ""))
                    except (ValueError, AttributeError):
                        return None

                # ``write_bench_markdown`` emits one column per (metric, mode):
                # ``time_<mode>_us`` / ``tflops_<mode>`` / ``vs_<baseline>_<mode>``
                # (headers are lower-cased above), with no separate ``mode``
                # column. Discover the modes from the time columns. The older
                # flat ``time_us``/``tflops``/``mode`` layout is still accepted so
                # previously-written reports keep parsing.
                per_mode = {
                    m.group(1)
                    for m in (re.match(r"time_(.+)_us$", c) for c in row)
                    if m
                }
                if per_mode:
                    for mode in per_mode:
                        vs = next(
                            (
                                v
                                for k, v in row.items()
                                if k.startswith("vs_") and k.endswith(f"_{mode}")
                            ),
                            "-",
                        )
                        impls.setdefault(impl_name, {})[mode] = {
                            "time_us": _float(row.get(f"time_{mode}_us", "")),
                            "tflops": _float(row.get(f"tflops_{mode}", "")),
                            "verify": row.get("verify", "N/A"),
                            "vs_baseline": vs,
                        }
                else:
                    mode = row.get("mode", "graph").strip()
                    impls.setdefault(impl_name, {})[mode] = {
                        "time_us": _float(row.get("time_us", "")),
                        "tflops": _float(row.get("tflops", "")),
                        "verify": row.get("verify", "N/A"),
                        "vs_baseline": row.get("vs_baseline", "-"),
                    }
            else:
                if in_table and line and not line.startswith("|"):
                    in_table = False

        if not impls:
            continue

        # Map impl names to categories; keep best (highest TFLOPs) per category.
        categories: dict[str, str] = {}
        for impl_name, modes in impls.items():
            best_tf = max((m.get("tflops") or 0.0 for m in modes.values()), default=0.0)
            for prefix, label in cats.items():
                if impl_name.startswith(prefix):
                    prev = categories.get(label)
                    if prev is None:
                        categories[label] = impl_name
                    else:
                        prev_tf = max(
                            (m.get("tflops") or 0.0 for m in impls[prev].values()),
                            default=0.0,
                        )
                        if best_tf > prev_tf:
                            categories[label] = impl_name
                    break

        results.append({"shape": heading, "impls": impls, "categories": categories})

    return results


# --------------------------------------------------------------------------- #
# Bar chart
# --------------------------------------------------------------------------- #
def make_bar_chart(
    results: list[dict],
    out_png: str,
    title: str = "Kernel comparison",
    mode: str = "graph",
    baseline_label: str = _DEFAULT_BASELINE,
    impl_categories: dict[str, str] | None = None,
) -> None:
    """Produce a grouped bar chart of TFLOPs per shape.

    Args:
        results: from :func:`parse_bench_md`.
        out_png: output PNG path.
        title: chart title (problem-specific description supplied by caller).
        mode: timing mode to plot (``"graph"`` or ``"eager"``).
        baseline_label: category label used as the speedup reference.
        impl_categories: prefix → label mapping.
    """
    if not _HAS_MPL:
        print("matplotlib not available; skipping plot.", file=sys.stderr)
        return

    import math

    cats = impl_categories or _DEFAULT_CATEGORIES
    ordered_labels = list(dict.fromkeys(cats.values()))
    colors = {lbl: _DEFAULT_COLORS.get(lbl, "#888888") for lbl in ordered_labels}

    shapes = [r["shape"] for r in results]
    n_shapes = len(shapes)
    n_cats = len(ordered_labels)
    bar_w = 0.8 / max(n_cats, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_shapes * 1.2), 5))

    # Collect baseline TFLOPs per shape for speedup annotation.
    baseline_tflops = []
    for r in results:
        base_impl = r["categories"].get(baseline_label)
        tf = None
        if base_impl and mode in r["impls"].get(base_impl, {}):
            tf = r["impls"][base_impl][mode].get("tflops")
        baseline_tflops.append(tf)

    for ci, label in enumerate(ordered_labels):
        tflops_vals = []
        bar_labels = []
        for ri, r in enumerate(results):
            impl = r["categories"].get(label)
            tf = None
            if impl and mode in r["impls"].get(impl, {}):
                tf = r["impls"][impl][mode].get("tflops")
            tflops_vals.append(tf if tf is not None else float("nan"))

            base = baseline_tflops[ri]
            if tf is not None and label == baseline_label:
                bar_labels.append(f"{tf:.1f}")
            elif tf is not None and base and base > 0:
                bar_labels.append(f"×{tf / base:.2f}")
            else:
                bar_labels.append("")

        if all(math.isnan(v) for v in tflops_vals):
            continue

        xs = [i + (ci - n_cats / 2 + 0.5) * bar_w for i in range(n_shapes)]
        bars = ax.bar(xs, tflops_vals, width=bar_w * 0.9, label=label,
                      color=colors[label], alpha=0.85)
        for bar, lbl in zip(bars, bar_labels):
            h = bar.get_height()
            if lbl and not math.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.01 * max(ax.get_ylim()[1], 1),
                    lbl, ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(range(n_shapes))
    ax.set_xticklabels(shapes, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("TFLOPs/s")
    ax.set_title(f"{title} ({mode})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Plot written to {out_png}")


# --------------------------------------------------------------------------- #
# Summary Markdown table
# --------------------------------------------------------------------------- #
def make_summary_md(
    results: list[dict],
    out_md: str,
    png_path: str | None,
    src_md: str,
    title: str = "Summary",
    mode: str = "graph",
    baseline_label: str = _DEFAULT_BASELINE,
    impl_categories: dict[str, str] | None = None,
) -> None:
    """Write a summary Markdown table (best impl per category per shape)."""
    cats = impl_categories or _DEFAULT_CATEGORIES
    ordered_labels = list(dict.fromkeys(cats.values()))
    non_baseline = [l for l in ordered_labels if l != baseline_label]

    lines: list[str] = [f"# {title} ({mode})\n"]
    if png_path:
        rel = os.path.relpath(png_path, os.path.dirname(out_md))
        lines += [f"![Performance plot]({rel})\n"]
    lines += [f"Source: `{src_md}`\n"]

    header_cols = (
        ["Shape"]
        + [f"{lbl} impl" for lbl in ordered_labels]
        + [f"{lbl} TFLOPs" for lbl in non_baseline]
        + [f"{lbl} vs {baseline_label}" for lbl in non_baseline]
    )
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    for r in results:
        base_impl = r["categories"].get(baseline_label)
        base_tf = None
        if base_impl and mode in r["impls"].get(base_impl, {}):
            base_tf = r["impls"][base_impl][mode].get("tflops")

        row_cells = [r["shape"]]
        for label in ordered_labels:
            row_cells.append(r["categories"].get(label) or "—")
        for label in non_baseline:
            impl = r["categories"].get(label)
            tf = None
            if impl and mode in r["impls"].get(impl, {}):
                tf = r["impls"][impl][mode].get("tflops")
            row_cells.append(f"{tf:.3f}" if tf is not None else "—")
        for label in non_baseline:
            impl = r["categories"].get(label)
            tf = None
            if impl and mode in r["impls"].get(impl, {}):
                tf = r["impls"][impl][mode].get("tflops")
            if tf is not None and base_tf and base_tf > 0:
                row_cells.append(f"×{tf / base_tf:.2f}")
            else:
                row_cells.append("—")

        lines.append("| " + " | ".join(row_cells) + " |")

    with open(out_md, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Summary written to {out_md}")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = _argparse_stdlib.ArgumentParser(
        description="Plot FlyDSL bench results from a Markdown report."
    )
    parser.add_argument("input_md", help="Markdown report produced by a bench script.")
    parser.add_argument("--out-png", default=None, help="Output PNG path.")
    parser.add_argument("--out-md", default=None, help="Output summary Markdown path.")
    parser.add_argument(
        "--title",
        default="Kernel comparison",
        help="Chart title / report heading (default: 'Kernel comparison').",
    )
    parser.add_argument(
        "--mode",
        default="graph",
        choices=("graph", "eager"),
        help="Timing mode column to plot (default: graph).",
    )
    parser.add_argument(
        "--baseline",
        default=_DEFAULT_BASELINE,
        metavar="LABEL",
        help=(
            f"Category label used as the speedup baseline "
            f"(default: '{_DEFAULT_BASELINE}'). Must match one of the category labels."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        metavar="PREFIX=LABEL",
        help=(
            "Override impl-prefix→category mapping, e.g. "
            "'triton=Triton' 'flydsl=FlyDSL' 'hip=HIP'. "
            "Uses built-in defaults when not given."
        ),
    )
    args = parser.parse_args(argv)

    src = args.input_md
    stem = Path(src).stem
    out_dir = str(Path(src).parent)
    out_png = args.out_png or os.path.join(out_dir, f"{stem}-plot.png")
    out_md  = args.out_md  or os.path.join(out_dir, f"{stem}-summary.md")

    impl_categories = None
    if args.categories:
        impl_categories = {}
        for item in args.categories:
            if "=" in item:
                k, v = item.split("=", 1)
                impl_categories[k.strip()] = v.strip()

    results = parse_bench_md(src, impl_categories)
    if not results:
        print("No shape sections found in the Markdown file.", file=sys.stderr)
        sys.exit(1)

    make_bar_chart(results, out_png, title=args.title, mode=args.mode,
                   baseline_label=args.baseline, impl_categories=impl_categories)
    make_summary_md(results, out_md, out_png, src, title=args.title, mode=args.mode,
                    baseline_label=args.baseline, impl_categories=impl_categories)


if __name__ == "__main__":
    main()
