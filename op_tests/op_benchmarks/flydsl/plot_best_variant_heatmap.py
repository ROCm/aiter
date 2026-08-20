#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Best-variant map over the (H, N) grid, for K5 or for fused K5+K6.

For each measured (H, N) shape, colour the point by which variant
(bv16 / bv32 / bv64 / bv64w8) is FASTEST (min graph-time over the equal/ragged
serving distributions). Overlays the H*N grid-size threshold curves that motivate
the selection rule shared by both kernels:

    H*N <= 32  -> bv16 ,  <= 80 -> bv32 ,  > 80 -> bv64w8

so the reader can see the best-variant regions fall into H*N bands (points sit on
constant-H*N hyperbolas N = C/H).

Usage:
    PYTHONPATH=. python3 op_tests/op_benchmarks/flydsl/plot_best_variant_heatmap.py \
        op_tests/dump_data/<sweep>.md [--kernel k5|fused] [-o out.png]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from utils.plot_perf import parse_bench_md

FUSED_VARIANTS = ("bv16", "bv32", "bv64", "bv64w8")
_COMMON_SEQDISTS = frozenset({"equal", "ragged"})

# Per-shape impl-column prefix for each kernel's variant sweep. A k5 sweep names
# its columns ``flydsl:bv32``; a fused sweep ``K5K6_flydsl_fused:bv32``.
_IMPL_PREFIX = {"k5": "flydsl:", "fused": "K5K6_flydsl_fused:"}
_KERNEL_TITLE = {"k5": "GDN K5", "fused": "GDN K5 + K6 fused"}

# The two H*N cut points of the shared selection rule (chunk_gated_delta_h_gfx942
# ._HN_BV32 / ._HN_BV64W8). Both sit in a measurement gap, so the drawn curve is
# a representative of an interval, not a knife edge.
_HN_BV32 = 32
_HN_BV64W8 = 80

# Categorical palette (validated brand-neutral slots): one hue per variant, in
# fixed order. Distinct in hue AND ordered by tile size so the map reads as a
# progression. Marker doubles the encoding for CVD/print safety.
VAR_STYLE = {
    "bv16": ("#2a78d6", "o"),  # blue
    "bv32": ("#28a745", "s"),  # green
    "bv64": ("#8a5cf6", "D"),  # purple
    "bv64w8": ("#eb6834", "^"),  # orange
}
INK = "#0b0b0b"
SEC = "#52514e"
MUT = "#8a887f"
GRID = "#e5e4df"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep", help="path to a k5 or fused sweep .md")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--mode", default="graph", choices=["graph", "eager"])
    ap.add_argument(
        "--kernel",
        default="fused",
        choices=["k5", "fused"],
        help="which kernel's per-variant columns to read (default: fused)",
    )
    args = ap.parse_args()
    prefix = _IMPL_PREFIX[args.kernel]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        return

    res = parse_bench_md(args.sweep)
    # (H, N) -> {variant: best time over common seqdists}
    cell: dict = defaultdict(dict)
    for r in res:
        h = r["shape"]
        mh = re.search(r"H=(\d+)", h)
        mn = re.search(r"N=(\d+)", h)
        if not (mh and mn):
            continue
        H, N = int(mh.group(1)), int(mn.group(1))
        seq = "equal"
        ms = re.search(r"seqs=(\w+)", h)
        if ms:
            seq = ms.group(1)
        if seq not in _COMMON_SEQDISTS:
            continue
        for v in FUSED_VARIANTS:
            md = r["impls"].get(f"{prefix}{v}", {}).get(args.mode)
            if md and md.get("time_us"):
                try:
                    t = float(md["time_us"])
                except ValueError:
                    continue
                cur = cell[(H, N)].get(v)
                cell[(H, N)][v] = t if cur is None else min(cur, t)

    pts = []
    for (H, N), tv in cell.items():
        if not tv:
            continue
        best = min(tv, key=tv.get)
        pts.append((H, N, best))
    if not pts:
        print(
            f"no {args.kernel} per-variant timing found "
            f"(looked for '{prefix}<variant>' columns)."
        )
        return

    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    # H*N threshold hyperbolas: N = C / H for each cut point of the rule.
    import numpy as np

    Hs = sorted({H for H, _, _ in pts})
    hx = np.linspace(min(Hs) * 0.8, max(Hs) * 1.2, 200)
    for C, lab, yfrac in (
        (_HN_BV32, f"H·N = {_HN_BV32}  (bv16 | bv32)", 0.60),
        (_HN_BV64W8, f"H·N = {_HN_BV64W8}  (bv32 | bv64w8)", 1.13),
    ):
        ax.plot(C / hx, hx, color=SEC, lw=1.2, ls=(0, (5, 3)), zorder=1)
        # label placed along each curve at a distinct height to avoid overlap
        yl = max(Hs) * yfrac
        ax.text(C / yl * 1.05, yl, lab, color=SEC, fontsize=8.5, ha="left", va="center")

    # jitter overlapping (H,N) points a touch (log space) so duplicates show
    seen: dict = defaultdict(int)
    for H, N, best in pts:
        k = (H, N)
        seen[k] += 1
        j = seen[k] - 1
        jitter = 1.0 + (0.0 if j == 0 else (1 if j % 2 else -1) * ((j + 1) // 2) * 0.03)
        color, marker = VAR_STYLE[best]
        ax.scatter(
            [N * jitter],
            [H],
            s=95,
            c=color,
            marker=marker,
            edgecolors="#fcfcfb",
            linewidths=1.2,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "N  (sequence count, log scale)", fontsize=12, color=INK, fontweight="bold"
    )
    ax.set_ylabel(
        "H  (head count, log scale)", fontsize=12, color=INK, fontweight="bold"
    )
    ax.set_title(
        f"{_KERNEL_TITLE[args.kernel]}: fastest variant over (H, N)",
        fontsize=13,
        color=INK,
        fontweight="bold",
        pad=12,
    )

    # ticks at the actual measured values
    Ns = sorted({N for _, N, _ in pts})
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_yticks(Hs)
    ax.set_yticklabels([str(h) for h in Hs])
    ax.tick_params(colors=SEC)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUT)
    ax.grid(True, which="major", color=GRID, lw=0.7, zorder=0)

    # legend: colour+marker = variant (identity never colour-alone)
    handles = [
        mlines.Line2D(
            [],
            [],
            marker=VAR_STYLE[v][1],
            ls="",
            ms=9,
            mfc=VAR_STYLE[v][0],
            mec="#fcfcfb",
            label=v,
        )
        for v in FUSED_VARIANTS
        if any(b == v for _, _, b in pts)
    ]
    ax.legend(
        handles=handles,
        title="fastest variant",
        loc="upper right",
        frameon=False,
        fontsize=10,
    )

    out = args.out or str(
        Path(args.sweep).with_name(Path(args.sweep).stem + "-best-variant-map.png")
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
