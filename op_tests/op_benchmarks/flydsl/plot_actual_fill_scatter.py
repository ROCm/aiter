#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scatter of fused-K5+K6 speedup vs the ACTUAL grid fill of the best instance.

Companion to ``utils/plot_perf.make_fill_scatter``, which plots against the
*shape-intrinsic* fill @ BV=64 (``2*N*H/CU``, independent of the chosen tile).
This variant instead uses the fill of the variant that is actually fastest for
each shape:

    fill = ceil(V / BV_best) * N * H / CU_count

where ``BV_best`` is the BV of the best fused variant for that shape. Because
smaller tiles emit more CTAs (bv16 = 4x the CTAs of bv64), shapes whose best
instance is bv16/bv32 shift RIGHT relative to the fill64 plot. The point is to
check whether the actual-fill axis separates fusion wins from losses with a
simple threshold (the motivating hypothesis: only very small fills need the
un-fused path).

Usage:
    PYTHONPATH=. python3 op_tests/op_benchmarks/flydsl/plot_actual_fill_scatter.py \
        op_tests/dump_data/<sweep>.md [-o out.png] [--mode graph] [--cu 304]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from utils.plot_perf import _bv_of_impl, parse_bench_md

# Match the existing fill scatter's validated palette/markers for consistency.
WIN = "#2a78d6"
LOSS = "#eb6834"
INK = "#0b0b0b"
SEC = "#52514e"
MUT = "#8a887f"
PAT_MARKER = {
    "equal": "o",
    "ragged": "s",
    "bimodal": "D",
    "skew": "^",
    "skew_last": "v",
}
_FUSED_PREFIX = "K5K6_flydsl_fused"


def _variant_speedup(impls: dict, mode: str, tag: str):
    """Speedup of one specific fused variant (e.g. ``bv64w8``), or None."""
    md = impls.get(_FUSED_PREFIX + ":" + tag, {}).get(mode)
    if not md:
        return None
    m = re.search(r"([\d.]+)", (md.get("vs_baseline", "") or "").replace("×", "x"))
    return float(m.group(1)) if m else None


def _best_fused(impls: dict, mode: str):
    """(bv, speedup) of the fastest fused variant for one shape, or (None, None)."""
    best_sp = None
    best_bv = None
    for name, modes in impls.items():
        if not name.startswith(_FUSED_PREFIX):
            continue
        md = modes.get(mode)
        if not md:
            continue
        vs = md.get("vs_baseline", "-")
        m = re.search(r"([\d.]+)", vs.replace("×", "x")) if vs else None
        if not m:
            continue
        sp = float(m.group(1))
        if best_sp is None or sp > best_sp:
            best_sp = sp
            best_bv = _bv_of_impl(name)
    return best_bv, best_sp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep", help="path to a fused-sweep .md")
    ap.add_argument("-o", "--out", default=None, help="output png path")
    ap.add_argument("--mode", default="graph", choices=["graph", "eager"])
    ap.add_argument("--cu", type=int, default=304, help="CU count (gfx942=304)")
    ap.add_argument(
        "--fill-bv",
        default="selected",
        choices=["selected", "best"],
        help="Which BV sets the x-axis fill. 'selected' (default) = the BV the "
        "runtime routing actually picks (select_fused_variant), matching "
        "should_use_fused_gfx942. 'best' = the empirically fastest variant's BV.",
    )
    args = ap.parse_args()

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        return

    # For the 'selected' axis: the BV the runtime routing actually uses.
    _select_fused_variant = None
    _bv_of_variant = None
    if args.fill_bv == "selected":
        from aiter.ops.flydsl.kernels.chunk_gated_delta_h_gfx942 import (
            select_fused_variant as _select_fused_variant,
        )
        from aiter.ops.flydsl.kernels.k5_variants import (
            _bv_of_variant as _bv_of_variant,
        )

    results = parse_bench_md(args.sweep)
    pts = []
    for r in results:
        head = r["shape"]
        mh = re.search(r"H=(\d+)", head)
        mn = re.search(r"N=(\d+)", head)
        mv = re.search(r"V=(\d+)", head)
        if not (mh and mn):
            continue
        H = int(mh.group(1))
        N = int(mn.group(1))
        V = int(mv.group(1)) if mv else 128
        mp = re.search(r"seqs=(\w+)", head)
        pat = mp.group(1) if mp else "equal"
        best_bv, best_sp = _best_fused(r["impls"], args.mode)
        if best_bv is None or best_sp is None:
            continue
        if args.fill_bv == "selected":
            # What the heuristic actually delivers: the variant it PICKS sets
            # both the fill (x) and the speedup (y).
            tag = _select_fused_variant(H=H, N=N, V=V)
            if tag is not None:
                bv = _bv_of_variant(tag)
                sp = _variant_speedup(r["impls"], args.mode, tag)
                if sp is None:  # variant not in this sweep -> fall back
                    sp = best_sp
            else:
                bv, sp = 64, best_sp
        else:
            # Best achievable: the empirically-fastest variant sets x and y.
            bv, sp = best_bv, best_sp
        # ACTUAL fill: more CTAs for smaller BV.
        fill = (-(-V // bv)) * N * H / args.cu
        pts.append({"H": H, "N": N, "sp": sp, "fill": fill, "pat": pat, "bv": bv})

    if not pts:
        print("no fused points found.")
        return

    win = [p for p in pts if p["sp"] >= 1.0]
    loss = [p for p in pts if p["sp"] < 1.0]

    fig, ax = plt.subplots(figsize=(9.8, 5.9), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhspan(0.0, 1.0, color=LOSS, alpha=0.05, zorder=0)
    ax.axhline(1.0, color=SEC, lw=1.5, ls=(0, (5, 4)), zorder=2)

    seen: dict = defaultdict(int)
    for p in pts:
        key = (round(p["fill"], 3), p["sp"] < 1.0)
        seen[key] += 1
        j = seen[key] - 1
        x = (
            p["fill"]
            if j == 0
            else p["fill"] * (1.0 + ((1 if j % 2 else -1) * ((j + 1) // 2) * 0.02))
        )
        color = WIN if p["sp"] >= 1.0 else LOSS
        marker = PAT_MARKER.get(p["pat"], "o")
        ax.scatter(
            [x],
            [p["sp"]],
            s=52,
            c=color,
            marker=marker,
            edgecolors="#fcfcfb",
            linewidths=1.1,
            zorder=3,
        )

    allfill = [p["fill"] for p in pts]
    allsp = [p["sp"] for p in pts]
    ax.set_xscale("log")
    xlo, xhi = min(allfill) * 0.7, max(allfill) * 1.3
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(min(0.4, min(allsp) - 0.05), max(allsp) + 0.15)

    # Decision threshold: fuse iff fill >= 0.45 (should_use_fused_gfx942).
    thr = 0.45
    if xlo < thr < xhi:
        ax.axvline(thr, color=SEC, lw=1, ls=":", zorder=1)
        ax.text(
            thr * 1.03,
            max(allsp) + 0.02,
            "fill = 0.45",
            fontsize=9.5,
            color=SEC,
            va="top",
        )

    if args.fill_bv == "selected":
        variant_word = "heuristic-selected"
    else:
        variant_word = "best"
    ax.set_xlabel(
        f"grid fill of {variant_word} variant  " f"(⌈V/BV⌉·N·H / {args.cu}; log scale)",
        fontsize=11.5,
        color=INK,
        fontweight="bold",
    )
    ax.set_ylabel(
        "fused speedup vs baseline", fontsize=12, color=INK, fontweight="bold"
    )
    ax.set_title(
        f"GDN K5 + K6 fused ({variant_word} variant)",
        fontsize=13,
        color=INK,
        fontweight="bold",
        pad=12,
    )

    outcome_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            mfc=WIN,
            mec="#fcfcfb",
            label=f"fusion faster ({len(win)})",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=8,
            mfc=LOSS,
            mec="#fcfcfb",
            label=f"fusion slower ({len(loss)})",
        ),
    ]
    pats_present = [p for p in PAT_MARKER if any(x["pat"] == p for x in pts)]
    pat_handles = [
        mlines.Line2D(
            [], [], marker=PAT_MARKER[p], ls="", ms=8, mfc=SEC, mec="#fcfcfb", label=p
        )
        for p in pats_present
    ]
    leg1 = ax.legend(
        handles=outcome_handles,
        title="outcome (colour)",
        loc="upper left",
        frameon=False,
        fontsize=10,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=pat_handles,
        title="seq pattern (shape)",
        loc="lower right",
        frameon=False,
        fontsize=9.5,
    )

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUT)
    ax.tick_params(colors=SEC)
    ax.grid(True, which="major", color="#e5e4df", lw=0.8, zorder=0)

    suffix = (
        "-actual-fill-scatter.png"
        if args.fill_bv == "best"
        else "-selected-fill-scatter.png"
    )
    out = args.out or str(Path(args.sweep).with_name(Path(args.sweep).stem + suffix))
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    # Also print the loss points so the threshold claim is auditable.
    if loss:
        print("\nfusion-slower points (shape actual-fill, bv_best, speedup):")
        for p in sorted(loss, key=lambda x: x["fill"]):
            print(
                f"  H={p['H']:<3} N={p['N']:<2} {p['pat']:<9} "
                f"fill={p['fill']:.3f} bv={p['bv']:<3} sp={p['sp']:.2f}"
            )


if __name__ == "__main__":
    main()
