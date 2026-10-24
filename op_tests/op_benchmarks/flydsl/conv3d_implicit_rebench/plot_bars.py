#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bar chart of every measured case of one model/resolution, from bench_all.py results.

Three panels share the case axis, rows sorted by ms/pass (same order as the
report's tables):
  1. per-call kernel time -- burst bar, sustained as a tick;
  2. kernel ms per pass (kernel us x calls);
  3. MFMA% bar with HBM% as a marker (both against theoretical peak).
Bars are coloured by bound. The SVG is written by hand like plot_roofline.py,
so the output is byte-stable; axis text is English so the figure renders
without a CJK font.

    python3 plot_bars.py                      # data/ -> figures/wan21_368x544_bars.svg
    python3 plot_bars.py --model qwenimage --res 1024x1024
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load

C_COM, C_MEM = "#2f6f9f", "#b5651d"
C_SUS, C_HBM = "#222222", "#c0392b"
C_AXIS, C_GRID, C_TEXT, C_MUTE = "#444444", "#e6e6e6", "#333333", "#777777"

ROW_H, BAR_H = 22, 13
PAD_T, PAD_B, PAD_L = 58, 112, 10
PANEL_W, GAP = 250, 46
COL_GAP = 12


def _triple(a, b, c):
    return f"{a}" if a == b == c else f"{a},{b},{c}"


def _kernel(k):
    s = f"{k[6]}×{k[7]}×{k[8]}"
    if not k[15] == k[16] == k[17] == 1:
        s += f" d{_triple(*k[15:18])}"
    if k[18] != 1:
        s += f" g{k[18]}"
    return s


# Label columns in front of the bars: (header, width px, anchor, text of one row).
# key = conv_kernels.TUNED_KEY_COLUMNS:
#   N C D H W K kT kH kW sd sh sw pd ph pw dd dh dw groups bias
LABEL_COLS = (
    ("case", 118, "start", lambda r: r.case),
    ("Cin→Cout", 56, "end", lambda r: f"{r.key[1]}→{r.key[5]}"),
    ("kernel", 44, "end", lambda r: _kernel(r.key)),
    ("stride", 40, "end", lambda r: _triple(*r.key[9:12])),
    ("pad", 36, "end", lambda r: _triple(*r.key[12:15])),
    (
        "input N×D×H×W",
        88,
        "end",
        lambda r: "×".join(str(v) for v in r.key[0:1] + r.key[2:5]),
    ),
    ("GEMM M×N×K", 104, "end", lambda r: f"{r.M}×{r.N}×{r.K}"),
    ("calls", 32, "end", lambda r: f"×{r.calls}"),
)
LABEL_W = PAD_L + sum(c[1] for c in LABEL_COLS) + COL_GAP * len(LABEL_COLS) + 14


def nice_max(v):
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * exp >= v:
            return m * exp
    return 10 * exp


def ticks(vmax, n=5):
    step = vmax / n
    return [step * i for i in range(n + 1)]


def fmt(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default=os.path.join(here, "data"))
    ap.add_argument("--model", default="wan21")
    ap.add_argument("--res", default="368x544")
    ap.add_argument("--title", default=None)
    ap.add_argument(
        "--out", default=None, help="default: figures/<model>_<res>_bars.svg"
    )
    args = ap.parse_args()
    if args.out is None:
        args.out = os.path.join(here, "figures", f"{args.model}_{args.res}_bars.svg")

    meta, df = load(args.data)
    s = df[(df.model == args.model) & (df.res == args.res)].sort_values(
        "ms_pass_k", ascending=False
    )
    rows = list(s.itertuples())
    n = len(rows)
    ph = n * ROW_H
    W = LABEL_W + 3 * PANEL_W + 2 * GAP + 30
    H = PAD_T + ph + PAD_B
    x0 = [LABEL_W + i * (PANEL_W + GAP) for i in range(3)]

    us_max = nice_max(max(max(r.kernel_us, r.kernel_us_sus) for r in rows) * 1.12)
    ms_max = nice_max(max(r.ms_pass_k for r in rows) * 1.15)
    pct_max = nice_max(max(max(r.mfma_k, r.hbm_k) for r in rows) * 1.12)
    scales = (us_max, ms_max, pct_max)

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">'
        ),
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    model = {"wan21": "Wan2.1 VAE", "qwenimage": "Qwen-Image VAE"}.get(
        args.model, args.model
    )
    frames = " @81f" if args.model == "wan21" else ""
    title = args.title or (
        f"{model} {args.res}{frames}: measured conv3d kernels "
        f"({meta.get('gfx')}, HIP {meta.get('hip_visible_devices')} / PCI {meta.get('pci')})"
    )
    out.append(
        f'<text x="{PAD_L}" y="22" font-size="14" font-weight="bold" fill="{C_TEXT}">{esc(title)}</text>'
    )
    heads = (
        "Kernel time per call (us)",
        "Kernel time per pass (ms)",
        "Utilisation vs theoretical peak (%)",
    )
    for i, h in enumerate(heads):
        out.append(
            f'<text x="{x0[i]}" y="{PAD_T - 22}" font-size="12" fill="{C_TEXT}">{h}</text>'
        )

    # label column x anchors: left edge for "start", right edge for "end"
    col_x, x = [], PAD_L
    for _, width, anchor, _ in LABEL_COLS:
        col_x.append(x if anchor == "start" else x + width)
        x += width + COL_GAP
    for (head, _, anchor, _), cx in zip(LABEL_COLS, col_x):
        out.append(
            f'<text x="{cx}" y="{PAD_T - 10}" text-anchor="{anchor}" font-size="10.5" '
            f'font-weight="bold" fill="{C_MUTE}">{esc(head)}</text>'
        )
    out.append(
        f'<line x1="{PAD_L}" y1="{PAD_T - 4}" x2="{LABEL_W - 14}" y2="{PAD_T - 4}" stroke="{C_GRID}"/>'
    )

    for j in range(1, n, 2):
        out.append(
            f'<rect x="4" y="{PAD_T + j * ROW_H}" width="{W - 8}" height="{ROW_H}" fill="#f7f7f7"/>'
        )

    # grid + tick labels
    for i, vmax in enumerate(scales):
        for t in ticks(vmax):
            x = x0[i] + t / vmax * PANEL_W
            out.append(
                f'<line x1="{x:.1f}" y1="{PAD_T - 6}" x2="{x:.1f}" y2="{PAD_T + ph}" stroke="{C_GRID}"/>'
            )
            out.append(
                f'<text x="{x:.1f}" y="{PAD_T + ph + 15}" text-anchor="middle" font-size="10.5" '
                f'fill="{C_MUTE}">{t:g}</text>'
            )
        out.append(
            f'<line x1="{x0[i]}" y1="{PAD_T - 6}" x2="{x0[i]}" y2="{PAD_T + ph}" stroke="{C_AXIS}"/>'
        )

    for j, r in enumerate(rows):
        yc = PAD_T + j * ROW_H + ROW_H / 2
        yb = yc - BAR_H / 2
        col = C_COM if r.bound == "compute" else C_MEM
        for ci, ((_, _, anchor, text), cx) in enumerate(zip(LABEL_COLS, col_x)):
            fill = C_TEXT if ci in (0, 6) else C_MUTE
            out.append(
                f'<text x="{cx}" y="{yc + 4:.1f}" text-anchor="{anchor}" font-size="10.5" '
                f'fill="{fill}">{esc(text(r))}</text>'
            )
        # panel 1: burst bar + sustained tick
        w = r.kernel_us / us_max * PANEL_W
        xs = x0[0] + r.kernel_us_sus / us_max * PANEL_W
        out.append(
            f'<rect x="{x0[0]}" y="{yb:.1f}" width="{w:.1f}" height="{BAR_H}" fill="{col}"/>'
        )
        out.append(
            f'<line x1="{xs:.1f}" y1="{yb - 2:.1f}" x2="{xs:.1f}" y2="{yb + BAR_H + 2:.1f}" '
            f'stroke="{C_SUS}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{max(x0[0] + w, xs) + 5:.1f}" y="{yc + 4:.1f}" font-size="10.5" '
            f'fill="{C_TEXT}">{fmt(r.kernel_us)}</text>'
        )
        # panel 2: ms/pass
        w = r.ms_pass_k / ms_max * PANEL_W
        out.append(
            f'<rect x="{x0[1]}" y="{yb:.1f}" width="{w:.1f}" height="{BAR_H}" fill="{col}"/>'
        )
        out.append(
            f'<text x="{x0[1] + w + 5:.1f}" y="{yc + 4:.1f}" font-size="10.5" fill="{C_TEXT}">'
            f'{fmt(r.ms_pass_k)} <tspan fill="{C_MUTE}">({r.ms_pass_k / s.ms_pass_k.sum() * 100:.1f}%)</tspan></text>'
        )
        # panel 3: MFMA% bar + HBM% diamond
        w = r.mfma_k / pct_max * PANEL_W
        xh = x0[2] + r.hbm_k / pct_max * PANEL_W
        out.append(
            f'<rect x="{x0[2]}" y="{yb:.1f}" width="{w:.1f}" height="{BAR_H}" fill="{col}"/>'
        )
        d = 4.5
        out.append(
            f'<path d="M{xh:.1f},{yc - d:.1f} L{xh + d:.1f},{yc:.1f} L{xh:.1f},{yc + d:.1f} '
            f'L{xh - d:.1f},{yc:.1f} Z" fill="{C_HBM}" stroke="#ffffff" stroke-width="0.8"/>'
        )
        out.append(
            f'<text x="{max(x0[2] + w, xh + d) + 5:.1f}" y="{yc + 4:.1f}" font-size="10.5" '
            f'fill="{C_TEXT}">{r.mfma_k:.1f} / <tspan fill="{C_HBM}">{r.hbm_k:.1f}</tspan></text>'
        )

    # legend
    ly = PAD_T + ph + 40
    items = [
        ("rect", C_COM, "compute-bound (AI >= 312.5)"),
        ("rect", C_MEM, "memory-bound"),
        ("tick", C_SUS, "sustained (after 1 s of the same conv)"),
        ("diamond", C_HBM, "HBM% (label: MFMA% / HBM%)"),
    ]
    lx = LABEL_W
    for kind, c, label in items:
        if kind == "rect":
            out.append(
                f'<rect x="{lx}" y="{ly - 9}" width="14" height="11" fill="{c}"/>'
            )
        elif kind == "tick":
            out.append(
                f'<line x1="{lx + 7}" y1="{ly - 11}" x2="{lx + 7}" y2="{ly + 3}" stroke="{c}" stroke-width="2"/>'
            )
        else:
            out.append(
                f'<path d="M{lx + 7},{ly - 9} L{lx + 12},{ly - 4} L{lx + 7},{ly + 1} L{lx + 2},{ly - 4} Z" fill="{c}"/>'
            )
        out.append(
            f'<text x="{lx + 20}" y="{ly}" font-size="11" fill="{C_TEXT}">{esc(label)}</text>'
        )
        lx += 20 + 7 * len(label) + 28
    notes = (
        "Bars: kernel-only, weight cache warm, burst (1 s idle before each window); median of 3 windows x 101 calls.",
        (
            "Rows sorted by ms/pass; xN = calls per pass; % in panel 2 = share of this resolution's total. "
            "Peaks: 2500 TFLOPS / 8000 GB/s."
        ),
        "Implicit GEMM: M = N·Do·Ho·Wo, N = Cout / groups, K = (Cin / groups)·kT·kH·kW.",
        "Stride / pad: one value when equal in D, H, W, else D,H,W. Dilation and groups are 1 unless shown after the kernel.",
    )
    for i, t in enumerate(notes):
        out.append(
            f'<text x="{LABEL_W}" y="{ly + 18 + i * 14}" font-size="10.5" fill="{C_MUTE}">{esc(t)}</text>'
        )
    out.append("</svg>")
    with open(args.out, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote {args.out}: {n} rows, total {s.ms_pass_k.sum():.1f} ms/pass")


if __name__ == "__main__":
    main()
