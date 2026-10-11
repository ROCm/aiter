#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Render the conv3d roofline figures for conv3d_implicit_roofline_headroom.md.

Reads the two tuned conv3d CSVs and writes self-contained SVGs into figures/
next to this script. No GPU, no benchmark run, no plotting library -- the SVG is emitted by
hand so the output is byte-stable and reviewable in a diff.

Axis labels are deliberately English: the SVG has to render wherever the doc is
read, and a CJK glyph missing from the viewer's font stack would leave holes.
The Chinese captions live in the markdown instead.

Usage::

    python plot_roofline.py [--configs <repo>/aiter/configs/model_configs]
"""

import argparse
import math
import os

import pandas as pd

# gfx950 / MI355X, 256 CU. Theory first, then what the same card actually
# reaches (torch.mm 8192^3 and a device copy); both ridge points are ~312.
PEAK_TF, PEAK_BW = 2500.0, 8000.0
ACH_TF, ACH_BW = 1468.0, 4700.0
RIDGE = PEAK_TF * 1000 / PEAK_BW

# Flat palette, no gradients. Wan and Qwen only; resolution is not encoded
# because the two resolutions of one model land on the same trend line.
C_WAN, C_QWEN = "#2f6f9f", "#b5651d"
C_AXIS, C_GRID, C_TEXT, C_MUTE = "#444444", "#e6e6e6", "#333333", "#777777"

# How often one pass runs each shape, keyed by the CSV's own problem columns.
# Source: the sweep builders in op_tests/test_flydsl_conv_implicit.py
# (wan_vae_conv3d / wan_vae_aux / wan_vae_decode / qwen_vae_conv2d) at
# Wan 81 frames and the two default resolutions per model. Inlined so this
# script needs nothing but the CSVs.
CALLS = {
    (1, 3, 6, 370, 546, 96, 3, 1, 0): ("conv_in", 20, "368x544"),
    (1, 96, 6, 370, 546, 96, 3, 1, 0): ("down_0_1", 80, "368x544"),
    (1, 96, 1, 369, 545, 96, 1, 2, 0): ("resample_96_t1", 1, "368x544"),
    (4, 96, 1, 369, 545, 96, 1, 2, 0): ("resample_96", 20, "368x544"),
    (1, 96, 6, 186, 274, 192, 3, 1, 0): ("down_3_in", 20, "368x544"),
    (1, 192, 6, 186, 274, 192, 3, 1, 0): ("down_3_4", 60, "368x544"),
    (1, 192, 1, 185, 273, 192, 1, 2, 0): ("resample_192_t1", 1, "368x544"),
    (4, 192, 1, 185, 273, 192, 1, 2, 0): ("resample_192", 20, "368x544"),
    (1, 192, 4, 94, 138, 384, 3, 1, 0): ("down_6_in", 20, "368x544"),
    (1, 384, 4, 94, 138, 384, 3, 1, 0): ("down_6_7", 60, "368x544"),
    (1, 384, 1, 93, 137, 384, 1, 2, 0): ("resample_384_t1", 1, "368x544"),
    (2, 384, 1, 93, 137, 384, 1, 2, 0): ("resample_384", 20, "368x544"),
    (1, 384, 3, 48, 70, 384, 3, 1, 0): ("down_9_10_mid", 160, "368x544"),
    (1, 384, 3, 48, 70, 32, 3, 1, 0): ("conv_out", 20, "368x544"),
    (1, 16, 3, 48, 70, 384, 3, 1, 0): ("wdec_conv_in", 20, "368x544"),
    (1, 384, 1, 92, 136, 192, 1, 1, 1): ("wdec_up_384_L2_t1", 1, "368x544"),
    (2, 384, 1, 92, 136, 192, 1, 1, 1): ("wdec_up_384_L2", 20, "368x544"),
    (1, 384, 1, 184, 272, 192, 1, 1, 1): ("wdec_up_384_L1_t1", 1, "368x544"),
    (4, 384, 1, 184, 272, 192, 1, 1, 1): ("wdec_up_384_L1", 20, "368x544"),
    (1, 192, 1, 368, 544, 96, 1, 1, 1): ("wdec_up_192_L0_t1", 1, "368x544"),
    (4, 192, 1, 368, 544, 96, 1, 1, 1): ("wdec_up_192_L0", 20, "368x544"),
    (1, 96, 6, 370, 546, 3, 3, 1, 0): ("wdec_conv_out", 20, "368x544"),
    (1, 3, 6, 482, 834, 96, 3, 1, 0): ("conv_in", 20, "480x832"),
    (1, 96, 6, 482, 834, 96, 3, 1, 0): ("down_0_1", 80, "480x832"),
    (1, 96, 1, 481, 833, 96, 1, 2, 0): ("resample_96_t1", 1, "480x832"),
    (4, 96, 1, 481, 833, 96, 1, 2, 0): ("resample_96", 20, "480x832"),
    (1, 96, 6, 242, 418, 192, 3, 1, 0): ("down_3_in", 20, "480x832"),
    (1, 192, 6, 242, 418, 192, 3, 1, 0): ("down_3_4", 60, "480x832"),
    (1, 192, 1, 241, 417, 192, 1, 2, 0): ("resample_192_t1", 1, "480x832"),
    (4, 192, 1, 241, 417, 192, 1, 2, 0): ("resample_192", 20, "480x832"),
    (1, 192, 4, 122, 210, 384, 3, 1, 0): ("down_6_in", 20, "480x832"),
    (1, 384, 4, 122, 210, 384, 3, 1, 0): ("down_6_7", 60, "480x832"),
    (1, 384, 1, 121, 209, 384, 1, 2, 0): ("resample_384_t1", 1, "480x832"),
    (2, 384, 1, 121, 209, 384, 1, 2, 0): ("resample_384", 20, "480x832"),
    (1, 384, 3, 62, 106, 384, 3, 1, 0): ("down_9_10_mid", 160, "480x832"),
    (1, 384, 3, 62, 106, 32, 3, 1, 0): ("conv_out", 20, "480x832"),
    (1, 16, 3, 62, 106, 384, 3, 1, 0): ("wdec_conv_in", 20, "480x832"),
    (1, 384, 1, 120, 208, 192, 1, 1, 1): ("wdec_up_384_L2_t1", 1, "480x832"),
    (2, 384, 1, 120, 208, 192, 1, 1, 1): ("wdec_up_384_L2", 20, "480x832"),
    (1, 384, 1, 240, 416, 192, 1, 1, 1): ("wdec_up_384_L1_t1", 1, "480x832"),
    (4, 384, 1, 240, 416, 192, 1, 1, 1): ("wdec_up_384_L1", 20, "480x832"),
    (1, 192, 1, 480, 832, 96, 1, 1, 1): ("wdec_up_192_L0_t1", 1, "480x832"),
    (4, 192, 1, 480, 832, 96, 1, 1, 1): ("wdec_up_192_L0", 20, "480x832"),
    (1, 96, 6, 482, 834, 3, 3, 1, 0): ("wdec_conv_out", 20, "480x832"),
    (1, 3, 1, 1024, 1024, 96, 1, 1, 1): ("enc_conv_in", 1, "1024x1024"),
    (1, 96, 1, 1024, 1024, 96, 1, 1, 1): ("res_96_L0", 10, "1024x1024"),
    (1, 96, 1, 1025, 1025, 96, 1, 2, 0): ("enc_down_96", 1, "1024x1024"),
    (1, 96, 1, 512, 512, 192, 1, 1, 1): ("down_96_192", 1, "1024x1024"),
    (1, 192, 1, 512, 512, 192, 1, 1, 1): ("res_192_L1", 9, "1024x1024"),
    (1, 192, 1, 513, 513, 192, 1, 2, 0): ("enc_down_192", 1, "1024x1024"),
    (1, 192, 1, 256, 256, 384, 1, 1, 1): ("down_192_384", 2, "1024x1024"),
    (1, 384, 1, 256, 256, 384, 1, 1, 1): ("res_384_L2", 8, "1024x1024"),
    (1, 384, 1, 257, 257, 384, 1, 2, 0): ("enc_down_384", 1, "1024x1024"),
    (1, 384, 1, 128, 128, 384, 1, 1, 1): ("res_384_L3", 18, "1024x1024"),
    (1, 384, 1, 128, 128, 32, 1, 1, 1): ("enc_conv_out", 1, "1024x1024"),
    (1, 16, 1, 128, 128, 384, 1, 1, 1): ("dec_conv_in", 1, "1024x1024"),
    (1, 384, 1, 256, 256, 192, 1, 1, 1): ("dec_up_384_L2", 1, "1024x1024"),
    (1, 384, 1, 512, 512, 192, 1, 1, 1): ("dec_up_384_L1", 1, "1024x1024"),
    (1, 192, 1, 1024, 1024, 96, 1, 1, 1): ("dec_up_192_L0", 1, "1024x1024"),
    (1, 96, 1, 1024, 1024, 3, 1, 1, 1): ("dec_conv_out", 1, "1024x1024"),
    (1, 3, 1, 1328, 1328, 96, 1, 1, 1): ("enc_conv_in", 1, "1328x1328"),
    (1, 96, 1, 1328, 1328, 96, 1, 1, 1): ("res_96_L0", 10, "1328x1328"),
    (1, 96, 1, 1329, 1329, 96, 1, 2, 0): ("enc_down_96", 1, "1328x1328"),
    (1, 96, 1, 664, 664, 192, 1, 1, 1): ("down_96_192", 1, "1328x1328"),
    (1, 192, 1, 664, 664, 192, 1, 1, 1): ("res_192_L1", 9, "1328x1328"),
    (1, 192, 1, 665, 665, 192, 1, 2, 0): ("enc_down_192", 1, "1328x1328"),
    (1, 192, 1, 332, 332, 384, 1, 1, 1): ("down_192_384", 2, "1328x1328"),
    (1, 384, 1, 332, 332, 384, 1, 1, 1): ("res_384_L2", 8, "1328x1328"),
    (1, 384, 1, 333, 333, 384, 1, 2, 0): ("enc_down_384", 1, "1328x1328"),
    (1, 384, 1, 166, 166, 384, 1, 1, 1): ("res_384_L3", 18, "1328x1328"),
    (1, 384, 1, 166, 166, 32, 1, 1, 1): ("enc_conv_out", 1, "1328x1328"),
    (1, 16, 1, 166, 166, 384, 1, 1, 1): ("dec_conv_in", 1, "1328x1328"),
    (1, 384, 1, 332, 332, 192, 1, 1, 1): ("dec_up_384_L2", 1, "1328x1328"),
    (1, 384, 1, 664, 664, 192, 1, 1, 1): ("dec_up_384_L1", 1, "1328x1328"),
    (1, 192, 1, 1328, 1328, 96, 1, 1, 1): ("dec_up_192_L0", 1, "1328x1328"),
    (1, 96, 1, 1328, 1328, 3, 1, 1, 1): ("dec_conv_out", 1, "1328x1328"),
}

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_CONFIGS = os.path.join(REPO, "aiter", "configs", "model_configs")


def load(configs_dir):
    frames = []
    for name, model in (
        ("wan21_vae_bf16_tuned_conv3d.csv", "wan"),
        ("qwenimage_vae_bf16_tuned_conv3d.csv", "qwen"),
    ):
        f = pd.read_csv(os.path.join(configs_dir, name))
        f["model"] = model
        frames.append(f)
    d = pd.concat(frames, ignore_index=True)

    d["AI"] = d.tflops * 1e12 / (d.bw * 1e9)
    d["roofTF"] = d.AI.clip(upper=PEAK_TF / 8) * 8
    keys = [
        (r.N, r.C, r.D, r.H, r.W, r.K, r.kT, r.stride_h, r.pad_h)
        for r in d.itertuples()
    ]
    missing = [k for k in keys if k not in CALLS]
    assert not missing, f"CALLS has no row for {missing}"
    meta = [CALLS[k] for k in keys]
    d["case"] = [m[0] for m in meta]
    d["calls"] = [m[1] for m in meta]
    d["res"] = [m[2] for m in meta]
    d["headroom"] = d.us * (1 - d.tflops / d.roofTF) * d.calls / 1000
    return d


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def roofline_svg(d):
    W, H = 940, 560
    padL, padR, padT, padB = 78, 24, 26, 64
    pw, ph = W - padL - padR, H - padT - padB
    # x runs to 6000 rather than the data max (3180) so the labels of the
    # right-most rows still fit to the right of their marker; flipping them to
    # the left would draw them straight through the cluster at 1000-2000.
    x0, x1 = math.log10(20), math.log10(6000)
    y0, y1 = math.log10(15), math.log10(2600)

    def sx(ai):
        return padL + (math.log10(ai) - x0) / (x1 - x0) * pw

    def sy(tf):
        return padT + ph - (math.log10(tf) - y0) / (y1 - y0) * ph

    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">'
        ),
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]

    for v in (25, 100, 400, 1468, 2500):
        y = sy(v)
        o.append(
            f'<line x1="{padL}" y1="{y:.1f}" x2="{W - padR}" y2="{y:.1f}" '
            f'stroke="{C_GRID}"/>'
        )
        o.append(
            f'<text x="{padL - 9}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="{C_MUTE}">{v}</text>'
        )
    for v in (20, 100, 312, 1000, 3000, 6000):
        x = sx(v)
        o.append(
            f'<line x1="{x:.1f}" y1="{padT}" x2="{x:.1f}" y2="{padT + ph}" '
            f'stroke="{C_GRID}"/>'
        )
        o.append(
            f'<text x="{x:.1f}" y="{padT + ph + 18}" text-anchor="middle" '
            f'font-size="11" fill="{C_MUTE}">{v}</text>'
        )

    # The two rooflines: flat at peak past the ridge, bandwidth-limited before it.
    for peak_tf, peak_bw, colour, dash, label in (
        (PEAK_TF, PEAK_BW, C_AXIS, "", "peak 2500 TFLOPS / 8.0 TB/s (theoretical)"),
        (
            ACH_TF,
            ACH_BW,
            C_MUTE,
            "4 4",
            "1468 TFLOPS / 4.7 TB/s (measured on this card)",
        ),
    ):
        ridge = peak_tf * 1000 / peak_bw
        pts = " ".join(
            f"{px:.1f},{py:.1f}"
            for px, py in (
                (sx(20), sy(20 * peak_bw / 1000)),
                (sx(ridge), sy(peak_tf)),
                (sx(6000), sy(peak_tf)),
            )
        )
        da = f' stroke-dasharray="{dash}"' if dash else ""
        o.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="1.6"{da}/>'
        )
        o.append(
            f'<text x="{W - padR - 4}" y="{sy(peak_tf) - 7:.1f}" text-anchor="end" '
            f'font-size="11" fill="{colour}">{esc(label)}</text>'
        )

    o.append(
        f'<line x1="{sx(RIDGE):.1f}" y1="{padT}" x2="{sx(RIDGE):.1f}" '
        f'y2="{padT + ph}" stroke="{C_MUTE}" stroke-dasharray="2 4"/>'
    )
    o.append(
        f'<text x="{sx(RIDGE) + 5:.1f}" y="{padT + 12}" font-size="11" '
        f'fill="{C_MUTE}">ridge 312</text>'
    )

    # Small points first so the big ones stay legible on top.
    for r in sorted(d.itertuples(), key=lambda r: r.headroom):
        colour = C_WAN if r.model == "wan" else C_QWEN
        rad = 3.2 + min(8.0, math.sqrt(max(r.headroom, 0)) * 1.7)
        o.append(
            f'<circle cx="{sx(r.AI):.1f}" cy="{sy(r.tflops):.1f}" r="{rad:.1f}" '
            f'fill="{colour}" fill-opacity="0.45" stroke="{colour}" '
            f'stroke-width="1.2"/>'
        )

    # Labels for the rows worth naming, with a leader line and a greedy
    # vertical push so the cluster above 1000 FLOP/byte stays readable. Without
    # this the six 20-80 ms rows sit within a few pixels of each other.
    FONT, GAP = 10.5, 13.0
    labelled = sorted(
        (r for r in d.itertuples() if r.headroom >= 10),
        key=lambda r: sy(r.tflops),
    )
    placed = []
    for r in labelled:
        text = f"{r.case} @{r.res.split('x')[0]}"
        width = len(text) * FONT * 0.54
        px, py = sx(r.AI), sy(r.tflops)
        rad = 3.2 + min(8.0, math.sqrt(max(r.headroom, 0)) * 1.7)
        ly = py
        if placed and ly < placed[-1] + GAP:
            ly = placed[-1] + GAP
        placed.append(ly)
        # Flip to the left of the marker when the text would run off the plot.
        if px + rad + 6 + width > W - padR:
            tx, anchor, elbow = px - rad - 6, "end", px - rad - 3
        else:
            tx, anchor, elbow = px + rad + 6, "start", px + rad + 3
        colour = C_WAN if r.model == "wan" else C_QWEN
        if abs(ly - py) > 2:
            o.append(
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{elbow:.1f}" '
                f'y2="{ly:.1f}" stroke="{colour}" stroke-width="0.7" '
                f'stroke-opacity="0.55"/>'
            )
        o.append(
            f'<text x="{tx:.1f}" y="{ly + 3.5:.1f}" text-anchor="{anchor}" '
            f'font-size="{FONT}" fill="{C_TEXT}">{esc(text)}</text>'
        )

    o.append(
        f'<text x="{padL + pw / 2:.1f}" y="{H - 26}" text-anchor="middle" '
        f'font-size="12" fill="{C_TEXT}">Arithmetic intensity '
        f"(FLOP/byte, compulsory x+w+y traffic only) &#8212; log scale</text>"
    )
    o.append(
        f'<text x="18" y="{padT + ph / 2:.1f}" text-anchor="middle" font-size="12" '
        f'fill="{C_TEXT}" transform="rotate(-90 18 {padT + ph / 2:.1f})">'
        f"Achieved throughput (TFLOPS) &#8212; log scale</text>"
    )

    lx, ly = padL + 10, padT + ph - 54
    for i, (colour, label) in enumerate(
        ((C_WAN, "Wan2.1 VAE (44 rows)"), (C_QWEN, "Qwen-Image VAE (32 rows)"))
    ):
        o.append(
            f'<circle cx="{lx}" cy="{ly + i * 17}" r="5" fill="{colour}" '
            f'fill-opacity="0.45" stroke="{colour}" stroke-width="1.2"/>'
        )
        o.append(
            f'<text x="{lx + 12}" y="{ly + i * 17 + 4}" font-size="11" '
            f'fill="{C_TEXT}">{esc(label)}</text>'
        )
    o.append(
        f'<text x="{lx}" y="{ly + 2 * 17 + 18}" font-size="10.5" fill="{C_MUTE}">'
        f"marker area scales with reclaimable time vs. that row&apos;s own "
        f"roofline; rows above 10 ms/pass are labelled</text>"
    )
    o.append("</svg>")
    return "\n".join(o)


def reuse_bins_svg(d):
    """MFMA utilisation (masked columns removed) against operand reuse."""
    d = d.copy()
    d["mi_m"] = d.tile_m // d.wave_m // 16
    d["mi_n"] = d.tile_n // d.wave_n // 16
    d["reuse"] = d.mi_m * d.mi_n / (d.mi_m + d.mi_n)
    kg = d.K // d.groups
    d["nfill"] = kg / ((kg / d.tile_n).apply(math.ceil) * d.tile_n)
    d["adj"] = d.tflops / PEAK_TF * 100 / d.nfill
    cb = d[d.AI >= RIDGE]
    edges = [(0, 0.9), (0.9, 1.4), (1.4, 1.8), (1.8, 2.5)]
    bins = []
    for lo, hi in edges:
        g = cb[(cb.reuse > lo) & (cb.reuse <= hi)]
        bins.append((f"({lo:g}, {hi:g}]", len(g), g.adj.mean(), g.adj.max()))

    W, H = 640, 300
    padL, padR, padT, padB = 56, 20, 26, 62
    pw, ph = W - padL - padR, H - padT - padB
    ymax = 40.0
    o = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif">'
        ),
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
    ]
    for v in (0, 10, 20, 30, 40):
        y = padT + ph - v / ymax * ph
        o.append(
            f'<line x1="{padL}" y1="{y:.1f}" x2="{W - padR}" y2="{y:.1f}" '
            f'stroke="{C_GRID}"/>'
        )
        o.append(
            f'<text x="{padL - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{C_MUTE}">{v}%</text>'
        )
    slot = pw / len(bins)
    for i, (label, n, mean, mx) in enumerate(bins):
        cx = padL + slot * (i + 0.5)
        for j, (val, colour, tag) in enumerate(
            ((mean, C_WAN, "mean"), (mx, C_QWEN, "best"))
        ):
            bw_ = slot * 0.26
            bx = cx - bw_ * 1.05 + j * bw_ * 1.1
            bh = val / ymax * ph
            o.append(
                f'<rect x="{bx:.1f}" y="{padT + ph - bh:.1f}" width="{bw_:.1f}" '
                f'height="{bh:.1f}" fill="{colour}" fill-opacity="0.55" '
                f'stroke="{colour}" stroke-width="1"/>'
            )
            o.append(
                f'<text x="{bx + bw_ / 2:.1f}" y="{padT + ph - bh - 5:.1f}" '
                f'text-anchor="middle" font-size="10.5" fill="{C_TEXT}">'
                f"{val:.1f}</text>"
            )
            del tag
        o.append(
            f'<text x="{cx:.1f}" y="{padT + ph + 17:.1f}" text-anchor="middle" '
            f'font-size="11" fill="{C_TEXT}">{esc(label)}</text>'
        )
        o.append(
            f'<text x="{cx:.1f}" y="{padT + ph + 31:.1f}" text-anchor="middle" '
            f'font-size="10" fill="{C_MUTE}">n={n}</text>'
        )
    o.append(
        f'<text x="{padL + pw / 2:.1f}" y="{H - 8}" text-anchor="middle" '
        f'font-size="12" fill="{C_TEXT}">MFMA operand reuse '
        f"(mi_m&#183;mi_n)/(mi_m+mi_n), 56 compute-bound rows</text>"
    )
    o.append(
        f'<text x="14" y="{padT + ph / 2:.1f}" text-anchor="middle" font-size="12" '
        f'fill="{C_TEXT}" transform="rotate(-90 14 {padT + ph / 2:.1f})">'
        f"MFMA util., masked cols removed</text>"
    )
    lx = padL + 8
    for i, (colour, label) in enumerate(
        ((C_WAN, "bin mean"), (C_QWEN, "best row in bin"))
    ):
        o.append(
            f'<rect x="{lx}" y="{padT + i * 16}" width="9" height="9" '
            f'fill="{colour}" fill-opacity="0.55" stroke="{colour}"/>'
        )
        o.append(
            f'<text x="{lx + 14}" y="{padT + i * 16 + 8}" font-size="11" '
            f'fill="{C_TEXT}">{esc(label)}</text>'
        )
    o.append("</svg>")
    return "\n".join(o)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=DEFAULT_CONFIGS)
    args = p.parse_args()

    d = load(args.configs)
    for name, svg in (
        ("conv3d_roofline.svg", roofline_svg(d)),
        ("conv3d_reuse_bins.svg", reuse_bins_svg(d)),
    ):
        path = os.path.join(HERE, "figures", name)
        with open(path, "w") as fh:
            fh.write(svg + "\n")
        print(f"wrote {path}")

    tot, head = (d.us * d.calls).sum() / 1000, d.headroom.sum()
    print(
        f"{len(d)} rows | {tot:.1f} ms across all four passes | {head:.1f} ms below roofline"
    )


if __name__ == "__main__":
    main()
