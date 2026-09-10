#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Plot conv3d_implicit against hipBLASLt from bench_conv_vs_hipblaslt.py.

Usage::

    python op_tests/op_benchmarks/flydsl/wan21_vae_conv/plot_conv_vs_hipblaslt.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "conv3d_implicit_vs_hipblaslt.png"
WIN, LOSE, NEUTRAL = "#2a6f6f", "#b44a3c", "#5c5c5c"
WIN_PALE, LOSE_PALE = "#9ec9c9", "#e8b4ad"
# The three buckets differ in what the kernel is actually asked to do, so the y
# labels carry the distinction rather than a fourth colour.
KIND_TAG = {"conv3d": "3D", "resample": "2D s2", "time": "3×1×1"}

rows = json.loads((HERE / "conv_vs_hipblaslt.json").read_text())
rows = [r for r in rows if r.get("ratio")]


def lab(r):
    return f"[{KIND_TAG[r['kind']]}] {r['sid']}  {r['M']}×{r['N']}×{r['K']}"


plt.rcParams.update(
    {
        "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
        "font.size": 11,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    }
)

max_f = max(r["freq"] for r in rows)


def bar_h(f):
    return 0.22 + 0.72 * (f / max_f)


fig, ax = plt.subplots(figsize=(13.4, 9.0))
gap = 0.22
centers, cursor = [], 0.0
for r in rows:
    h = bar_h(r["freq"])
    cursor += h / 2
    centers.append(cursor)
    cursor += h / 2 + gap
centers = np.array(centers)

for yc, r in zip(centers, rows):
    h = bar_h(r["freq"])
    win = r["ratio"] >= 1.0
    ax.barh(yc, r["ratio"], height=h, color=WIN if win else LOSE, zorder=2)
    # pale overlay: the same ratio once the NCDHW->NDHWC transpose is charged too
    ax.barh(
        yc, r["ratio_all"], height=h, color=WIN_PALE if win else LOSE_PALE, zorder=3
    )
    share = 1 - r["conv"] / r["conv_all"]
    note = f"{r['ratio']:.2f}  ×{r['freq']}"
    if share > 0.15:
        note += f"   转置 {share * 100:.0f}%"
    ax.text(
        r["ratio"] + 0.03,
        yc,
        note,
        va="center",
        fontsize=10,
        color=WIN if win else LOSE,
        zorder=5,
    )

ax.axvline(1.0, color=NEUTRAL, ls="--", lw=1.2, zorder=4)
ax.set_yticks(centers)
ax.set_yticklabels([lab(r) for r in rows], fontsize=10)
ax.set_ylim(centers[-1] + bar_h(rows[-1]["freq"]) / 2 + gap, -gap)
ax.set_xlabel("加速比  hipBLASLt on-device µs / conv3d_implicit kernel on-device µs")
ax.set_xlim(0, max(r["ratio"] for r in rows) * 1.28)
ax.set_title(
    "Wan2.1 VAE：aiter conv3d_implicit 对标 hipBLASLt"
    "（条宽 = 每次 encode 调用次数；同 M/N/K，同 FLOP；gfx950）"
)
fig.subplots_adjust(left=0.34)
ax.legend(
    handles=[
        Patch(facecolor=WIN, label="conv kernel 更快（≥1）"),
        Patch(facecolor=LOSE, label="conv kernel 更慢（<1）"),
        Patch(facecolor=WIN_PALE, label="计入 NCDHW→NDHWC 转置后（赢）"),
        Patch(facecolor=LOSE_PALE, label="计入转置后（输）"),
    ],
    loc="center right",
    bbox_to_anchor=(1.0, 0.35),
    frameon=True,
    framealpha=0.95,
    edgecolor="none",
    fontsize=10,
)

nwin = sum(r["ratio"] >= 1.0 for r in rows)
wsum = sum(r["freq"] for r in rows)
weighted = sum(r["freq"] * r["ratio"] for r in rows) / wsum
c3 = [r for r in rows if r["kind"] == "conv3d"]
w3 = sum(r["freq"] * r["ratio"] for r in c3) / sum(r["freq"] for r in c3)
fig.text(
    0.01,
    0.012,
    f"深色为 conv 主 kernel（与 Qwen-Image 图同口径）：{len(rows)} 个 shape 中 "
    f"{nwin} 个跑赢，按调用次数加权 {weighted:.3f}×；其中真 3D 卷积那 {len(c3)} 个"
    f"加权 {w3:.3f}×。浅色叠加了 NCDHW→NDHWC 转置——kernel 内部是 channels-last，"
    "NCDHW 输入需先转置。",
    color=NEUTRAL,
    fontsize=8,
)
fig.text(
    0.01,
    -0.004,
    "非同类对比：hipBLASLt 读已物化的 M×K 矩阵，3×3×3 卷积从约 27 倍小的原张量 gather"
    "（Qwen 的 3×3 只有 9 倍），物化代价未计入 hipBLASLt，故此图的差距应大于 2D 图。"
    ' 若上游已是 NDHWC（input_layout="NDHWC"），转置那部分不发生。',
    color=NEUTRAL,
    fontsize=8,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(f"wrote {OUT}")
