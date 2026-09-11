#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Three-way view of the Qwen-Image VAE shapes, from bench_three_way.py.

Usage::

    python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/plot_three_way.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures" / "three_way_vs_hipblaslt.png"
CONV, GEMM, NEUTRAL = "#2a6f6f", "#6aaee0", "#5c5c5c"
CONV_LOSE, GEMM_LOSE = "#b44a3c", "#e0a860"

rows = [r for r in json.loads((HERE / "three_way.json").read_text()) if r.get("conv")]
rows.sort(key=lambda r: -r["conv_ratio"])

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

fig, ax = plt.subplots(figsize=(13.6, 11.0))
H = 0.36
centers = np.arange(len(rows)) * 1.05

for yc, r in zip(centers, rows):
    cr = r["conv_ratio"]
    ax.barh(yc + H / 2, cr, height=H, color=CONV if cr >= 1 else CONV_LOSE, zorder=2)
    ax.text(
        cr + 0.02,
        yc + H / 2,
        f"{cr:.2f}",
        va="center",
        fontsize=9,
        color=CONV if cr >= 1 else CONV_LOSE,
    )
    gr = r.get("gemm_ratio")
    if gr:
        ax.barh(
            yc - H / 2, gr, height=H, color=GEMM if gr >= 1 else GEMM_LOSE, zorder=2
        )
        ax.text(
            gr + 0.02, yc - H / 2, f"{gr:.2f}", va="center", fontsize=9, color=NEUTRAL
        )
    else:
        ax.text(
            0.02, yc - H / 2, "GEMM 不支持", va="center", fontsize=8, color="#9a9a9a"
        )

ax.axvline(1.0, color=NEUTRAL, ls="--", lw=1.2, zorder=4)
ax.set_yticks(centers)
ax.set_yticklabels(
    [f"{r['sid']}  {r['M']}×{r['N']}×{r['K']}  ×{r['freq']}" for r in rows],
    fontsize=9.5,
)
ax.set_ylim(centers[-1] + 0.8, centers[0] - 0.8)
ax.set_xlabel("相对 hipBLASLt 的加速比   hipBLASLt µs / 本实现 µs   （>1 更快）")
ax.set_xlim(0, max(r["conv_ratio"] for r in rows) * 1.2)
ax.set_title(
    "Qwen-Image VAE：conv3d_implicit、FlyDSL A16W16 GEMM、hipBLASLt 三方对比"
    "（统一轮换口径 + 调优选解；gfx950）"
)
fig.subplots_adjust(left=0.34)
ax.legend(
    handles=[
        Patch(facecolor=CONV, label="conv3d_implicit（≥1 跑赢 hipBLASLt）"),
        Patch(facecolor=CONV_LOSE, label="conv3d_implicit（<1）"),
        Patch(facecolor=GEMM, label="FlyDSL A16W16 GEMM（≥1）"),
        Patch(facecolor=GEMM_LOSE, label="FlyDSL A16W16 GEMM（<1）"),
    ],
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    edgecolor="none",
    fontsize=9.5,
)

cw = sum(r["conv_ratio"] >= 1 for r in rows)
gs = [r for r in rows if r.get("gemm_ratio")]
gw = sum(r["gemm_ratio"] >= 1 for r in gs)
both = [r for r in gs]
conv_beats_gemm = sum(r["conv_ratio"] > r["gemm_ratio"] for r in both)
fig.text(
    0.01,
    0.012,
    f"三条统一为轮换口径：两条 GEMM 取 tuner 的调优选解（hipBLASLt 遍历 solution index），"
    f"conv 用等效的输入轮换，同一次会话内先后测完于空闲机。{len(rows)} 个卷积中 conv 跑赢 "
    f"hipBLASLt 的有 {cw} 个；GEMM 覆盖 {len(gs)} 个（K 非 64 的倍数即不支持），其中 {gw} 个跑赢。"
    f"两者都可用的 {len(both)} 个里，conv 更快的有 {conv_beats_gemm} 个。",
    color=NEUTRAL,
    fontsize=8,
)
fig.text(
    0.01,
    -0.004,
    "conv 只计其主 kernel，不含 NCHW→NHWC 转置，以便与两条 GEMM 可比；"
    "两条 GEMM 读已物化的 M×K 矩阵，im2col 的物化代价均未计入。"
    "轮换对 conv 几乎无影响（工作集仅约 21 MB，4 份仍在 256 MB 缓存内），"
    "却让两条 GEMM 真正走 HBM——这正是 conv 免于 im2col 物化的直接体现。",
    color=NEUTRAL,
    fontsize=8,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(f"wrote {OUT}")
