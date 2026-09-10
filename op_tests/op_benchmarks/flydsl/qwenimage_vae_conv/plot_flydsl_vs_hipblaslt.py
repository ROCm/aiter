#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL A16W16 (after the pruner fix) vs hipBLASLt on the VAE GEMM shapes.

Usage::

    python op_tests/op_benchmarks/flydsl/qwenimage_vae_conv/plot_flydsl_vs_hipblaslt.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "figures" / "flydsl_after_fix_vs_hipblaslt.png"
WIN, LOSE, DEF_WIN, DEF_LOSE, NEUTRAL = (
    "#2a6f6f",
    "#b44a3c",
    "#9ec9c9",
    "#e8b4ad",
    "#5c5c5c",
)

# the 18 VAE convolutions, folded onto their equivalent GEMM shapes
SHAPES = [
    ("3→96 @1024²", 3, 96, 1024, 1, 1, 1),
    ("96→96 @1024²", 96, 96, 1024, 1, 1, 10),
    ("96→192 @512²", 96, 192, 512, 1, 1, 1),
    ("192→192 @512²", 192, 192, 512, 1, 1, 9),
    ("192→384 @256²", 192, 384, 256, 1, 1, 2),
    ("384→384 @256²", 384, 384, 256, 1, 1, 8),
    ("384→384 @128²", 384, 384, 128, 1, 1, 18),
    ("384→32 @128²", 384, 32, 128, 1, 1, 1),
    ("16→384 @128²", 16, 384, 128, 1, 1, 1),
    ("96→3 @1024²", 96, 3, 1024, 1, 1, 1),
    ("96→96 @1025² s2", 96, 96, 1025, 2, 0, 1),
    ("192→192 @513² s2", 192, 192, 513, 2, 0, 1),
    ("384→384 @257² s2", 384, 384, 257, 2, 0, 1),
    ("384→192 @256²", 384, 192, 256, 1, 1, 1),
    ("384→192 @512²", 384, 192, 512, 1, 1, 1),
    ("192→96 @1024²", 192, 96, 1024, 1, 1, 1),
    ("384→384 @166²", 384, 384, 166, 1, 1, 18),
    ("96→96 @1328²", 96, 96, 1328, 1, 1, 10),
]

meta = {}
freq = defaultdict(int)
for name, cin, cout, hin, stride, pad, f in SHAPES:
    hout = (hin + 2 * pad - 3) // stride + 1
    key = (hout * hout, cout, cin * 9)
    freq[key] += f
    meta.setdefault(key, []).append(name)

fly = pd.read_csv(HERE / "flydsl_pruner_before_after.csv")
fly = {(int(r.M), int(r.N), int(r.K)): r for r in fly.itertuples()}
hip = pd.read_csv(
    ROOT / "aiter" / "configs" / "model_configs" / "qwenimage_vae_bf16_tuned_gemm.csv"
)
hip = {(int(r.M), int(r.N), int(r.K)): float(r.us) for r in hip.itertuples()}

rows = []
for key, names in meta.items():
    r = fly.get(key)
    rows.append(
        {
            "label": f"{' / '.join(names)}  {key[0]}×{key[1]}×{key[2]}",
            "freq": freq[key],
            "ok": r is not None,
            "after": (hip[key] / r.flydsl_us_after) if r is not None else None,
            "before": (hip[key] / r.flydsl_us_before) if r is not None else None,
            "hti": bool(r.hti_after) if r is not None else False,
        }
    )
rows.sort(key=lambda x: -(x["after"] or -1))

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


fig, ax = plt.subplots(figsize=(13.6, 10.2))
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
    if not r["ok"]:
        ax.text(
            0.02,
            yc,
            f"不支持  ×{r['freq']}",
            va="center",
            fontsize=9,
            color="#9a9a9a",
            zorder=5,
        )
        continue
    win = r["after"] >= 1.0
    # dark = after the fix, pale = before; whichever is longer draws first
    lo, hi = sorted((r["before"], r["after"]))
    ax.barh(
        yc,
        hi,
        height=h,
        color=(WIN if win else LOSE)
        if hi == r["after"]
        else (DEF_WIN if win else DEF_LOSE),
        zorder=2,
    )
    ax.barh(
        yc,
        lo,
        height=h,
        color=(WIN if win else LOSE)
        if lo == r["after"]
        else (DEF_WIN if win else DEF_LOSE),
        zorder=3,
    )
    note = f"{r['after']:.2f}  ×{r['freq']}"
    gain = r["after"] / r["before"] - 1
    if gain > 0.02:
        note += f"   +{gain * 100:.0f}%"
    if r["hti"]:
        note += "  HTI"
    ax.text(
        hi + 0.02,
        yc,
        note,
        va="center",
        fontsize=10,
        color=WIN if win else LOSE,
        zorder=5,
    )

ax.axvline(1.0, color=NEUTRAL, ls="--", lw=1.2, zorder=4)
ax.set_yticks(centers)
ax.set_yticklabels([r["label"] for r in rows], fontsize=9.5)
ax.set_ylim(centers[-1] + bar_h(rows[-1]["freq"]) / 2 + gap, -gap)
ax.set_xlabel("GEMM 加速比   hipBLASLt on-device µs / FlyDSL A16W16 on-device µs")
ax.set_xlim(0, 1.45)
ax.set_title(
    "剪枝器修复后的 FlyDSL A16W16 对标 hipBLASLt"
    "（条宽 = 调用次数；同 M/N/K，同 FLOP；gfx950）"
)
fig.subplots_adjust(left=0.34)
ax.legend(
    handles=[
        Patch(facecolor=WIN, label="修复后 ≥1 跑赢 hipBLASLt"),
        Patch(facecolor=LOSE, label="修复后 <1 落后"),
        Patch(facecolor=DEF_WIN, label="修复前（赢，浅青绿）"),
        Patch(facecolor=DEF_LOSE, label="修复前（输，浅红）"),
    ],
    loc="center right",
    bbox_to_anchor=(1.0, 0.30),
    frameon=True,
    framealpha=0.95,
    edgecolor="none",
    fontsize=10,
)
ok = [r for r in rows if r["ok"]]
nwin = sum(r["after"] >= 1.0 for r in ok)
fig.text(
    0.01,
    0.012,
    f"深色为剪枝器修复后，浅色为修复前；条尾百分比是本次修复的增益，"
    f"「HTI」标记修复后重新选中 half-tile-interleaved 的 shape。"
    f"17 个 GEMM shape 中 {len(ok)} 个有可用配置，其中 {nwin} 个跑赢 hipBLASLt。"
    " 每侧为空闲机上四次交替调优取优（原始值见 CSV 的 *_reps 列）：7 提速 0 回退，"
    "几何平均 1.114×，其中解锁 HTI 的 5 个为 1.174×。",
    color=NEUTRAL,
    fontsize=8,
)
fig.text(
    0.01,
    -0.004,
    "K=27 / K=144 / N=3 等不满足 DMA 或 K-tail 约束的 shape 标「不支持」。"
    "非同类对比：hipBLASLt 读已物化的 M×K 矩阵，物化代价未计入。",
    color=NEUTRAL,
    fontsize=8,
)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
plt.close(fig)
print(f"wrote {OUT}")
