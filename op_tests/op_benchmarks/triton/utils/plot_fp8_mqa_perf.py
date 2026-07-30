#!/usr/bin/env python3
"""
Bar plot: Triton vs GEAK v4 vs Best FlyDSL mfma
Data extracted from bench-fp8-mqa-kernel-all.md (graph timing, AMD MI325X gfx942).
Only shapes where GEAK v4 ran (H=64, not SKIP/FAIL).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --------------------------------------------------------------------------
# Data: (shape_label, dtype, triton_tflops, geak_tflops, best_mfma_tflops, best_mfma_variant)
# best_mfma = max TFLOPs over all passing mfma variants per shape/dtype.
# GEAK: SKIP for H=128, FAIL for Shape10 (32768²) → excluded from plot.
# --------------------------------------------------------------------------

rows = [
    # shape, dtype, triton, geak, best_mfma, best_variant
    ("1024²\nH64 D128", "fnuz/fnuz",  310.5, 162.3,  225.1, "mfma32x32x16_r2_w2"),
    ("1024²\nH64 D128", "fn/fnuz",    110.0,  85.9,  215.9, "mfma32x32x16_r2_w2"),
    ("2048²\nH64 D128", "fnuz/fnuz",  394.5, 358.5,  445.1, "mfma32x32x16_r2_w2"),
    ("2048²\nH64 D128", "fn/fnuz",    202.3, 190.9,  426.6, "mfma16x16x32_r2_w4"),
    ("4096²\nH64 D128", "fnuz/fnuz",  463.1, 583.1,  721.4, "mfma32x32x16_r2_w2"),
    ("4096²\nH64 D128", "fn/fnuz",    306.7, 355.7,  695.0, "mfma32x32x16_r2_w2"),
    ("1024×4096\nH64 D128","fnuz/fnuz",581.0, 483.4,  550.1, "mfma16x16x32_r2_w4"),
    ("1024×4096\nH64 D128","fn/fnuz",  388.1, 348.8,  519.9, "mfma16x16x32_r2_w4"),
    ("4096×1024\nH64 D128","fnuz/fnuz",140.0, 199.3,  213.9, "mfma32x32x16_r1_w1"),
    ("4096×1024\nH64 D128","fn/fnuz",   39.6,  43.4,  196.1, "mfma32x32x16_r1_w1"),
    ("4096×8192\nH64 D128","fnuz/fnuz",486.1, 711.8,  785.1, "mfma16x16x32_r4_w2"),
    ("4096×8192\nH64 D128","fn/fnuz",  410.5, 559.2,  772.3, "mfma16x16x32_r4_w2"),
    ("8192×4096\nH64 D128","fnuz/fnuz",397.5, 541.2,  671.0, "mfma32x32x16_r2_w2"),
    ("8192×4096\nH64 D128","fn/fnuz",  198.7, 229.4,  647.5, "mfma32x32x16_r2_w2"),
    ("8192²\nH64 D128",    "fnuz/fnuz",485.6, 675.4,  872.5, "mfma16x16x32_r4_w2"),
    ("8192²\nH64 D128",    "fn/fnuz",  370.6, 478.6,  862.0, "mfma16x16x32_r4_w2"),
    ("16384²\nH64 D128",   "fnuz/fnuz",482.0, 733.2,  896.4, "mfma16x16x32_r4_w2"),
    ("16384²\nH64 D128",   "fn/fnuz",  410.4, 590.9,  891.4, "mfma16x16x32_r4_w2"),
]

shape_labels = [r[0] for r in rows]
dtypes = [r[1] for r in rows]
triton_vals = np.array([r[2] for r in rows])
geak_vals   = np.array([r[3] for r in rows])
mfma_vals   = np.array([r[4] for r in rows])

n = len(rows)
x = np.arange(n)
w = 0.26  # bar width

COLORS = {
    "triton": "#4878CF",   # blue
    "geak":   "#6ACC65",   # green
    "mfma":   "#D65F5F",   # red/salmon
}

fig, ax = plt.subplots(figsize=(22, 7))

bars_triton = ax.bar(x - w,     triton_vals, w, label="Triton",          color=COLORS["triton"], alpha=0.9, zorder=3)
bars_geak   = ax.bar(x,         geak_vals,   w, label="GEAK v4",         color=COLORS["geak"],   alpha=0.9, zorder=3)
bars_mfma   = ax.bar(x + w,     mfma_vals,   w, label="Best FlyDSL mfma",color=COLORS["mfma"],   alpha=0.9, zorder=3)

# --- perf-improvement tags on top of each bar ---
def add_labels(bars, ref_vals, ax, color, show_speedup=False, ref_bars=None):
    for bar, val, ref in zip(bars, [b.get_height() for b in bars], ref_vals):
        h = bar.get_height()
        if show_speedup and ref > 0:
            tag = f"{val/ref:.2f}×"
        else:
            tag = f"{h:.0f}"
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, tag,
                ha="center", va="bottom", fontsize=6.5, color=color, fontweight="bold", rotation=90)

# For Triton: show absolute TFLOPS
add_labels(bars_triton, triton_vals, ax, COLORS["triton"])
# For GEAK: show speedup vs Triton
add_labels(bars_geak,   triton_vals, ax, COLORS["geak"], show_speedup=True, ref_bars=geak_vals)
# For mfma: show speedup vs Triton
add_labels(bars_mfma,   triton_vals, ax, COLORS["mfma"], show_speedup=True, ref_bars=mfma_vals)

# x-axis: group by shape+dtype
tick_labels = [f"{s}\n{d}" for s, d in zip(shape_labels, dtypes)]
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=7.5)
ax.set_ylabel("TFLOPS (higher is better)", fontsize=11)
ax.set_title("FP8 MQA Logits — Kernel Performance: Triton vs GEAK v4 vs Best FlyDSL mfma\n"
             "AMD MI325X (gfx942) · bs=1 · H=64 · D=128 · graph mode (warmup=10, bench=20×50 replays)",
             fontsize=10)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(0, max(mfma_vals.max(), geak_vals.max(), triton_vals.max()) * 1.22)

# Legend
legend_handles = [
    mpatches.Patch(color=COLORS["triton"], label="Triton (bar = abs TFLOPS)"),
    mpatches.Patch(color=COLORS["geak"],   label="GEAK v4 (tag = speedup vs Triton)"),
    mpatches.Patch(color=COLORS["mfma"],   label="Best FlyDSL mfma (tag = speedup vs Triton)"),
]
ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

plt.tight_layout()
out = "bench-fp8-mqa-perf-plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
