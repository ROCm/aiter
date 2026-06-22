# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Turn a rocprof-compute roofline workload (roofline.csv empirical ceilings +
# pmc_perf.csv per-dispatch counters) into a numeric per-kernel roofline table
# and a roofline PNG. The FLOP / byte / arithmetic-intensity formulas mirror
# rocprof-compute's own utils/roofline_calc.py:calc_ai_profile so the numbers
# match the tool's roofline plot.
#
# Run in the profiler driver venv (has pandas + matplotlib):
#   rocprof_venv/bin/python roofline_report.py workloads/<name> [--png out.png]
#                                               [--filter grad_]

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LDS_BANKS_PER_CU = 32  # CDNA


def _col(df, name):
    return df[name] if name in df.columns else 0.0


def kernel_metrics(pmc_csv, name_filter):
    df = pd.read_csv(pmc_csv)
    df = df[df["Kernel_Name"].str.contains(name_filter, na=False)] if name_filter else df
    out = []
    for kname, g in df.groupby("Kernel_Name"):
        calls = len(g)
        valu = (
            64 * (_col(g, "SQ_INSTS_VALU_ADD_F16") + _col(g, "SQ_INSTS_VALU_MUL_F16")
                  + 2 * _col(g, "SQ_INSTS_VALU_FMA_F16") + _col(g, "SQ_INSTS_VALU_TRANS_F16"))
            + 64 * (_col(g, "SQ_INSTS_VALU_ADD_F32") + _col(g, "SQ_INSTS_VALU_MUL_F32")
                    + 2 * _col(g, "SQ_INSTS_VALU_FMA_F32") + _col(g, "SQ_INSTS_VALU_TRANS_F32"))
            + 64 * (_col(g, "SQ_INSTS_VALU_ADD_F64") + _col(g, "SQ_INSTS_VALU_MUL_F64")
                    + 2 * _col(g, "SQ_INSTS_VALU_FMA_F64") + _col(g, "SQ_INSTS_VALU_TRANS_F64"))
        ).sum()
        mfma = (
            (_col(g, "SQ_INSTS_VALU_MFMA_MOPS_F16") + _col(g, "SQ_INSTS_VALU_MFMA_MOPS_BF16")
             + _col(g, "SQ_INSTS_VALU_MFMA_MOPS_F32") + _col(g, "SQ_INSTS_VALU_MFMA_MOPS_F64")
             + _col(g, "SQ_INSTS_VALU_MFMA_MOPS_F8")) * 512
        ).sum()
        total_flops = valu + mfma
        # gfx942 HBM bytes (matches roofline_calc.py non-MI200 branch).
        hbm = (
            _col(g, "TCC_BUBBLE_sum") * 128
            + _col(g, "TCC_EA0_RDREQ_32B_sum") * 32
            + (_col(g, "TCC_EA0_RDREQ_sum") - _col(g, "TCC_BUBBLE_sum") - _col(g, "TCC_EA0_RDREQ_32B_sum")) * 64
            + (_col(g, "TCC_EA0_WRREQ_sum") - _col(g, "TCC_EA0_WRREQ_64B_sum")) * 32
            + _col(g, "TCC_EA0_WRREQ_64B_sum") * 64
        ).sum()
        l2 = ((_col(g, "TCP_TCC_WRITE_REQ_sum") + _col(g, "TCP_TCC_ATOMIC_WITH_RET_REQ_sum")
               + _col(g, "TCP_TCC_ATOMIC_WITHOUT_RET_REQ_sum") + _col(g, "TCP_TCC_READ_REQ_sum")) * 64).sum()
        l1 = (_col(g, "TCP_TOTAL_CACHE_ACCESSES_sum") * 64).sum()
        dur = (g["End_Timestamp"] - g["Start_Timestamp"]).sum()  # ns, summed over calls
        avg_dur = dur / calls
        # total_flops and dur are both summed over calls, so the ratio is the
        # achieved rate (FLOP/ns == GFLOP/s); matches rocprof-compute.
        perf = total_flops / dur if dur else 0.0
        out.append({
            "kernel": kname, "calls": calls, "avg_us": avg_dur / 1e3,
            "GFLOPs": perf, "valu_flops": valu / calls, "mfma_flops": mfma / calls,
            "hbm_bytes": hbm / calls, "l2_bytes": l2 / calls, "l1_bytes": l1 / calls,
            "AI_hbm": total_flops / hbm if hbm else 0.0,
            "AI_l2": total_flops / l2 if l2 else 0.0,
            "AI_l1": total_flops / l1 if l1 else 0.0,
        })
    out.sort(key=lambda r: r["avg_us"] * r["calls"], reverse=True)
    return out


def read_ceilings(roof_csv):
    row = pd.read_csv(roof_csv).iloc[0]
    return {
        "HBM": row["HBMBw"], "L2": row["L2Bw"], "L1": row["L1Bw"], "LDS": row["LDSBw"],
        "VALU_FP32": row["FP32Flops"], "VALU_BF16": row["BF16Flops"],
        "MFMA_BF16": row["MFMABF16Flops"], "MFMA_F32": row["MFMAF32Flops"],
    }


def render(kernels, ceil, png):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ai = [1e-2 * 1.2 ** i for i in range(80)]
    # Bandwidth ceilings (sloped) and compute ceilings (flat), GFLOP/s vs FLOP/byte.
    for lbl, bw, c in [("HBM", ceil["HBM"], "tab:red"), ("L2", ceil["L2"], "tab:blue"),
                       ("L1", ceil["L1"], "tab:green")]:
        ax.plot(ai, [bw * x for x in ai], c=c, lw=1.3, label=f"{lbl} {bw/1e3:.1f} TB/s")
    for lbl, pk, c, ls in [("MFMA bf16", ceil["MFMA_BF16"], "black", "-"),
                           ("VALU fp32", ceil["VALU_FP32"], "dimgray", "--")]:
        ax.axhline(pk, c=c, ls=ls, lw=1.3, label=f"{lbl} {pk/1e3:.0f} TFLOP/s")
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    for i, k in enumerate(kernels):
        if k["GFLOPs"] <= 0 or k["AI_hbm"] <= 0:
            continue
        short = k["kernel"].split("(")[0][:26]
        ax.plot(k["AI_hbm"], k["GFLOPs"], markers[i % len(markers)], ms=11,
                label=f"{short}  {k['GFLOPs']/1e3:.1f} TF/s @ AI {k['AI_hbm']:.1f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / HBM byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("MI300X empirical roofline - jagged_dense_bmm backward")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(top=ceil["MFMA_BF16"] * 2)
    fig.tight_layout(); fig.savefig(png, dpi=130)
    print(f"[png] {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workload", help="workload dir (contains pmc_perf.csv + roofline.csv)")
    p.add_argument("--filter", default="grad_", help="substring filter on kernel name")
    p.add_argument("--png", default=None)
    args = p.parse_args()

    pmc = os.path.join(args.workload, "pmc_perf.csv")
    roof = os.path.join(args.workload, "roofline.csv")
    ceil = read_ceilings(roof)
    kernels = kernel_metrics(pmc, args.filter)

    print(f"\nMI300X empirical ceilings: HBM {ceil['HBM']/1e3:.2f} TB/s | "
          f"L2 {ceil['L2']/1e3:.1f} TB/s | VALU fp32 {ceil['VALU_FP32']/1e3:.0f} TFLOP/s | "
          f"MFMA bf16 {ceil['MFMA_BF16']/1e3:.0f} TFLOP/s\n")
    hdr = (f"{'kernel':<30}{'calls':>6}{'us':>9}{'TFLOP/s':>9}{'engine':>7}"
           f"{'AI_hbm':>8}{'%cmpPk':>8}{'%hbmBW':>8}{'%roof':>7}")
    print(hdr); print("-" * len(hdr))
    for k in kernels:
        mfma = k["mfma_flops"] > k["valu_flops"]
        engine = "MFMA" if mfma else "VALU"
        cmp_peak = ceil["MFMA_BF16"] if mfma else ceil["VALU_FP32"]
        achieved_bw = (k["GFLOPs"] / k["AI_hbm"]) if k["AI_hbm"] else 0.0  # GB/s
        roof_at_ai = min(cmp_peak, k["AI_hbm"] * ceil["HBM"]) if k["AI_hbm"] else cmp_peak
        pct_cmp = 100 * k["GFLOPs"] / cmp_peak
        pct_bw = 100 * achieved_bw / ceil["HBM"] if achieved_bw else 0
        pct_roof = 100 * k["GFLOPs"] / roof_at_ai if roof_at_ai else 0
        print(f"{k['kernel'].split('(')[0][:29]:<30}{k['calls']:>6}{k['avg_us']:>9.2f}"
              f"{k['GFLOPs']/1e3:>9.2f}{engine:>7}{k['AI_hbm']:>8.1f}"
              f"{pct_cmp:>8.1f}{pct_bw:>8.1f}{pct_roof:>7.1f}")
    print("\n%cmpPk = % of relevant compute peak (MFMA bf16 / VALU fp32)")
    print("%hbmBW = % of peak HBM bandwidth   %roof = % of the binding roofline ceiling at that AI")

    if args.png:
        render(kernels, ceil, args.png)


if __name__ == "__main__":
    main()
