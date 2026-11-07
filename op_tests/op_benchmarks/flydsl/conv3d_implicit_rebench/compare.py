#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Row-by-row comparison of two or more bench_all.py result dirs.

    python3 compare.py [label=]<base_dir> [label=]<new_dir> [[label=]<other_dir> ...]

Prints markdown: per-column deltas of new vs base (and of every extra dir vs
new), plus the rows whose kernel-only time moved by more than 3%.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load

COLS = (
    ("e2e_us", "e2e（CSV 协议）"),
    ("kernel_us", "kernel-only burst"),
    ("kernel_us_sus", "kernel-only sustained"),
)


def keyed(df):
    df = df.copy()
    df["k"] = df.case + "@" + df.res
    return df.set_index("k")


def delta_table(a, b, la, lb):
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    print(f"#### {lb} 相对 {la}（{len(j)} 行）")
    print()
    print(
        f"| 列 | 合计 ms/pass（{la}） | 合计 ms/pass（{lb}） | 合计差 | 逐行差 最小 | 中位 | 最大 | |Δ|≤2% 行数 |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for c, name in COLS:
        d = (j[f"{c}_b"] / j[f"{c}_a"] - 1) * 100
        sa = (j[f"{c}_a"] * j.calls_a).sum() / 1e3
        sb = (j[f"{c}_b"] * j.calls_a).sum() / 1e3
        print(
            f"| {name} | {sa:.1f} | {sb:.1f} | {(sb / sa - 1) * 100:+.1f}% | "
            f"{d.min():+.1f}% | {d.median():+.1f}% | {d.max():+.1f}% | {int((d.abs() <= 2).sum())} |"
        )
    print()
    d = (j.kernel_us_b / j.kernel_us_a - 1) * 100
    big = j[d.abs() > 3].assign(d=d[d.abs() > 3]).sort_values("d")
    if len(big):
        print(f"kernel-only burst 变化超过 ±3% 的行（{len(big)} 行）：")
        print()
        print(f"| case | {la} us | {lb} us | Δ | {lb} 3 次 |")
        print("|---|---:|---:|---:|---|")
        for k, r in big.iterrows():
            reps = " / ".join(f"{v:.1f}" for v in r.rep_kernel_us_b)
            print(
                f"| `{k}` | {r.kernel_us_a:.1f} | {r.kernel_us_b:.1f} | {r.d:+.1f}% | {reps} |"
            )
        print()
    return j


def main():
    args = [
        a.split("=", 1) if "=" in a else [os.path.basename(os.path.normpath(a)), a]
        for a in sys.argv[1:]
    ]
    labels = [a[0] for a in args]
    dfs = []
    for lb, d in args:
        meta, df = load(d)
        print(
            f"- `{lb}` (`{d}`): PCI {meta.get('pci')}, "
            f"mm {meta['mm_tflops']:.0f}/{meta['mm_tflops_sus']:.0f} TFLOPS, "
            f"copy {meta['copy_gbs']:.0f}/{meta['copy_gbs_sus']:.0f} GB/s (burst/sustained)"
        )
        dfs.append(keyed(df))
    print()
    delta_table(dfs[0], dfs[1], labels[0], labels[1])
    for df, lb in zip(dfs[2:], labels[2:]):
        delta_table(dfs[1], df, labels[1], lb)


if __name__ == "__main__":
    main()
