#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Emit mha_performance_comparison.md: MFMA vs JAX vs TE/CK reference tables plus ck_pr_6764 timings.

Reads timings CSV: seq,fwd_ms,bwd_ms (from run_mha_performance_comparison.sh).
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

# Reference: bs=2048, nheads=32, hdim=128, bfloat16, causal=False, seqlen_q == seqlen_kv
JAX_FWD_MS = {
    1: 0.051,
    2: 0.233,
    3: 0.256,
    4: 0.286,
    5: 0.324,
    6: 0.354,
    7: 0.389,
    8: 0.415,
    9: 0.496,
    10: 0.516,
    11: 0.565,
    12: 0.582,
    13: 0.637,
    14: 0.649,
    15: 0.702,
    16: 0.686,
    17: 0.793,
}
TE_CK_FWD_MS = {
    1: 1.275,
    2: 1.298,
    3: 1.306,
    4: 1.316,
    5: 1.366,
    6: 1.375,
    7: 1.389,
    8: 1.379,
    9: 1.403,
    10: 1.404,
    11: 1.409,
    12: 1.423,
    13: 1.424,
    14: 1.433,
    15: 1.436,
    16: 1.439,
    17: 1.452,
}
MFMA_FWD_MS = {
    1: 0.029,
    2: 0.047,
    3: 0.063,
    4: 0.080,
    5: 0.262,
    6: 0.272,
    7: 0.290,
    8: 0.291,
    9: 0.304,
    10: 0.320,
    11: 0.330,
    12: 0.335,
    13: 0.358,
    14: 0.359,
    15: 0.380,
    16: 0.386,
    17: 0.401,
}
JAX_BWD_MS = {
    1: 0.087,
    2: 0.350,
    3: 0.415,
    4: 0.461,
    5: 0.509,
    6: 0.519,
    7: 0.604,
    8: 0.835,
    9: 0.790,
    10: 0.731,
    11: 0.894,
    12: 0.922,
    13: 0.987,
    14: 0.924,
    15: 1.084,
    16: 1.185,
    17: 1.246,
}
TE_CK_BWD_MS = {
    1: 2.317,
    2: 2.377,
    3: 2.472,
    4: 2.548,
    5: 2.606,
    6: 2.669,
    7: 2.705,
    8: 2.737,
    9: 2.851,
    10: 2.877,
    11: 2.899,
    12: 2.957,
    13: 3.000,
    14: 3.017,
    15: 3.083,
    16: 3.096,
    17: 3.441,
}
MFMA_BWD_MS = {
    1: 0.264,
    2: 0.308,
    3: 0.386,
    4: 0.463,
    5: 0.503,
    6: 0.523,
    7: 0.561,
    8: 0.570,
    9: 0.606,
    10: 0.634,
    11: 0.657,
    12: 0.677,
    13: 0.734,
    14: 0.753,
    15: 0.792,
    16: 0.813,
    17: 0.998,
}


def mfma_fwd_kernel(seq: int) -> str:
    return "mfma_4x4" if seq <= 4 else "mfma_16x16"


def geom_mean_ratios(ratios: list[float]) -> float:
    logs = [math.log(r) for r in ratios if r > 0]
    if not logs:
        return float("nan")
    return math.exp(sum(logs) / len(logs))


def load_timings(path: Path) -> dict[int, tuple[float | None, float | None]]:
    out: dict[int, tuple[float | None, float | None]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            seq = int(row["seq"])
            try:
                fwd = float(row["fwd_ms"]) if row.get("fwd_ms", "").strip() else None
            except ValueError:
                fwd = None
            try:
                bwd = float(row["bwd_ms"]) if row.get("bwd_ms", "").strip() else None
            except ValueError:
                bwd = None
            out[seq] = (fwd, bwd)
    return out


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def write_md(timings: dict[int, tuple[float | None, float | None]], out: Path) -> None:
    lines: list[str] = [
        "# Performance Comparison: MFMA Kernel vs JAX vs TransformerEngine (CK) vs ck_pr_6764",
        "",
        "**Config:** bs=2048, nheads=32, hdim=128, bfloat16, causal=False, seqlen_q == seqlen_kv",
        "",
        "**MFMA kernel selection:** mfma_4x4 for seq 1–4, mfma_16x16 for seq 5–17",
        "",
        "**ck_pr_6764:** measured with `run_mha_performance_comparison.sh` (asm v2 `fwd_v3=0`, `bwd_v3=0`, CK path).",
        "",
        "## Forward Pass (mean time in ms)",
        "",
        "| seq | JAX (ms) | TE/CK (ms) | MFMA (ms) | kernel | vs JAX | vs TE/CK | ck_pr_6764 (ms) |",
        "|----:|---------:|-----------:|----------:|:-------|-------:|---------:|----------------:|",
    ]

    mfma_fwd_rat_jax: list[float] = []
    mfma_fwd_rat_te: list[float] = []
    ck_fwd_rat_jax: list[float] = []
    ck_fwd_rat_te: list[float] = []

    for s in range(1, 18):
        jax, te, mfma = JAX_FWD_MS[s], TE_CK_FWD_MS[s], MFMA_FWD_MS[s]
        rj, rt = jax / mfma, te / mfma
        mfma_fwd_rat_jax.append(rj)
        mfma_fwd_rat_te.append(rt)
        ck_f, _ = timings.get(s, (None, None))
        ck_col = fmt_ms(ck_f)
        if ck_f is not None and ck_f > 0:
            ck_fwd_rat_jax.append(jax / ck_f)
            ck_fwd_rat_te.append(te / ck_f)
        lines.append(
            f"| {s} | {jax:.3f} | {te:.3f} | {mfma:.3f} | {mfma_fwd_kernel(s)} | "
            f"{rj:.2f}x | {rt:.2f}x | {ck_col} |"
        )

    lines += [
        "",
        "## Backward Pass (mean time in ms)",
        "",
        "| seq | JAX (ms) | TE/CK (ms) | MFMA (ms) | vs JAX | vs TE/CK | ck_pr_6764 (ms) |",
        "|----:|---------:|-----------:|----------:|-------:|---------:|----------------:|",
    ]

    mfma_bwd_rat_jax: list[float] = []
    mfma_bwd_rat_te: list[float] = []
    ck_bwd_rat_jax: list[float] = []
    ck_bwd_rat_te: list[float] = []

    for s in range(1, 18):
        jax, te, mfma = JAX_BWD_MS[s], TE_CK_BWD_MS[s], MFMA_BWD_MS[s]
        rj, rt = jax / mfma, te / mfma
        mfma_bwd_rat_jax.append(rj)
        mfma_bwd_rat_te.append(rt)
        _, ck_b = timings.get(s, (None, None))
        ck_col = fmt_ms(ck_b)
        if ck_b is not None and ck_b > 0:
            ck_bwd_rat_jax.append(jax / ck_b)
            ck_bwd_rat_te.append(te / ck_b)
        lines.append(
            f"| {s} | {jax:.3f} | {te:.3f} | {mfma:.3f} | {rj:.2f}x | {rt:.2f}x | {ck_col} |"
        )

    gj_mfma_f = geom_mean_ratios(mfma_fwd_rat_jax)
    gt_mfma_f = geom_mean_ratios(mfma_fwd_rat_te)
    gj_mfma_b = geom_mean_ratios(mfma_bwd_rat_jax)
    gt_mfma_b = geom_mean_ratios(mfma_bwd_rat_te)

    lines += [
        "",
        "## Summary",
        "",
        "### Forward (MFMA reference)",
        f"- **vs JAX:** {min(mfma_fwd_rat_jax):.2f}x -- {max(mfma_fwd_rat_jax):.2f}x faster (geometric mean ~{gj_mfma_f:.1f}x)",
        f"- **vs TE/CK:** {min(mfma_fwd_rat_te):.2f}x -- {max(mfma_fwd_rat_te):.2f}x faster (geometric mean ~{gt_mfma_f:.1f}x)",
        "",
        "### Backward (MFMA reference)",
        f"- **vs JAX:** {min(mfma_bwd_rat_jax):.2f}x -- {max(mfma_bwd_rat_jax):.2f}x (geometric mean ~{gj_mfma_b:.1f}x)",
        f"- **vs TE/CK:** {min(mfma_bwd_rat_te):.2f}x -- {max(mfma_bwd_rat_te):.2f}x faster (geometric mean ~{gt_mfma_b:.1f}x)",
        "",
    ]

    if len(ck_fwd_rat_jax) == 17:
        gj = geom_mean_ratios(ck_fwd_rat_jax)
        gt = geom_mean_ratios(ck_fwd_rat_te)
        lines += [
            "### Forward (ck_pr_6764, geometric mean vs references)",
            f"- **vs JAX:** ~{gj:.1f}x",
            f"- **vs TE/CK:** ~{gt:.1f}x",
            "",
        ]
    if len(ck_bwd_rat_jax) == 17:
        gj = geom_mean_ratios(ck_bwd_rat_jax)
        gt = geom_mean_ratios(ck_bwd_rat_te)
        lines += [
            "### Backward (ck_pr_6764, geometric mean vs references)",
            f"- **vs JAX:** ~{gj:.1f}x",
            f"- **vs TE/CK:** ~{gt:.1f}x",
            "",
        ]

    lines += [
        "### Notes",
        "- JAX numbers are from TransformerEngine benchmark on the same GPU class",
        "- TE/CK = TransformerEngine using CK (Composable Kernel) backend",
        "- MFMA forward uses mfma_4x4x4 for seq 1–4, mfma_16x16x16 for seq 5–17",
        "- MFMA backward uses mfma_16x16x16 for all sequence lengths",
        "- Forward speedup columns (vs JAX / vs TE/CK) are reference_mean / MFMA time (higher = MFMA faster)",
        "- Backward vs columns use the same ratio convention as the MFMA reference table",
        "- Backward seq=1 is slower than JAX in the MFMA reference because JAX uses a highly optimized scalar path for single-token attention",
        "- **ck_pr_6764** timings are produced by this repo’s MHA v2 benchmark (`mask=0`, batch layout BHSD)",
        "",
    ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--timings",
        type=Path,
        required=True,
        help="CSV with columns seq,fwd_ms,bwd_ms",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=here / "mha_performance_comparison.md",
        help="Output markdown path",
    )
    args = ap.parse_args()
    timings = load_timings(args.timings)
    write_md(timings, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
