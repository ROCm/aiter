#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Emit markdown for MHA performance comparisons.

--kind self_attn: MFMA / JAX / TE·CK reference + ck_pr_6764 (CSV: seq,fwd_ms,bwd_ms).

--kind cross_attn: cross-attention ck_pr_6764 (CSV: batch,s_kv,fwd_ms,bwd_ms[,group_fwd]; legacy: s_kv,... defaults batch=2048).
  Optional --jax-timings: batch,s_kv,jax_fwd_ms,jax_bwd_ms. Section order follows first occurrence of each batch in the timings CSV (matches shell sweep order).
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


def load_self_timings(path: Path) -> dict[int, tuple[float | None, float | None]]:
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


def load_jax_cross_timings(path: Path) -> tuple[dict[tuple[int, int], tuple[float, float]], dict[int, tuple[float, float]]]:
    """Return (jax_by_batch_skv, jax_by_skv_legacy_no_batch_column)."""
    by_bs: dict[tuple[int, int], tuple[float, float]] = {}
    by_p: dict[int, tuple[float, float]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or ()
        has_batch = "batch" in fields
        for row in reader:
            p = int(row["s_kv"])
            try:
                jf = float(row["jax_fwd_ms"]) if row.get("jax_fwd_ms", "").strip() else 0.0
                jb = float(row["jax_bwd_ms"]) if row.get("jax_bwd_ms", "").strip() else 0.0
            except ValueError:
                continue
            if has_batch and row.get("batch", "").strip().isdigit():
                b = int(row["batch"])
                by_bs[(b, p)] = (jf, jb)
            else:
                by_p[p] = (jf, jb)
    return by_bs, by_p


def load_cross_timings(ck_path: Path, jax_path: Path | None = None) -> tuple[list[dict[str, str | int | float | None]], list[int]]:
    rows: list[dict[str, str | int | float | None]] = []
    batch_order: list[int] = []
    seen_batch: set[int] = set()
    with ck_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or ()
        has_batch = "batch" in fields
        for row in reader:
            p = int(row["s_kv"])
            b_raw = row.get("batch", "").strip() if has_batch else ""
            batch = int(b_raw) if b_raw.isdigit() else 2048
            if batch not in seen_batch:
                seen_batch.add(batch)
                batch_order.append(batch)
            try:
                fwd = float(row["fwd_ms"]) if row.get("fwd_ms", "").strip() else None
            except ValueError:
                fwd = None
            try:
                bwd = float(row["bwd_ms"]) if row.get("bwd_ms", "").strip() else None
            except ValueError:
                bwd = None
            gf = int(row["group_fwd"]) if row.get("group_fwd", "").strip().isdigit() else 0
            rows.append({"batch": batch, "s_kv": p, "fwd_ms": fwd, "bwd_ms": bwd, "group_fwd": gf})
    rows.sort(key=lambda r: (int(r["batch"]), int(r["s_kv"])))

    if jax_path is not None and jax_path.exists():
        jm_bs, jm_p = load_jax_cross_timings(jax_path)
        for r in rows:
            b, p = int(r["batch"]), int(r["s_kv"])
            if (b, p) in jm_bs:
                r["jax_fwd_ms"], r["jax_bwd_ms"] = jm_bs[(b, p)]
            elif p in jm_p:
                r["jax_fwd_ms"], r["jax_bwd_ms"] = jm_p[p]
    return rows, batch_order


def fmt_ms(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}"


def fmt_ck_vs_mfma(ck: float | None, mfma: float) -> str:
    """Time ratio ck_pr_6764 / MFMA (>1 means ck slower than MFMA reference)."""
    if ck is None or ck <= 0 or mfma <= 0:
        return "—"
    return f"{ck / mfma:.2f}x"


def write_self_attn_md(timings: dict[int, tuple[float | None, float | None]], out: Path) -> None:
    lines: list[str] = [
        "# Performance Comparison: MFMA Kernel vs JAX vs TransformerEngine (CK) vs ck_pr_6764 (self-attention)",
        "",
        "**Config:** bs=2048, nheads=32, hdim=128, bfloat16, causal=False, seqlen_q == seqlen_kv",
        "",
        "**MFMA kernel selection:** mfma_4x4 for seq 1–4, mfma_16x16 for seq 5–17",
        "",
        "**ck_pr_6764:** `run_mha_performance_comparison_self_attn.sh` (asm v2 `fwd_v3=0`, `bwd_v3=0`).",
        "",
        "## Forward Pass (mean time in ms)",
        "",
        "| seq | JAX (ms) | TE/CK (ms) | MFMA (ms) | kernel | vs JAX | vs TE/CK | ck_pr_6764 (ms) | ck vs MFMA |",
        "|----:|---------:|-----------:|----------:|:-------|-------:|---------:|----------------:|-----------:|",
    ]

    mfma_fwd_rat_jax: list[float] = []
    mfma_fwd_rat_te: list[float] = []
    ck_fwd_rat_jax: list[float] = []
    ck_fwd_rat_te: list[float] = []
    ck_fwd_vs_mfma: list[float] = []

    for s in range(1, 18):
        jax, te, mfma = JAX_FWD_MS[s], TE_CK_FWD_MS[s], MFMA_FWD_MS[s]
        rj, rt = jax / mfma, te / mfma
        mfma_fwd_rat_jax.append(rj)
        mfma_fwd_rat_te.append(rt)
        ck_f, _ = timings.get(s, (None, None))
        ck_col = fmt_ms(ck_f)
        vm = fmt_ck_vs_mfma(ck_f, mfma)
        if ck_f is not None and ck_f > 0:
            ck_fwd_rat_jax.append(jax / ck_f)
            ck_fwd_rat_te.append(te / ck_f)
            ck_fwd_vs_mfma.append(ck_f / mfma)
        lines.append(
            f"| {s} | {jax:.3f} | {te:.3f} | {mfma:.3f} | {mfma_fwd_kernel(s)} | "
            f"{rj:.2f}x | {rt:.2f}x | {ck_col} | {vm} |"
        )

    lines += [
        "",
        "## Backward Pass (mean time in ms)",
        "",
        "| seq | JAX (ms) | TE/CK (ms) | MFMA (ms) | vs JAX | vs TE/CK | ck_pr_6764 (ms) | ck vs MFMA |",
        "|----:|---------:|-----------:|----------:|-------:|---------:|----------------:|-----------:|",
    ]

    mfma_bwd_rat_jax: list[float] = []
    mfma_bwd_rat_te: list[float] = []
    ck_bwd_rat_jax: list[float] = []
    ck_bwd_rat_te: list[float] = []
    ck_bwd_vs_mfma: list[float] = []

    for s in range(1, 18):
        jax, te, mfma = JAX_BWD_MS[s], TE_CK_BWD_MS[s], MFMA_BWD_MS[s]
        rj, rt = jax / mfma, te / mfma
        mfma_bwd_rat_jax.append(rj)
        mfma_bwd_rat_te.append(rt)
        _, ck_b = timings.get(s, (None, None))
        ck_col = fmt_ms(ck_b)
        vm = fmt_ck_vs_mfma(ck_b, mfma)
        if ck_b is not None and ck_b > 0:
            ck_bwd_rat_jax.append(jax / ck_b)
            ck_bwd_rat_te.append(te / ck_b)
            ck_bwd_vs_mfma.append(ck_b / mfma)
        lines.append(
            f"| {s} | {jax:.3f} | {te:.3f} | {mfma:.3f} | {rj:.2f}x | {rt:.2f}x | {ck_col} | {vm} |"
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
        gm = geom_mean_ratios(ck_fwd_vs_mfma)
        lines += [
            "### Forward (ck_pr_6764, geometric mean vs references)",
            f"- **vs JAX:** ~{gj:.1f}x",
            f"- **vs TE/CK:** ~{gt:.1f}x",
            f"- **ck vs MFMA** (time ratio): ~{gm:.1f}x",
            "",
        ]
    if len(ck_bwd_rat_jax) == 17:
        gj = geom_mean_ratios(ck_bwd_rat_jax)
        gt = geom_mean_ratios(ck_bwd_rat_te)
        gm = geom_mean_ratios(ck_bwd_vs_mfma)
        lines += [
            "### Backward (ck_pr_6764, geometric mean vs references)",
            f"- **vs JAX:** ~{gj:.1f}x",
            f"- **vs TE/CK:** ~{gt:.1f}x",
            f"- **ck vs MFMA** (time ratio): ~{gm:.1f}x",
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
        "- **ck vs MFMA** is ck_pr_6764 wall time / MFMA reference time (<1 means ck is faster than MFMA column)",
        "- **ck_pr_6764** timings: `mask=0`, BHSD (`iperm=1`, `operm=1`)",
        "",
    ]

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt_speedup_jax_over_ck(jax_ms: float | None, ck_ms: float | None) -> str:
    """jax_ms / ck_ms (>1 means ck_pr_6764 is faster than JAX unfused)."""
    if jax_ms is None or ck_ms is None or jax_ms <= 0 or ck_ms <= 0:
        return "—"
    return f"{jax_ms / ck_ms:.2f}x"


def write_cross_attn_md(
    rows: list[dict[str, str | int | float | None]],
    out: Path,
    batch_order: list[int],
) -> None:
    if not batch_order:
        batch_order = sorted({int(r["batch"]) for r in rows})
    any_group = any(int(r.get("group_fwd") or 0) == 1 for r in rows)

    lines: list[str] = [
        "# Cross-attention: CK vs JAX unfused",
        "",
        "**Layout:** CK **BHSD** (`benchmark_mha_fwd.cpp`, `benchmark_mha_bwd.cpp`: `-iperm=1 -operm=1`). JAX unfused **BSHD** (`jax_unfused_attention.py`).",
        "",
        "**Files & run:**",
        "- `run_mha_performance_comparison_cross_attn.sh` — CK + optional JAX, then `write_mha_performance_comparison_md.py --kind cross_attn`. From repo root: `cd op_tests/cpp/mha && ./run_mha_performance_comparison_cross_attn.sh`.",
        "- `run_jax_unfused_cross_attn_benchmark.py` — JAX timings only (env `JAX_UNFUSED_*` set by the shell).",
        "",
        "`jax/ck` = jax_time / ck_time (>1 ⇒ CK faster).",
        "",
        "Each **Configuration *n*** is one batch size **B**; numbering follows the order of batch blocks in the timing CSV (same as `CROSS_ATTN_BATCHES` in `run_mha_performance_comparison_cross_attn.sh` when that script produced the CSV).",
        "",
    ]
    if any_group:
        lines += [
            "> Some rows used `CROSS_ATTN_GROUP_FWD=1` (forward group mode); backward stays uniform `s_kv=P`.",
            "",
        ]

    for idx, b in enumerate(batch_order, start=1):
        batch_rows = sorted((r for r in rows if int(r["batch"]) == b), key=lambda r: int(r["s_kv"]))
        has_jax_b = any(r.get("jax_fwd_ms") is not None for r in batch_rows)

        lines += [f"## Configuration {idx} — B={b}", ""]

        lines += [
            "### Forward",
            "",
            "**CK:** `benchmark_mha_fwd`, bf16, BHSD, **B** as in section title, `s_q=1`, `s_kv=P`, `h=32`, `d=128`, `mask=0`, `fwd_v3=0`; warmup/repeat from `run_mha_performance_comparison_cross_attn.sh`.",
            "",
            "**JAX:** `run_jax_unfused_cross_attn_benchmark.py` + `jax_unfused_attention.py`, BSHD, same **B** / `s_q` / `s_kv` / `h` / `d`; non-causal; softmax `1/sqrt(d)` when `JAX_UNFUSED_SM_SCALE=ck`.",
            "",
        ]
        if has_jax_b:
            lines += [
                "| s_kv (P) | ck fwd (ms) | jax unfused fwd (ms) | jax/ck fwd | group fwd |",
                "|---------:|--------------:|---------------------:|-----------:|:---------:|",
            ]
            for r in batch_rows:
                p = int(r["s_kv"])
                gf = "yes" if int(r.get("group_fwd") or 0) == 1 else "no"
                fv = r.get("fwd_ms")
                jf = r.get("jax_fwd_ms")
                ck_f = float(fv) if isinstance(fv, (int, float)) else None
                jax_f = float(jf) if isinstance(jf, (int, float)) else None
                lines.append(
                    f"| {p} | {fmt_ms(ck_f)} | {fmt_ms(jax_f)} | {fmt_speedup_jax_over_ck(jax_f, ck_f)} | {gf} |"
                )
        else:
            lines += [
                "| s_kv (P) | ck fwd (ms) | group fwd |",
                "|---------:|--------------:|:---------:|",
            ]
            for r in batch_rows:
                p = int(r["s_kv"])
                gf = "yes" if int(r.get("group_fwd") or 0) == 1 else "no"
                fv = r.get("fwd_ms")
                fwd_s = fmt_ms(float(fv)) if isinstance(fv, (int, float)) else "—"
                lines.append(f"| {p} | {fwd_s} | {gf} |")

        lines += [
            "",
            "### Backward",
            "",
            "**CK:** `benchmark_mha_bwd`, bf16, BHSD, **B** as in section title, `s_q=1`, `s_kv=P`, `h=32`, `d=128`, `mask=0`, `bwd_v3=0`, `mode=0`; warmup/repeat from same shell.",
            "",
            "**JAX:** same script as forward; `vjp` on unfused forward, then `jit(pullback)` timed for `do`.",
            "",
        ]
        if has_jax_b:
            lines += [
                "| s_kv (P) | ck bwd (ms) | jax unfused bwd (ms) | jax/ck bwd | group fwd |",
                "|---------:|--------------:|---------------------:|-----------:|:---------:|",
            ]
            for r in batch_rows:
                p = int(r["s_kv"])
                gf = "yes" if int(r.get("group_fwd") or 0) == 1 else "no"
                bv = r.get("bwd_ms")
                jb = r.get("jax_bwd_ms")
                ck_b = float(bv) if isinstance(bv, (int, float)) else None
                jax_b = float(jb) if isinstance(jb, (int, float)) else None
                lines.append(
                    f"| {p} | {fmt_ms(ck_b)} | {fmt_ms(jax_b)} | {fmt_speedup_jax_over_ck(jax_b, ck_b)} | {gf} |"
                )
        else:
            lines += [
                "| s_kv (P) | ck bwd (ms) | group fwd |",
                "|---------:|--------------:|:---------:|",
            ]
            for r in batch_rows:
                p = int(r["s_kv"])
                gf = "yes" if int(r.get("group_fwd") or 0) == 1 else "no"
                bv = r.get("bwd_ms")
                bwd_s = fmt_ms(float(bv)) if isinstance(bv, (int, float)) else "—"
                lines.append(f"| {p} | {bwd_s} | {gf} |")

        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kind",
        choices=("self_attn", "cross_attn"),
        required=True,
    )
    ap.add_argument("--timings", type=Path, required=True, help="Input timings CSV")
    ap.add_argument(
        "--jax-timings",
        type=Path,
        default=None,
        help="Optional JAX unfused CSV (cross_attn): batch,s_kv,jax_fwd_ms,jax_bwd_ms (legacy: s_kv,jax_fwd_ms,jax_bwd_ms)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Output markdown path")
    args = ap.parse_args()

    if args.out is None:
        args.out = (
            here / "mha_performance_comparison_self_attn.md"
            if args.kind == "self_attn"
            else here / "mha_performance_comparison_cross_attn.md"
        )

    if args.kind == "self_attn":
        write_self_attn_md(load_self_timings(args.timings), args.out)
    else:
        jax_p = args.jax_timings if args.jax_timings is not None else None
        rows, batch_order = load_cross_timings(args.timings, jax_p)
        write_cross_attn_md(rows, args.out, batch_order)

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
