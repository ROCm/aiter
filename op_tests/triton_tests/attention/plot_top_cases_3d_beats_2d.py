#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


def is_ok(status: str) -> bool:
    return status == "ok" or status.startswith("ok:")


def to_int(x: str) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def to_float(x: str) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class CaseCmp:
    idx: str
    count: int
    max_q: int
    max_k: int
    is_decode: bool
    ms_2d: float
    ms_3d: float
    abs_gain_ms: float
    rel_gain_pct: float


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect_cases(rows: list[dict[str, str]]) -> list[CaseCmp]:
    by_idx: dict[str, dict[str, dict[str, str]]] = {}
    meta: dict[str, dict[str, str]] = {}
    for r in rows:
        idx = r["idx"]
        b = r["backend"]
        by_idx.setdefault(idx, {})[b] = r
        meta.setdefault(idx, r)

    out: list[CaseCmp] = []
    for idx, mp in by_idx.items():
        r2 = mp.get("unified_force_2d")
        r3 = mp.get("unified_force_3d")
        if r2 is None or r3 is None:
            continue
        if not (is_ok(r2["status"]) and is_ok(r3["status"])):
            continue
        ms2 = to_float(r2["ms"])
        ms3 = to_float(r3["ms"])
        if ms2 is None or ms3 is None:
            continue
        if ms3 >= ms2:
            continue
        m = meta[idx]
        abs_gain = ms2 - ms3
        rel_gain = 100.0 * abs_gain / max(ms2, 1e-12)
        out.append(
            CaseCmp(
                idx=idx,
                count=to_int(m["count"]),
                max_q=to_int(m["max_seqlen_q"]),
                max_k=to_int(m["max_seqlen_k"]),
                is_decode=str(m.get("is_decode", "")).lower() == "true",
                ms_2d=ms2,
                ms_3d=ms3,
                abs_gain_ms=abs_gain,
                rel_gain_pct=rel_gain,
            )
        )
    return out


def plot_cases(cases: list[CaseCmp], out_png: Path, top_n: int, sort_by: str) -> None:
    if sort_by == "relative":
        cases = sorted(cases, key=lambda c: (c.rel_gain_pct, c.abs_gain_ms), reverse=True)
    else:
        cases = sorted(cases, key=lambda c: (c.abs_gain_ms, c.rel_gain_pct), reverse=True)
    cases = cases[:top_n]

    if not cases:
        print("No cases where 3D beats 2D.")
        return

    labels = [
        f"idx{c.idx}|{'dec' if c.is_decode else 'pre'}|q{c.max_q}|k{c.max_k}|n{c.count}"
        for c in cases
    ]
    ms2 = [c.ms_2d for c in cases]
    ms3 = [c.ms_3d for c in cases]

    fig, ax = plt.subplots(figsize=(max(12, top_n * 1.2), 6))
    xs = list(range(len(cases)))
    w = 0.38
    ax.bar([x - w / 2 for x in xs], ms2, width=w, label="unified_force_2d")
    ax.bar([x + w / 2 for x in xs], ms3, width=w, label="unified_force_3d")
    ax.set_title("Top cases where forced 3D beats forced 2D")
    ax.set_ylabel("ms")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot top cases where 3D outperforms 2D.")
    ap.add_argument("--csv", type=Path, required=True, help="Benchmark CSV path.")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/workspaces/workspace/attn_charts/top_cases_3d_beats_2d.png"),
        help="Output PNG path.",
    )
    ap.add_argument("--top-n", type=int, default=30, help="How many top cases to plot.")
    ap.add_argument(
        "--sort-by",
        choices=["absolute", "relative"],
        default="absolute",
        help="Sort by absolute ms gain or relative percent gain.",
    )
    args = ap.parse_args()

    rows = read_rows(args.csv)
    cases = collect_cases(rows)
    decode_cnt = sum(1 for c in cases if c.is_decode)
    prefill_cnt = len(cases) - decode_cnt
    print(f"cases_where_3d_beats_2d={len(cases)} (decode={decode_cnt}, prefill={prefill_cnt})")
    if cases:
        best_abs = max(cases, key=lambda c: c.abs_gain_ms)
        best_rel = max(cases, key=lambda c: c.rel_gain_pct)
        print(
            f"best_abs_gain: idx={best_abs.idx} gain_ms={best_abs.abs_gain_ms:.6f} "
            f"({best_abs.rel_gain_pct:.2f}%)"
        )
        print(
            f"best_rel_gain: idx={best_rel.idx} gain_ms={best_rel.abs_gain_ms:.6f} "
            f"({best_rel.rel_gain_pct:.2f}%)"
        )
    plot_cases(cases, args.out, top_n=args.top_n, sort_by=args.sort_by)
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
