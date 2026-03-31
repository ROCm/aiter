#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def plot_grouped(rows: list[dict[str, str]], out_png: Path, title: str, top_n: int) -> None:
    rows = [r for r in rows if r.get("status", "") == "ok"]
    rows = [r for r in rows if to_float(r.get("ms_2d", "")) is not None and to_float(r.get("ms_3d", "")) is not None]
    rows = sorted(rows, key=lambda r: abs(to_float(r["delta_ms_3d_minus_2d"])), reverse=True)[:top_n]
    if not rows:
        print(f"No rows to plot for {title}")
        return

    labels = [
        f"idx{r['idx']}|{r['phase']}|q{r['max_q']}|k{r['max_k']}"
        for r in rows
    ]
    ms2d = [to_float(r["ms_2d"]) for r in rows]
    ms3d = [to_float(r["ms_3d"]) for r in rows]

    xs = list(range(len(rows)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(12, len(rows) * 1.1), 5))
    ax.bar([x - w / 2 for x in xs], ms2d, width=w, label="unified_force_2d")
    ax.bar([x + w / 2 for x in xs], ms3d, width=w, label="unified_force_3d")
    ax.set_title(title)
    ax.set_ylabel("ms")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Wrote: {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot unified 2D vs 3D microbench results.")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("/workspaces/workspace/attn_charts"))
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    rows = read_rows(args.csv)
    all_ok = [r for r in rows if r.get("status", "") == "ok"]
    only_3d_better = [r for r in all_ok if r.get("winner", "") == "3d"]

    print(f"ok_rows={len(all_ok)} rows_where_3d_better={len(only_3d_better)}")

    plot_grouped(
        all_ok,
        args.out_dir / "top_cases_unified_2d_vs_3d_microbench.png",
        "Top cases by |delta| (unified forced 2D vs forced 3D)",
        top_n=args.top_n,
    )
    plot_grouped(
        only_3d_better,
        args.out_dir / "top_cases_unified_3d_better_only_microbench.png",
        "Cases where forced 3D beats forced 2D",
        top_n=max(1, args.top_n),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
