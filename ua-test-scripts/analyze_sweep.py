#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Render a sweep_amir_shapes.csv into a readable Markdown summary.

Usage:
    ./analyze_sweep.py                                   # default CSV path
    ./analyze_sweep.py --pre  pre.csv  --post post.csv   # before/after compare
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "sweep_amir_shapes.csv"


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                r["batch"] = int(r["batch"])
                r["sq"] = int(r["sq"])
                r["sk"] = int(r["sk"])
                r["num_splits"] = int(r["num_splits"]) if r["num_splits"] else None
                r["ck_ms"] = float(r["ck_ms"]) if r["ck_ms"] else None
                r["triton_ms"] = float(r["triton_ms"]) if r["triton_ms"] else None
                r["speedup"] = float(r["speedup"]) if r["speedup"] else None
            except (ValueError, KeyError):
                pass
            rows.append(r)
    return rows


def _pct(sp: float) -> str:
    """Speedup as a +/-% delta vs Triton."""
    return f"{(sp - 1) * 100:+.0f}%"


def _emit_decode_grid(rows: list[dict]) -> str:
    """One row per batch, one column per Sk, cell = CK speedup over Triton."""
    decode = [r for r in rows if r["phase"] == "decode" and r["speedup"] is not None]
    batches = sorted({r["batch"] for r in decode})
    sks     = sorted({r["sk"]    for r in decode})

    lines = []
    hdr = "| batch \\ Sk | " + " | ".join(f"{sk:>7}" for sk in sks) + " |"
    sep = "|" + "---|" * (len(sks) + 1)
    lines.append(hdr); lines.append(sep)
    for b in batches:
        cells = []
        for sk in sks:
            r = next((r for r in decode if r["batch"]==b and r["sk"]==sk), None)
            if r is None or r["speedup"] is None:
                cells.append("   —   ")
            else:
                sp = r["speedup"]
                # Bold wins of >=10%, italicize losses
                tag = f"{sp:.2f}x"
                if sp < 0.95:
                    tag = f"**{tag}**"   # Triton wins meaningfully
                cells.append(f"{tag:>7}")
        lines.append(f"| {b:>4}      | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _emit_prefill_grid(rows: list[dict]) -> str:
    prefill = [r for r in rows if r["phase"] == "prefill" and r["speedup"] is not None]
    batches = sorted({r["batch"] for r in prefill})
    sks     = sorted({r["sk"]    for r in prefill})
    lines = []
    hdr = "| batch \\ Sq=Sk | " + " | ".join(f"{sk:>5}" for sk in sks) + " |"
    sep = "|" + "---|" * (len(sks) + 1)
    lines.append(hdr); lines.append(sep)
    for b in batches:
        cells = []
        for sk in sks:
            r = next((r for r in prefill if r["batch"]==b and r["sk"]==sk), None)
            if r is None or r["speedup"] is None:
                cells.append("   —   ")
            else:
                sp = r["speedup"]
                tag = f"{sp:.2f}x"
                if sp < 0.95:
                    tag = f"**{tag}**"
                cells.append(f"{tag:>5}")
        lines.append(f"| {b:>4}          | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _summary_stats(rows: list[dict], phase: str) -> str:
    cells = [r for r in rows if r["phase"] == phase and r["speedup"] is not None]
    if not cells:
        return f"_{phase}: no data_"
    sps = sorted(r["speedup"] for r in cells)
    wins  = sum(1 for s in sps if s >= 1.0)
    n     = len(sps)
    med   = sps[n // 2]
    mn, mx = min(sps), max(sps)
    geomean = 1.0
    for s in sps: geomean *= s
    geomean = geomean ** (1.0 / n)
    return (f"  - cells: {n}\n"
            f"  - CK wins: {wins}/{n} ({100*wins/n:.0f}%)\n"
            f"  - geomean speedup: {geomean:.2f}x\n"
            f"  - median:          {med:.2f}x\n"
            f"  - range:           {mn:.2f}x .. {mx:.2f}x")


def _emit_compare(pre: list[dict], post: list[dict]) -> str:
    """Show shapes where speedup changed by >=5% pre→post."""
    pre_idx  = {(r["phase"], r["batch"], r["sq"], r["sk"]): r for r in pre}
    post_idx = {(r["phase"], r["batch"], r["sq"], r["sk"]): r for r in post}
    keys = sorted(set(pre_idx) & set(post_idx))
    moved = []
    for k in keys:
        r0, r1 = pre_idx[k], post_idx[k]
        sp0, sp1 = r0.get("speedup"), r1.get("speedup")
        ns0, ns1 = r0.get("num_splits"), r1.get("num_splits")
        if sp0 is None or sp1 is None:
            continue
        if abs(sp1 - sp0) >= 0.05:
            moved.append((k, sp0, sp1, ns0, ns1,
                          r0.get("ck_ms"), r1.get("ck_ms")))
    if not moved:
        return "_No cell shifted by ≥5% — heuristic change is a no-op._"
    moved.sort(key=lambda x: x[2] - x[1], reverse=True)
    lines = ["| phase | batch | sq | sk | splits pre→post | CK ms pre→post | speedup pre→post |",
             "|---|---|---|---|---|---|---|"]
    for (phase, b, sq, sk), sp0, sp1, ns0, ns1, ck0, ck1 in moved:
        lines.append(
            f"| {phase} | {b} | {sq} | {sk} | {ns0}→{ns1} | "
            f"{ck0:.3f}→{ck1:.3f} ms | {sp0:.2f}x → {sp1:.2f}x ({_pct(sp1/sp0 - 0)}) |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="single CSV to report on")
    ap.add_argument("--pre",  help="pre-fix CSV for compare mode")
    ap.add_argument("--post", help="post-fix CSV for compare mode")
    args = ap.parse_args()

    if args.pre and args.post:
        pre  = _load(Path(args.pre))
        post = _load(Path(args.post))
        print("# Sweep — Before vs After heuristic fix\n")
        print("## Decode (post-fix) speedup grid (CK / Triton)\n")
        print(_emit_decode_grid(post))
        print("\n## Prefill (post-fix) speedup grid (CK / Triton)\n")
        print(_emit_prefill_grid(post))
        print("\n## Summary (post-fix)")
        print("### Decode:")
        print(_summary_stats(post, "decode"))
        print("### Prefill:")
        print(_summary_stats(post, "prefill"))
        print("\n## Shifts pre → post  (|Δspeedup| ≥ 0.05)\n")
        print(_emit_compare(pre, post))
        return

    rows = _load(Path(args.csv))
    print("# Sweep summary\n")
    print("## Decode speedup grid (CK / Triton)\n")
    print(_emit_decode_grid(rows))
    print("\n## Prefill speedup grid\n")
    print(_emit_prefill_grid(rows))
    print("\n## Aggregate")
    print("### Decode:")
    print(_summary_stats(rows, "decode"))
    print("### Prefill:")
    print(_summary_stats(rows, "prefill"))


if __name__ == "__main__":
    main()
