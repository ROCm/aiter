#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@dataclass
class RowDecision:
    idx: int
    observed_kind: str
    predicted_kind: str
    phase: str
    max_q: int
    max_k: int
    num_seqs: int
    num_q_heads: int
    num_kv_heads: int
    block_q: int
    total_num_q_blocks: int
    num_2d_prgms: int
    target_num_prgms: int
    sliding_window_gt_0: bool
    max_k_le_512: bool
    num_2d_gt_target: bool


def phase_name(max_q: int, window_size: tuple[int, int]) -> str:
    if max_q == 1:
        return "decode_windowed" if window_size != (-1, -1) else "decode_global"
    return "prefill"


def decide_row(r: dict[str, Any], num_sms: int) -> RowDecision:
    q_shape = tuple(r["q_shape"])
    k_shape = tuple(r["k_shape"])
    max_q = int(r["max_seqlen_q"])
    max_k = int(r["max_seqlen_k"])
    num_seqs = int(r["num_seqs"])
    num_q_heads = int(r["num_query_heads"])
    num_kv_heads = int(r["num_kv_heads"])
    window_size = tuple(r["window_size"])

    num_queries_per_kv = num_q_heads // num_kv_heads
    block_m = 16 if num_queries_per_kv <= 16 else next_pow2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv

    total_num_q_blocks = q_shape[0] // block_q + num_seqs
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    target_num_prgms = num_sms * 4

    sliding_window = 1 + int(window_size[0])
    sliding_window_gt_0 = sliding_window > 0
    max_k_le_512 = max_k <= 512
    num_2d_gt_target = num_2d_prgms > target_num_prgms
    use_2d = sliding_window_gt_0 or max_k_le_512 or num_2d_gt_target

    return RowDecision(
        idx=-1,
        observed_kind=str(r.get("kind", "unknown")),
        predicted_kind="2d" if use_2d else "3d",
        phase=phase_name(max_q, window_size),
        max_q=max_q,
        max_k=max_k,
        num_seqs=num_seqs,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        block_q=block_q,
        total_num_q_blocks=total_num_q_blocks,
        num_2d_prgms=num_2d_prgms,
        target_num_prgms=target_num_prgms,
        sliding_window_gt_0=sliding_window_gt_0,
        max_k_le_512=max_k_le_512,
        num_2d_gt_target=num_2d_gt_target,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Replay unified_attention selector on JSONL cases."
    )
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument(
        "--num-sms",
        type=int,
        default=304,
        help=(
            "GPU CU/SM count used by target_num_prgms = num_sms*4. "
            "Set this to your MI350 value if different."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/workspaces/workspace/unified_selector_analysis.csv"),
        help="Write per-row decision table CSV.",
    )
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with args.jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("No rows in JSONL.")
        return 1

    decisions: list[RowDecision] = []
    for i, r in enumerate(rows):
        d = decide_row(r, num_sms=args.num_sms)
        d.idx = i
        decisions.append(d)

    # Global agreement with observed kind.
    agree = sum(1 for d in decisions if d.observed_kind == d.predicted_kind)
    total = len(decisions)
    print(f"rows={total} num_sms={args.num_sms} target_num_prgms={args.num_sms * 4}")
    print(f"selector agreement vs observed kind: {agree}/{total} ({100.0*agree/total:.2f}%)")

    # Phase x predicted kind table.
    tbl: dict[str, Counter[str]] = defaultdict(Counter)
    for d in decisions:
        tbl[d.phase][d.predicted_kind] += 1
    print("\nPredicted kernel by phase:")
    print("phase            2d     3d    total")
    print("-------------------------------------")
    for phase in ["prefill", "decode_global", "decode_windowed"]:
        c2 = tbl[phase]["2d"]
        c3 = tbl[phase]["3d"]
        print(f"{phase:14s} {c2:5d} {c3:6d} {c2+c3:7d}")

    # Why 2d fired (predicted).
    why = Counter()
    for d in decisions:
        if d.predicted_kind != "2d":
            continue
        if d.sliding_window_gt_0:
            why["sliding_window_gt_0"] += 1
        elif d.max_k_le_512:
            why["max_k_le_512"] += 1
        elif d.num_2d_gt_target:
            why["num_2d_prgms_gt_target"] += 1
        else:
            why["other"] += 1
    print("\nPredicted 2D reason counts:")
    for k, v in why.items():
        print(f"- {k}: {v}")

    # Write detailed CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "phase",
                "observed_kind",
                "predicted_kind",
                "max_q",
                "max_k",
                "num_seqs",
                "num_q_heads",
                "num_kv_heads",
                "block_q",
                "total_num_q_blocks",
                "num_2d_prgms",
                "target_num_prgms",
                "sliding_window_gt_0",
                "max_k_le_512",
                "num_2d_prgms_gt_target",
            ]
        )
        for d in decisions:
            w.writerow(
                [
                    d.idx,
                    d.phase,
                    d.observed_kind,
                    d.predicted_kind,
                    d.max_q,
                    d.max_k,
                    d.num_seqs,
                    d.num_q_heads,
                    d.num_kv_heads,
                    d.block_q,
                    d.total_num_q_blocks,
                    d.num_2d_prgms,
                    d.target_num_prgms,
                    d.sliding_window_gt_0,
                    d.max_k_le_512,
                    d.num_2d_gt_target,
                ]
            )
    print(f"\nWrote: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
