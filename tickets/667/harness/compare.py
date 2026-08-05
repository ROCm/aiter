#!/usr/bin/env python3
"""Compare FlyDSL vs CK warp-decode benchmark results.

Parses stdout from bench_flydsl_wd.py and bench_ck_tile_warp_decode,
joins on (shape, B, kernel_category), and prints a Markdown comparison table.

Usage:
  # Capture outputs first:
  python bench_flydsl_wd.py --shapes deepseek-v3 qwen3next --batches 1 2 4 8 > flydsl.txt
  CK_WARP_DECODE_BENCH_SHAPES=deepseek-v3,qwen3next CK_WARP_DECODE_BENCH_BATCHES=1,2,4,8 \\
    /path/to/bench_ck_tile_warp_decode > ck.txt

  # Then compare:
  python compare.py --flydsl flydsl.txt --ck ck.txt

  # Or pipe both directly:
  python compare.py --flydsl <(python bench_flydsl_wd.py ...) --ck <(CK_... ./bench_ck_tile_warp_decode)

CK kernel tag → category mapping:
  gate_up_bf16, gate_bf16_d2  →  gate_up_bf16   (bf16 act × fp8 weight)
  gate_up_fp8, gate_fp8_d2    →  gate_up_fp8    (fp8 act × fp8 weight)
  down_h2_d2                  →  down_h2_d2
  down_fp4_h2                 →  down_fp4_h2

FlyDSL kernel tag → category mapping:
  gate_up_bf16x2              →  gate_up_bf16   (bf16 act × bf16 weight, Task 0)
"""

import argparse
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ── Row parsers ───────────────────────────────────────────────────────────────

# CK bench output format:
#   shape              B              kernel        ms     TFLOP/s      GB/s
#   deepseek-v3        1    gate_up_bf16        0.0123       12.34       567.8

_CK_ROW = re.compile(
    r"^(\S+)\s+(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)

# FlyDSL bench output format (same columns):
#   deepseek-v3            1  gate_up_bf16x2            0.0456      10.23       345.6

_FLYDSL_ROW = re.compile(
    r"^(\S+)\s+(\d+)\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


def _parse_rows(text: str, regex) -> List[Dict]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        m = regex.match(line)
        if m:
            rows.append(
                dict(
                    shape=m.group(1),
                    B=int(m.group(2)),
                    kernel=m.group(3),
                    ms=float(m.group(4)),
                    tflops=float(m.group(5)),
                    gbs=float(m.group(6)),
                )
            )
    return rows


# ── Category normalization ────────────────────────────────────────────────────

_CK_CATEGORY = {
    "gate_up_bf16": "gate_up_bf16",
    "gate_bf16_d2": "gate_up_bf16",
    "gate_up_fp8":  "gate_up_fp8",
    "gate_fp8_d2":  "gate_up_fp8",
    "down_h2_d2":   "down_h2_d2",
    "down_fp4_h2":  "down_fp4_h2",
    "down_reduce":  "down_reduce",
    "down_d2":      "down_d2",
}

_FLYDSL_CATEGORY = {
    "gate_up_bf16x2":  "gate_up_bf16",   # BF16×BF16 scaffold
    "gate_up_bf16":    "gate_up_bf16",
    "gate_bf16x_fp8w": "gate_up_bf16",   # BF16×FP8 — matches CK gate_bf16_d2
    "gate_up_fp8":     "gate_up_fp8",
    "down_h2_d2":      "down_h2_d2",
    "down_h2_fp8w_d2": "down_h2_d2",    # FP8 weights down H2 — matches CK down_h2_d2
    "down_fp4_h2":     "down_fp4_h2",
}


def _best_by_category(rows: List[Dict], cat_map: Dict) -> Dict[Tuple, Dict]:
    """For each (shape, B, category), keep the row with the lowest ms."""
    best = {}
    for row in rows:
        cat = cat_map.get(row["kernel"])
        if cat is None:
            continue
        key = (row["shape"], row["B"], cat)
        if key not in best or row["ms"] < best[key]["ms"]:
            best[key] = row
    return best


# ── Markdown table ────────────────────────────────────────────────────────────

def _print_table(ck_best, flydsl_best):
    # Collect all keys
    all_keys = sorted(set(ck_best.keys()) | set(flydsl_best.keys()))
    if not all_keys:
        print("No matching rows found.")
        return

    header = (
        f"| {'shape':<18} | {'B':>4} | {'category':<16} "
        f"| {'FlyDSL ms':>10} | {'FlyDSL GB/s':>12} "
        f"| {'CK ms':>10} | {'CK GB/s':>10} | {'FlyDSL/CK':>10} |"
    )
    sep = "|" + "|".join(["-" * (len(c) + 2) for c in header.split("|")[1:-1]]) + "|"
    print(header)
    print(sep)

    for shape, B, cat in all_keys:
        ck_row = ck_best.get((shape, B, cat))
        fd_row = flydsl_best.get((shape, B, cat))

        ck_ms_s  = f"{ck_row['ms']:.4f}"  if ck_row else "—"
        ck_gbs_s = f"{ck_row['gbs']:.1f}" if ck_row else "—"
        fd_ms_s  = f"{fd_row['ms']:.4f}"  if fd_row else "—"
        fd_gbs_s = f"{fd_row['gbs']:.1f}" if fd_row else "—"

        ratio_s = "—"
        if ck_row and fd_row and ck_row["ms"] > 0:
            ratio = fd_row["ms"] / ck_row["ms"]
            ratio_s = f"{ratio:.2f}×"

        print(
            f"| {shape:<18} | {B:>4} | {cat:<16} "
            f"| {fd_ms_s:>10} | {fd_gbs_s:>12} "
            f"| {ck_ms_s:>10} | {ck_gbs_s:>10} | {ratio_s:>10} |"
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare FlyDSL vs CK bench results")
    parser.add_argument("--flydsl", metavar="FILE", help="FlyDSL bench output file")
    parser.add_argument("--ck",     metavar="FILE", help="CK bench output file")
    args = parser.parse_args()

    if args.flydsl:
        with open(args.flydsl) as f:
            fd_text = f.read()
    else:
        print("No --flydsl file given; FlyDSL column will be empty.", file=sys.stderr)
        fd_text = ""

    if args.ck:
        with open(args.ck) as f:
            ck_text = f.read()
    else:
        print("No --ck file given; CK column will be empty.", file=sys.stderr)
        ck_text = ""

    fd_rows = _parse_rows(fd_text, _FLYDSL_ROW)
    ck_rows = _parse_rows(ck_text, _CK_ROW)

    fd_best = _best_by_category(fd_rows, _FLYDSL_CATEGORY)
    ck_best = _best_by_category(ck_rows, _CK_CATEGORY)

    print(f"\nParsed {len(fd_rows)} FlyDSL rows, {len(ck_rows)} CK rows.\n")
    _print_table(ck_best, fd_best)


if __name__ == "__main__":
    main()
