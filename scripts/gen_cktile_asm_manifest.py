#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import argparse
import csv
import os
import re
from typing import List


def extract_kernel_symbols(s_path: str) -> List[str]:
    pat = re.compile(r"^\s*\.amdhsa_kernel\s+(.+?)\s*$")
    out: List[str] = []
    with open(s_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            sym = m.group(1).strip()
            # Skip helper/runtime symbols that aren't GEMM entry kernels.
            if "flush_cache" in sym:
                continue
            out.append(sym)
    # Stable unique order
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def main() -> int:
    p = argparse.ArgumentParser(description="Generate i8gemm_cktile manifest rows from .s")
    p.add_argument("--s", required=True, help="Path to CKTile .s file")
    p.add_argument("--co-name", required=True, help="Target .co filename")
    p.add_argument("--tile-m", required=True, type=int)
    p.add_argument("--tile-n", required=True, type=int)
    p.add_argument("--tile-k", required=True, type=int)
    p.add_argument("--splitk", default=0, type=int, help="0/1 capability flag")
    p.add_argument("--block-size", default=256, type=int)
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--append", action="store_true", help="Append rows to existing CSV")
    args = p.parse_args()

    symbols = extract_kernel_symbols(args.s)
    if not symbols:
        raise RuntimeError(f"No .amdhsa_kernel symbols found in {args.s}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_header = not (args.append and os.path.exists(args.out))
    mode = "a" if args.append else "w"

    with open(args.out, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["tile_m", "tile_n", "tile_k", "splitK", "block_size", "knl_name", "co_name"])
        for sym in symbols:
            w.writerow([
                args.tile_m,
                args.tile_n,
                args.tile_k,
                args.splitk,
                args.block_size,
                sym,
                args.co_name,
            ])

    print(f"Wrote {len(symbols)} kernel rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
