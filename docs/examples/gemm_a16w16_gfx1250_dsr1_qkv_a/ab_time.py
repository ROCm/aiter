# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Interleaved old-vs-new timing of the config the loader picks for each M.

Both configs are timed in one process on one GPU, alternating in ABBA order per round, so clock
and thermal drift hit both alike; the result per M is the median over rounds.

    HIP_VISIBLE_DEVICES=0 python3 ab_time.py --out results/ab_gpu0.json
"""

import argparse
import json
import statistics

import torch
from common import (
    load_tables,
    make_cold_copies,
    make_inputs,
    resolve,
    time_with_cuda_graph,
)

MS = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--M", type=int, nargs="+", default=MS)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tables = load_tables()
    results = []
    for M in args.M:
        copies = make_cold_copies(make_inputs(M))
        picked = {label: resolve(table, M) for label, table in tables.items()}
        times = {"old": [], "new": []}
        for r in range(args.rounds):
            for label in ("old", "new") if r % 2 == 0 else ("new", "old"):
                times[label].append(time_with_cuda_graph(picked[label][1], copies)[0])
        old_us, new_us = (statistics.median(times[k]) for k in ("old", "new"))
        row = {"M": M, "copies": len(copies), "speedup": round(old_us / new_us, 3)}
        for label, us in (("old", old_us), ("new", new_us)):
            row[f"{label}_bucket"], row[f"{label}_cfg"] = picked[label]
            row[f"{label}_us"] = round(us, 3)
            row[f"{label}_spread"] = [
                round(min(times[label]), 3),
                round(max(times[label]), 3),
            ]
        results.append(row)
        print(
            f"M={M:6d} old {old_us:9.3f} us {row['old_spread']}  "
            f"new {new_us:9.3f} us {row['new_spread']}  {row['speedup']:.3f}x",
            flush=True,
        )
        del copies
        torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
