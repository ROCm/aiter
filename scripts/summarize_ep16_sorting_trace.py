#!/usr/bin/env python3
import csv
import glob
import os
import sys


base = sys.argv[1]
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 20

print("token_per_rank,sorting_kernel_us")
for bs_dir in sorted(glob.glob(os.path.join(base, "bs*")), key=lambda p: int(os.path.basename(p)[2:])):
    bs = int(os.path.basename(bs_dir)[2:])
    rank_totals = []
    for path in glob.glob(os.path.join(bs_dir, "**", "*kernel_trace.csv"), recursive=True):
        groups = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                name = row["Kernel_Name"]
                duration = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
                if any(key in name for key in (
                    "opus_moe_sorting_entry",
                    "mxfp4_moe_sort_kernel",
                    "fused_mx_quant_moe_sort_kernel",
                )):
                    groups.setdefault(name, []).append(duration)
        if not groups:
            continue
        # The trace has 101 measured MoE calls. A kernel may launch once or
        # twice per call; infer that multiplicity from its occurrence count.
        total_ns = 0
        for durations in groups.values():
            launches_per_call = max(1, round(len(durations) / 101))
            total_ns += sum(durations[-tail * launches_per_call :])
        rank_totals.append(total_ns / tail / 1000.0)
    if rank_totals:
        print(f"{bs},{sum(rank_totals) / len(rank_totals):.4f}")
