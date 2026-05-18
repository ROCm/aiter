#!/usr/bin/env python3
"""
Aggregate rocprofv3 PMC counter CSV (long-format) into a per-kernel,
per-counter summary, skip warmup, and print a comparison table.

Usage:
  python3 aggregate_pmc.py <pmc_counter_collection.csv> \
                          --kernels '<CK_pattern>' '<Triton_pattern>' \
                          --labels CK Triton \
                          --warmup N
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv_file", help="pmc_counter_collection.csv from rocprofv3")
    p.add_argument(
        "--kernels",
        nargs="+",
        default=["UnifiedAttentionKernel", "kernel_unified_attention_3d"],
        help="Substring patterns to identify each backend's kernel.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=["CK", "Triton"],
        help="Display labels for each pattern (same length as --kernels).",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup *dispatches per kernel* to skip (per counter).",
    )
    return p.parse_args()


def aggregate(csv_path: str, patterns, warmup: int):
    # data[label][counter] = list of values across dispatches (post-warmup)
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # dispatch_count[label][counter] tracks how many dispatches we've seen
    seen: dict[tuple[str, str], int] = defaultdict(int)

    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            name = row["Kernel_Name"]
            label = None
            for lbl, pat in patterns:
                if pat in name:
                    label = lbl
                    break
            if label is None:
                continue
            counter = row["Counter_Name"]
            try:
                value = float(row["Counter_Value"])
            except ValueError:
                continue
            key = (label, counter)
            seen[key] += 1
            if seen[key] <= warmup:
                continue
            data[label][counter].append(value)
    return data


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def print_table(data, labels):
    counters = sorted({c for d in data.values() for c in d.keys()})
    if not counters:
        print("No counters captured for the given kernels.")
        return

    # Header
    col_w = 22
    print(f"\n{'Counter':<28} | " + " | ".join(f"{lbl:>{col_w}}" for lbl in labels))
    print("-" * (28 + 3 + (col_w + 3) * len(labels)))
    for c in counters:
        row_vals = []
        for lbl in labels:
            xs = data.get(lbl, {}).get(c, [])
            row_vals.append(median(xs))
        print(
            f"{c:<28} | "
            + " | ".join(f"{v:>{col_w},.1f}" for v in row_vals)
        )
    print()


def derived_metrics(data, labels):
    """Compute derived metrics that summarise compute/memory mix."""

    def get(label, counter):
        xs = data.get(label, {}).get(counter, [])
        return median(xs) if xs else float("nan")

    rows = []
    for lbl in labels:
        active = get(lbl, "GRBM_GUI_ACTIVE")  # cycles GPU was active (max over CUs)
        valu = get(lbl, "SQ_INSTS_VALU")
        mfma = get(lbl, "SQ_INSTS_MFMA")
        lds = get(lbl, "SQ_INSTS_LDS")
        salu = get(lbl, "SQ_INSTS_SALU")
        vmem = get(lbl, "SQ_INSTS_VMEM")
        cvt = get(lbl, "SQ_INSTS_VALU_CVT")
        waves = get(lbl, "SQ_WAVES")
        wait_lds = get(lbl, "SQ_WAIT_INST_LDS")
        wait_any = get(lbl, "SQ_WAIT_INST_ANY")

        total_insts = sum(
            x for x in [valu, mfma, lds, salu, vmem] if x == x
        )

        def pct(x, total):
            return (x / total * 100) if total else float("nan")

        rows.append(
            (
                lbl,
                active,
                waves,
                pct(valu, total_insts),
                pct(mfma, total_insts),
                pct(lds, total_insts),
                pct(salu, total_insts),
                pct(vmem, total_insts),
                pct(cvt, valu) if valu == valu and valu > 0 else float("nan"),
                pct(wait_lds, active) if active == active and active > 0 else float("nan"),
                pct(wait_any, active) if active == active and active > 0 else float("nan"),
            )
        )

    headers = [
        "Backend",
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "%VALU",
        "%MFMA",
        "%LDS",
        "%SALU",
        "%VMEM",
        "%CVT/VALU",
        "%LDS_wait",
        "%any_wait",
    ]
    print("Derived metrics (median over post-warmup dispatches)")
    print("  - %X = SQ_INSTS_X / sum(SQ_INSTS_VALU+MFMA+LDS+SALU+VMEM)")
    print("  - %CVT/VALU = fraction of VALU that are fp8/bf16 cvt insts")
    print("  - %LDS_wait = SQ_WAIT_INST_LDS / GRBM_GUI_ACTIVE")
    print("  - %any_wait = SQ_WAIT_INST_ANY / GRBM_GUI_ACTIVE")
    print()
    fmt = "{:<10} {:>16,.0f} {:>10,.0f}" + " {:>7.2f}%" * 5 + " {:>10.2f}%" * 3
    print(
        f"{'Backend':<10} {'GRBM_GUI_ACTIVE':>16} {'SQ_WAVES':>10} "
        f"{'%VALU':>8} {'%MFMA':>8} {'%LDS':>8} {'%SALU':>8} {'%VMEM':>8} "
        f"{'%CVT/VALU':>11} {'%LDS_wait':>11} {'%any_wait':>11}"
    )
    print("-" * 130)
    for r in rows:
        print(fmt.format(*r))
    print()


def main():
    args = parse_args()
    if len(args.kernels) != len(args.labels):
        raise SystemExit("--kernels and --labels must have the same length")
    patterns = list(zip(args.labels, args.kernels))

    data = aggregate(args.csv_file, patterns, args.warmup)

    if not any(data.values()):
        print("No matching kernels found. Patterns:", patterns)
        return

    print_table(data, args.labels)
    derived_metrics(data, args.labels)


if __name__ == "__main__":
    main()
