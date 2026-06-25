#!/usr/bin/env python3
"""Drive kernel-only rocprofv3 traces for the three decode top-k kernels.

For every (kernel, k, L) cell this launches a fresh process under
``rocprofv3 --kernel-trace`` so the trace contains only that kernel's
dispatches, then aggregates the measured (warm) kernel durations.

Timing is taken exclusively from the rocprof kernel trace timestamps
(End-Start, in ns) -- no host-side timing.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "op_tests" / "topk_decode_rocprof_runner.py"

# Substring/regex that identifies the kernel-under-test in the mangled name.
KERNEL_MATCH = {
    "flydsl": re.compile(r"topk_per_row_decode_radix_unordered"),
    "flydsl_coop_atomic": re.compile(r"topk_per_row_decode_radix_coop_k512_.*atomic_hist"),
    "flydsl_coop_partial": re.compile(r"topk_per_row_decode_radix_coop_k512_.*partial_hist"),
    "flydsl_coop_local_merge": re.compile(
        r"topk_per_row_decode_radix_coop_local_topk_merge_k512"
    ),
    "flydsl_coop_local_merge_p4": re.compile(
        r"topk_per_row_decode_radix_coop_local_topk_merge_k512"
    ),
    "flydsl_coop_local_merge_p8": re.compile(
        r"topk_per_row_decode_radix_coop_local_topk_merge_k512"
    ),
    "flydsl_coop_local_merge_p16": re.compile(
        r"topk_per_row_decode_radix_coop_local_topk_merge_k512"
    ),
    "flydsl_coop_local_merge_p32": re.compile(
        r"topk_per_row_decode_radix_coop_local_topk_merge_k512"
    ),
    "flydsl_aiter_persistent": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g\d+"
    ),
    "flydsl_aiter_persistent_stage1": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g\d+_v\d+_stage1"
    ),
    "flydsl_aiter_persistent_stage2": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g\d+_v\d+_stage2"
    ),
    "flydsl_aiter_persistent_stage4": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g\d+_v\d+_stage4"
    ),
    "flydsl_aiter_persistent_stage8": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g\d+_v\d+_stage8"
    ),
    "flydsl_aiter_persistent_tiered": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g\d+_v\d+_stage\d+.*_tiered"
    ),
    "flydsl_aiter_persistent_p2": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g2"
    ),
    "flydsl_aiter_persistent_p4": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g4"
    ),
    "flydsl_aiter_persistent_p8": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g8"
    ),
    "flydsl_aiter_persistent_p16": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g16"
    ),
    "flydsl_aiter_persistent_p30": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g30"
    ),
    "flydsl_aiter_persistent_p32": re.compile(
        r"topk_per_row_decode_aiter_persistent_k512_bpp\d+_g32"
    ),
    "aiter_hip": re.compile(r"radix_topk_one_block_kernel|radix_kernel|last_filter_kernel"),
    "vllm": re.compile(r"topKPerRowDecode"),
}

KERNEL_ENV = {
    "flydsl_coop_atomic": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "8",
        "FLYDSL_TOPK_COOP_MODE": "histogram",
        "FLYDSL_TOPK_COOP_ATOMIC_HIST": "1",
    },
    "flydsl_coop_partial": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "8",
        "FLYDSL_TOPK_COOP_MODE": "histogram",
        "FLYDSL_TOPK_COOP_ATOMIC_HIST": "0",
    },
    "flydsl_coop_local_merge": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "8",
        "FLYDSL_TOPK_COOP_MODE": "local_topk_merge",
    },
    "flydsl_coop_local_merge_p4": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "4",
        "FLYDSL_TOPK_COOP_MODE": "local_topk_merge",
    },
    "flydsl_coop_local_merge_p8": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "8",
        "FLYDSL_TOPK_COOP_MODE": "local_topk_merge",
    },
    "flydsl_coop_local_merge_p16": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "16",
        "FLYDSL_TOPK_COOP_MODE": "local_topk_merge",
    },
    "flydsl_coop_local_merge_p32": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "32",
        "FLYDSL_TOPK_COOP_MODE": "local_topk_merge",
    },
    "flydsl_aiter_persistent": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
    "flydsl_aiter_persistent_stage1": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
        "FLYDSL_TOPK_AITER_PERSISTENT_SCAN_STAGES": "1",
    },
    "flydsl_aiter_persistent_stage2": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
        "FLYDSL_TOPK_AITER_PERSISTENT_SCAN_STAGES": "2",
    },
    "flydsl_aiter_persistent_stage4": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
        "FLYDSL_TOPK_AITER_PERSISTENT_SCAN_STAGES": "4",
    },
    "flydsl_aiter_persistent_stage8": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
        "FLYDSL_TOPK_AITER_PERSISTENT_SCAN_STAGES": "8",
    },
    "flydsl_aiter_persistent_tiered": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent_tiered",
    },
    "flydsl_aiter_persistent_p2": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "2",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
    "flydsl_aiter_persistent_p4": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "4",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
    "flydsl_aiter_persistent_p8": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "8",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
    "flydsl_aiter_persistent_p16": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "16",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
    "flydsl_aiter_persistent_p30": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "30",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
    "flydsl_aiter_persistent_p32": {
        "FLYDSL_TOPK_COOP": "1",
        "FLYDSL_TOPK_COOP_MIN_ROW_LEN": "0",
        "FLYDSL_TOPK_COOP_PARTITIONS": "32",
        "FLYDSL_TOPK_COOP_MODE": "aiter_persistent",
    },
}


def run_cell(
    kernel: str,
    k: int,
    L: int,
    num_rows: int,
    max_width: int | None,
    iters: int,
    warmup: int,
    outdir: Path,
    distribution: str,
    pmc: list[str] | None,
    pmc_kernel_regex: str | None,
    stat: str = "median",
) -> tuple[float, int, str]:
    width_tag = "" if max_width is None else f"_W{max_width}"
    tag = f"{kernel}_k{k}_rows{num_rows}_L{L}{width_tag}"
    out_file = outdir / tag
    for f in glob.glob(str(out_file) + "*"):
        os.remove(f)
    cmd = [
        "rocprofv3", "--kernel-trace", "--output-format", "csv",
        "--output-file", str(out_file),
    ]
    if pmc:
        cmd += ["--pmc", *pmc]
    if pmc_kernel_regex:
        cmd += ["--kernel-include-regex", pmc_kernel_regex]
    cmd += [
        "--",
        sys.executable, str(RUNNER),
        "--kernel", kernel, "--k", str(k), "--L", str(L),
        "--num-rows", str(num_rows),
        "--iters", str(iters), "--warmup", str(warmup),
        "--distribution", distribution,
    ]
    if max_width is not None:
        cmd += ["--max-width", str(max_width)]
    env = os.environ.copy()
    env.update(KERNEL_ENV.get(kernel, {}))
    log = subprocess.run(cmd, capture_output=True, text=True, env=env)
    csvs = glob.glob(str(out_file) + "*kernel_trace.csv")
    if not csvs:
        return float("nan"), 0, f"no trace ({log.returncode}): {log.stderr[-200:]}"

    rows = list(csv.DictReader(open(csvs[0])))
    pat = KERNEL_MATCH[kernel]
    matched = [r for r in rows if pat.search(r["Kernel_Name"])]
    if not matched:
        names = {r["Kernel_Name"][:60] for r in rows}
        return float("nan"), 0, f"no matching kernel; saw {names}"

    matched.sort(key=lambda r: int(r["Start_Timestamp"]))
    total_launches = warmup + iters
    kpc = max(1, len(matched) // total_launches)  # kernels per call (multi-block > 1)
    # Drop warmup launches, keep the measured tail.
    measured = matched[warmup * kpc:]
    durs_ns = [int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in measured]
    # Per-call time = sum of the kpc kernels in each call.
    per_call = [sum(durs_ns[i:i + kpc]) for i in range(0, len(durs_ns), kpc)]
    if not per_call:
        return float("nan"), kpc, "no measured launches"
    # ``min`` is robust to noisy co-tenant GPU contention (sharing CUs only ever
    # inflates a kernel's measured duration), while ``median`` is the default.
    reduce_fn = min if stat == "min" else statistics.median
    note = "" if kpc == 1 else f"{kpc} kernels/call summed"
    if pmc:
        pmc_csvs = glob.glob(str(out_file) + "*counter_collection.csv")
        if not pmc_csvs:
            note = f"{note}; no counter CSV".strip("; ")
        else:
            counter_rows = list(csv.DictReader(open(pmc_csvs[0])))
            counter_matched = [r for r in counter_rows if pat.search(r["Kernel_Name"])]
            if not counter_matched:
                names = {r["Kernel_Name"][:60] for r in counter_rows}
                note = f"{note}; no matching counters; saw {names}".strip("; ")
            else:
                resource = counter_matched[-1]
                counter_totals: dict[str, float] = {}
                for row in counter_matched:
                    counter_totals[row["Counter_Name"]] = counter_totals.get(
                        row["Counter_Name"], 0.0
                    ) + float(row["Counter_Value"])
                counter_note = ", ".join(
                    f"{name}={value:.0f}" for name, value in sorted(counter_totals.items())
                )
                note = (
                    f"{note}; VGPR={resource.get('VGPR_Count')}, "
                    f"SGPR={resource.get('SGPR_Count')}, "
                    f"LDS={resource.get('LDS_Block_Size')}, {counter_note}"
                ).strip("; ")
    return reduce_fn(per_call) / 1000.0, kpc, note  # us


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="+", default=["flydsl", "aiter_hip", "vllm"])
    ap.add_argument("--k", type=int, nargs="+", default=[512])
    # Dense L grid for the K=512 tiered decode study; extra density around the
    # 12K-16K short/cooperative crossover and across the long tier.
    ap.add_argument(
        "--L",
        type=int,
        nargs="+",
        default=[
            1024, 2048, 4096, 8192, 12288, 14336, 16384,
            24576, 32768, 49152, 65536, 120000,
        ],
    )
    ap.add_argument("--num-rows", type=int, default=1)
    ap.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Physical logits width for padded-width runs; defaults to each L.",
    )
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--stat",
        choices=["median", "min"],
        default="median",
        help="Reduction over per-call kernel durations (min is robust to GPU contention).",
    )
    ap.add_argument("--distribution", default="random")
    ap.add_argument("--outdir", default="/tmp/topk_rocprof")
    ap.add_argument(
        "--pmc",
        nargs="*",
        default=None,
        help="Optional rocprofv3 counters to collect with kernel trace.",
    )
    ap.add_argument(
        "--pmc-kernel-regex",
        default=None,
        help="Optional rocprofv3 kernel include regex; default captures all and filters post-hoc.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results: dict[tuple[str, int, int], tuple[float, int, str]] = {}
    for kernel in args.kernels:
        for k in args.k:
            for L in args.L:
                res = run_cell(
                    kernel,
                    k,
                    L,
                    args.num_rows,
                    args.max_width,
                    args.iters,
                    args.warmup,
                    outdir,
                    args.distribution,
                    args.pmc,
                    args.pmc_kernel_regex,
                    args.stat,
                )
                results[(kernel, k, L)] = res
                print(
                    f"[{kernel:9s} k={k:4d} rows={args.num_rows:2d} L={L:6d}] "
                    f"{res[0]:7.2f} us  {res[2]}",
                    flush=True,
                )

    # Markdown table: rows = (k, L), columns = kernels.
    print(
        "\n## Kernel-only %s latency (us), distribution=%s, num_rows=%d, max_width=%s, iters=%d"
        % (args.stat, args.distribution, args.num_rows, args.max_width or "L", args.iters)
    )
    header = "| k | num_rows | L | " + " | ".join(args.kernels) + " |"
    sep = "|" + "---|" * (3 + len(args.kernels))
    print(header)
    print(sep)
    for k in args.k:
        for L in args.L:
            cells = []
            for kernel in args.kernels:
                v = results[(kernel, k, L)][0]
                cells.append("na" if v != v else f"{v:.2f}")
            print(f"| {k} | {args.num_rows} | {L} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
