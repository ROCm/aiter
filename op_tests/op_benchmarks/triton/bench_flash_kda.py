# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for the FlashKDA two-kernel forward pass.

Sweeps sequence length and head count by default; supports custom shapes via
--shape. Only FlashKDA is timed -- run bench_chunk_delta_attn.py for the default
pipeline and compare the Time(ms) column. TFLOPS uses the same approximation as
that script but at C=32, the chunk size FlashKDA requires, so the two TFLOPS
columns count different amounts of intra-chunk work.

Usage examples
--------------
# Sweep default shapes, report time / TFLOPS / BW
python bench_flash_kda.py

# Single shape: B=1 T=16384 H=12 K=128 V=128
python bench_flash_kda.py --shape 1 16384 12 128 128

# Pin segment lengths instead of letting the heuristic choose
python bench_flash_kda.py --seg-sweep

# Save CSV
python bench_flash_kda.py -o
"""

import argparse
import math
import os
import sys

# Skip CK/HIP native .so loading – Triton kernels only
os.environ.setdefault("AITER_TRITON_ONLY", "1")


# Ensure repo root is on the path when running this script directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
    FLASH_KDA_CHUNK,
    flash_kda_fwd,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (B, T, H, K, V) – H=12 is the Kimi K3 shape this path was built for; the rest
# bracket it so a regression in the segmentation heuristic shows up as well as a
# kernel one. K and V are fixed at 128, the only width FlashKDA supports.
DEFAULT_SHAPES = [
    (1, 16384, 8, 128, 128),
    (1, 16384, 12, 128, 128),
    (1, 16384, 64, 128, 128),
    (1, 16384, 96, 128, 128),
    (1, 8192, 12, 128, 128),
    (1, 8192, 16, 128, 128),
    (1, 8192, 32, 128, 128),
    (1, 8192, 64, 128, 128),
    (1, 32768, 128, 128, 128),
    (1, 65536, 12, 128, 128),
    (2, 16384, 32, 128, 128),
    (4, 4096, 16, 128, 128),
    (8, 2048, 16, 128, 128),
]
# Segment lengths in chunks; 0 means one segment, i.e. segmentation off.
SEG_SWEEP = [0, 16, 32, 64, 128]

K_DIM = 128
LOWER_BOUND = -5.0
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_inputs(B, T, H, seed=42):
    """Create raw benchmark inputs (gate, l2norm and beta sigmoid are in-kernel)."""
    torch.manual_seed(seed)
    K = V = K_DIM
    return {
        "q": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE),
        "k": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE),
        "v": torch.randn(B, T, H, V, device=DEVICE, dtype=DTYPE),
        "g": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) * 0.1,
        "beta": torch.randn(B, T, H, device=DEVICE, dtype=torch.float32),
        "A_log": torch.randn(H, device=DEVICE, dtype=torch.float32).abs() * 0.5,
        "dt_bias": torch.randn(H * K, device=DEVICE, dtype=torch.float32) * 0.1,
        "scale": 1.0 / math.sqrt(K),
    }


def _flops(B, T, H, K, V, chunk_size):
    """
    Approximate FLOPs for the chunk_delta_attn forward pass.

    Dominant terms:
      - Intra-chunk QK:       B * T * H * chunk_size * K * 2
      - Intra-chunk AV:       B * T * H * chunk_size * V * 2
      - Inter-chunk KV state: B * T * H * K * V * 2
      - Output QH:            B * T * H * K * V * 2
    """
    return B * T * H * 2 * (chunk_size * K + chunk_size * V + 2 * K * V)


def _bytes(B, T, H, K, V, HV, elem_bytes=2):
    """Approximate memory traffic (inputs read + output written)."""
    read = (
        B * T * H * K * elem_bytes  # q
        + B * T * H * K * elem_bytes  # k
        + B * T * HV * V * elem_bytes  # v
        + B * T * HV * K * elem_bytes  # g
        + B * T * HV * elem_bytes  # beta
    )
    write = B * T * HV * V * elem_bytes  # o
    return read + write


def _seg_label(chunks_per_seg):
    if chunks_per_seg is None:
        return "auto"
    return "1seg" if chunks_per_seg <= 0 else str(chunks_per_seg)


def run_benchmark(args):
    if args.shape is not None:
        B, T, H, K, V = args.shape
        if K != K_DIM or V != K_DIM:
            raise SystemExit(f"flash_kda only supports K=V={K_DIM}, got K={K} V={V}")
        x_vals_list = [(B, T, H, K, V)]
    else:
        x_vals_list = DEFAULT_SHAPES

    header = f"{'B':>4} {'T':>6} {'H':>4} {'K':>4} {'V':>4}  {'Time(ms)':>10}  {'TFLOPS':>8}  {'BW(GB/s)':>10}"
    if args.seg_sweep:
        header += f"  {'seg':>6}"
    print(header)
    print("-" * len(header))

    rows = []
    for B, T, H, K, V in x_vals_list:
        HV = H
        t = _make_inputs(B, T, H)

        if args.seg_sweep:
            # A segment longer than the sequence would leave the grid empty.
            segs = [None] + [m for m in SEG_SWEEP if not m or m * FLASH_KDA_CHUNK <= T]
        else:
            segs = [None]

        for m in segs:

            def fn(t=t, m=m):
                return flash_kda_fwd(**t, lower_bound=LOWER_BOUND, chunks_per_seg=m)

            # Time-based warmup: run until WARMUP_MS elapsed so JIT/autotune completes.
            _elapsed = 0.0
            while _elapsed < args.warmup_ms:
                _t0 = torch.cuda.Event(enable_timing=True)
                _t1 = torch.cuda.Event(enable_timing=True)
                _t0.record()
                fn()
                _t1.record()
                torch.cuda.synchronize()
                _elapsed += _t0.elapsed_time(_t1)

            ms = triton.testing.do_bench(fn, warmup=0, rep=args.rep_ms)
            tflops = _flops(B, T, H, K, V, FLASH_KDA_CHUNK) / ms * 1e-9
            bw = _bytes(B, T, H, K, V, HV) / (ms * 1e-3) * 1e-9

            line = f"{B:>4} {T:>6} {H:>4} {K:>4} {V:>4}  {ms:>10.4f}  {tflops:>8.2f}  {bw:>10.1f}"
            row = [B, T, H, K, V, ms, tflops, bw]
            if args.seg_sweep:
                line += f"  {_seg_label(m):>6}"
                row.append(_seg_label(m))
            print(line)
            rows.append(tuple(row))

    if args.o:
        import csv

        fname = f"{get_caller_name_no_ext()}.csv"
        cols = ["B", "T", "H", "K", "V", "Time_ms", "TFLOPS", "BW_GBs"]
        if args.seg_sweep:
            cols.append("seg")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        print(f"\nSaved to {fname}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashKDA forward",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        metavar=("B", "T", "H", "K", "V"),
        help=f"Single shape to benchmark instead of the default sweep. K and V must be {K_DIM}.",
    )
    parser.add_argument(
        "--seg-sweep",
        action="store_true",
        help="Also time pinned segment lengths, which is how the defaults in "
        "_choose_chunks_per_seg were calibrated. Rerun after changing anything "
        "that moves the crossover: segment cost, the recurrence kernel's block "
        "count, or the hardware.",
    )
    parser.add_argument(
        "-o",
        action="store_true",
        help="Write results to a CSV file in the current directory.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=float,
        default=300.0,
        help="Warmup budget in ms (time-based, ensures JIT/autotune completes).",
    )
    parser.add_argument(
        "--rep-ms",
        type=float,
        default=500.0,
        help="Measurement budget in ms passed to triton.testing.do_bench.",
    )
    return parser.parse_args(args=args)


def main(args=None):
    run_benchmark(parse_args(args=args))


if __name__ == "__main__":
    main()
