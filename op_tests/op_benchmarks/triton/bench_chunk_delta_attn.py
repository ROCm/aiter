# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for chunk_delta_attn forward pass.

Sweeps over sequence length (T) by default; supports custom shapes via --shape.

Usage examples
--------------
# Sweep default shapes, report time / TFLOPS / BW
python bench_chunk_delta_attn.py

# Single shape: B=2 T=4096 H=16 K=64 V=64
python bench_chunk_delta_attn.py --shape 2 4096 16 64 64

# Save CSV
python bench_chunk_delta_attn.py -o
"""

import argparse
import math
import os
import sys

# Skip CK/HIP native .so loading – Triton kernels only
os.environ.setdefault("AITER_TRITON_ONLY", "1")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")


# Ensure repo root is on the path when running this script directly
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import triton

from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_delta_attn_fwd
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (B, T, H, K, V)  – representative prefill shapes
DEFAULT_SHAPES = [
    (1, 512, 16, 64, 64),
    (1, 1024, 16, 64, 64),
    (1, 2048, 16, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 16, 64, 64),
    (1, 16384, 16, 64, 64),
    (2, 2048, 16, 64, 64),
    (2, 4096, 16, 64, 64),
    (4, 2048, 16, 64, 64),
    (4, 4096, 16, 64, 64),
    # K=V=128, sweep H and T
    (1, 8192, 32, 128, 128),
    (1, 8192, 64, 128, 128),
    (1, 8192, 96, 128, 128),
    (1, 8192, 128, 128, 128),
    (1, 16384, 32, 128, 128),
    (1, 16384, 64, 128, 128),
    (1, 16384, 96, 128, 128),
    (1, 16384, 128, 128, 128),
]

CHUNK_SIZE = 64
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_inputs(B, T, H, K, V, HV=None):
    """Create raw benchmark inputs (l2norm + beta sigmoid applied inside fn)."""
    if HV is None:
        HV = H
    scale = 1.0 / math.sqrt(K)
    torch.manual_seed(0)
    q = torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, T, HV, V, device=DEVICE, dtype=DTYPE)
    g = torch.randn(B, T, HV, K, device=DEVICE, dtype=DTYPE) * 0.1
    beta = torch.randn(B, T, HV, device=DEVICE, dtype=DTYPE)
    A_log = torch.randn(HV, device=DEVICE, dtype=torch.float32).abs() * 0.5
    dt_bias = torch.randn(HV * K, device=DEVICE, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, A_log, dt_bias, scale


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


def run_benchmark(args):
    if args.shape is not None:
        B, T, H, K, V = args.shape
        x_vals_list = [(B, T, H, K, V)]
    else:
        x_vals_list = DEFAULT_SHAPES

    header = f"{'B':>4} {'T':>6} {'H':>4} {'K':>4} {'V':>4}  {'Time(ms)':>10}  {'TFLOPS':>8}  {'BW(GB/s)':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for B, T, H, K, V in x_vals_list:
        HV = H
        q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(B, T, H, K, V, HV)

        def fn(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale, A_log=A_log, dt_bias=dt_bias
        ):
            return chunk_delta_attn_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                chunk_size=CHUNK_SIZE,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=-5.0,
                safe_gate=True,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
            )

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
        tflops = _flops(B, T, H, K, V, CHUNK_SIZE) / ms * 1e-9
        bw = _bytes(B, T, H, K, V, HV) / (ms * 1e-3) * 1e-9

        print(
            f"{B:>4} {T:>6} {H:>4} {K:>4} {V:>4}  {ms:>10.4f}  {tflops:>8.2f}  {bw:>10.1f}"
        )
        rows.append((B, T, H, K, V, ms, tflops, bw))

    if args.o:
        import csv

        fname = f"{get_caller_name_no_ext()}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["B", "T", "H", "K", "V", "Time_ms", "TFLOPS", "BW_GBs"])
            w.writerows(rows)
        print(f"\nSaved to {fname}")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="Benchmark chunk_delta_attn forward",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        metavar=("B", "T", "H", "K", "V"),
        help="Single shape to benchmark instead of the default sweep.",
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
        help="Warmup budget in ms (time-based, ensures JIT/Gluon compile completes).",
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
