# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Performance benchmark for the chunk_delta_attn backward pass.

Runs ``chunk_kimi_delta_attn`` with inputs that require grad and times the
forward (autograd path) and the backward separately. The backward always
recomputes the forward intermediates, so its time includes that recompute.

Usage examples
--------------
# Sweep default shapes
python bench_chunk_delta_attn_bwd.py

# Single shape: B=2 T=4096 H=16 K=64 V=64
python bench_chunk_delta_attn_bwd.py --shape 2 4096 16 64 64

# Without safe_gate (unbounded gate)
python bench_chunk_delta_attn_bwd.py --no-safe-gate

# Save CSV
python bench_chunk_delta_attn_bwd.py -o
"""

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

from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# (B, T, H, K, V)  – representative training shapes
DEFAULT_SHAPES = [
    (1, 2048, 16, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 16, 64, 64),
    (2, 4096, 16, 64, 64),
    (4, 4096, 16, 64, 64),
    # K=V=128, sweep H and T
    (1, 4096, 32, 128, 128),
    (1, 8192, 32, 128, 128),
    (1, 8192, 64, 128, 128),
    (2, 8192, 32, 128, 128),
]

DTYPE = torch.bfloat16
DEVICE = "cuda"


def _make_inputs(B, T, H, K, V):
    """Raw leaf inputs that require grad (gate, l2norm and beta sigmoid in kernel)."""
    torch.manual_seed(0)
    x = {
        "q": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE),
        "k": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE),
        "v": torch.randn(B, T, H, V, device=DEVICE, dtype=DTYPE),
        "g": torch.randn(B, T, H, K, device=DEVICE, dtype=DTYPE) * 0.1,
        "beta": torch.randn(B, T, H, device=DEVICE, dtype=DTYPE),
        "A_log": torch.randn(H, device=DEVICE, dtype=torch.float32).abs() * 0.5,
        "dt_bias": torch.randn(H * K, device=DEVICE, dtype=torch.float32) * 0.1,
    }
    return {n: t.requires_grad_(True) for n, t in x.items()}


def _warmup(fn, warmup_ms):
    """Run ``fn`` until ``warmup_ms`` elapsed, so JIT compiles complete."""
    elapsed = 0.0
    while elapsed < warmup_ms:
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        elapsed += t0.elapsed_time(t1)


def run_benchmark(args):
    if args.shape is not None:
        x_vals_list = [tuple(args.shape)]
    else:
        x_vals_list = DEFAULT_SHAPES

    header = f"{'B':>4} {'T':>6} {'H':>4} {'K':>4} {'V':>4}  {'Fwd(ms)':>10}  {'Bwd(ms)':>10}"
    print(header)
    print("-" * len(header))

    rows = []
    for B, T, H, K, V in x_vals_list:
        leaves = _make_inputs(B, T, H, K, V)
        kwargs = {
            "scale": 1.0 / math.sqrt(K),
            "use_gate_in_kernel": True,
            "use_qk_l2norm_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "safe_gate": not args.no_safe_gate,
            "lower_bound": None if args.no_safe_gate else -5.0,
        }

        def fwd(leaves=leaves, kwargs=kwargs):
            return chunk_kimi_delta_attn(**leaves, **kwargs)[0]

        o = fwd()
        do = torch.randn_like(o)
        inputs = tuple(leaves.values())

        def bwd(o=o, do=do, inputs=inputs):
            return torch.autograd.grad(o, inputs, do, retain_graph=True)

        _warmup(fwd, args.warmup_ms)
        _warmup(bwd, args.warmup_ms)
        fwd_ms = triton.testing.do_bench(fwd, warmup=0, rep=args.rep_ms)
        bwd_ms = triton.testing.do_bench(bwd, warmup=0, rep=args.rep_ms)

        print(f"{B:>4} {T:>6} {H:>4} {K:>4} {V:>4}  {fwd_ms:>10.4f}  {bwd_ms:>10.4f}")
        rows.append((B, T, H, K, V, fwd_ms, bwd_ms))

    if args.o:
        import csv

        fname = f"{get_caller_name_no_ext()}.csv"
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["B", "T", "H", "K", "V", "Fwd_ms", "Bwd_ms"])
            w.writerows(rows)
        print(f"\nSaved to {fname}")


def parse_args(args=None):
    parser = get_parser(kernel_name="chunk_delta_attn backward")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=5,
        metavar=("B", "T", "H", "K", "V"),
        help="Single shape to benchmark instead of the default sweep.",
    )
    parser.add_argument(
        "--no-safe-gate",
        action="store_true",
        help="Run without safe_gate (and without lower_bound).",
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
        help="Warmup budget in ms (time-based, ensures JIT compile completes).",
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
