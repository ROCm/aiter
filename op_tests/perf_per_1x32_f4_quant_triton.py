# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark ``per_1x32_f4_quant_triton`` (Triton mxfp4 quant, quant.py:616).

Profiles the dynamic per-1x32 fp4 quantization kernel across representative MoE
activation shapes using ``aiter.test_common.run_perftest`` (device-time, with
arg rotation to defeat caching), and reports effective memory bandwidth.

Run on the GPU box:
    python op_tests/perf_per_1x32_f4_quant_triton.py
    python op_tests/perf_per_1x32_f4_quant_triton.py -m 256 -n 4096
    python op_tests/perf_per_1x32_f4_quant_triton.py --shuffle --iters 200
"""

import argparse
import os
import sys

# Allow `import aiter` on boxes with triton<3.6 (gluon's hard requirement) so
# aiter.test_common.run_perftest is importable. Harmless otherwise.
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import torch
from aiter import dtypes
from aiter.ops.quant import per_1x32_f4_quant_triton
from aiter.test_common import run_perftest


def _bench_one(M, N, dtype, shuffle, iters, warmup, rotate, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((M, N), generator=g, device="cuda", dtype=dtype)

    (y, scale), avg_us = run_perftest(
        per_1x32_f4_quant_triton,
        x,
        quant_dtype=dtypes.fp4x2,
        shuffle=shuffle,
        num_iters=iters,
        num_warmup=warmup,
        num_rotate_args=rotate,
    )

    # Bytes moved: read x (M*N*elem), write packed fp4 (M*N/2) + e8m0 scale (M*N/32).
    in_bytes = x.numel() * x.element_size()
    out_bytes = (M * N) // 2 + (M * N) // 32
    total_gb = (in_bytes + out_bytes) / 1e9
    bw = total_gb / (avg_us * 1e-6) if avg_us > 0 else 0.0

    print(
        f"  {str(dtype):14} {M:>6}x{N:<6} shuffle={int(shuffle)} "
        f"y{tuple(y.shape)} s{tuple(scale.shape)} "
        f"-> {avg_us:8.2f} us/iter   {bw:7.1f} GB/s"
    )
    return avg_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--rows", type=int, action="append", default=None)
    ap.add_argument("-n", "--cols", type=int, action="append", default=None)
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="benchmark the shuffle=True scale layout (default: both off/on)",
    )
    ap.add_argument("--iters", type=int, default=101)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--rotate",
        type=int,
        default=4,
        help="num_rotate_args for run_perftest (defeat L2 reuse)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA/ROCm GPU")

    # Default sweep: decode (M=1), small/medium batch, real MoE hidden dims.
    if args.rows and args.cols:
        shapes = [(m, n) for m in args.rows for n in args.cols]
    else:
        shapes = [(1, 4096), (64, 4096), (160, 2880), (256, 512), (4096, 7168)]

    shuffles = (True,) if args.shuffle else (False, True)

    print(
        "Benchmarking per_1x32_f4_quant_triton "
        f"(iters={args.iters}, warmup={args.warmup}, rotate={args.rotate})\n"
    )
    for dtype in (torch.bfloat16, torch.float16):
        for shuffle in shuffles:
            for M, N in shapes:
                _bench_one(
                    M, N, dtype, shuffle, args.iters, args.warmup, args.rotate
                )
    print("\ndone")
    sys.exit(0)


if __name__ == "__main__":
    main()
