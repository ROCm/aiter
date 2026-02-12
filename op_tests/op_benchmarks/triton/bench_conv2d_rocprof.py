#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Minimal benchmark script for rocprof profiling.

Usage:
    rocprof --stats python bench_conv2d_rocprof.py --layout nchw
    rocprof --stats python bench_conv2d_rocprof.py --layout nhwc
    rocprof --stats python bench_conv2d_rocprof.py --layout torch

Then inspect rocprof_results.stats.csv for kernel timings.
"""

import argparse
import torch
import torch.nn.functional as F

from aiter.ops.triton.conv2d import conv2d as triton_conv2d

# Config: B=1, H=W=27, C_in=256, C_out=128, k=3x3, stride=1, pad=1
B, C_in, H, W = 1, 256, 27, 27
C_out, kH, kW = 128, 3, 3
STRIDE, PAD = 1, 1
WARMUP, REPS = 10, 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["nchw", "nhwc", "torch"], default="nchw")
    args = parser.parse_args()

    dtype = torch.float16

    w = torch.randn(C_out, C_in, kH, kW, dtype=dtype, device="cuda")
    b = torch.randn(C_out, dtype=dtype, device="cuda")

    if args.layout == "nchw":
        x = torch.randn(B, C_in, H, W, dtype=dtype, device="cuda")
        fn = lambda: triton_conv2d(x, w, b, stride=STRIDE, padding=PAD, layout="nchw")
    elif args.layout == "nhwc":
        x = torch.randn(B, H, W, C_in, dtype=dtype, device="cuda")
        fn = lambda: triton_conv2d(x, w, b, stride=STRIDE, padding=PAD, layout="nhwc")
    else:  # torch
        x = torch.randn(B, C_in, H, W, dtype=dtype, device="cuda")
        fn = lambda: F.conv2d(x, w, b, stride=STRIDE, padding=PAD)

    # Warmup
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    # Timed reps
    for _ in range(REPS):
        fn()
    torch.cuda.synchronize()

    print(f"Done: {args.layout}, {REPS} reps")


if __name__ == "__main__":
    main()
