# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Explicit gfx1250 TDM benchmark; no tuning-table or dispatch changes."""

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from aiter.ops.flydsl import flydsl_gemm_a16w16_tdm as gemm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="8192,8192,8192", help="M,N,K")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=31)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("Skipped: gfx1250 ROCm GPU required")
        return
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
        print("Skipped: this explicit TDM backend requires gfx1250")
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    m, n, k = map(int, args.shape.split(","))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    torch.set_num_threads(1)
    torch.manual_seed(83)
    a = torch.randn(m, k, device="cuda", dtype=dtype) * 0.1
    b = (torch.randn(n, k, device="cuda", dtype=dtype) * 0.1).t()
    ref = a.float() @ b.float()

    # Each baseline has two independently captured graphs. Randomize order.
    configurations = [
        ("simple", {"block_m": 256, "block_n": 256, "interleave": False}),
        ("simpleAA", {"block_m": 256, "block_n": 256, "interleave": False}),
        ("interleaved", {"block_m": 256, "block_n": 256}),
    ]
    if n % 2 == 0 and k % 1024 == 0 and k >= 2048:
        configurations += [
            (f"split{sk}", {"block_m": 128, "block_n": 128, "split_k": sk})
            for sk in (1, 2, 4, 8)
        ]
    graphs, errors = [], []
    outputs = []
    try:
        for _, cfg in configurations:
            y = gemm(a, b, group_m=4, **cfg)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(4):
                    y = gemm(a, b, group_m=4, **cfg)
            graph.replay()
            torch.cuda.synchronize()
            relative_l2 = ((y.float() - ref).norm() / ref.norm()).item()
            if not torch.isfinite(y).all() or relative_l2 >= 0.03:
                raise AssertionError(f"numerical check failed: {relative_l2}")
            graphs.append(graph)
            outputs.append(y)
            errors.append(relative_l2)
        samples = [[] for _ in graphs]
        rng = random.Random(222)
        for _ in range(args.repeats):
            order = list(range(len(graphs)))
            rng.shuffle(order)
            for i in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                samples[i].append(start.elapsed_time(end) * 1000 / 4)
        us = [statistics.median(s) for s in samples]
        result = {
            "shape": [m, n, k],
            "dtype": args.dtype,
            "configs": configurations,
            "us": us,
            "tflops": [2 * m * n * k / t / 1e6 for t in us],
            "relative_l2": errors,
            "samples_us": samples,
            "warning": "Packed atomic split-K is rounded and nondeterministic; see docs.",
        }
        print(json.dumps(result, indent=2))
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        for graph in graphs:
            graph.reset()


if __name__ == "__main__":
    main()
