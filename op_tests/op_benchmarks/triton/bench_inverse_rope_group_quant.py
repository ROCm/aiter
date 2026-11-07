# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the inverse RoPE + FP8 group quant of attention output rows.

The defaults model the DeepSeek-V4.1 wo_a input: 64 heads of 512 values, a
64-value GPT-J rope tail, 8 output groups and MXFP8 scales. The benchmark
uses preallocated outputs, as a graph-captured caller does.

Usage:
    python bench_inverse_rope_group_quant.py -S 192,2234,16384 --metric time
"""

import argparse

import torch
import triton

from aiter.ops.triton.rope.inverse_rope_group_quant import inverse_rope_group_quant
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)


def parse_int_or_list(value):
    if "," in value:
        return [int(item) for item in value.split(",")]
    return int(value)


def benchmark_rows(rows, metric, args, **_):
    device = torch.device("cuda")
    H, D, G, qgs = args.num_heads, args.head_dim, args.num_groups, args.group_size
    o = torch.randn(rows, H, D, device=device).to(torch.bfloat16)
    positions = torch.randint(0, args.max_pos, (rows,), device=device)
    angle = torch.randn(args.max_pos, args.rope_dim // 2, device=device)
    cos_sin = torch.cat((angle.cos(), angle.sin()), dim=-1)
    cos, sin = cos_sin[:, : args.rope_dim // 2], cos_sin[:, args.rope_dim // 2 :]
    x_fp8 = torch.empty(rows, G, H * D // G, dtype=get_fp8_e4m3_dtype(), device=device)
    x_scale = torch.empty(
        rows, G, H * D // G // qgs, dtype=torch.float8_e8m0fnu, device=device
    )

    def run():
        inverse_rope_group_quant(
            o, positions, cos, sin, G, qgs, x_fp8=x_fp8, x_scale=x_scale
        )

    time_ms = triton.testing.do_bench(run, warmup=25, rep=100)
    if metric == "time":
        return time_ms
    if metric == "bandwidth":
        moved = sum(t.numel() * t.element_size() for t in (o, x_fp8, x_scale))
        return moved / (time_ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"Unknown metric: {metric}")


def main():
    parser = argparse.ArgumentParser(
        prog="Benchmark inverse RoPE + FP8 group quant",
        allow_abbrev=False,
    )
    parser.add_argument("-S", type=parse_int_or_list, default=[192, 2234, 16384])
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--rope-dim", type=int, default=64)
    parser.add_argument("--num-groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=32, choices=(32, 128))
    parser.add_argument("--max-pos", type=int, default=65536)
    parser.add_argument(
        "--metric", type=str, default="time", choices=["time", "bandwidth"]
    )
    parser.add_argument("-o", action="store_true", help="Write the results to CSV.")
    args = parser.parse_args()

    rows = args.S if isinstance(args.S, list) else [args.S]
    benchmark = triton.testing.Benchmark(
        x_names=["rows"],
        x_vals=rows,
        line_arg="metric",
        line_vals=[args.metric],
        line_names=["Time (ms)" if args.metric == "time" else "Bandwidth (GB/s)"],
        styles=[("green", "-")],
        ylabel="ms" if args.metric == "time" else "GB/s",
        plot_name=get_caller_name_no_ext(),
        args={"args": args},
    )
    triton.testing.perf_report([benchmark])(benchmark_rows).run(
        save_path="." if args.o else None, print_data=True
    )


if __name__ == "__main__":
    main()
