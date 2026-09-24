# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for moe_gemm_mxfp8.

Usage:
    python bench_moe_gemm_mxfp8.py
    python bench_moe_gemm_mxfp8.py -metric bandwidth
"""

import argparse

import torch
import triton

from aiter.ops.triton.moe.moe_gemm_mxfp8 import moe_gemm_mxfp8
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# DSv4-style MoE shapes: (E, N, K, tokens_per_expert)
_SHAPES = [
    (8, 7168, 2048, 64, "DSv4-decode"),
    (8, 7168, 2048, 256, "DSv4-prefill-small"),
    (8, 7168, 2048, 1024, "DSv4-prefill-large"),
    (64, 2048, 7168, 64, "DSv4-E64-decode"),
]
_QBS = 32


def _make_mxfp8(E, N, K, tpe, device="cuda"):
    total = E * tpe
    gs = torch.full((E,), tpe, dtype=torch.int32, device=device)
    lhs = torch.randint(-3, 4, (total, K), dtype=torch.int8, device=device).to(
        torch.float8_e4m3fnuz
    )
    rhs = torch.randint(-3, 4, (E, N, K), dtype=torch.int8, device=device).to(
        torch.float8_e4m3fnuz
    )
    xs = torch.randint(120, 135, (total, K // _QBS), dtype=torch.uint8, device=device)
    ws = torch.randint(120, 135, (E, N, K // _QBS), dtype=torch.uint8, device=device)
    return lhs, rhs, xs, ws, gs


def benchmark(args):
    unit = "ms" if args.metric == "time" else "GB/s"
    x_vals = [(E, N, K, tpe, label) for E, N, K, tpe, label in _SHAPES]

    config = triton.testing.Benchmark(
        x_names=["E", "N", "K", "tpe", "label"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["mxfp8"],
        line_names=[f"mxfp8 ({unit})"],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(E, N, K, tpe, label, provider):
        total = E * tpe
        lhs, rhs, xs, ws, gs = _make_mxfp8(E, N, K, tpe)
        fn = lambda: moe_gemm_mxfp8(lhs, rhs, xs, ws, gs, quant_block_size=_QBS)
        # reads: lhs(total*K) + rhs(E*N*K) + xs(total*K/QBS) + ws(E*N*K/QBS)
        # writes: out(total*N)
        mem = (
            total * K
            + E * N * K
            + total * K // _QBS
            + E * N * K // _QBS
            + total * N * 2  # bf16 out
        )
        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MoE GEMM mxfp8", allow_abbrev=False
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
