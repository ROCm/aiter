# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused LayerNorm + table-indexed AdaLN modulation against the
eager expression, which materialises the normalised row and both gathered
modulation tables at [M, N].
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
    fused_layernorm_indexed_adaln,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_rmsnorm_indexed_adaln import make

_PROVIDERS = ("fused", "eager")
# [M, N] arrays moved. The kernel walks the row twice, once for the statistics
# and once to normalise and modulate, and writes the output: 2 reads + 1 write.
# Eager writes the normalised row, both gathered tables and the two products,
# and reads each one back. Its count is an upper bound; a cache hit on a short
# row moves less.
_ARRAYS = {"fused": 3, "eager": 12}
# (tokens, hidden, table rows) for the diffusion transformers this op targets.
# Tokens is batch x sequence. The table holds one row per batch sample.
_SHAPES = (
    (8192, 1152, 2),  # PixArt-XL 1024px
    (2048, 2432, 2),  # SD3.5-Large image stream
    (666, 2432, 2),  # SD3.5-Large context stream
    (32760, 1536, 1),  # Wan2.1-1.3B, 81 frames at 480p
    (4096, 3072, 1),  # FLUX.1 image stream
)
_HIDDEN = 1536


def bench_fn_impl(M, N, G, provider, metric, args):
    x, weight, shift, scale, indices, bias = make(
        M, N, G, torch.bfloat16, "cuda", runs=not args.scattered
    )
    if args.affine == "none":
        weight, bias = None, None
    elif args.affine == "weight":
        bias = None
    mem = _ARRAYS[provider] * M * N * x.element_size()

    if provider == "fused":
        out = torch.empty_like(x)

        def fn():
            return fused_layernorm_indexed_adaln(
                x, shift, scale, indices, weight=weight, bias=bias, out=out
            )

    else:

        def fn():
            normed = torch.nn.functional.layer_norm(x, (N,), weight, bias, 1e-6)
            return normed * (1 + scale[indices]) + shift[indices]

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]
    x_vals = [(args.M, args.N, args.G)] if args.M else list(_SHAPES)

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N", "G"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext()
        + ("_scattered" if args.scattered else "_runs"),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, G, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(M, N, G, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused layernorm indexed adaln",
        description="Benchmark layernorm(x) * (1 + scale[indices]) + shift[indices], "
        "fused against eager",
        allow_abbrev=False,
    )
    parser.add_argument("-M", type=int, default=None, help="Number of tokens")
    parser.add_argument("-N", type=int, default=_HIDDEN, help="Hidden size")
    parser.add_argument(
        "-G", type=int, default=2, help="Modulation table rows, with -M"
    )
    parser.add_argument(
        "--affine",
        type=str,
        default="none",
        choices=["none", "weight", "weight_bias"],
        help="LayerNorm affine parameters (default: none, as in the DiT blocks)",
    )
    parser.add_argument(
        "--scattered",
        action="store_true",
        help="Random index per token instead of runs of one index",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused kernel, eager expression, or both",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "time", "bandwidth"],
        help="Metric to report (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for fused_layernorm_indexed_adaln...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
