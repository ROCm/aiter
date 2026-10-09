# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the one-launch FP8 MLA prefill q/k/v prep."""

import argparse
import sys

import torch
import triton

from aiter import dtypes
from aiter.ops.triton.fusions.fused_mla_prefill_qkv_fp8 import (
    fused_mla_prefill_qkv_fp8,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_mla_prefill_qkv_fp8 import (
    generate_inputs,
    torch_mla_prefill_qkv_fp8_ref,
)

arg_to_torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}

_PROVIDERS = ("fused", "eager")
# Kimi-K3 heads per rank at TP8 and TP4, padded to the FP8 prefill head count.
_HEADS = ((12, 16), (24, 32))
_TOKENS = (1112, 4096, 7784, 8192, 16384)


def get_benchmark_shapes(args):
    """Return [(T, H, H_out), ...] for the current CLI args."""
    heads = [(args.H, args.H_out)] if args.H else _HEADS
    tokens = [args.T] if args.T else _TOKENS
    return [(t, h, h_out) for h, h_out in heads for t in tokens]


def bench_fn_impl(T, H, H_out, provider, metric, args):
    q, k_nope, v, k_pe = generate_inputs(T, H, arg_to_torch_dtype[args.dtype])
    read = sum(t.numel() for t in (q, k_nope, v, k_pe)) * q.element_size()
    written = T * H_out * (2 * q.shape[-1] + v.shape[-1])  # one byte per FP8 value
    if provider == "fused":

        def fn():
            return fused_mla_prefill_qkv_fp8(q, k_nope, v, k_pe, H_out, dtypes.fp8)

    else:

        def fn():
            return torch_mla_prefill_qkv_fp8_ref(q, k_nope, v, k_pe, H_out, dtypes.fp8)

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return (read + written) / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["T", "H", "H_out"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(T, H, H_out, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(T, H, H_out, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark fused MLA prefill q/k/v FP8 prep",
        description="Benchmark the Triton fused MLA prefill q/k/v FP8 prep kernel",
        allow_abbrev=False,
    )
    parser.add_argument("-T", type=int, default=None, help="Number of tokens")
    parser.add_argument("-H", type=int, default=None, help="Real heads per rank")
    parser.add_argument("--H_out", type=int, default=16, help="Padded head count")
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused kernel, the eager concat/repeat/cast it replaces, or both",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
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
        print("Retrieving VGPR usage for fused_mla_prefill_qkv_fp8 Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
