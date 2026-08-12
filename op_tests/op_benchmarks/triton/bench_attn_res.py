# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the attention-residual (residual candidate gating) kernel.

Sweeps the two residual layouts over (N tokens, D hidden, L candidates) and
reports time and effective HBM bandwidth.

Usage (from the repository root):
  python op_tests/op_benchmarks/triton/bench_attn_res.py
  python op_tests/op_benchmarks/triton/bench_attn_res.py --layout packed
  python op_tests/op_benchmarks/triton/bench_attn_res.py -N 8192 -D 7168 -L 8
  python op_tests/op_benchmarks/triton/bench_attn_res.py --metric bandwidth --onorm
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --add-hidden
  python op_tests/op_benchmarks/triton/bench_attn_res.py --op gate --sweep decode

Bandwidth uses the MINIMUM required traffic (read each residual once + write
the output once) for every layout, so a faster configuration always reports a
higher number. The sequence layout runs a two-pass kernel and therefore reads
the residual twice; its real HBM throughput is about 2x the reported value.
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.fusions.attn_res import attn_res_fwd, attn_res_gate
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_attn_res import (
    generate_attn_res_gate_inputs,
    generate_attn_res_inputs,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_LAYOUTS = ("sequence", "packed")


# Prefill-sized token counts, plus the decode range where the grid is small
# enough that launch overhead and occupancy, not bandwidth, set the runtime.
_PREFILL_TOKENS = (1166, 8192, 16384)
_DECODE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)


def get_benchmark_shapes(args):
    """Return [(N, D, L), ...] for the current CLI args."""
    if args.N and args.D and args.L:
        return [(args.N, args.D, args.L)]
    tokens = ()
    if args.sweep in ("decode", "all"):
        tokens += _DECODE_TOKENS
    if args.sweep in ("prefill", "all"):
        tokens += _PREFILL_TOKENS
    return [(N, 7168, L) for N in tokens for L in (2, 4, 8)]


def bench_attn_res_fn(N, D, L, layout, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()
    # Minimum required traffic: every candidate read once + output written once.
    mem = (N * L * D + N * D) * elem_size

    if args.op == "gate":
        # L counts the prefix, so the packed block holds L - 1 rows.
        prefix, block_residual, score_weight, add_hidden = (
            generate_attn_res_gate_inputs(N, D, L - 1, dtype, with_add=args.add_hidden)
        )
        if args.add_hidden:
            mem += 2 * N * D * elem_size  # add_hidden read + prefix write-back
        output_rms_weight = (
            torch.randn(D, dtype=dtype, device="cuda") if args.onorm else None
        )
        fn = lambda: attn_res_gate(
            prefix,
            block_residual,
            score_weight,
            args.eps,
            add_hidden,
            output_rms_weight=output_rms_weight,
            scale=args.scale,
        )
    else:
        query, residuals, rms_weight, output_rms_weight = generate_attn_res_inputs(
            N, D, L, dtype, with_onorm=args.onorm
        )
        if layout == "packed":
            residuals = torch.stack(residuals, dim=-2).contiguous()
        fn = lambda: attn_res_fwd(
            query,
            residuals,
            rms_weight,
            output_rms_weight,
            args.eps,
            args.scale,
            layout=layout,
        )

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)

    if metric == "time":
        return ms
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    raise ValueError("Unknown metric: " + metric)


def run_benchmark(args):
    if args.op == "gate":
        layouts = ("packed",)  # the gate is packed-only
    else:
        layouts = _LAYOUTS if args.layout == "all" else (args.layout,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{layout}_{metric}" for metric in metrics for layout in layouts]

    benchmark = triton.testing.Benchmark(
        x_names=["N", "D", "L"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.op}_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, D, L, provider):
        layout, metric = provider.rsplit("_", 1)
        return bench_attn_res_fn(N, D, L, layout, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark attention-residual",
        description="Benchmark the Triton attention-residual (attn_res) kernel",
        allow_abbrev=False,
    )
    parser.add_argument("-N", type=int, default=None, help="Number of tokens")
    parser.add_argument("-D", type=int, default=None, help="Hidden size")
    parser.add_argument("-L", type=int, default=None, help="Residual candidates")
    parser.add_argument(
        "--op",
        type=str,
        default="fwd",
        choices=["fwd", "gate"],
        help="attn_res_fwd (both layouts) or attn_res_gate (packed, inference)",
    )
    parser.add_argument(
        "--add-hidden",
        dest="add_hidden",
        action="store_true",
        default=False,
        help="For --op gate: fold the prefix += hidden add into the kernel",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="prefill",
        choices=["prefill", "decode", "all"],
        help="Token counts to sweep: prefill sizes, the decode range (1-128), or both",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="all",
        choices=[*_LAYOUTS, "all"],
        help="Residual layout: sequence (2-pass), packed (1-pass), or all",
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
    parser.add_argument(
        "--onorm", action="store_true", default=False, help="Enable the output RMSNorm"
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument("--scale", type=float, default=1.0, help="Logit scale")
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
        print("Retrieving VGPR usage for attn_res Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
