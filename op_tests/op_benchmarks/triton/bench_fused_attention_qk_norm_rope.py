# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused Qwen3-Next attention prologue (Q/K norm + partial RoPE).

The op launches one token-persistent kernel for every token count; the
defaults sweep decode and prefill sizes. Outputs are preallocated inside the
wrapper, matching the graph-captured integration.

Usage:
    python bench_fused_attention_qk_norm_rope.py \
        -M 1,1024,12288,16384,32768 --metric time
"""

import functools

import triton

from aiter.ops.triton.attention.fused_attention_qk_norm_rope import (
    fused_attention_qk_norm_rope,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.test_fused_attention_qk_norm_rope import (
    HEAD_DIM,
    QUERY_HEADS,
    _make_inputs,
)

_POOL_KEYS = ["packed", "q_norm", "k_norm", "positions", "cos_sin_cache"]


def parse_int_or_list(value):
    if "," in value:
        return [int(item) for item in value.split(",")]
    return int(value)


def get_x_vals(args):
    token_counts = args.M if isinstance(args.M, list) else [args.M]
    return [(tokens,) for tokens in token_counts]


def benchmark_prologue(tokens, metric, bencher, **_):
    bencher.build_pool([functools.partial(_make_inputs, tokens)], [_POOL_KEYS])

    def run(packed, q_norm, k_norm, positions, cos_sin_cache):
        fused_attention_qk_norm_rope(packed, q_norm, k_norm, positions, cos_sin_cache)

    time_ms = bencher.bench(run)
    if metric == "time":
        return time_ms
    if metric == "bandwidth":
        # The op reads Q, gate and K and writes their normed and passthrough
        # copies. V is returned as a view, so it is neither read nor written.
        moved = tokens * (2 * QUERY_HEADS + 1) * HEAD_DIM
        return 2 * moved * 2 / time_ms * 1.0e-6
    raise ValueError(f"Unsupported metric: {metric}")


def run_benchmark(args):
    ylabel = {"time": "Time (ms)", "bandwidth": "Bandwidth (GB/s)"}[args.metric]
    benchmark = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=get_x_vals(args),
        line_arg="provider",
        line_vals=[args.metric],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={
            "metric": args.metric,
            "bencher": RotatingBufferBench.from_dict(
                {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
            ),
        },
    )
    triton.testing.perf_report([benchmark])(benchmark_prologue).run(
        save_path="." if args.output else None,
        print_data=True,
    )


def main():
    parser = get_parser(kernel_name="fused attention Q/K norm + RoPE")
    parser.description = __doc__
    parser.add_argument(
        "-M",
        type=parse_int_or_list,
        default=[1, 1024, 8192, 12288, 14336, 16384, 24576, 32768],
        help="token counts to sweep (comma-separated)",
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument("--output", action="store_true", help="save the plot")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
