# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the batch-invariant MXFP4 MoE forward (Gluon moe_mxfp4_bi_forward, gfx950)."""

import functools

import triton

from aiter.ops.triton.moe.batch_invariant.moe_mxfp4_bi import moe_mxfp4_bi_forward
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.moe.batch_invariant.test_moe_mxfp4_bi import (
    generate_moe_mxfp4_bi_inputs,
)

_TOKENS = [1, 8, 64, 256, 1024, 4096]
_KEYS = ["x", "w13", "w13_scale", "w2", "w2_scale", "topk_weights", "topk_ids"]


def run_benchmark(args) -> None:
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    benchmark = triton.testing.Benchmark(
        x_names=["T"],
        x_vals=[args.tokens] if args.tokens else _TOKENS,
        line_arg="provider",
        line_vals=["gluon"],
        line_names=["moe_mxfp4_bi"],
        styles=[("green", "-")],
        ylabel="Time (ms)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(T: int, provider: str) -> float:
        gen = functools.partial(
            generate_moe_mxfp4_bi_inputs, T, args.experts, h=args.hidden, n=args.inter
        )
        bencher.build_pool([gen], [_KEYS])
        fn = functools.partial(
            moe_mxfp4_bi_forward, situ_beta=4.0, situ_linear_beta=7.0
        )
        return bencher.bench(fn)

    bench.run(print_data=True, show_plots=False)


def main(args=None) -> None:
    parser = get_parser("MoE MXFP4 batch invariant")
    parser.add_argument(
        "--tokens", type=int, default=None, help="One token count (default: a sweep)."
    )
    parser.add_argument("--experts", type=int, default=112, help="Local experts.")
    parser.add_argument("--hidden", type=int, default=3584, help="Expert input size H.")
    parser.add_argument(
        "--inter",
        type=int,
        default=3072,
        help="Expert intermediate size N on this rank.",
    )
    run_benchmark(parser.parse_args(args))


if __name__ == "__main__":
    main()
