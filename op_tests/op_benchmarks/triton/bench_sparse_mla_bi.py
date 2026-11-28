# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the batch-invariant sparse MLA forward (Gluon sparse_mla_bi_fwd, gfx950)."""

import functools

import triton

from aiter.ops.triton.attention.batch_invariant.sparse_mla_bi import sparse_mla_bi_fwd
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.batch_invariant.test_sparse_mla_bi import (
    generate_sparse_mla_bi_inputs,
)

_ROWS = [1, 8, 64, 256, 1024, 4096]
_KEYS = ["q", "kv_buffer", "kv_indptr", "kv_indices"]


def run_benchmark(args) -> None:
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    benchmark = triton.testing.Benchmark(
        x_names=["C"],
        x_vals=[args.rows] if args.rows else _ROWS,
        line_arg="provider",
        line_vals=["gluon"],
        line_names=["sparse_mla_bi"],
        styles=[("green", "-")],
        ylabel="Time (ms)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(C: int, provider: str) -> float:
        gen = functools.partial(
            generate_sparse_mla_bi_inputs,
            C,
            args.heads,
            topk=args.topk,
            fp8=not args.bf16_cache,
        )
        bencher.build_pool([gen], [_KEYS])
        fn = functools.partial(sparse_mla_bi_fwd, softmax_scale=192**-0.5)
        return bencher.bench(fn)

    bench.run(print_data=True, show_plots=False)


def main(args=None) -> None:
    parser = get_parser("Sparse MLA batch invariant")
    parser.add_argument(
        "--rows", type=int, default=None, help="One query-row count (default: a sweep)."
    )
    parser.add_argument("--heads", type=int, default=96, help="Heads on this rank.")
    parser.add_argument("--topk", type=int, default=2048, help="Keys per row.")
    parser.add_argument(
        "--bf16-cache", action="store_true", help="bf16 cache (default fp8_ds_mla)."
    )
    run_benchmark(parser.parse_args(args))


if __name__ == "__main__":
    main()
