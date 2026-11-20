# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the batch-invariant DSA indexer (Gluon dsa_indexer_bi, gfx950)."""

import functools

import triton

from aiter.ops.triton.attention.batch_invariant.dsa_indexer_bi import (
    dsa_indexer_bi,
    dsa_mqa_logits_bi,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.attention.batch_invariant.test_dsa_indexer_bi import (
    generate_dsa_indexer_bi_cache,
    generate_dsa_indexer_bi_rows,
)

# (requests, keys per request, rows per request): decode, then prefill chunks
_SHAPES = [
    (1, 8192, 1),
    (1, 131072, 1),
    (64, 8192, 1),
    (64, 65536, 1),
    (256, 16384, 1),
    (1, 8192, 4096),
    (1, 65536, 4096),
]
_ROT = ["q", "weights", "row_req", "row_len"]
_FIXED = ["index_cache", "block_table"]


def run_benchmark(args) -> None:
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )
    shapes = [(args.reqs, args.req_len, args.rows)] if args.req_len else _SHAPES
    benchmark = triton.testing.Benchmark(
        x_names=["R", "L", "rows"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=["indexer", "logits"],
        line_names=["dsa_indexer_bi", "dsa_mqa_logits_bi"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="Time (ms)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(R: int, L: int, rows: int, provider: str) -> float:
        bencher.build_pool(
            [functools.partial(generate_dsa_indexer_bi_rows, R, L, rows, H=args.heads)],
            [_ROT],
            inp_gen_fixed_fns=[functools.partial(generate_dsa_indexer_bi_cache, R, L)],
            inp_fixed_keys=[_FIXED],
        )
        if provider == "indexer":
            fn = functools.partial(dsa_indexer_bi, topk=args.topk, max_len=L)
        else:
            fn = functools.partial(dsa_mqa_logits_bi, max_len=L)
        return bencher.bench(fn)

    bench.run(print_data=True, show_plots=False)


def main(args=None) -> None:
    parser = get_parser("DSA indexer batch invariant")
    parser.add_argument("--reqs", type=int, default=1, help="Requests.")
    parser.add_argument(
        "--req-len", type=int, default=None, help="Keys per request (default: a sweep)."
    )
    parser.add_argument("--rows", type=int, default=1, help="Query rows per request.")
    parser.add_argument("--heads", type=int, default=32, help="Index heads H.")
    parser.add_argument("--topk", type=int, default=2048, help="Keys selected per row.")
    run_benchmark(parser.parse_args(args))


if __name__ == "__main__":
    main()
