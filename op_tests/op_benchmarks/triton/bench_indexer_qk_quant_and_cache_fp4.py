# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import triton

from aiter.ops.triton.fusions.indexer_qk_quant_and_cache_fp4 import (
    indexer_qk_quant_and_cache_fp4,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)
from op_tests.triton_tests.fusions.test_indexer_qk_quant_and_cache_fp4 import (
    generate_inputs,
)


def run_benchmark(args):
    if args.backend not in (None, "triton"):
        raise ValueError("The OPUS FP4 writer uses the Triton backend")
    shapes = (
        [tuple(args.shape)]
        if args.shape
        else [
            (m, h, r)
            for m in (6, 64, 256, 2048)
            for h, r in ((32, 0), (32, 64), (64, 64))
        ]
    )

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "H", "rope_dim"],
            x_vals=shapes,
            line_arg="provider",
            line_vals=["triton"],
            line_names=["OPUS writer"],
            styles=[("green", "-")],
            ylabel="Time (ms)",
            plot_name="indexer_qk_quant_and_cache_fp4",
            args={},
        )
    )
    def bench(M, H, rope_dim, provider):
        bencher = RotatingBufferBench.from_dict(
            get_rot_bufs_kwargs(args) | get_do_bench_graph_kwargs(args)
        )
        generator = functools.partial(generate_inputs, M, H, rope_dim)
        bencher.build_pool([lambda: (generator(),)], [["inputs"]])
        return bencher.bench(lambda inputs: indexer_qk_quant_and_cache_fp4(**inputs))

    bench.run(save_path="." if args.o else None, print_data=True)


if __name__ == "__main__":
    parser = get_parser("indexer_qk_quant_and_cache_fp4")
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "H", "ROPE"))
    parser.add_argument("-o", action="store_true", help="Write results to CSV")
    override_parser_action(parser, action="--metric", default="time", choices=["time"])
    run_benchmark(parser.parse_args())
