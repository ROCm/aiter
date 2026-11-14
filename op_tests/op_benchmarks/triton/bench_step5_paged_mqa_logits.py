# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the Step-5 CSA paged MQA logits kernel.

The default shape is one prefill chunk of the Step-5 indexer: 4096 query rows
that share one page table, with causal lengths up to 8000 regions.
"""

import triton

from aiter.ops.triton.attention.step5_paged_mqa_logits import step5_paged_mqa_logits
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_step5_paged_mqa_logits import (
    HEAD_DIM,
    make_inputs,
)


def run_benchmark(args):
    x_names = ["rows", "regions", "heads", "page_size", "layout"]
    x_vals = [[args.rows, args.regions, args.heads, args.page_size, args.layout]]
    ylabel = {"time": "Time (us)", "throughput": "TFLOPS"}[args.metric]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals,
        line_arg="unit",
        line_vals=[ylabel],
        line_names=[ylabel],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_step5_paged_mqa_logits(
        rows, regions, heads, page_size, layout, metric, **kwargs
    ):
        inputs = make_inputs(rows, 1, heads, page_size, regions, layout)
        seq_lens = inputs[-1]

        def fn():
            return step5_paged_mqa_logits(*inputs, regions)

        time_ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if metric == "time":
            return time_ms * 1e3
        flops = 2.0 * heads * HEAD_DIM * seq_lens.sum().item()
        return flops / (time_ms * 1e-3) / 1e12

    bench_step5_paged_mqa_logits.run(save_path="." if args.o else None, print_data=True)


def main():
    parser = get_parser(kernel_name="Step-5 Paged MQA Logits")
    parser.add_argument("--rows", type=int, default=4096, help="query rows")
    parser.add_argument("--regions", type=int, default=8000, help="max_seq_len")
    parser.add_argument("--heads", type=int, choices=[4, 8, 16], default=16)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument(
        "--layout",
        choices=["shared", "mixed", "distinct"],
        default="shared",
        help="page mapping of the rows, see make_inputs in the unit test",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "throughput"], default="time"
    )
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
