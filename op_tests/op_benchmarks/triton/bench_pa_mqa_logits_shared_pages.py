# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the shared-pages FP8 paged MQA logits kernel.

The default shape is one Step-5 CSA prefill indexer chunk: 4096 query rows
that share one page table, with causal lengths up to 8000 keys.
"""

import torch
import triton

from aiter.ops.triton.attention.pa_mqa_logits_shared_pages import (
    pa_mqa_logits_shared_pages,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_pa_mqa_logits_shared_pages import (
    make_inputs,
)


def run_benchmark(args):
    x_names = ["rows", "keys", "heads", "head_dim", "page_size", "layout"]
    x_vals = [
        [args.rows, args.keys, args.heads, args.head_dim, args.page_size, args.layout]
    ]
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
    def bench_pa_mqa_logits_shared_pages(
        rows, keys, heads, head_dim, page_size, layout, metric, **kwargs
    ):
        q, kv_cache, weights, lengths, block_tables = make_inputs(
            rows, 1, heads, page_size, keys, layout, head_dim=head_dim
        )
        out = torch.empty((rows, keys), dtype=torch.float32, device="cuda")

        def fn():
            return pa_mqa_logits_shared_pages(
                q, kv_cache, weights, out, lengths, block_tables, keys
            )

        time_ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if metric == "time":
            return time_ms * 1e3
        flops = 2.0 * heads * head_dim * lengths.sum().item()
        return flops / (time_ms * 1e-3) / 1e12

    bench_pa_mqa_logits_shared_pages.run(
        save_path="." if args.o else None, print_data=True
    )


def main():
    parser = get_parser(kernel_name="Shared-pages Paged MQA Logits")
    parser.add_argument("--rows", type=int, default=4096, help="query rows")
    parser.add_argument("--keys", type=int, default=8000, help="max_model_len")
    parser.add_argument("--heads", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--head-dim", type=int, choices=[128, 256], default=256)
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
