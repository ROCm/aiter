# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused_bf16_bmm_nvfp4_quant_mla: the R1 MLA W_UK absorb of the
NoPE query, quantized into the NVFP4 MLA query planes. Inputs rotate through
a pool, so each call reads its W_UK from HBM, as each decode layer does."""

import functools
import sys

import triton

from aiter.ops.triton.fusions.fused_bf16_bmm_nvfp4_quant_mla import (
    D_NOPE,
    KV_LORA,
    fused_bf16_bmm_nvfp4_quant_mla,
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
from op_tests.triton_tests.fusions.test_fused_bf16_bmm_nvfp4_quant_mla import (
    generate_inputs,
)

KEYS = ["q_nope", "w_uk", "q_nope_fp4", "q_aux"]


def bench_fn(bencher, batch, heads, metric):
    # q_b_proj stores NoPE and RoPE together in [B, H, 192].
    gen = functools.partial(generate_inputs, batch, heads, q_width=192)
    rot_bufs = bencher.build_pool([gen], [KEYS])
    ms = bencher.bench(fused_bf16_bmm_nvfp4_quant_mla)
    flops = 2.0 * batch * heads * KV_LORA * D_NOPE
    q = rot_bufs[0]["q_nope"]
    mem = (
        batch * heads * D_NOPE * q.element_size()
        + heads * KV_LORA * D_NOPE * rot_bufs[0]["w_uk"].element_size()
        + batch * heads * (KV_LORA // 2 + KV_LORA // 16)
    )
    if metric == "time":
        return ms
    if metric == "throughput":
        return flops / ms * 1e-9
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    raise ValueError("Unknown metric: " + metric)


def run_benchmark(args):
    batches = (
        [args.M]
        if args.M
        else [1, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2200, 2240, 2300]
    )
    benchmark = triton.testing.Benchmark(
        x_names=["batch"],
        x_vals=batches,
        line_arg="provider",
        line_vals=["gluon"],
        line_names=["fused_bf16_bmm_nvfp4_quant_mla"],
        ylabel={"time": "Time (ms)", "throughput": "TFLOPS", "bandwidth": "GB/s"}[
            args.metric
        ],
        plot_name=get_caller_name_no_ext(),
        args={"heads": args.qh, "metric": args.metric},
    )
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench(batch, heads, metric, provider):
        return bench_fn(bencher, batch, heads, metric)

    bench.run(save_path="." if args.o else None, print_data=True)


def parse_args(argv=None):
    parser = get_parser("fused_bf16_bmm_nvfp4_quant_mla")
    parser.add_argument(
        "-M", type=int, default=None, help="Batch rows (default: a sweep)."
    )
    parser.add_argument("--qh", type=int, default=128, help="Query heads.")
    parser.add_argument("-o", action="store_true", help="Write results to a CSV file.")
    return parser.parse_args(argv)


def main(argv=None):
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
