# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark fused RMSNorm, residual add, RMSNorm for Muse shapes.

Use bench_rmsnorm.py and bench_fused_rmsnorm_add.py --add-residual to benchmark
the existing RMSNorm and RMSNorm-plus-add operators separately.
"""

import functools

import torch
import triton

from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
    fused_rmsnorm_add_rmsnorm,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)
from op_tests.triton_tests.normalization.test_fused_rmsnorm_add_rmsnorm import (
    generate_fused_rmsnorm_add_rmsnorm_inputs,
)

POST_EPS = 1e-8
PRE_EPS = 1e-5


def get_x_vals():
    # Muse-Glimmer-30B hidden width: decode batches through prefill chunks.
    return [(m, 6656) for m in (1, 8, 32, 64, 128, 256, 1024, 4096)]


def _generate_inputs(M, N):
    x, residual, post_weight, pre_weight = generate_fused_rmsnorm_add_rmsnorm_inputs(
        M, N
    )
    residual_out = torch.empty_like(x)
    return x, residual, post_weight, pre_weight, residual_out


def bench_fused_rmsnorm_add_rmsnorm_fn(
    M, N, metric, kwargs_rot_bufs, kwargs_do_bench_graph
):
    _p_gen = functools.partial(_generate_inputs, M, N)
    _out_val_keys = ["x", "residual", "post_weight", "pre_weight", "residual_out"]
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_val_keys], **kwargs_rot_bufs
    )

    # Shapes/dtypes are identical across the pool, so any entry works for sizing.
    sample = rot_bufs[0]

    def _run(x, residual, post_weight, pre_weight, residual_out):
        fused_rmsnorm_add_rmsnorm(
            x, residual, post_weight, pre_weight, POST_EPS, PRE_EPS, residual_out
        )

    ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms
    if metric == "bandwidth":
        bytes_moved = (4 * M * N + 2 * N) * sample["x"].element_size()
        return bytes_moved / (ms * 1e-3) * 1e-9
    raise ValueError(f"Unknown metric: {metric}")


def run_benchmark(args):
    x_vals = [tuple(args.shape)] if args.shape is not None else get_x_vals()
    unit = "Time_(ms)" if args.metric == "time" else "Bandwidth_(GB/s)"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="unit",
        line_vals=[unit],
        line_names=[unit],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_rmsnorm_add_rmsnorm(M, N, metric, **kwargs):
        return bench_fused_rmsnorm_add_rmsnorm_fn(
            M, N, metric, kwargs_rot_bufs, kwargs_do_bench_graph
        )

    bench_fused_rmsnorm_add_rmsnorm.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args(args: list[str] | None = None):
    parser = get_parser(kernel_name="FusedRMSNormAddRMSNorm")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument("-o", action="store_true", help="write results to CSV")
    parsed = parser.parse_args(args=args)
    if parsed.shape is not None and min(parsed.shape) <= 0:
        parser.error("--shape M N requires positive dimensions")
    return parsed


def main(args: list[str] | None = None):
    parsed = parse_args(args)
    run_benchmark(parsed)


if __name__ == "__main__":
    main()
