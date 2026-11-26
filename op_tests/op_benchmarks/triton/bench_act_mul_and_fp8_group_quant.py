# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused activation, multiply and FP8 group quantization kernel.

Times with rotating input buffers and CUDA graph replay (``do_bench_aiter_triton``), so
host launch cost is not counted.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.fusions.activation import act_mul_and_fp8_group_quant
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
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
    print_vgpr,
)

_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
_ROWS = (1, 128, 2048, 8192, 32768)
_WIDTHS = (2176, 4352, 8704, 17408)


def _generate_inputs(rows: int, width: int, dtype: torch.dtype):
    return (torch.randn((rows, 2 * width), dtype=dtype, device="cuda"),)


def _benchmark(rows: int, width: int, metric: str, args):
    dtype_quant = get_fp8_e4m3_dtype()
    gen = functools.partial(_generate_inputs, rows, width, _DTYPES[args.dtype])
    rot_bufs = generate_rotating_buffers_pool(
        [gen], [["x"]], **get_rot_bufs_kwargs(args)
    )

    def fn(x):
        act_mul_and_fp8_group_quant(x, args.activation, args.group_size, dtype_quant)

    ms = do_bench_aiter_triton(fn, rot_bufs, **get_do_bench_graph_kwargs(args))
    if metric == "time":
        return ms * 1000
    if metric == "bandwidth":
        # Read gate and up; write one FP8 byte per output and one fp32 scale per group.
        scales = rows * triton.cdiv(width, args.group_size)
        x_bytes = rot_bufs[0]["x"].element_size()
        logical_bytes = rows * width * (2 * x_bytes + 1) + 4 * scales
        return logical_bytes / (ms * 1e-3) * 1e-9
    raise ValueError(f"unknown metric: {metric}")


def run_benchmark(args):
    rows = (args.rows,) if args.rows is not None else _ROWS
    widths = (args.width,) if args.width is not None else _WIDTHS
    unit = "us" if args.metric == "time" else "GB/s"

    benchmark = triton.testing.Benchmark(
        x_names=["rows", "width"],
        x_vals=[(m, n) for n in widths for m in rows],
        line_arg="metric",
        line_vals=[args.metric],
        line_names=[args.metric],
        styles=[("green", "-")],
        ylabel=unit,
        plot_name=f"{get_caller_name_no_ext()}_{args.dtype}_{args.metric}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(rows, width, metric):
        return _benchmark(rows, width, metric, args)

    bench_fn.run(save_path="." if args.output else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="Act-Mul and FP8 Group Quant")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument(
        "--width", type=int, default=None, help="N // 2, the output width"
    )
    parser.add_argument(
        "--activation", choices=("silu", "gelu", "gelu_tanh"), default="silu"
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument("--print-vgpr", action="store_true")
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
    else:
        run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
