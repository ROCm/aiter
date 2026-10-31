# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for MXFP8 convert kernels.

``to`` exercises ``_convert_to_mxfp8_kernel``; ``from`` exercises
``_convert_from_mxfp8_kernel``. use_asm is left at the wrapper default
(auto-selected: ASM on gfx950, portable path elsewhere).
"""

import functools

import torch
import triton

from aiter.ops.triton.quant.quant_mxfp8 import convert_from_mxfp8, convert_to_mxfp8
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


def _generate_mxfp8_to_pool_inputs(M, N, in_dtype):
    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    return (x,)


def _generate_mxfp8_from_pool_inputs(M, N, in_dtype, fp8_dtype):
    x = torch.randn(M, N, device="cuda", dtype=in_dtype)
    y, s = convert_to_mxfp8(x, fp8_dtype, quant_block_size=32)
    return y, s


def benchmark(args):
    in_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    fp8_dtype = torch.float8_e4m3fn
    # M, N must tile to (block_m=64, block_n=64).
    x_vals = [(m, n) for m in [4096, 8192] for n in [4096, 8192, 16384]]
    unit = "ms" if args.metric == "time" else "GB/s"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    config = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["to", "from"],
        line_names=[f"to ({unit})", f"from ({unit})"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, N, provider):
        if provider == "to":
            _p_gen = functools.partial(_generate_mxfp8_to_pool_inputs, M, N, in_dtype)
            _out_val_keys = ["x"]

            def kernel_fn(x):
                convert_to_mxfp8(x, fp8_dtype, quant_block_size=32)

        else:
            _p_gen = functools.partial(
                _generate_mxfp8_from_pool_inputs, M, N, in_dtype, fp8_dtype
            )
            _out_val_keys = ["y", "s"]

            def kernel_fn(y, s):
                convert_from_mxfp8(y, s, in_dtype, quant_block_size=32)

        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], **kwargs_rot_bufs
        )
        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms
        # bytes: high-precision payload + fp8 payload + one e8m0 scale byte per 32
        gb = (M * N * (in_dtype.itemsize + 1) + M * triton.cdiv(N, 32)) * 1e-9
        return gb / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="MXFP8 convert")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
