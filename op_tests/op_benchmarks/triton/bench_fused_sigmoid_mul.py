# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused x * sigmoid(gate) kernel.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul
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
from op_tests.triton_tests.fusions.test_fused_sigmoid_mul import (
    generate_fused_sigmoid_mul_inputs,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PROVIDERS = ("fused", "eager")
_ARRAYS = {"fused": 3, "eager": 5}
_GATE_WIDTH = 4096
# Kimi-K3 at TP8: the gate is a 1536-wide slice of the 3648-wide merged q_a|kv_a|gate row.
_STRIDED_WIDTH, _STRIDED_ROW = 1536, 3648
_DECODE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
_PREFILL_TOKENS = (1166, 4096, 8192, 16384)


def get_benchmark_shapes(args):
    """Return [(N, D), ...] for the current CLI args."""
    if args.N and args.D:
        return [(args.N, args.D)]
    tokens = ()
    if args.sweep in ("decode", "all"):
        tokens += _DECODE_TOKENS
    if args.sweep in ("prefill", "all"):
        tokens += _PREFILL_TOKENS
    width = _STRIDED_WIDTH if args.layout == "strided" else _GATE_WIDTH
    return [(n, width) for n in tokens]


def _generate_inputs(N, D, dtype, layout, row_stride, x_row_stride):
    x, gate = generate_fused_sigmoid_mul_inputs((N, D), dtype)
    if layout == "strided":
        gate = torch.randn(N, row_stride, dtype=dtype, device=x.device)[:, -D:]
        if x_row_stride:
            x = torch.randn(N, x_row_stride, dtype=dtype, device=x.device)[:, :D]
    out = torch.empty_like(x)
    return x, gate, out


def bench_fused_sigmoid_mul_fn(N, D, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    _p_gen = functools.partial(
        _generate_inputs, N, D, dtype, args.layout, args.row_stride, args.x_row_stride
    )
    _out_val_keys = ["x", "gate", "out"]
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [_out_val_keys], **kwargs_rot_bufs
    )

    # Shapes/dtypes are identical across the pool, so any entry works for sizing.
    sample = rot_bufs[0]
    elem_size = sample["x"].element_size()
    mem = _ARRAYS[provider] * N * D * elem_size

    if provider == "fused":

        def _run(x, gate, out):
            fused_sigmoid_mul(x, gate, out)

    else:

        def _run(x, gate, out):
            x * gate.sigmoid()

    ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["N", "D"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.layout}_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(N, D, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fused_sigmoid_mul_fn(N, D, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="fused sigmoid-mul")
    parser.add_argument("-N", type=int, default=None, help="Number of tokens")
    parser.add_argument("-D", type=int, default=None, help="Gated hidden width")
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["prefill", "decode", "all"],
        help="Token counts to sweep: prefill sizes, the decode range (1-128), or both",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused kernel, eager two-pass baseline, or both",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="contiguous",
        choices=["contiguous", "strided"],
        help="contiguous inputs, or the gate as a column slice of a wider row",
    )
    parser.add_argument(
        "--row_stride",
        type=int,
        default=_STRIDED_ROW,
        help="Row stride of the strided gate (default: K3 merged row)",
    )
    parser.add_argument(
        "--x_row_stride",
        type=int,
        default=0,
        help="Row stride of a strided x, 0 keeps x dense (K3 padded 16 heads: 2048)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    override_parser_action(
        parser, action="--metric", choices=["all", "time", "bandwidth"], default="all"
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for fused_sigmoid_mul Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
