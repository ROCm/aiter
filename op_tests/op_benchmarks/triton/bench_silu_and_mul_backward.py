# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused SiLU-and-multiply backward kernel.

Reports the public API, its preallocated-output path, and an FP32 eager reference.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.fusions.activation import silu_and_mul_backward
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
    "fp32": torch.float32,
}
_PROVIDERS = ("aiter", "aiter-preallocated", "torch")
_ROWS = (1, 128, 2048, 8192)
_WIDTHS = (64, 128, 256, 512, 4096, 12288)


def _torch_reference(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    width = x.size(-1) // 2
    gate, up = x[..., :width].float(), x[..., width:].float()
    grad = grad_output.float()
    sigmoid_gate = torch.sigmoid(gate)
    grad_gate = grad * up * sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    grad_up = grad * gate * sigmoid_gate
    return torch.cat((grad_gate, grad_up), dim=-1).to(x.dtype)


def _generate_pool_entry(dtype, rows, width):
    x = torch.randn((rows, 2 * width), dtype=dtype, device="cuda")
    grad_output = torch.randn((rows, width), dtype=dtype, device=x.device)
    out = torch.empty_like(x)
    return x, grad_output, out


_POOL_FIELDS = ["x", "grad_output", "out"]


def _make_kernel_fn(provider):
    if provider == "aiter-preallocated":

        def kernel_fn(x, grad_output, out, **_):
            silu_and_mul_backward(grad_output, x, out=out)

    elif provider == "aiter":

        def kernel_fn(x, grad_output, **_):
            silu_and_mul_backward(grad_output, x)

    else:

        def kernel_fn(x, grad_output, **_):
            _torch_reference(grad_output, x)

    return kernel_fn


def _benchmark(
    rows: int,
    width: int,
    provider: str,
    metric: str,
    args,
    kwargs_rot_bufs: dict,
    kwargs_do_bench_graph: dict,
):
    dtype = _DTYPES[args.dtype]
    pool_gen = functools.partial(_generate_pool_entry, dtype, rows, width)
    rot_bufs = generate_rotating_buffers_pool(
        [pool_gen], [_POOL_FIELDS], **kwargs_rot_bufs
    )
    kernel_fn = _make_kernel_fn(provider)

    ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms * 1000
    if metric == "bandwidth":
        # Three logical input reads and two gradient writes per gate/up pair.
        logical_bytes = 5 * rows * width * rot_bufs[0]["x"].element_size()
        return logical_bytes / (ms * 1e-3) * 1e-9
    raise ValueError(f"unknown metric: {metric}")


def run_benchmark(args):
    rows = (args.rows,) if args.rows is not None else _ROWS
    widths = (args.width,) if args.width is not None else _WIDTHS
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    lines = [f"{provider}_{args.metric}" for provider in providers]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

    benchmark = triton.testing.Benchmark(
        x_names=["rows", "width"],
        x_vals=[(m, n) for n in widths for m in rows],
        line_arg="provider_metric",
        line_vals=lines,
        line_names=lines,
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
        ][: len(lines)],
        ylabel="us" if args.metric == "time" else "GB/s",
        plot_name=f"{get_caller_name_no_ext()}_{args.dtype}_{args.metric}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(rows, width, provider_metric):
        provider, metric = provider_metric.rsplit("_", 1)
        return _benchmark(
            rows, width, provider, metric, args, kwargs_rot_bufs, kwargs_do_bench_graph
        )

    bench_fn.run(save_path="." if args.output else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="SiLU-and-Mul Backward")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--provider", choices=(*_PROVIDERS, "all"), default="all")
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Unused: do_bench_aiter_triton's warmup is controlled by "
        "--graph-warmup-iters. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Unused: do_bench_aiter_triton's replay count is a fixed "
        "constant, not an adaptive rep target. Kept for CLI compatibility.",
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
