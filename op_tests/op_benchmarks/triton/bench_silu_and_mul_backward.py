# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the fused SiLU-and-multiply backward kernel."""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.activation import silu_and_mul_backward
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)

_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
_PROVIDERS = ("aiter", "torch")
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


def _benchmark(rows: int, width: int, provider: str, metric: str, args):
    dtype = _DTYPES[args.dtype]
    x = torch.randn((rows, 2 * width), dtype=dtype, device="cuda")
    grad_output = torch.randn((rows, width), dtype=dtype, device=x.device)
    if provider == "aiter":
        out = torch.empty_like(x)

        def fn():
            return silu_and_mul_backward(grad_output, x, out=out)

    else:

        def fn():
            return _torch_reference(grad_output, x)

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000
    if metric == "bandwidth":
        logical_bytes = 5 * rows * width * x.element_size()
        return logical_bytes / (ms * 1e-3) * 1e-9
    raise ValueError(f"unknown metric: {metric}")


def run_benchmark(args):
    rows = (args.rows,) if args.rows is not None else _ROWS
    widths = (args.width,) if args.width is not None else _WIDTHS
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    lines = [f"{provider}_{metric}" for metric in metrics for provider in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["rows", "width"],
        x_vals=[(m, n) for n in widths for m in rows],
        line_arg="provider_metric",
        line_vals=lines,
        line_names=lines,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("orange", "-")][
            : len(lines)
        ],
        ylabel="",
        plot_name=f"{get_caller_name_no_ext()}_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(rows, width, provider_metric):
        provider, metric = provider_metric.rsplit("_", 1)
        return _benchmark(rows, width, provider, metric, args)

    bench_fn.run(save_path="." if args.output else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark fused SiLU-and-multiply backward",
        allow_abbrev=False,
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--provider", choices=(*_PROVIDERS, "all"), default="all")
    parser.add_argument("--metric", choices=("time", "bandwidth", "all"), default="all")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
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
