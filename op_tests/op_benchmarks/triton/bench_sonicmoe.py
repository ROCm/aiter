# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark SonicMoE forward and training throughput."""

import argparse

import torch
import triton

from aiter.ops.triton.sonicmoe import (
    SonicMoEActivationType,
    moe_TC_softmax_topk_layer,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# tokens, hidden, intermediate, experts, top-k, label
_SHAPES = [
    (1024, 2048, 768, 16, 2, "Qwen3-small"),
    (4096, 2048, 768, 16, 2, "Qwen3-train"),
    (4096, 4096, 14336, 8, 2, "Mixtral-train"),
]


def _setup(tokens, hidden, intermediate, experts, top_k):
    dtype = torch.bfloat16
    x = (torch.randn(tokens, hidden, device="cuda", dtype=dtype) * 0.1).requires_grad_()
    router = torch.randn(experts, hidden, device="cuda", dtype=dtype) * 0.02
    w1 = (
        torch.randn(2 * intermediate, hidden, experts, device="cuda", dtype=dtype)
        * 0.02
    ).requires_grad_()
    w2 = (
        torch.randn(hidden, intermediate, experts, device="cuda", dtype=dtype) * 0.02
    ).requires_grad_()
    grad = torch.randn_like(x)
    return x, router, w1, w2, grad


def benchmark(args):
    unit = {"time": "ms", "throughput": "TFLOPS"}[args.metric]
    config = triton.testing.Benchmark(
        x_names=["tokens", "hidden", "intermediate", "experts", "top_k", "label"],
        x_vals=list(_SHAPES),
        line_arg="provider",
        line_vals=["forward", "forward_backward"],
        line_names=[f"forward ({unit})", f"forward + backward ({unit})"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(tokens, hidden, intermediate, experts, top_k, label, provider):
        x, router, w1, w2, grad = _setup(tokens, hidden, intermediate, experts, top_k)

        def forward():
            return moe_TC_softmax_topk_layer(
                x,
                router,
                w1,
                None,
                w2,
                None,
                top_k,
                torch.cuda.current_stream().cuda_stream,
                SonicMoEActivationType.SWIGLU,
            )[0]

        if provider == "forward":
            fn = forward
            flops = 4 * tokens * top_k * hidden * intermediate
        else:

            def fn():
                out = forward()
                torch.autograd.grad(out, (x, w1, w2), grad)

            flops = 12 * tokens * top_k * hidden * intermediate

        ms = triton.testing.do_bench(fn, warmup=10, rep=50)
        if args.metric == "time":
            return ms
        return flops * 1e-12 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark SonicMoE", allow_abbrev=False)
    parser.add_argument(
        "-metric",
        choices=["time", "throughput"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    benchmark(parse_args())


if __name__ == "__main__":
    main()
