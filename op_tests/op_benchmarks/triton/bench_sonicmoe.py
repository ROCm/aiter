# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark SonicMoE forward and training throughput."""

import functools

import torch
import triton

from aiter.ops.triton.moe.sonicmoe import (
    SonicMoEActivationType,
    moe_TC_softmax_topk_layer,
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


_SONICMOE_POOL_FIELDS = ["x", "router", "w1", "w2", "grad"]


def benchmark(args):
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
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
        pool_gen = functools.partial(
            _setup, tokens, hidden, intermediate, experts, top_k
        )
        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_SONICMOE_POOL_FIELDS], **kwargs_rot_bufs
        )

        def kernel_fn(x, router, w1, w2, grad):
            out = moe_TC_softmax_topk_layer(
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
            if provider == "forward_backward":
                torch.autograd.grad(out, (x, w1, w2), grad)

        if provider == "forward":
            flops = 6 * tokens * top_k * hidden * intermediate
        else:
            flops = 18 * tokens * top_k * hidden * intermediate

        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms
        return flops * 1e-12 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="SonicMoE")
    override_parser_action(
        parser, action="--metric", choices=["time", "throughput"], default="time"
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    benchmark(parse_args())


if __name__ == "__main__":
    main()
