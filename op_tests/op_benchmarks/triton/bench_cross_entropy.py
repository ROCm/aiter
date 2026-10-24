# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for the vocab-parallel cross-entropy Triton kernels.

Single-GPU (dist_group=None) path. The forward provider exercises the
``_ce_local_softmax_stats_kernel`` + ``_ce_fused_loss_grad_kernel`` pair; the
backward provider exercises ``_ce_grad_scale_kernel``.
"""

import functools

import torch
import triton

from aiter.ops.triton.cross_entropy import (
    cross_entropy_backward,
    cross_entropy_forward,
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


def _bytes_moved(B_SQ, V, dtype):
    # forward reads logits once and writes the gradient in-place once.
    return 2 * B_SQ * V * torch.tensor([], dtype=dtype, device="cuda").element_size()


def _generate_ce_fwd_pool_inputs(B_SQ, V, dtype):
    logits = torch.randn(B_SQ, 1, V, dtype=dtype, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")
    return logits, target


def _generate_ce_bwd_pool_inputs(B_SQ, V, dtype):
    logits = torch.randn(B_SQ, 1, V, dtype=dtype, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")
    _, grad = cross_entropy_forward(logits, target, 0.0, True, None, -100)
    grad_output = torch.tensor(2.0, device="cuda")
    return grad, grad_output


def _generate_ce_fwd_pool_inputs(B_SQ, V, dtype):
    logits = torch.randn(B_SQ, 1, V, dtype=dtype, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")
    return logits, target


def _generate_ce_bwd_pool_inputs(B_SQ, V, dtype):
    logits = torch.randn(B_SQ, 1, V, dtype=dtype, device="cuda")
    target = torch.randint(0, V, (B_SQ, 1), device="cuda")
    _, grad = cross_entropy_forward(logits, target, 0.0, True, None, -100)
    grad_output = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    return grad, grad_output


def benchmark(args):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    x_names = ["B_SQ", "V"]
    x_vals_list = [
        (bsq, v)
        for v in [32000, 128256, 151936]  # Llama / Qwen-class vocab sizes
        for bsq in [4096, 8192, 16384]
    ]
    unit = "ms" if args.metric == "time" else "GB/s"
    line_vals = ["fwd", "bwd"]

    config = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        line_arg="provider",
        line_vals=line_vals,
        line_names=[f"{p} ({unit})" for p in line_vals],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(B_SQ, V, provider):
        torch.manual_seed(0)

        if provider == "fwd":
            _p_gen = functools.partial(_generate_ce_fwd_pool_inputs, B_SQ, V, dtype)
            _out_val_keys = ["logits", "target"]

            def kernel_fn(logits, target):
                # validate_targets=False: the default path's range check is a
                # host sync (target.all()), which CUDA graph capture forbids.
                cross_entropy_forward(
                    logits, target, 0.0, True, None, -100, validate_targets=False
                )

        else:  # bwd: scale the stored gradient (grad_output != 1 to force kernel)
            _p_gen = functools.partial(_generate_ce_bwd_pool_inputs, B_SQ, V, dtype)
            _out_val_keys = ["grad", "grad_output"]

            def kernel_fn(grad, grad_output):
                cross_entropy_backward(grad, grad_output)

        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], **kwargs_rot_bufs
        )
        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms
        return _bytes_moved(B_SQ, V, dtype) * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="cross-entropy")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    # This kernel only supports time/bandwidth (no throughput/FLOPs notion),
    # so narrow get_parser()'s shared --metric choices/default to match.
    override_parser_action(
        parser,
        action="--metric",
        choices=["time", "bandwidth"],
        default="time",
        help="Metric for the kernel benchmark.",
    )
    # V up to ~152k makes a single logits/grad buffer multi-GB; a floor of 4
    # pool entries could quadruple an already multi-GB config and OOM. Default
    # to a smaller floor, as large tensors already exceed L2 capacity on their
    # own and get little extra benefit from rotation anyway.
    override_parser_action(parser, action="--rot-bufs-min-pool-len", default=2)
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to CSV"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
