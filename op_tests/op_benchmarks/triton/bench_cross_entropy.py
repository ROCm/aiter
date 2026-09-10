# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for the vocab-parallel cross-entropy Triton kernels.

Single-GPU (dist_group=None) path. The forward provider exercises the
``_ce_local_softmax_stats_kernel`` + ``_ce_fused_loss_grad_kernel`` pair; the
backward provider exercises ``_ce_grad_scale_kernel``.
"""

import argparse

import torch
import triton

from aiter.ops.triton.cross_entropy import (
    cross_entropy_backward,
    cross_entropy_forward,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)


def _bytes_moved(B_SQ, V, dtype):
    # forward reads logits once and writes the gradient in-place once.
    return 2 * B_SQ * V * torch.tensor([], dtype=dtype).element_size()


def benchmark(args):
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
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
        logits = torch.randn(B_SQ, 1, V, dtype=dtype, device="cuda")
        target = torch.randint(0, V, (B_SQ, 1), device="cuda")

        if provider == "fwd":

            def fn():
                cross_entropy_forward(logits.clone(), target, 0.0, True, None, -100)

        else:  # bwd: scale the stored gradient (grad_output != 1 to force kernel)
            _, grad = cross_entropy_forward(
                logits.clone(), target, 0.0, True, None, -100
            )
            grad_output = torch.tensor(2.0, device="cuda")

            def fn():
                cross_entropy_backward(grad.clone(), grad_output)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return _bytes_moved(B_SQ, V, dtype) * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark cross-entropy", allow_abbrev=False)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
        help="Metric for the kernel benchmark.",
    )
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
