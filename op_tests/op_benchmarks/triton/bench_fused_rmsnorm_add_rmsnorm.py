# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare the fused sandwich norm with two AITER normalization calls.

The separate calls round the first RMSNorm output and effective weights to
BF16. Both implementations compute the same operation within BF16 tolerance.
Timing uses GPU graph replay to exclude Python launch overhead.
"""

import argparse

import torch
import triton

from aiter.ops.triton.normalization.fused_rmsnorm_add import fused_rmsnorm_add
from aiter.ops.triton.normalization.fused_rmsnorm_add_rmsnorm import (
    fused_rmsnorm_add_rmsnorm,
)
from aiter.ops.triton.normalization.rmsnorm import rms_norm
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext
from op_tests.triton_tests.normalization.test_fused_rmsnorm_add_rmsnorm import (
    generate_fused_rmsnorm_add_rmsnorm_inputs,
)

POST_EPS = 1e-8
PRE_EPS = 1e-5


def get_x_vals():
    # Muse-Glimmer-30B hidden width: decode batches through prefill chunks.
    return [(m, 6656) for m in (1, 8, 32, 64, 128, 256, 1024, 4096)]


def bench_fused_rmsnorm_add_rmsnorm_fn(M, N, provider, metric):
    x, residual, post_weight, pre_weight = generate_fused_rmsnorm_add_rmsnorm_inputs(
        M, N
    )
    residual_out = torch.empty_like(x)
    # The two existing ops multiply by weight directly. Prepare their effective
    # Muse weights outside the timed region to measure only the two norm calls.
    post_scale = (post_weight.float() + 1.0).to(torch.bfloat16)
    pre_scale = (pre_weight.float() + 1.0).to(torch.bfloat16)

    def fused():
        return fused_rmsnorm_add_rmsnorm(
            x, residual, post_weight, pre_weight, POST_EPS, PRE_EPS, residual_out
        )

    def separate():
        post = rms_norm(x, post_scale, POST_EPS)
        return fused_rmsnorm_add(post, pre_scale, PRE_EPS, res1=residual)

    fused_norm = fused()
    separate_norm, separate_residual = separate()
    torch.testing.assert_close(fused_norm, separate_norm, atol=0.05, rtol=0.05)
    torch.testing.assert_close(residual_out, separate_residual, atol=0.05, rtol=0.05)

    fn = fused if provider == "fused" else separate
    ms = triton.testing.do_bench_cudagraph(fn, rep=100)
    if metric == "time":
        return ms
    if metric == "bandwidth":
        # Effective bandwidth uses the same logical two inputs, two weights,
        # and two outputs for both implementations. Separate calls move more
        # intermediate data, which the elapsed time already reflects.
        logical_bytes = (4 * M * N + 2 * N) * x.element_size()
        return logical_bytes / (ms * 1e-3) * 1e-9
    raise ValueError(f"Unknown metric: {metric}")


def run_benchmark(args):
    x_vals = [tuple(args.shape)] if args.shape is not None else get_x_vals()
    unit = "Time_(ms)" if args.metric == "time" else "Effective_Bandwidth_(GB/s)"
    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["fused", "separate"],
        line_names=["fused", "rms_norm + fused_rmsnorm_add"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={"metric": args.metric},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fused_rmsnorm_add_rmsnorm(M, N, provider, metric):
        return bench_fused_rmsnorm_add_rmsnorm_fn(M, N, provider, metric)

    bench_fused_rmsnorm_add_rmsnorm.run(
        save_path="." if args.o else None, print_data=True, show_plots=False
    )


def parse_args(args: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="Benchmark FusedRMSNormAddRMSNorm",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        metavar=("M", "N"),
        help="user-defined shape to benchmark",
    )
    parser.add_argument(
        "--metric",
        choices=["time", "bandwidth"],
        default="time",
        help="metric to report",
    )
    parser.add_argument("-o", action="store_true", help="write results to CSV")
    parsed = parser.parse_args(args=args)
    if parsed.shape is not None and min(parsed.shape) <= 0:
        parser.error("--shape M N requires positive dimensions")
    return parsed


def main(args: list[str] | None = None):
    parsed = parse_args(args)
    with torch.no_grad():
        run_benchmark(parsed)


if __name__ == "__main__":
    main()
