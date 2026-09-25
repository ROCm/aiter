# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the fused delayed mHC seam kernel: mhc_fused_post_pre_delayed_rmsnorm.

Usage:
    python bench_mhc_fused_post_pre_delayed_rmsnorm.py
    python bench_mhc_fused_post_pre_delayed_rmsnorm.py --mode no_post
    python bench_mhc_fused_post_pre_delayed_rmsnorm.py -metric bandwidth

Timing uses ``do_bench_cudagraph``, so host launch overhead is excluded (serving runs
decode under CUDA graphs).
"""

import argparse

import torch
import triton

from aiter.ops.triton.fusions.mhc_fused_post_pre_delayed_rmsnorm import (
    mhc_fused_post_pre_delayed_rmsnorm,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# DeepSeek-V4.1 representative shapes: (M, n, C), decode batches and prefill chunks
_SHAPES = [
    (1, 4, 5120),
    (64, 4, 5120),
    (192, 4, 5120),
    (384, 4, 5120),
    (768, 4, 5120),
    (4096, 4, 5120),
    (16384, 4, 5120),
]

# rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat
_HC_ARGS = (1e-6, 1e-6, 1e-6, 2.0, 20)


def _make_inputs(M, n, C, device="cuda"):
    rows = 2 * n + n * n
    residual = torch.randn(M, n, C, dtype=torch.bfloat16, device=device)
    y = torch.randn(M, C, dtype=torch.bfloat16, device=device)
    fn = torch.randn(rows, n * C, dtype=torch.float32, device=device) * 0.02
    scale = torch.ones(3, dtype=torch.float32, device=device)
    base = torch.zeros(rows, dtype=torch.float32, device=device)
    pre = torch.rand(M, n, dtype=torch.float32, device=device)
    post = torch.rand(M, n, 1, dtype=torch.float32, device=device)
    comb = torch.softmax(torch.randn(M, n, n, device=device), dim=-1)
    w = torch.ones(C, dtype=torch.bfloat16, device=device)
    return residual, y, fn, scale, base, pre, post, comb, w


def benchmark(args):
    mode = args.mode
    unit = "ms" if args.metric == "time" else "GB/s"

    config = triton.testing.Benchmark(
        x_names=["M", "n", "C"],
        x_vals=_SHAPES,
        line_arg="provider",
        line_vals=[mode],
        line_names=[f"{mode} ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, n, C, provider):
        residual, y, fn, scale, base, pre, post, comb, w = _make_inputs(M, n, C)
        post_kw = {}
        if mode == "post":
            post_kw = {"sublayer_out": y, "post_layer_mix": post, "comb_res_mix": comb}
        fn_b = lambda: mhc_fused_post_pre_delayed_rmsnorm(
            residual, fn, scale, base, *_HC_ARGS, pre_mix=pre, norm_weight=w, **post_kw
        )
        rows = 2 * n + n * n
        # reads: residual + fn + scale + base + pre_mix + norm_weight;
        # writes: layer_input + post_mix + comb_mix + next_pre_mix
        mem = (
            M * n * C * residual.element_size()
            + rows * n * C * 4
            + 3 * 4
            + rows * 4
            + M * n * 4
            + C * w.element_size()
            + M * C * residual.element_size()
            + M * rows * 4
        )
        if mode == "post":
            # reads: sublayer_out + post_mix + comb_mix; writes: the new residual
            mem += (
                M * C * residual.element_size()
                + M * n * 4
                + M * n * n * 4
                + M * n * C * residual.element_size()
            )

        ms = triton.testing.do_bench_cudagraph(fn_b, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark the fused delayed mHC seam", allow_abbrev=False
    )
    parser.add_argument(
        "--mode",
        choices=["post", "no_post"],
        default="post",
        help="post: a regular seam (default); no_post: an Engram seam without the post block",
    )
    parser.add_argument(
        "-metric",
        nargs="?",
        const="time",
        choices=["time", "bandwidth"],
        default="time",
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
