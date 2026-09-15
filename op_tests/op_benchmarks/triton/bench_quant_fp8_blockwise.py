# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for the FP8 block-wise quantization kernels.

One provider per kernel:
  blockwise  -> quant_fp8_blockwise_kernel
  weight     -> quant_fp8_blockwise_for_weight_kernel
  act_grad   -> quant_fp8_blockwise_for_act_grad_kernel
  requant    -> requant_fp8_row_to_col_kernel
  segment_m  -> quant_fp8_blockwise_segment_m_kernel
"""

import argparse
import math

import torch
import triton

from aiter.ops.triton.quant.quant_fp8_blockwise import (
    quant_fp8_blockwise,
    quant_fp8_blockwise_for_act_grad,
    quant_fp8_blockwise_for_weight,
    quant_fp8_blockwise_segment_m,
    requant_fp8_row_to_col,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

_BS = 128


def _segment_indptrs(M, nseg, device):
    """Equal-length segments over M rows -> (seg_indptr, scales_seg_indptr)."""
    base = M // nseg
    lens = [base] * (nseg - 1) + [M - base * (nseg - 1)]
    seg = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device=device).int()
    blocks = [math.ceil(length / _BS) for length in lens]
    sseg = torch.tensor([0] + list(torch.tensor(blocks).cumsum(0)), device=device).int()
    return seg, sseg


def benchmark(args):
    x_vals = [(m, n) for m in [4096, 8192] for n in [4096, 7168, 8192]]
    unit = "ms" if args.metric == "time" else "GB/s"
    providers = ["blockwise", "weight", "act_grad", "requant", "segment_m"]

    config = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=providers,
        line_names=[f"{p} ({unit})" for p in providers],
        styles=[
            ("green", "-"),
            ("blue", "-"),
            ("red", "-"),
            ("orange", "-"),
            ("purple", "-"),
        ],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(M, N, provider):
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        if provider == "blockwise":
            fn = lambda: quant_fp8_blockwise(x, block_size=_BS, axis=1)
        elif provider == "weight":
            w = x.unsqueeze(0)  # [1, M, N]
            fn = lambda: quant_fp8_blockwise_for_weight(w, block_size=_BS)
        elif provider == "act_grad":
            fn = lambda: quant_fp8_blockwise_for_act_grad(x, block_size=_BS)
        elif provider == "requant":
            x_row, s_row = quant_fp8_blockwise(x, block_size=_BS, axis=1)
            fn = lambda: requant_fp8_row_to_col(x_row, s_row, block_size=_BS)
        else:  # segment_m
            seg, sseg = _segment_indptrs(M, nseg=4, device=x.device)
            fn = lambda: quant_fp8_blockwise_segment_m(x, 4, seg, sseg, block_size=_BS)

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        # Per-provider bytes moved (hp=2, fp8=1, scale fp32=4), incl. scales.
        hp, fp8, f32 = x.element_size(), 1, 4
        nb, mb = math.ceil(N / _BS), math.ceil(M / _BS)
        if provider == "blockwise":
            b = M * N * hp + M * N * fp8 + M * nb * f32
        elif provider == "weight":
            b = M * N * hp + M * N * fp8 + mb * nb * f32
        elif provider == "act_grad":
            b = M * N * hp + 2 * M * N * fp8 + M * nb * f32 + mb * N * f32
        elif provider == "requant":  # reads fp8 + row scales, writes fp8 + col scales
            b = M * N * fp8 + M * nb * f32 + M * N * fp8 + mb * N * f32
        else:  # segment_m
            b = M * N * hp + M * N * fp8 + mb * N * f32
        return b * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FP8 blockwise quant", allow_abbrev=False
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
