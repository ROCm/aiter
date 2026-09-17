# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for DSV4 indexer ops: dsv4_indexer (fwd+topk) and indexer_bwd.

Usage:
    python bench_dsv4_indexer.py
    python bench_dsv4_indexer.py --op bwd
    python bench_dsv4_indexer.py -metric bandwidth
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer, indexer_bwd
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# DSV4 representative shapes: (S, H, Hd, P, compress_ratio, topk)
_SHAPES = [
    (512, 4, 128, 128, 4, 32),
    (1024, 4, 128, 256, 4, 64),
    (2048, 4, 128, 512, 4, 128),
    (4096, 4, 128, 1024, 4, 256),
    (8192, 4, 128, 2048, 4, 512),
]


def _make_inputs(S, H, Hd, P, device="cuda"):
    q = torch.randn(S, H, Hd, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn(P, Hd, dtype=torch.bfloat16, device=device) * 0.1
    w = torch.ones(S, H, dtype=torch.float32, device=device)
    return q, k, w


def benchmark(args):
    op = args.op
    unit = "ms" if args.metric == "time" else "GB/s"

    x_vals = [(S, H, Hd, P, cr, topk) for S, H, Hd, P, cr, topk in _SHAPES]

    config = triton.testing.Benchmark(
        x_names=["S", "H", "Hd", "P", "compress_ratio", "topk"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=[op],
        line_names=[f"{op} ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(S, H, Hd, P, compress_ratio, topk, provider):
        q, k, w = _make_inputs(S, H, Hd, P)
        if op == "fwd":
            fn = lambda: dsv4_indexer(q, k, w, compress_ratio, topk)
            # reads: q + k + w; writes: scores + indices
            mem = (
                S * H * Hd * q.element_size()
                + P * Hd * k.element_size()
                + S * H * 4
                + S * min(topk, P) * (4 + 4)
            )
        else:  # bwd
            _scores, _ = dsv4_indexer(q, k, w, compress_ratio, topk)
            d_scores = torch.randn(S, P, dtype=torch.float32, device="cuda")
            fn = lambda: indexer_bwd(q, k, w, d_scores, compress_ratio)
            mem = (
                S * H * Hd * q.element_size()
                + P * Hd * k.element_size()
                + S * H * 4
                + S * P * 4
                + S * H * Hd * 4  # dq
                + P * Hd * 4  # dk
                + S * H * 4  # dw
            )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(prog="Benchmark DSV4 Indexer", allow_abbrev=False)
    parser.add_argument(
        "--op",
        choices=["fwd", "bwd"],
        default="fwd",
        help="Which op to benchmark (default: fwd)",
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
