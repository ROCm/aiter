# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for DSV4 indexer ops: dsv4_indexer (fwd+topk) and indexer_bwd.

Usage:
    python bench_dsv4_indexer.py
    python bench_dsv4_indexer.py --op bwd
    python bench_dsv4_indexer.py --metric bandwidth
"""

import functools

import torch
import triton

from aiter.ops.triton.attention.dsv4_indexer import dsv4_indexer, indexer_bwd
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


def _generate_dsv4_indexer_bwd_pool_inputs(S, H, Hd, P, compress_ratio, topk):
    q, k, w = _make_inputs(S, H, Hd, P)
    d_scores = torch.randn(S, P, dtype=torch.float32, device="cuda")
    return q, k, w, d_scores


def benchmark(args):
    op = args.op
    unit = "ms" if args.metric == "time" else "GB/s"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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
        hp = torch.bfloat16.itemsize
        if op == "fwd":
            _p_gen = functools.partial(_make_inputs, S, H, Hd, P)
            _out_val_keys = ["q", "k", "w"]

            def kernel_fn(q, k, w):
                dsv4_indexer(q, k, w, compress_ratio, topk)

            # reads: q + k + w; writes: scores + indices
            mem = S * H * Hd * hp + P * Hd * hp + S * H * 4 + S * min(topk, P) * (4 + 4)
        else:  # bwd
            _p_gen = functools.partial(
                _generate_dsv4_indexer_bwd_pool_inputs,
                S,
                H,
                Hd,
                P,
                compress_ratio,
                topk,
            )
            _out_val_keys = ["q", "k", "w", "d_scores"]

            def kernel_fn(q, k, w, d_scores):
                indexer_bwd(q, k, w, d_scores, compress_ratio)

            mem = (
                S * H * Hd * hp
                + P * Hd * hp
                + S * H * 4
                + S * P * 4
                + S * H * Hd * 4  # dq
                + P * Hd * 4  # dk
                + S * H * 4  # dw
            )

        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], **kwargs_rot_bufs
        )
        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="DSV4 Indexer")
    parser.add_argument(
        "--op",
        choices=["fwd", "bwd"],
        default="fwd",
        help="Which op to benchmark (default: fwd)",
    )
    override_parser_action(
        parser, action="--metric", choices=["time", "bandwidth"], default="time"
    )
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    benchmark(args)


if __name__ == "__main__":
    main()
