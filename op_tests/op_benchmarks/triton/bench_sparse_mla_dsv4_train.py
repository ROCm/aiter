# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for DSV4 sparse-MLA training kernels: forward and backward.

Usage:
    python bench_sparse_mla_dsv4_train.py
    python bench_sparse_mla_dsv4_train.py --op bwd
    python bench_sparse_mla_dsv4_train.py -metric bandwidth
"""

import argparse

import torch
import triton

from aiter.ops.triton.attention.sparse_mla_dsv4_train import (
    sparse_mla_bwd,
    sparse_mla_fwd,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import get_caller_name_no_ext

# Representative DSV4 shapes: (N, H, D, N_kv, topk)
_SHAPES = [
    (512, 16, 512, 4096, 64),
    (1024, 16, 512, 4096, 64),
    (2048, 16, 512, 4096, 128),
    (4096, 16, 512, 8192, 128),
    (8192, 16, 512, 8192, 256),
]


def _make_inputs(N, H, D, N_kv, topk, device="cuda"):
    q = torch.randn(N, H, D, dtype=torch.bfloat16, device=device) * 0.1
    kv = torch.randn(N_kv, D, dtype=torch.bfloat16, device=device) * 0.1
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)
    indices = torch.randint(0, N_kv, (N, topk), dtype=torch.int32, device=device)
    scale = 1.0 / (D**0.5)
    return q, kv, attn_sink, indices, scale


def benchmark(args):
    op = args.op
    unit = "ms" if args.metric == "time" else "GB/s"

    x_vals = [(N, H, D, N_kv, topk) for N, H, D, N_kv, topk in _SHAPES]

    config = triton.testing.Benchmark(
        x_names=["N", "H", "D", "N_kv", "topk"],
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
    def _run(N, H, D, N_kv, topk, provider):
        q, kv, attn_sink, indices, scale = _make_inputs(N, H, D, N_kv, topk)
        elem = q.element_size()

        if op == "fwd":
            fn = lambda: sparse_mla_fwd(q, kv, attn_sink, indices, scale)
            # reads: q + kv (gathered topk) + attn_sink; writes: o + lse
            mem = (
                N * H * D * elem
                + N * topk * D * elem
                + H * 4
                + N * H * D * elem  # o
                + N * H * 4  # lse
            )
        else:  # bwd via sparse_mla_dsv4_train autograd
            q_g = q.detach().requires_grad_(True)
            kv_g = kv.detach().requires_grad_(True)
            o, lse = sparse_mla_fwd(q_g, kv_g, attn_sink, indices, scale)
            do = torch.randn_like(o)

            fn = lambda: sparse_mla_bwd(
                q_g, kv_g, o, do, indices, lse, attn_sink, scale
            )
            mem = (
                N * H * D * elem  # q
                + N_kv * D * elem  # kv
                + N * H * D * elem  # o
                + N * H * D * elem  # do
                + N * topk * 4  # indices
                + N * H * 4  # lse
                + N * H * D * elem  # dq
                + N_kv * D * elem  # dkv
            )

        ms = triton.testing.do_bench(fn, warmup=25, rep=100)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark DSV4 sparse-MLA training", allow_abbrev=False
    )
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
