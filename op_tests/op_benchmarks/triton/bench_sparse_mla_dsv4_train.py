# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for DSV4 sparse-MLA training kernels: forward and backward.

Usage:
    python bench_sparse_mla_dsv4_train.py
    python bench_sparse_mla_dsv4_train.py --op bwd
    python bench_sparse_mla_dsv4_train.py --metric bandwidth
"""

import functools
import itertools

import torch
import triton

from aiter.ops.triton.attention.sparse_mla_dsv4_train import (
    sparse_mla_bwd,
    sparse_mla_fwd,
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


def _generate_fwd_pool_inputs(N, H, D, N_kv, topk, scale):
    q, kv, attn_sink, indices, _ = _make_inputs(N, H, D, N_kv, topk)
    return q, kv, attn_sink, indices


def _generate_bwd_pool_inputs(N, H, D, N_kv, topk, scale):
    q, kv, attn_sink, indices, _ = _make_inputs(N, H, D, N_kv, topk)
    o, lse = sparse_mla_fwd(q, kv, attn_sink, indices, scale)
    do = torch.randn_like(o)
    return q, kv, o, do, indices, lse, attn_sink


def _make_fwd_kernel_fn(scale):
    def kernel_fn(q, kv, attn_sink, indices):
        sparse_mla_fwd(q, kv, attn_sink, indices, scale)

    return kernel_fn


def _make_bwd_kernel_fn(scale):
    def kernel_fn(q, kv, o, do, indices, lse, attn_sink):
        sparse_mla_bwd(q, kv, o, do, indices, lse, attn_sink, scale)

    return kernel_fn


def _rotating_eager_call(kernel_fn, rot_bufs):
    """Cycle through a rotating-buffer pool without CUDA graph capture.

    sparse_mla_bwd's _build_inverted_topk compacts a boolean mask
    (``positions[valid]``), a data-dependent output shape that CUDA graph
    capture cannot record, so do_bench_aiter_triton is not usable here. This
    keeps the same cache-defeating pool rotation under plain do_bench.
    """
    pool_len = len(rot_bufs)
    counter = itertools.count()

    def fn():
        kernel_fn(**rot_bufs[next(counter) % pool_len])

    return fn


def benchmark(args):
    op = args.op
    unit = "ms" if args.metric == "time" else "GB/s"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)

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
        scale = 1.0 / (D**0.5)
        elem = torch.bfloat16.itemsize

        if op == "fwd":
            pool_gen = functools.partial(
                _generate_fwd_pool_inputs, N, H, D, N_kv, topk, scale
            )
            _out_val_keys = ["q", "kv", "attn_sink", "indices"]
            kernel_fn = _make_fwd_kernel_fn(scale)
            # reads: q + kv (gathered topk) + attn_sink; writes: o + lse
            mem = (
                N * H * D * elem
                + N * topk * D * elem
                + H * 4
                + N * H * D * elem  # o
                + N * H * 4  # lse
            )
        else:  # bwd via sparse_mla_dsv4_train
            pool_gen = functools.partial(
                _generate_bwd_pool_inputs, N, H, D, N_kv, topk, scale
            )
            _out_val_keys = ["q", "kv", "o", "do", "indices", "lse", "attn_sink"]
            kernel_fn = _make_bwd_kernel_fn(scale)
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

        rot_bufs = generate_rotating_buffers_pool(
            [pool_gen], [_out_val_keys], **kwargs_rot_bufs
        )
        if op == "fwd":
            ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        else:
            ms = triton.testing.do_bench(
                _rotating_eager_call(kernel_fn, rot_bufs), warmup=25, rep=100
            )
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="Benchmark DSV4 sparse-MLA training")
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
