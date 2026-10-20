# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark for moe_wgrad.

Usage:
    python bench_moe_wgrad.py
    python bench_moe_wgrad.py --metric bandwidth
"""

import functools

import torch
import triton

from aiter.ops.moe_op import moe_align_block_size
from aiter.ops.triton.moe.moe_wgrad import moe_wgrad
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

# (num_tokens, E, N, K, top_k, label)
_SHAPES = [
    (4096, 8, 7168, 2048, 2, "DSv4-decode"),
    (16384, 8, 7168, 2048, 2, "DSv4-prefill"),
    (4096, 64, 2048, 7168, 6, "DSv4-E64-topk6"),
]
_BLOCK_SIZE_M = 64


def _setup_routing(num_tokens, E, top_k, device="cuda"):
    """Build moe_align_block_size's sorted-token routing once, shared by
    every pool entry.

    Rotating sorted_ids/expert_ids/ntpp across CUDA graph replay triggers a
    real GPU memory access fault in moe_wgrad (reproduced with as few as two
    distinct routing draws) -- a kernel/autotune interaction, not something
    this benchmark controls. Routing stays fixed; grad/inp still rotate,
    which is also where almost all the read/write traffic is.
    """
    torch.manual_seed(0)
    scores = torch.randn(num_tokens, E, device=device)
    topk_ids = scores.topk(top_k, dim=1).indices  # [num_tokens, top_k]

    # moe_align_block_size requires pre-allocated output buffers.
    max_blocks = triton.cdiv(num_tokens * top_k, _BLOCK_SIZE_M) + E
    sorted_ids = torch.empty(
        max_blocks * _BLOCK_SIZE_M, dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
    token_nums = torch.empty(E, dtype=torch.int32, device=device)
    ntpp = torch.empty(1, dtype=torch.int32, device=device)
    moe_align_block_size(
        topk_ids, E, _BLOCK_SIZE_M, sorted_ids, expert_ids, token_nums, ntpp
    )
    return sorted_ids, expert_ids, ntpp


def _generate_wgrad_pool_inputs(num_tokens, N, K, device="cuda"):
    grad = torch.randn(num_tokens, N, dtype=torch.bfloat16, device=device) * 0.1
    inp = torch.randn(num_tokens, K, dtype=torch.bfloat16, device=device) * 0.1
    return grad, inp


def benchmark(args):
    unit = "ms" if args.metric == "time" else "GB/s"
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    x_vals = list(_SHAPES)

    config = triton.testing.Benchmark(
        x_names=["num_tokens", "E", "N", "K", "top_k", "label"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["moe_wgrad"],
        line_names=[f"moe_wgrad ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(num_tokens, E, N, K, top_k, label, provider):
        sorted_ids, expert_ids, ntpp = _setup_routing(num_tokens, E, top_k)
        _p_gen = functools.partial(_generate_wgrad_pool_inputs, num_tokens, N, K)
        _out_val_keys = ["grad", "inp"]
        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], **kwargs_rot_bufs
        )

        def kernel_fn(grad, inp):
            moe_wgrad(
                grad,
                inp,
                sorted_ids,
                expert_ids,
                ntpp,
                num_experts=E,
                top_k=top_k,
                weight_shape=(E, N, K),
                block_size_m=_BLOCK_SIZE_M,
            )

        # reads: grad(T*N) + inp(T*K); writes: dW(E*N*K); bf16 = 2 bytes
        mem = (num_tokens * N + num_tokens * K + E * N * K) * 2
        ms = do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph)
        if args.metric == "time":
            return ms
        return mem * 1e-9 / (ms * 1e-3)

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="moe_wgrad")
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
