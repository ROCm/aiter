# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the gfx1250 Gluon MoE GEMM with 16-bit activations and weights."""

import argparse
import functools
import sys

import torch

from aiter import logger
from aiter.ops.triton.moe.moe_op_gemm_a16w16 import moe_gemm_a16w16
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

_SUPPORTED_ARCHS = ("gfx1250",)

# Names the values make_inputs returns, so the pool entry is the kernel's kwargs.
_INPUT_KEYS = [
    "x",
    "w",
    "routing_data",
    "gather_indx",
    "scatter_indx",
    "apply_swiglu",
]

# (n, k, n_expts_tot, n_expts_act)
DEFAULT_SHAPES = [
    (512, 512, 8, 2),
    (1024, 1024, 32, 4),
    (3072, 3072, 128, 4),
    (6144, 3072, 32, 4),
]
DEFAULT_M = [1, 16, 64, 256, 1024, 4096]


def make_inputs(m, n, k, n_expts_tot, n_expts_act, apply_swiglu, device):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    return x, w, routing_data, gather_idx, scatter_idx, apply_swiglu


def run_shape(m, n, k, n_expts_tot, n_expts_act, args, device):
    # Pin the pool length to --rotating-buffers, which the shape loop sizes.
    bencher = RotatingBufferBench()
    bencher.min_pool_len = args.rotating_buffers
    bencher.max_pool_len = args.rotating_buffers
    gen = functools.partial(
        make_inputs, m, n, k, n_expts_tot, n_expts_act, args.swiglu, device
    )
    rot_bufs = bencher.build_pool([gen], [_INPUT_KEYS])
    ms = bencher.bench(
        moe_gemm_a16w16,
        warmup_iters=args.warmup,
        warmup_replays=args.warmup_replays,
        kernels_per_graph=args.kernels_per_graph,
        num_replays=args.replays,
    )

    # each routed token does one (k x n) matvec
    routed = m * n_expts_act
    flops = 2.0 * routed * n * k
    # Only experts that receive a token have their weights read, so counting all
    # of them would overstate the traffic (and the GB/s) at small m.
    hist = rot_bufs[0]["routing_data"].expt_data.hist
    active_expts = int((hist > 0).sum())
    nbytes = (routed * k + routed * n) * 2 + active_expts * k * n * 2
    return ms, flops / (ms * 1e-3) / 1e12, nbytes / (ms * 1e-3) / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, nargs="+", default=DEFAULT_M)
    parser.add_argument("-N", type=int, default=None, help="override N")
    parser.add_argument("-K", type=int, default=None, help="override K")
    parser.add_argument("-E", type=int, default=None, help="override num experts")
    parser.add_argument("--topk", type=int, default=None, help="override n_expts_act")
    parser.add_argument("--swiglu", action="store_true", help="fuse swiglu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup-replays", type=int, default=3)
    parser.add_argument("--kernels-per-graph", type=int, default=4)
    parser.add_argument("--rotating-buffers", type=int, default=4)
    args = parser.parse_args()

    arch = get_arch()
    if arch not in _SUPPORTED_ARCHS:
        logger.warning(
            "moe_gemm_a16w16 is only implemented for %s, got %s",
            _SUPPORTED_ARCHS,
            arch,
        )
        return 1

    if args.N is not None and args.K is not None:
        shapes = [(args.N, args.K, args.E or 8, args.topk or 2)]
    else:
        shapes = DEFAULT_SHAPES

    logger.warning(
        "%-6s %-6s %-6s %-5s %-5s %-10s %-10s %-10s",
        "M",
        "N",
        "K",
        "E",
        "topk",
        "ms",
        "TFLOP/s",
        "GB/s",
    )
    for n, k, n_expts_tot, n_expts_act in shapes:
        for m in args.M:
            ms, tflops, gbps = run_shape(
                m, n, k, n_expts_tot, n_expts_act, args, args.device
            )
            logger.warning(
                "%-6d %-6d %-6d %-5d %-5d %-10.4f %-10.2f %-10.2f",
                m,
                n,
                k,
                n_expts_tot,
                n_expts_act,
                ms,
                tflops,
                gbps,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
