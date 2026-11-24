# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the 16-bit MoE GEMM helpers.

On gfx1250 this exercises the existing Gluon ``moe_gemm_a16w16`` path.
On gfx1201 it benchmarks the selected-large-M BF16 G1U1 fused-MoE helper.
"""

import argparse
import functools
import sys
from functools import partial

import torch
import triton

from aiter import dtypes, logger
from aiter.ops.triton.moe.moe_op_gemm_a16w16 import (
    fused_moe_bf16_g1u1_gfx1201_large_m,
    moe_gemm_a16w16,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)
_GFX1201_SUPPORTED_ARCHS = ("gfx1201",)
_GFX1201_SELECTED_TOKENS = [1152, 1543, 1728, 1799, 1803, 1815, 1837, 1945, 2048]
_MODEL_DIM = 2048
_INTER_DIM = 128
_EXPERTS = 256
_TOPK = 8

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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{value!r} is not positive")
    return parsed


def make_inputs(m, n, k, n_expts_tot, n_expts_act, apply_swiglu, device):
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    w = torch.randn((n_expts_tot, k, n), dtype=torch.bfloat16, device=device)
    return x, w, routing_data, gather_idx, scatter_idx, apply_swiglu


def _make_case_gfx1201(tokens: int, device: torch.device):
    torch.manual_seed(2026 + tokens)
    hidden = torch.randn((tokens, _MODEL_DIM), dtype=dtypes.bf16, device=device)
    w1 = torch.randn(
        (_EXPERTS, 2 * _INTER_DIM, _MODEL_DIM),
        dtype=dtypes.bf16,
        device=device,
    ).contiguous()
    w2 = torch.randn(
        (_EXPERTS, _MODEL_DIM, _INTER_DIM),
        dtype=dtypes.bf16,
        device=device,
    ).contiguous()
    topk_ids = torch.randint(
        0,
        _EXPERTS,
        (tokens, _TOPK),
        dtype=torch.int32,
        device=device,
    )
    topk_weight = torch.rand((tokens, _TOPK), dtype=torch.float32, device=device)
    topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    return hidden, w1, w2, topk_weight.contiguous(), topk_ids.contiguous()


def _flops_gfx1201(tokens: int) -> int:
    routed = tokens * _TOPK
    stage1 = 2 * routed * _MODEL_DIM * (2 * _INTER_DIM)
    stage2 = 2 * routed * _INTER_DIM * _MODEL_DIM
    return stage1 + stage2


def run_shape(m, n, k, n_expts_tot, n_expts_act, args, device):
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

    routed = m * n_expts_act
    flops = 2.0 * routed * n * k
    hist = rot_bufs[0]["routing_data"].expt_data.hist
    active_expts = int((hist > 0).sum())
    nbytes = (routed * k + routed * n) * 2 + active_expts * k * n * 2
    return ms, flops / (ms * 1e-3) / 1e12, nbytes / (ms * 1e-3) / 1e9


def benchmark_gfx1201(args):
    device = torch.device(args.device)
    metric_name = "time_ms" if args.metric == "time" else "tflops"
    print(f"tokens,{metric_name}", flush=True)
    for tokens in args.tokens:
        hidden, w1, w2, topk_weight, topk_ids = _make_case_gfx1201(tokens, device)
        fn = partial(
            fused_moe_bf16_g1u1_gfx1201_large_m,
            hidden,
            w1,
            w2,
            topk_weight,
            topk_ids,
        )
        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
        if args.metric == "time":
            value = ms
        else:
            value = _flops_gfx1201(tokens) / ms * 1e-9
        print(f"{tokens},{value:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", type=int, nargs="+", default=DEFAULT_M)
    parser.add_argument("-N", type=int, default=None, help="override N")
    parser.add_argument("-K", type=int, default=None, help="override K")
    parser.add_argument("-E", type=int, default=None, help="override num experts")
    parser.add_argument("--topk", type=int, default=None, help="override n_expts_act")
    parser.add_argument("--swiglu", action="store_true", help="fuse swiglu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--warmup-replays", type=int, default=3)
    parser.add_argument("--kernels-per-graph", type=int, default=4)
    parser.add_argument("--rotating-buffers", type=int, default=4)
    parser.add_argument(
        "--metric",
        choices=["time", "throughput"],
        default="time",
        help="gfx1201 mode only",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=positive_int,
        default=_GFX1201_SELECTED_TOKENS,
        help="gfx1201 token counts to benchmark",
    )
    parser.add_argument(
        "--rep",
        type=positive_int,
        default=100,
        help="gfx1201 mode only",
    )
    args = parser.parse_args()

    arch = get_arch()
    if arch in _GFX1201_SUPPORTED_ARCHS:
        benchmark_gfx1201(args)
        return 0

    if arch not in _GLUON_SUPPORTED_ARCHS:
        logger.warning(
            "bench_moe_gemm_a16w16 supports %s or %s, got %s",
            _GLUON_SUPPORTED_ARCHS,
            _GFX1201_SUPPORTED_ARCHS,
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
