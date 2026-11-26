# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the DiT producers with a dynamic per-token FP8 output against
the unfused path: the existing BF16 producer followed by
``dynamic_per_token_quant_fp8_i8``.

--op adaln:  fused_rmsnorm_indexed_adaln_fp8_per_token_quant
--op gated:  fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant
--op silu:   fused_silu_mul_fp8_per_token_quant
"""

import functools
import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.fusions.fused_indexed_gated_residual import (
    fused_indexed_gated_residual,
)
from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
    fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant,
    fused_rmsnorm_indexed_adaln,
    fused_rmsnorm_indexed_adaln_fp8_per_token_quant,
)
from aiter.ops.triton.quant.fused_fp8_quant import fused_silu_mul_fp8_per_token_quant
from aiter.ops.triton.quant.quant import dynamic_per_token_quant_fp8_i8
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
    override_parser_action,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

_PROVIDERS = ("fused", "unfused")
_FP8 = get_fp8_e4m3_dtype()
# MiniMax-H3: hidden 5376, MLP 14336; 4144 tokens for a 1024x1024 image,
# 57330 per rank for a 15 s 768p video with two-way sequence parallelism.
_DEFAULT_N = {"adaln": 5376, "gated": 5376, "silu": 14336}
_TOKENS = (1024, 4144, 16384, 57330)
# Bytes moved per [M, N] element, for element size e of the input dtype.
_BYTES = {
    ("adaln", "fused"): lambda e: e + 1,
    ("adaln", "unfused"): lambda e: 3 * e + 1,
    ("gated", "fused"): lambda e: 3 * e + 1,
    ("gated", "unfused"): lambda e: 6 * e + 1,
    ("silu", "fused"): lambda e: 2 * e + 1,
    ("silu", "unfused"): lambda e: 6 * e + 1,
}
_ROT_KEYS = {
    "adaln": ["x", "indices"],
    "gated": ["residual", "branch", "indices"],
    "silu": ["inp"],
}
_FIXED_KEYS = {
    "adaln": ["weight", "shift", "scale"],
    "gated": ["weight", "shift", "scale", "gate"],
    "silu": [],
}


def _indices(M, G):
    # Runs of one index per token block, as in a batch of G conditions.
    return (torch.arange(M, device="cuda") * G // max(M, 1)).to(torch.int64)


def _gen_rot(op, M, N, G, dtype):
    if op == "adaln":
        return torch.randn(M, N, device="cuda", dtype=dtype), _indices(M, G)
    if op == "gated":
        return (
            torch.randn(M, N, device="cuda", dtype=dtype),
            torch.randn(M, N, device="cuda", dtype=dtype),
            _indices(M, G),
        )
    return (torch.randn(M, 2 * N, device="cuda", dtype=dtype),)


def _gen_fixed(op, N, G, dtype):
    tables = (
        torch.rand(N, device="cuda", dtype=dtype) + 0.5,
        torch.randn(G, N, device="cuda", dtype=dtype),
        torch.randn(G, N, device="cuda", dtype=dtype) * 0.1,
    )
    if op == "gated":
        return (*tables, torch.randn(G, N, device="cuda", dtype=dtype) * 0.5)
    return tables


def _quant(y):
    q = torch.empty_like(y, dtype=_FP8)
    s = torch.empty(y.shape[0], dtype=torch.float32, device=y.device)
    return dynamic_per_token_quant_fp8_i8(q, y, s)


def _adaln_fused(x, weight, shift, scale, indices, **_):
    return fused_rmsnorm_indexed_adaln_fp8_per_token_quant(
        x, weight, shift, scale, indices
    )


def _adaln_unfused(x, weight, shift, scale, indices, **_):
    return _quant(fused_rmsnorm_indexed_adaln(x, weight, shift, scale, indices))


def _gated_fused(residual, branch, gate, weight, shift, scale, indices, **_):
    return fused_gated_residual_rmsnorm_indexed_adaln_fp8_per_token_quant(
        residual, gate, branch, weight, shift, scale, indices
    )


def _gated_unfused(residual, branch, gate, weight, shift, scale, indices, **_):
    updated = fused_indexed_gated_residual(
        residual, branch, gate, indices, round_intermediate=True
    )
    return updated, _quant(
        fused_rmsnorm_indexed_adaln(updated, weight, shift, scale, indices)
    )


def _silu_fused(inp, **_):
    return fused_silu_mul_fp8_per_token_quant(inp, dtype_quant=_FP8)


def _silu_unfused(inp, **_):
    gate, up = inp.chunk(2, dim=-1)
    return _quant(F.silu(gate) * up)


_FNS = {
    ("adaln", "fused"): _adaln_fused,
    ("adaln", "unfused"): _adaln_unfused,
    ("gated", "fused"): _gated_fused,
    ("gated", "unfused"): _gated_unfused,
    ("silu", "fused"): _silu_fused,
    ("silu", "unfused"): _silu_unfused,
}


def bench_fn_impl(bencher, M, N, provider, metric, args):
    dtype = torch.bfloat16
    op = args.op
    rot_keys, fixed_keys = _ROT_KEYS[op], _FIXED_KEYS[op]
    bencher.build_pool(
        [functools.partial(_gen_rot, op, M, N, args.G, dtype)],
        [rot_keys],
        [functools.partial(_gen_fixed, op, N, args.G, dtype)] if fixed_keys else None,
        [fixed_keys] if fixed_keys else None,
    )
    mem = _BYTES[(op, provider)](dtype.itemsize) * M * N

    ms = bencher.bench(_FNS[(op, provider)])
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]
    N = args.N or _DEFAULT_N[args.op]
    x_vals = [(args.M, N)] if args.M else [(m, N) for m in _TOKENS]

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")][
            : len(line_vals)
        ],
        ylabel="",
        plot_name=f"{get_caller_name_no_ext()}_{args.op}",
        args={},
    )

    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(M, N, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(bencher, M, N, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="DiT producers with per-token FP8 output")
    parser.add_argument(
        "--op",
        type=str,
        default="adaln",
        choices=["adaln", "gated", "silu"],
        help="Producer to benchmark",
    )
    parser.add_argument("-M", type=int, default=None, help="Number of tokens")
    parser.add_argument(
        "-N",
        type=int,
        default=None,
        help="Output width (default: 5376 for adaln/gated, 14336 for silu)",
    )
    parser.add_argument("-G", type=int, default=9, help="Modulation table rows")
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused FP8 producer, BF16 producer + per-token quant, or both",
    )
    override_parser_action(
        parser,
        action="--metric",
        choices=["all", "time", "bandwidth"],
        default="all",
        help="Metric to report (default: all)",
    )
    parser.add_argument(
        "-print_vgpr",
        action="store_true",
        default=False,
        help="Print VGPR usage for Triton kernels",
    )
    parser.add_argument(
        "-o", action="store_true", default=False, help="Write results to a CSV file"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_vgpr:
        print("Retrieving VGPR usage for the per-token FP8 producers...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
