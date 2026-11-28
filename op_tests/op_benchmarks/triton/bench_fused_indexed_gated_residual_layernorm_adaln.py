# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for the fused gated residual + LayerNorm + indexed adaLN kernel
against the eager expression and, where fused_layernorm_indexed_adaln is
available, against the two-kernel path (fused_indexed_gated_residual, then
fused_layernorm_indexed_adaln).
"""

import functools
import sys

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.fusions.fused_indexed_gated_residual import (
    fused_indexed_gated_residual,
)
from aiter.ops.triton.fusions.fused_indexed_gated_residual_layernorm_adaln import (
    fused_indexed_gated_residual_layernorm_adaln,
)
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
from op_tests.triton_tests.fusions.test_fused_indexed_gated_residual_layernorm_adaln import (
    make,
)

try:
    from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
        fused_layernorm_indexed_adaln,
    )
except ImportError:
    fused_layernorm_indexed_adaln = None

_PROVIDERS = ("fused", "eager") + (
    ("two_kernels",) if fused_layernorm_indexed_adaln is not None else ()
)
# [M, N] arrays moved: fused reads residual and x and writes h and out; the
# two-kernel path also re-reads h; eager runs nine ops, one pass each, and
# materialises the three gathered tables and every intermediate.
_ARRAYS = {"fused": 4, "two_kernels": 5, "eager": 19}
# (tokens, hidden): Wan 2.1 14B and 1.3B at 720p / 81 frames, Qwen-Image and
# FLUX.1 at 1024x1024, SD3.5 at 1024x1024 and 512x512.
_SHAPES = (
    (32760, 5120),
    (32760, 1536),
    (4096, 3072),
    (4608, 3072),
    (4096, 1536),
    (1024, 1536),
)

_OUT_VAL_KEYS = ["residual", "x", "gate", "shift", "scale", "indices"]


def _generate_pool_inputs(M, N, G, scattered):
    return make(M, N, G, torch.bfloat16, "cuda", runs=not scattered)


def _fused_fn(residual, x, gate, shift, scale, indices, **_):
    return fused_indexed_gated_residual_layernorm_adaln(
        residual, x, gate, shift, scale, indices
    )


def _two_kernels_fn(residual, x, gate, shift, scale, indices, **_):
    h = fused_indexed_gated_residual(residual, x, gate, indices)
    return fused_layernorm_indexed_adaln(h, shift, scale, indices), h


def _eager_fn(residual, x, gate, shift, scale, indices, **_):
    h = residual + gate[indices] * x
    normed = F.layer_norm(h, (h.shape[1],), eps=1e-6)
    return normed * (1 + scale[indices]) + shift[indices], h


_FNS = {"fused": _fused_fn, "two_kernels": _two_kernels_fn, "eager": _eager_fn}


def bench_fn_impl(bencher, M, N, provider, metric, args):
    _p_gen = functools.partial(_generate_pool_inputs, M, N, args.G, args.scattered)
    rot_bufs = bencher.build_pool([_p_gen], [_OUT_VAL_KEYS])
    mem = _ARRAYS[provider] * M * N * rot_bufs[0]["residual"].element_size()

    ms = bencher.bench(_FNS[provider])
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]
    if args.M:
        x_vals = [(args.M, args.N)]
    elif args.N:
        x_vals = [(m, args.N) for m, _ in _SHAPES]
    else:
        x_vals = list(_SHAPES)

    benchmark = triton.testing.Benchmark(
        x_names=["M", "N"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("yellow", "-"),
            ("black", "-"),
            ("cyan", "-"),
        ][: len(line_vals)],
        ylabel="",
        plot_name=get_caller_name_no_ext()
        + ("_scattered" if args.scattered else "_runs"),
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
    parser = get_parser(kernel_name="fused indexed gated residual layernorm adaln")
    parser.add_argument("-M", type=int, default=None, help="Number of tokens")
    parser.add_argument("-N", type=int, default=None, help="Hidden size")
    parser.add_argument("-G", type=int, default=1, help="Table rows")
    parser.add_argument(
        "--scattered",
        action="store_true",
        help="Random index per token instead of runs of one index",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="fused kernel, two-kernel path, eager expression, or all",
    )
    override_parser_action(
        parser,
        action="--metric",
        choices=["all", "time", "bandwidth"],
        default="all",
        help="Metric to report (default: all)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Unused: RotatingBufferBench.bench's warmup is controlled by "
        "--graph-warmup-iters. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Unused: RotatingBufferBench.bench's replay count is a fixed "
        "constant, not an adaptive rep target. Kept for CLI compatibility.",
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
    if args.M and not args.N:
        raise SystemExit("-M needs -N")
    if args.print_vgpr:
        print(
            "Retrieving VGPR usage for fused_indexed_gated_residual_layernorm_adaln..."
        )
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
