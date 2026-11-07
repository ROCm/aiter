# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for fused_qk_norm_rope_across_heads against the eager diffusers Wan
ops and against aiter rms_norm + rope_cached_fwd.
"""

import functools
import sys

import torch
import triton

from aiter.ops.triton.fusions.fused_qk_norm_rope_across_heads import (
    fused_qk_norm_rope_across_heads,
)
from aiter.ops.triton.normalization.rmsnorm import rms_norm
from aiter.ops.triton.rope.rope import RotateStyle, rope_cached_fwd
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
    print_vgpr,
)
from op_tests.triton_tests.fusions.test_fused_qk_norm_rope_across_heads import (
    generate_qk_norm_rope_inputs,
    torch_qk_norm_rope_ref,
)

arg_to_torch_dtype = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

_PROVIDERS = ("triton", "eager", "aiter_unfused")
_EPS = 1e-6
# (S_q, S_k, heads, rope): Wan attn1 (self, norm + RoPE) and attn2 (cross, norm
# only, 512 text tokens) of Wan2.1-1.3B (480p), Wan2.2-TI2V-5B and Wan2.1-14B (480p).
_WAN_SHAPES = [
    (32760, 32760, 12, 1),
    (32760, 512, 12, 0),
    (27280, 27280, 24, 1),
    (27280, 512, 24, 0),
    (32760, 32760, 40, 1),
    (32760, 512, 40, 0),
]


def get_benchmark_shapes(args):
    """Return [(S_q, S_k, heads, rope), ...] for the current CLI args."""
    if args.S_q:
        return [(args.S_q, args.S_k or args.S_q, args.heads, int(not args.no_rope))]
    return _WAN_SHAPES


def bench_fn_impl(S_q, S_k, heads, rope, provider, metric, args):
    dtype = arg_to_torch_dtype[args.dtype]
    hd = args.head_dim
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    _p_gen = functools.partial(
        generate_qk_norm_rope_inputs,
        1,
        S_q,
        S_k,
        heads,
        hd,
        dtype,
        bool(rope),
        False,
        interleaved=True,
    )
    rot_bufs = generate_rotating_buffers_pool(
        [_p_gen], [["q", "k", "wq", "wk", "cos", "sin"]], **kwargs_rot_bufs
    )
    q = rot_bufs[0]["q"]
    N = heads * hd
    # q and k are read once and written once.
    mem = 2 * (S_q + S_k) * N * q.element_size()

    if provider == "triton":

        def _run(q, k, wq, wk, cos, sin):
            fused_qk_norm_rope_across_heads(q, k, wq, wk, _EPS, hd, cos, sin)

    elif provider == "eager":

        def _run(q, k, wq, wk, cos, sin):
            torch_qk_norm_rope_ref(q, k, wq, wk, _EPS, hd, cos, sin)

    else:

        def _run(q, k, wq, wk, cos, sin):
            qn = rms_norm(q.view(S_q, N), wq, _EPS)
            kn = rms_norm(k.view(S_k, N), wk, _EPS)
            if rope:
                cs, sn = cos.view(S_q, 1, 1, hd), sin.view(S_q, 1, 1, hd)
                gptj = RotateStyle.GPTJ
                rope_cached_fwd(qn.view(S_q, 1, heads, hd), cs, sn, gptj, False, False)
                rope_cached_fwd(kn.view(S_k, 1, heads, hd), cs, sn, gptj, False, False)

    ms = do_bench_aiter_triton(_run, rot_bufs, **kwargs_do_bench_graph)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = _PROVIDERS if args.provider == "all" else (args.provider,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["S_q", "S_k", "heads", "rope"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")]
        * ((len(line_vals) + 3) // 4),
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.dtype}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(S_q, S_k, heads, rope, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn_impl(S_q, S_k, heads, rope, name, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = get_parser(kernel_name="fused QK norm across heads + RoPE")
    parser.add_argument("--S_q", type=int, default=None, help="Query tokens")
    parser.add_argument("--S_k", type=int, default=None, help="Key tokens")
    parser.add_argument("--heads", type=int, default=12, help="Attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dim")
    parser.add_argument("--no_rope", action="store_true", help="Norm only")
    parser.add_argument(
        "--provider",
        type=str,
        default="all",
        choices=[*_PROVIDERS, "all"],
        help="Triton kernel, eager PyTorch, aiter rms_norm + rope_cached_fwd, or all",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(arg_to_torch_dtype)
    )
    override_parser_action(
        parser, action="--metric", choices=["all", "time", "bandwidth"], default="all"
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
        print("Retrieving VGPR usage for fused_qk_norm_rope_across_heads kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
