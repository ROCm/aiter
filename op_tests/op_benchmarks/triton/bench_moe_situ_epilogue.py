# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the merged MoE SiTU Triton epilogue."""

import argparse
import functools

import torch
import triton

from aiter.ops.triton.moe.moe_situ_epilogue import (
    moe_situ_epilogue,
)
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
    get_caller_name_no_ext,
)

_M_VALUES = [1, 4, 16, 32, 64, 128, 192, 512, 1024, 1536, 2048]


def _generate_situ_epilogue_pool_inputs(
    M, projection_size, shared_size, num_experts, routed_size
):
    merged_projection = torch.randn(
        (M, projection_size), dtype=torch.float32, device="cuda"
    )
    shared = torch.empty((M, shared_size), dtype=torch.bfloat16, device="cuda")
    router = torch.empty((M, num_experts), dtype=torch.float32, device="cuda")
    routed = torch.empty((M, routed_size), dtype=torch.bfloat16, device="cuda")
    return merged_projection, shared, router, routed


def run_benchmark(args: argparse.Namespace) -> None:
    kwargs_do_bench_graph = get_do_bench_graph_kwargs(args)
    kwargs_rot_bufs = get_rot_bufs_kwargs(args)
    m_values = [args.m] if args.m is not None else _M_VALUES
    benchmark = triton.testing.Benchmark(
        x_names=["M"],
        x_vals=m_values,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton epilogue"],
        styles=[("green", "-")],
        ylabel="Time (us)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(M: int, provider: str) -> float:
        del provider
        projection_size = (
            2 * args.shared_intermediate_size
            + args.num_experts
            + args.routed_latent_size
        )
        _p_gen = functools.partial(
            _generate_situ_epilogue_pool_inputs,
            M,
            projection_size,
            args.shared_intermediate_size,
            args.num_experts,
            args.routed_latent_size,
        )
        _out_val_keys = ["merged_projection", "shared", "router", "routed"]
        rot_bufs = generate_rotating_buffers_pool(
            [_p_gen], [_out_val_keys], **kwargs_rot_bufs
        )

        def kernel_fn(merged_projection, shared, router, routed):
            moe_situ_epilogue(
                merged_projection,
                shared_intermediate_size=args.shared_intermediate_size,
                num_experts=args.num_experts,
                routed_latent_size=args.routed_latent_size,
                shared_out=shared,
                router_out=router,
                routed_out=routed,
            )

        return (
            do_bench_aiter_triton(kernel_fn, rot_bufs, **kwargs_do_bench_graph) * 1000
        )

    bench.run(save_path="." if args.output else None, print_data=True)


def parse_args() -> argparse.Namespace:
    parser = get_parser(kernel_name="merged MoE SiTU epilogue")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--shared-intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--routed-latent-size", type=int, default=3584)
    parser.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Unused: do_bench_aiter_triton's replay count is a fixed constant, "
        "not an adaptive rep-ms target. Kept for CLI compatibility.",
    )
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.manual_seed(0)
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
