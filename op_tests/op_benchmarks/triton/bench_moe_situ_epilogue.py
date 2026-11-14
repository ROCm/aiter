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
    get_caller_name_no_ext,
)
from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
    RotatingBufferBench,
)

_M_VALUES = [1, 4, 16, 32, 64, 128, 192, 512, 1024, 1536, 2048]

_OUT_VAL_KEYS = ["merged_projection", "shared_out", "router_out", "routed_out"]


def _generate_pool_inputs(M, shared_intermediate_size, num_experts, routed_latent_size):
    projection_size = 2 * shared_intermediate_size + num_experts + routed_latent_size
    merged_projection = torch.randn(
        (M, projection_size), dtype=torch.float32, device="cuda"
    )
    shared_out = torch.empty(
        (M, shared_intermediate_size), dtype=torch.bfloat16, device="cuda"
    )
    router_out = torch.empty((M, num_experts), dtype=torch.float32, device="cuda")
    routed_out = torch.empty(
        (M, routed_latent_size), dtype=torch.bfloat16, device="cuda"
    )
    return merged_projection, shared_out, router_out, routed_out


def run_benchmark(args: argparse.Namespace) -> None:
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
    bencher = RotatingBufferBench.from_dict(
        {**get_rot_bufs_kwargs(args), **get_do_bench_graph_kwargs(args)}
    )

    @triton.testing.perf_report([benchmark])
    def bench(M: int, provider: str) -> float:
        del provider
        _p_gen = functools.partial(
            _generate_pool_inputs,
            M,
            args.shared_intermediate_size,
            args.num_experts,
            args.routed_latent_size,
        )
        bencher.build_pool([_p_gen], [_OUT_VAL_KEYS])

        kernel_fn = functools.partial(
            moe_situ_epilogue,
            shared_intermediate_size=args.shared_intermediate_size,
            num_experts=args.num_experts,
            routed_latent_size=args.routed_latent_size,
        )
        return bencher.bench(kernel_fn) * 1000

    bench.run(save_path="." if args.output else None, print_data=True)


def parse_args() -> argparse.Namespace:
    parser = get_parser(kernel_name="merged MoE SiTU epilogue")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--shared-intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--routed-latent-size", type=int, default=3584)
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.manual_seed(0)
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
