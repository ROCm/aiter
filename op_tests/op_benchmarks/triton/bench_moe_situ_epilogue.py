# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the merged MoE SiTU Triton epilogue."""

import argparse

import torch
import triton

from aiter.ops.triton.moe.moe_situ_epilogue import (
    moe_situ_epilogue,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

_M_VALUES = [1, 4, 16, 32, 64, 128, 192, 512, 1024, 1536, 2048]


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

    @triton.testing.perf_report([benchmark])
    def bench(M: int, provider: str) -> float:
        del provider
        projection_size = (
            2 * args.shared_intermediate_size
            + args.num_experts
            + args.routed_latent_size
        )
        merged_projection = torch.randn(
            (M, projection_size), dtype=torch.float32, device="cuda"
        )
        shared = torch.empty(
            (M, args.shared_intermediate_size),
            dtype=torch.bfloat16,
            device=merged_projection.device,
        )
        router = torch.empty(
            (M, args.num_experts),
            dtype=torch.float32,
            device=merged_projection.device,
        )
        routed = torch.empty(
            (M, args.routed_latent_size),
            dtype=torch.bfloat16,
            device=merged_projection.device,
        )

        def fn():
            return moe_situ_epilogue(
                merged_projection,
                shared_intermediate_size=args.shared_intermediate_size,
                num_experts=args.num_experts,
                routed_latent_size=args.routed_latent_size,
                shared_out=shared,
                router_out=router,
                routed_out=routed,
            )

        return (
            triton.testing.do_bench_cudagraph(
                fn,
                rep=args.rep,
                return_mode="mean",
            )
            * 1000
        )

    bench.run(save_path="." if args.output else None, print_data=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Benchmark merged MoE SiTU epilogue",
        allow_abbrev=False,
    )
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--shared-intermediate-size", type=int, default=768)
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--routed-latent-size", type=int, default=3584)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.manual_seed(0)
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
