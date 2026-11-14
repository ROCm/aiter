# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the BF16 Triton stages of aiter.fused_moe on preshuffled weights."""

import argparse

import torch
import triton

from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.moe.fused_moe_a16w16 import (
    fused_moe_a16w16_stage1,
    fused_moe_a16w16_stage2,
    parse_triton_a16w16_kernel,
)
from op_tests.op_benchmarks.triton.utils.argparse import get_parser
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)

_M_VALUES = [1, 4, 16, 64, 256, 1024, 4096]


def run_benchmark(args: argparse.Namespace) -> None:
    block_m = parse_triton_a16w16_kernel(args.stage1)["bm"]
    if parse_triton_a16w16_kernel(args.stage2)["bm"] != block_m:
        raise ValueError("--stage1 and --stage2 need the same bm")
    m_values = [args.m] if args.m is not None else _M_VALUES
    benchmark = triton.testing.Benchmark(
        x_names=["M"],
        x_vals=m_values,
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton a16w16 2-stage"],
        styles=[("green", "-")],
        ylabel="Time (us)",
        plot_name=get_caller_name_no_ext(),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(M: int, provider: str) -> float:
        del provider
        D, inter, E, topk = args.model_dim, args.inter_dim, args.experts, args.topk
        x = torch.randn((M, D), dtype=torch.bfloat16, device="cuda")
        w1 = shuffle_weight(
            torch.randn((E, 2 * inter, D), dtype=torch.bfloat16, device="cuda")
            / D**0.5,
            (16, 16),
        )
        w2 = shuffle_weight(
            torch.randn((E, D, inter), dtype=torch.bfloat16, device="cuda")
            / inter**0.5,
            (16, 16),
        )
        logits = torch.randn((M, E), dtype=torch.float32, device="cuda")
        topk_weights, topk_ids = torch.topk(logits.softmax(-1), topk, dim=-1)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = moe_sorting(
            topk_ids.to(torch.int32),
            topk_weights,
            E,
            D,
            torch.bfloat16,
            block_m,
        )
        inter_states = torch.empty(
            (M, topk, inter), dtype=torch.bfloat16, device=x.device
        )

        def fn():
            fused_moe_a16w16_stage1(
                x,
                w1,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                inter_states,
                topk,
                block_m,
                args.stage1,
            )
            fused_moe_a16w16_stage2(
                inter_states,
                w2,
                sorted_ids,
                sorted_expert_ids,
                sorted_weights,
                num_valid_ids,
                out,
                topk,
                block_m,
                args.stage2,
            )

        return (
            triton.testing.do_bench_cudagraph(fn, rep=args.rep, return_mode="mean")
            * 1000
        )

    bench.run(save_path="." if args.output else None, print_data=True)


def parse_args() -> argparse.Namespace:
    parser = get_parser(kernel_name="fused MoE a16w16 (Triton stages)")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--model-dim", type=int, default=4096)
    parser.add_argument("--inter-dim", type=int, default=512)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--stage1", default="triton_moe1_a16w16_bm32_bn128_bk64_gm8_w4_s2"
    )
    parser.add_argument(
        "--stage2", default="triton_moe2_a16w16_bm32_bn128_bk64_gm8_w4_s2"
    )
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("-o", "--output", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.manual_seed(0)
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
