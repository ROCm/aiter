# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter import dtypes
from aiter.ops.triton.moe.moe_op_gemm_a16w16 import fused_moe_bf16_g1u1
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_parser,
    override_parser_action,
    positive_int,
)

_SELECTED_TOKENS = [1152, 1543, 1728, 1799, 1803, 1815, 1837, 1945, 2048]
_MODEL_DIM = 2048
_INTER_DIM = 128
_EXPERTS = 256
_TOPK = 8


def _make_case(tokens: int, device: torch.device):
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


def _flops(tokens: int) -> int:
    routed = tokens * _TOPK
    stage1 = 2 * routed * _MODEL_DIM * (2 * _INTER_DIM)
    stage2 = 2 * routed * _INTER_DIM * _MODEL_DIM
    return stage1 + stage2


def benchmark(args):
    device = torch.device("cuda")
    unit = "ms" if args.metric == "time" else "TFLOPS"
    config = triton.testing.Benchmark(
        x_names=["tokens"],
        x_vals=args.tokens,
        line_arg="provider",
        line_vals=["moe_gemm_a16w16"],
        line_names=[f"moe_gemm_a16w16 ({unit})"],
        styles=[("blue", "-")],
        ylabel=unit,
        plot_name="bench_moe_gemm_a16w16",
        args={},
    )

    @triton.testing.perf_report([config])
    def _run(tokens, provider):
        hidden, w1, w2, topk_weight, topk_ids = _make_case(tokens, device)
        fn = lambda: fused_moe_bf16_g1u1(hidden, w1, w2, topk_weight, topk_ids)
        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
        if args.metric == "time":
            return ms
        return _flops(tokens) / ms * 1e-9

    _run.run(save_path="." if args.o else None, print_data=True, show_plots=False)


def parse_args():
    parser = get_parser(kernel_name="moe_gemm_a16w16")
    override_parser_action(
        parser,
        action="--metric",
        choices=["time", "throughput"],
        default="time",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=positive_int,
        default=_SELECTED_TOKENS,
        help="Token counts to benchmark.",
    )
    parser.add_argument("--warmup", type=positive_int, default=25)
    parser.add_argument("--rep", type=positive_int, default=100)
    parser.add_argument("-o", action="store_true", default=False)
    return parser.parse_args()


def main():
    benchmark(parse_args())


if __name__ == "__main__":
    main()
