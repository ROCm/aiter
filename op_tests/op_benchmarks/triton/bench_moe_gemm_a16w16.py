# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from argparse import ArgumentParser, ArgumentTypeError
from functools import partial

import torch
import triton

from aiter import dtypes
from aiter.ops.triton.moe.moe_op_gemm_a16w16 import fused_moe_bf16_g1u1

_SELECTED_TOKENS = [1152, 1543, 1728, 1799, 1803, 1815, 1837, 1945, 2048]
_MODEL_DIM = 2048
_INTER_DIM = 128
_EXPERTS = 256
_TOPK = 8


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ArgumentTypeError(f"{value!r} is not an integer") from error
    if parsed <= 0:
        raise ArgumentTypeError(f"{value!r} is not positive")
    return parsed


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
    metric_name = "time_ms" if args.metric == "time" else "tflops"
    print(f"tokens,{metric_name}", flush=True)
    for tokens in args.tokens:
        hidden, w1, w2, topk_weight, topk_ids = _make_case(tokens, device)
        fn = partial(fused_moe_bf16_g1u1, hidden, w1, w2, topk_weight, topk_ids)
        ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
        if args.metric == "time":
            value = ms
        else:
            value = _flops(tokens) / ms * 1e-9
        print(f"{tokens},{value:.6f}", flush=True)


def parse_args():
    parser = ArgumentParser(
        prog="Benchmark moe_gemm_a16w16",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--metric",
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
    return parser.parse_args()


def main():
    benchmark(parse_args())


if __name__ == "__main__":
    main()
