# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the hierarchical-indexer candidate selection ops.

Sweeps decode (T=1) and prefill (T=4096) query counts over the compressed
length L and reports latency and effective HBM bandwidth.

Usage (from the repository root):
  python op_tests/op_benchmarks/triton/bench_candidate_blocks.py
  python op_tests/op_benchmarks/triton/bench_candidate_blocks.py --op scores
  python op_tests/op_benchmarks/triton/bench_candidate_blocks.py -T 1 -L 131072

Bandwidth counts the MINIMUM required traffic for each op, so a faster
configuration always reports a higher number.
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.candidate_blocks import (
    apply_candidate_mask,
    candidate_block_scores,
    candidate_mask_from_blocks,
    select_candidate_blocks,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)

_OPS = ("scores", "mask", "select", "apply")

# DSV4.1-Flash decode (one query) and prefill (a few thousand queries).
_TOKENS = (1, 4096)
_LENS = (64, 999, 8192, 131072)


def get_benchmark_shapes(args):
    if args.T and args.L:
        return [(args.T, args.L)]
    return [(T, L) for T in _TOKENS for L in _LENS]


def _inputs(T, L, block_size, topk_blocks):
    torch.manual_seed(0)
    logits = torch.randn(T, L, device="cuda", dtype=torch.float32)
    cl = torch.randint(0, L + 1, (T,), dtype=torch.int32, device="cuda")
    logits.masked_fill_(torch.arange(L, device="cuda") >= cl[:, None], -float("inf"))
    scores = candidate_block_scores(logits, cl, block_size)
    k = min(topk_blocks, scores.shape[1])
    idx = scores.topk(k, dim=-1).indices.to(torch.int32)
    mask = candidate_mask_from_blocks(idx, scores, L, block_size)
    return logits, cl, scores, idx, mask


def bench_candidate_blocks_fn(T, L, op, metric, args):
    bs, topk = args.block_size, args.topk_blocks
    num_blocks = triton.cdiv(L, bs)
    k = min(topk, num_blocks)
    logits, cl, scores, idx, mask = _inputs(T, L, bs, topk)

    if op == "scores":
        mem = T * (4 * L + 4 * num_blocks) + 4 * T
        fn = lambda: candidate_block_scores(logits, cl, bs)
    elif op == "mask":
        # zero the mask + read the picks and their scores + scatter the runs
        mem = T * (L + 4 * k + 4 * k + k * bs)
        fn = lambda: candidate_mask_from_blocks(idx, scores, L, bs)
    elif op == "select":
        mem = T * (4 * L + 8 * num_blocks + 4 * k + L + k * bs)
        fn = lambda: select_candidate_blocks(logits, cl, bs, topk)
    elif op == "apply":
        # read the mask once, rewrite only the cleared logits
        mem = T * L + 4 * int(mask.numel() - mask.sum().item())
        fn = lambda: apply_candidate_mask(logits, mask)
    else:
        raise ValueError(op)

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1e3  # us
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    raise ValueError("Unknown metric: " + metric)


def run_benchmark(args):
    ops = _OPS if args.op == "all" else (args.op,)
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{op}_{metric}" for metric in metrics for op in ops]

    benchmark = triton.testing.Benchmark(
        x_names=["T", "L"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=[
            f"{v} ({'us' if v.endswith('time') else 'GB/s'})" for v in line_vals
        ],
        styles=[("red", "-"), ("blue", "-"), ("green", "-"), ("yellow", "-")]
        * ((len(line_vals) + 3) // 4),
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_bs{args.block_size}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(T, L, provider):
        op, metric = provider.rsplit("_", 1)
        return bench_candidate_blocks_fn(T, L, op, metric, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark candidate_blocks",
        description="Benchmark the hierarchical-indexer candidate selection ops",
        allow_abbrev=False,
    )
    parser.add_argument("-T", type=int, default=None, help="Number of query tokens")
    parser.add_argument("-L", type=int, default=None, help="Compressed positions")
    parser.add_argument(
        "--op", type=str, default="all", choices=[*_OPS, "all"], help="Op to benchmark"
    )
    parser.add_argument("--block-size", type=int, default=8, dest="block_size")
    parser.add_argument("--topk-blocks", type=int, default=2048, dest="topk_blocks")
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "time", "bandwidth"],
        help="Metric to report (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
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
        print("Retrieving VGPR usage for candidate_blocks Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
