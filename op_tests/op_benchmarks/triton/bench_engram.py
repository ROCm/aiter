# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the Engram ops of DeepSeek-V4.1-Flash.

The lookup is a pure gather: it reads one fp8 row plus its e8m0 scales per
hash id and writes the dequantized bf16 row, so achieved GB/s against the
minimum required traffic is the only metric that matters. The table is sized
past the last-level cache by default, otherwise a small shard is measured
resident and the number means nothing.

Usage (from the repository root):
  python op_tests/op_benchmarks/triton/bench_engram.py
  python op_tests/op_benchmarks/triton/bench_engram.py --sweep decode
  python op_tests/op_benchmarks/triton/bench_engram.py --op gate --metric time
  python op_tests/op_benchmarks/triton/bench_engram.py --rows 16000000
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.fusions.engram import engram_embedding_lookup, engram_gate_apply
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)

# 3 n-gram orders x 8 hash heads, 256-wide rows, one e8m0 scale per 32 columns
_HASH_COLS = 24
_HEAD_DIM = 256
_SCALE_BLOCK = 32
# hc copies and hidden size of the residual stream the gate writes into
_HC_MULT = 4
_DIM = 5120

_DECODE_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
_PREFILL_TOKENS = (1024, 4096, 16384, 32768)


def get_benchmark_shapes(args):
    if args.tokens:
        return [args.tokens]
    tokens = ()
    if args.sweep in ("decode", "all"):
        tokens += _DECODE_TOKENS
    if args.sweep in ("prefill", "all"):
        tokens += _PREFILL_TOKENS
    return list(tokens)


def _lookup_inputs(n_tokens, rows, device):
    table = torch.randn(rows, _HEAD_DIM, device=device).to(torch.float8_e4m3fn)
    scale = torch.randint(
        118, 137, (rows, _HEAD_DIM // _SCALE_BLOCK), device=device, dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    hash_ids = torch.randint(
        0, rows, (1, n_tokens, _HASH_COLS), device=device, dtype=torch.int64
    )
    return hash_ids, table, scale


def _gate_inputs(n_tokens, device):
    h = torch.randn(1, n_tokens, _HC_MULT, _DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, n_tokens, _HC_MULT, _DIM, device=device)
    value = torch.randn(1, n_tokens, _DIM, device=device, dtype=torch.bfloat16)
    weight = torch.randn(_HC_MULT, _DIM, device=device)
    return h, key, value, weight


def bench_engram_fn(n_tokens, metric, args):
    device = "cuda"
    if args.op == "lookup":
        hash_ids, table, scale = _lookup_inputs(n_tokens, args.rows, device)
        # Minimum required traffic: ids in, one fp8 row + its scales gathered
        # per id, one bf16 row out. Duplicate ids are counted, so a workload
        # that hits in cache reports above HBM peak -- which is the point.
        gathered = n_tokens * _HASH_COLS
        mem = (
            gathered * 8
            + gathered * (_HEAD_DIM + _HEAD_DIM // _SCALE_BLOCK)
            + gathered * _HEAD_DIM * 2
        )

        def fn():
            return engram_embedding_lookup(hash_ids, table, scale)

    else:
        h, key, value, weight = _gate_inputs(n_tokens, device)
        rows = n_tokens * _HC_MULT * _DIM
        # h is read twice (reduce, then inject), key once, value once, out once
        mem = rows * 2 * 2 + rows * 4 + n_tokens * _DIM * 2 + rows * 2

        def fn():
            return engram_gate_apply(h, key, value, weight)

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms
    if metric == "bandwidth":
        return mem / (ms * 1e-3) * 1e-9
    raise ValueError("Unknown metric: " + metric)


def run_benchmark(args):
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)

    benchmark = triton.testing.Benchmark(
        x_names=["n_tokens"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=list(metrics),
        line_names=[f"{args.op}_{metric}" for metric in metrics],
        styles=[("red", "-"), ("blue", "-")][: len(metrics)],
        ylabel="",
        plot_name=get_caller_name_no_ext() + f"_{args.op}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench_fn(n_tokens, provider):
        return bench_engram_fn(n_tokens, provider, args)

    bench_fn.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark Engram",
        description="Benchmark the Triton Engram lookup and gate kernels",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--op",
        type=str,
        default="lookup",
        choices=["lookup", "gate"],
        help="engram_embedding_lookup (gather) or engram_gate_apply (inject)",
    )
    parser.add_argument(
        "--tokens", type=int, default=None, help="Single B*L point to run"
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default="all",
        choices=["decode", "prefill", "all"],
        help="B*L range: decode (1-128), prefill (1k-32k), or both",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4_000_000,
        help="Table rows on this rank (4M rows = 1 GB fp8, past any LLC)",
    )
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
        print("Retrieving VGPR usage for Engram Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
