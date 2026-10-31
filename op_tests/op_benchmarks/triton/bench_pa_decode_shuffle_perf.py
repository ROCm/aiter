# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for split-K sliding-window decode over the shuffled fp8 paged KV cache.
"""

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.pa_decode_shuffle_perf import (
    paged_attention_decode_shuffle_swa_perf,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
    print_vgpr,
)
from op_tests.triton_tests.attention.test_pa_decode_shuffle_perf import (
    make_shuffle_inputs,
)

_PROVIDERS = {"onepass": False, "split": True, "auto": None}
# (BS, SEQ_LEN, WINDOW)
_POINTS = (
    (1, 8192, 4096),
    (8, 8192, 4096),
    (32, 8192, 4096),
    (32, 2048, 1024),
    (32, 512, 256),
    (8, 32768, 0),
)

arg_to_torch_dtype = {
    "e4m3fn": torch.float8_e4m3fn,
    "e4m3fnuz": torch.float8_e4m3fnuz,
}


def get_benchmark_shapes(args):
    if args.b:
        return [(args.b, args.sq, args.window)]
    return list(_POINTS)


def bench_fn(BS, SEQ_LEN, WINDOW, provider, metric, args):
    q, kc, vc, bt, sl, ks, vs, _, _ = make_shuffle_inputs(
        [SEQ_LEN] * BS,
        args.hq,
        args.hk,
        args.d,
        args.block,
        arg_to_torch_dtype[args.kv_cache_dtype],
        args.per_token_scale,
    )
    out = torch.empty_like(q)
    ws = {}

    def fn():
        paged_attention_decode_shuffle_swa_perf(
            out,
            q,
            kc,
            vc,
            bt,
            sl,
            args.d**-0.5,
            ks,
            vs,
            WINDOW,
            partition_size=args.part,
            max_seq_len=SEQ_LEN,
            workspace=ws,
            force_split=_PROVIDERS[provider],
        )

    ms = triton.testing.do_bench(fn, warmup=args.warmup, rep=args.rep)
    if metric == "time":
        return ms * 1000  # us
    if metric == "bandwidth":
        keys = SEQ_LEN if WINDOW <= 0 else min(WINDOW, SEQ_LEN)
        # fp8 K + V for the window, plus query and output
        mem = BS * args.hk * keys * args.d * 2 * kc.element_size()
        mem += 2 * q.numel() * q.element_size()
        return mem / (ms * 1e-3) * 1e-9  # GB/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = list(_PROVIDERS) if args.provider == "all" else [args.provider]
    metrics = ("time", "bandwidth") if args.metric == "all" else (args.metric,)
    line_vals = [f"{p}_{m}" for m in metrics for p in providers]

    benchmark = triton.testing.Benchmark(
        x_names=["BS", "SEQ_LEN", "WINDOW"],
        x_vals=get_benchmark_shapes(args),
        line_arg="provider",
        line_vals=line_vals,
        line_names=line_vals,
        styles=[("red", "-"), ("blue", "-"), ("green", "-")] * 2,
        ylabel="us / GB/s",
        plot_name=get_caller_name_no_ext() + f"_HQ{args.hq}_HK{args.hk}_D{args.d}",
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def bench(BS, SEQ_LEN, WINDOW, provider):
        name, metric = provider.rsplit("_", 1)
        return bench_fn(BS, SEQ_LEN, WINDOW, name, metric, args)

    bench.run(save_path="." if args.o else None, print_data=True)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark pa_decode_shuffle_perf",
        description="Split-K sliding-window decode over the shuffled fp8 KV cache",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0, help="Batch size (0: default list)")
    parser.add_argument("-sq", type=int, default=8192, help="Context length")
    parser.add_argument("--window", type=int, default=4096, help="<= 0: full")
    parser.add_argument("-hq", type=int, default=32)
    parser.add_argument("-hk", type=int, default=8)
    parser.add_argument("-d", type=int, default=128)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--part", type=int, default=256, help="partition_size")
    parser.add_argument(
        "-kv_cache_dtype", default="e4m3fn", choices=list(arg_to_torch_dtype)
    )
    parser.add_argument("--per_token_scale", action="store_true")
    parser.add_argument("--provider", default="all", choices=[*_PROVIDERS, "all"])
    parser.add_argument("--metric", default="all", choices=["all", "time", "bandwidth"])
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
        print("Retrieving VGPR usage for pa_decode_shuffle_perf Triton kernels...")
        print_vgpr(lambda: run_benchmark(args), get_caller_name_no_ext())
        return 0
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
