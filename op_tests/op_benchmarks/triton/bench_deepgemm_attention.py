# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
import random

import torch
import triton

from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_schedule,
)
from aiter.test_common import run_perftest
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    get_caller_name_no_ext,
)
from op_tests.triton_tests.attention.test_deepgemm_attention import (
    apply_preshuffle,
    make_paged_inputs,
)

_METRIC_TO_UNIT = {
    "time": "Time_(us)",
    "throughput": "TFLOPS",
    "bandwidth": "Bandwidth_(GB/s)",
}


def create_paged_mqa_logits_configs(args: argparse.Namespace):
    x_names = ["batch_size", "next_n", "heads", "index_dim", "avg_kv_length"]

    if args.perf:
        x_vals_list = [
            (1, 2, 64, 128, 16384),
            (1, 2, 64, 128, 32768),
            (1, 2, 64, 128, 65536),
            (2, 2, 64, 128, 16384),
            (2, 2, 64, 128, 32768),
            (2, 2, 64, 128, 65536),
            (4, 2, 64, 128, 16384),
            (4, 2, 64, 128, 32768),
            (4, 2, 64, 128, 65536),
            (1, 1, 64, 128, 65536),
            (2, 1, 64, 128, 65536),
            (4, 1, 64, 128, 65536),
            (8, 1, 64, 128, 65536),
        ]
    else:
        x_vals_list = [
            (args.batch, args.mtp + 1, args.heads, args.index_dim, args.kv_length)
        ]

    if args.metric not in _METRIC_TO_UNIT:
        raise NotImplementedError(f"{args.metric} is not supported")
    unit = _METRIC_TO_UNIT[args.metric]

    return [
        triton.testing.Benchmark(
            x_names=x_names,
            x_vals=x_vals_list,
            line_arg="unit",
            line_vals=[unit],
            line_names=[unit],
            styles=[("green", "-")],
            ylabel=unit,
            plot_name=get_caller_name_no_ext(),
            args={"metric": args.metric},
        )
    ]


def _bandwidth_bytes(
    batch_size: int,
    next_n: int,
    heads: int,
    index_dim: int,
    context_lens: torch.Tensor,
) -> int:
    """Bytes moved by one kernel invocation (memory-bound op)."""
    total_ctx = int(context_lens.sum().item())
    q_bytes = batch_size * next_n * heads * index_dim  # fp8, 1 B/elem
    # KV is MQA (num_kv_heads=1): FP8 data + per-token FP32 scale.
    kv_bytes = total_ctx * (index_dim + 4)
    w_bytes = batch_size * next_n * heads * 4  # fp32
    # Output: only valid positions are written (one per query/key pair).
    out_bytes = next_n * total_ctx * 4  # fp32
    return q_bytes + kv_bytes + w_bytes + out_bytes


def _dump_aot_cache(args, cache_key, heads, index_dim, blocksize, enable_var_ctx):
    """Move the just-compiled triton cache entry under ./paged_mqa_logits/aot/."""
    triton_cache_dir = str(triton.knobs.cache.dir)
    aot_kernel_dir = "./paged_mqa_logits/aot"
    os.makedirs(aot_kernel_dir, exist_ok=True)

    padded_str = "T" if args.padding else "F"
    preshuffle_suffix = "_preshuffle" if args.kv_preshuffle else ""
    varctx_suffix = "_varctx" if enable_var_ctx else ""
    aot_name = (
        f"paged_mqa_logits{preshuffle_suffix}{varctx_suffix}"
        f"_{heads}x{args.chunk_k}x{index_dim}_B{blocksize}P{padded_str}W{args.wave_per_eu}"
    )

    src = os.path.join(triton_cache_dir, cache_key)
    dst = os.path.join(aot_kernel_dir, aot_name)
    if os.path.exists(dst):
        os.system(f"rm -rf {dst}")
    os.system(f"mv {src} {dst}")
    print(f"Moved cache from {src} to {dst}")
    os.system("zip -r paged_mqa_logits_aot_kernel paged_mqa_logits")


def run_benchmark(args: argparse.Namespace):
    @triton.testing.perf_report(create_paged_mqa_logits_configs(args))
    def bench_deepgemm_fp8_paged_mqa_logits(
        batch_size, next_n, heads, index_dim, avg_kv_length, metric, **kwargs
    ):
        torch.manual_seed(0)
        random.seed(0)

        blocksize = args.blocksize if args.kv_preshuffle else 1
        assert blocksize == 1 or (args.kv_preshuffle and blocksize % 16 == 0)

        inputs = make_paged_inputs(
            batch_size,
            next_n,
            heads,
            index_dim,
            avg_kv_length,
            blocksize=blocksize,
            padding=args.padding,
        )
        q_fp8 = inputs["q_fp8"]
        kv_cache_fp8 = inputs["kv_cache_fp8"]
        weights = inputs["weights"]
        context_lens = inputs["context_lens"]
        block_tables = inputs["block_tables"]
        max_model_len = inputs["max_model_len"]

        if args.kv_preshuffle:
            apply_preshuffle(kv_cache_fp8, blocksize, index_dim)

        safe_chunks_per_cta = None
        if args.var_ctx_opt:
            safe_chunks_per_cta = deepgemm_fp8_paged_mqa_logits_schedule(
                batch_size,
                next_n,
                context_lens,
                max_model_len,
                ChunkK=args.chunk_k,
                WavePerEU=args.wave_per_eu,
            )

        out_logits = torch.full(
            (batch_size * next_n, max_model_len),
            float("-inf"),
            device="cuda",
            dtype=torch.float32,
        )

        cache_key, elapsed_us = run_perftest(
            deepgemm_fp8_paged_mqa_logits,
            q_fp8,
            kv_cache_fp8,
            weights,
            out_logits,
            context_lens,
            block_tables,
            max_model_len,
            ChunkK=args.chunk_k,
            Preshuffle=args.kv_preshuffle,
            KVBlockSize=blocksize,
            WavePerEU=args.wave_per_eu,
            VarCtxSchedule=safe_chunks_per_cta,
        )

        if args.aot:
            _dump_aot_cache(
                args, cache_key, heads, index_dim, blocksize, args.var_ctx_opt
            )

        if metric == "time":
            return elapsed_us
        if metric == "throughput":
            total_flops = (
                2 * next_n * heads * index_dim * context_lens.float().sum().item()
            )
            return total_flops / elapsed_us * 1e-6  # TFLOPS = FLOPs / us * 1e-6
        if metric == "bandwidth":
            mem = _bandwidth_bytes(batch_size, next_n, heads, index_dim, context_lens)
            return mem / elapsed_us * 1e-3  # GB/s = B / us * 1e-3
        raise ValueError(f"Unknown metric: {metric}")

    bench_deepgemm_fp8_paged_mqa_logits.run(
        save_path="." if args.o else None, print_data=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepGEMM paged FP8 MQA logits benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-B", "--batch", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "-hq",
        "--heads",
        type=int,
        default=64,
        help="Number of query heads (equal to number of key/value heads)",
    )
    parser.add_argument(
        "--index_dim",
        type=int,
        default=128,
        help="Head dimension (dimension of query/key/value vectors)",
    )
    parser.add_argument(
        "-kv_length",
        type=int,
        default=4096,
        help="Sequence length (since this is decode, this is the length of the key/value sequence)",
    )
    parser.add_argument(
        "-mtp",
        type=int,
        default=0,
        help="Q sequence length (mtp + 1 == qo_len) in MTP mode",
    )
    parser.add_argument(
        "-p",
        "--padding",
        action="store_true",
        help="Padding the contiguous dimension of KVCache to multiple of 16 Bytes",
    )
    parser.add_argument(
        "-aot",
        action="store_true",
        help="Save compiled triton kernel for later AOT use",
    )
    parser.add_argument(
        "--perf",
        action="store_true",
        help="Run a predefined performance sweep instead of a single shape",
    )
    parser.add_argument(
        "--kv_preshuffle",
        action="store_true",
        help="Enable KV cache preshuffle, also change blocksize to 16",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=16,
        help="KVCache block size, only used when kv_preshuffle is enabled, must be multiple of 16",
    )
    parser.add_argument(
        "--var_ctx_opt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable variable-context-length scheduling",
    )
    parser.add_argument(
        "--chunk_k",
        type=int,
        default=128,
        help="ChunkK tile size",
    )
    parser.add_argument(
        "--wave_per_eu",
        type=int,
        default=5,
        help="waves_per_eu hint passed to the kernel",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=list(_METRIC_TO_UNIT.keys()),
        default="bandwidth",
        help="Metric to report (default: bandwidth, since this kernel is memory-bound)",
    )
    parser.add_argument(
        "-o", action="store_true", help="Write performance results to CSV file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
