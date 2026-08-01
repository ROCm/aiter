#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark for MXFP4 MoE stage-1 GEMM with compact activation scales.

Separates the two effects of letting GEMM1 read compact per-token E8M0 scales
instead of a sorted-row scale tensor:

    (a) tuned params + conventional sorted scales   <- baseline
    (b) tuned params + compact scales, no retune    <- cost of compact addressing
    (c) branch's compact tile/wave overrides        <- what `auto` mode ships

(a) and (b) call flydsl_moe_stage1 directly with the tuned row's parameters, so
they differ only in scale layout. (c) goes through _flydsl_stage1_wrapper, which
applies the compact-specific tile_n / waves_per_eu / b_nt / xcd_swizzle
overrides. Each variant is CUDA-graph captured and replayed.

Note this measures GEMM1 in isolation. It does not include the scale-layout
conversion that (a) would need in a real decode step, so the (b)/(c) ratios are
a lower bound on the end-to-end benefit -- see bench_mxfp4_moe_decode_layer.py
for the full-layer picture.

Usage:
    # Run with default parameters (Kimi decode shape)
    python bench_mxfp4_moe_compact_scale_gemm1.py

    # Run selected token counts
    python bench_mxfp4_moe_compact_scale_gemm1.py -t 16 32

    # Save results to CSV
    python bench_mxfp4_moe_compact_scale_gemm1.py -o results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import _flydsl_stage1_wrapper, moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, get_flydsl_kernel_params
from aiter.ops.quant import mxfp4_moe_sort_fwd, per_1x32_f4_quant_hip
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

DEVICE = "cuda"

# Default sweep parameters (Kimi K2 TP8 shard, E=384/topk=8)
MODEL_DIM = 7168
INTER_DIM = 256
EXPERTS = 384
TOPK = 8
BLOCK_M = 32

# Tuned block_m=32 stage-1 rows this shape dispatches to, per token count.
TUNED_KERNELS = {
    1: "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_kw4",
    2: "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3_bnt0_kw4",
    4: "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w2_kw4",
    8: "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w4_xcd4_kw4_fp4",
    16: "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w4_fp4",
    32: "flydsl_moe1_afp4_wfp4_bf16_t32x64x256_w4",
    64: "flydsl_moe1_afp4_wfp4_bf16_t32x32x256_w3",
    128: "flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4",
}


def time_graph(fn, iters: int, warmup: int, reps: int) -> float:
    """CUDA-graph capture `fn`, then return the best per-replay time in us."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()

    best = float("inf")
    for _ in range(reps):
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters * 1000.0)
    return best


def _prepare_weights(experts: int, inter_dim: int, model_dim: int):
    """Quantize and pre-shuffle the gate/up weight for the a4w4 stage-1 kernel."""
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = (
        torch.randn(
            (experts, 2 * inter_dim, model_dim), dtype=dtypes.bf16, device=DEVICE
        )
        * 0.1
    )
    w1_qt, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    del w1
    torch.cuda.empty_cache()
    w1_qt = shuffle_weight_a16w4(
        w1_qt.view(experts, 2 * inter_dim, model_dim // 2), 16, True
    )
    w1_scale = shuffle_scale_a16w4(w1_scale, experts, True)
    return w1_qt, w1_scale


def bench_compact_scale_gemm1_latency(
    tokens: int,
    kernel_name: str,
    w1_qt: torch.Tensor,
    w1_scale: torch.Tensor,
    experts: int = EXPERTS,
    topk: int = TOPK,
    model_dim: int = MODEL_DIM,
    block_m: int = BLOCK_M,
    iters: int = 300,
    warmup: int = 30,
    reps: int = 3,
) -> tuple[float, float, float]:
    """
    Benchmark the three stage-1 GEMM variants for one token count.

    Returns:
        Tuple of (tuned+conventional, tuned+compact, override+compact)
        latencies in microseconds.
    """
    torch.manual_seed(1000 + tokens)
    hidden = torch.randn((tokens, model_dim), dtype=dtypes.bf16, device=DEVICE) * 0.1
    topk_ids = torch.stack(
        [
            torch.randperm(experts, dtype=dtypes.i32, device=DEVICE)[:topk]
            for _ in range(tokens)
        ]
    )
    topk_weights = torch.rand((tokens, topk), dtype=dtypes.fp32, device=DEVICE)
    sorted_ids, _sorted_weights, sorted_expert_ids, num_valid_ids, _buf = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, dtypes.bf16, block_m
    )

    act_quant, compact_scale = per_1x32_f4_quant_hip(hidden, shuffle=False)
    sorted_scale = mxfp4_moe_sort_fwd(
        compact_scale,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=tokens,
        cols=model_dim,
    )

    params = get_flydsl_kernel_params(kernel_name)
    raw_kwargs = {
        "a": act_quant,
        "w1": w1_qt,
        "sorted_token_ids": sorted_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "out": None,
        "topk": topk,
        "tile_m": params["tile_m"],
        "tile_n": params["tile_n"],
        "tile_k": params["tile_k"],
        "a_dtype": params["a_dtype"],
        "b_dtype": params["b_dtype"],
        "out_dtype": params["out_dtype"],
        "act": "silu",
        "w1_scale": w1_scale,
        "sorted_weights": None,
        "use_async_copy": True,
        "k_batch": params.get("k_batch", 1),
        "waves_per_eu": params.get("waves_per_eu", 3),
        "b_nt": params.get("b_nt", 2),
        "gate_mode": params.get("gate_mode", "separated"),
        "xcd_swizzle": params.get("xcd_swizzle", 0),
        "k_wave": params.get("k_wave", 1),
    }
    wrapper_kwargs = {
        "w2": None,
        "sorted_token_ids": sorted_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "out": None,
        "topk": topk,
        "kernelName": kernel_name,
        "w1_scale": w1_scale,
        "sorted_weights": None,
    }

    conventional_us = time_graph(
        lambda: flydsl_moe_stage1(
            a1_scale=sorted_scale, a_scale_compact=False, **raw_kwargs
        ),
        iters,
        warmup,
        reps,
    )
    compact_us = time_graph(
        lambda: flydsl_moe_stage1(
            a1_scale=compact_scale, a_scale_compact=True, **raw_kwargs
        ),
        iters,
        warmup,
        reps,
    )
    override_us = time_graph(
        lambda: _flydsl_stage1_wrapper(
            hidden_states=act_quant,
            w1=w1_qt,
            a1_scale=compact_scale,
            a1_scale_compact=True,
            **wrapper_kwargs,
        ),
        iters,
        warmup,
        reps,
    )
    return conventional_us, compact_us, override_us


def run_benchmark(args):
    """Run the GEMM1 benchmark across every requested token count."""
    print(f"arch: {torch.cuda.get_device_properties(0).gcnArchName}")
    print(
        f"E={args.experts} topk={args.topk} model_dim={args.model_dim} "
        f"inter_dim={args.inter_dim} block_m={args.block_m}"
    )

    w1_qt, w1_scale = _prepare_weights(args.experts, args.inter_dim, args.model_dim)

    results = []
    header = (
        f"{'M':>8} {'(a) tuned+conv':>16} {'(b) tuned+compact':>19} "
        f"{'(c) override+compact':>22} {'b/a':>8} {'c/a':>8}"
    )
    print()
    print(header)
    print("-" * len(header))

    for tokens in args.tokens:
        kernel_name = TUNED_KERNELS.get(tokens)
        if kernel_name is None:
            print(
                f"{tokens:>8}  no tuned block_m=32 kernel recorded; "
                f"add one to TUNED_KERNELS to benchmark this token count"
            )
            continue
        conv_us, compact_us, override_us = bench_compact_scale_gemm1_latency(
            tokens,
            kernel_name,
            w1_qt,
            w1_scale,
            experts=args.experts,
            topk=args.topk,
            model_dim=args.model_dim,
            block_m=args.block_m,
            iters=args.iters,
            warmup=args.warmup,
            reps=args.reps,
        )
        results.append((tokens, kernel_name, conv_us, compact_us, override_us))
        print(
            f"{tokens:>8} {conv_us:>16.2f} {compact_us:>19.2f} "
            f"{override_us:>22.2f} {conv_us / compact_us:>7.2f}x "
            f"{conv_us / override_us:>7.2f}x"
        )

    if args.o:
        _save_results_csv(args.o, results)


def _save_results_csv(filepath: str, results: list):
    """Save benchmark results to CSV file."""
    path = Path(filepath)
    with open(path, "w") as f:
        f.write(
            "tokens,kernel_name,tuned_conventional_us,tuned_compact_us,"
            "override_compact_us,speedup_compact,speedup_override\n"
        )
        f.writelines(
            f"{tokens},{kernel_name},{conv_us:.4f},{compact_us:.4f},"
            f"{override_us:.4f},{conv_us / compact_us:.4f},"
            f"{conv_us / override_us:.4f}\n"
            for tokens, kernel_name, conv_us, compact_us, override_us in results
        )
    print(f"\nResults saved to {path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark MXFP4 MoE Compact-Scale GEMM1",
        description=(
            "Benchmark MXFP4 MoE stage-1 GEMM reading compact per-token scales "
            "against the conventional sorted-row scale layout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="+",
        default=sorted(TUNED_KERNELS),
        help="Token counts to sweep.",
    )
    parser.add_argument(
        "-e",
        "--experts",
        type=int,
        default=EXPERTS,
        help="Number of experts.",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        default=TOPK,
        help="Experts routed per token.",
    )
    parser.add_argument(
        "--model-dim",
        type=int,
        dest="model_dim",
        default=MODEL_DIM,
        help="Model hidden dimension.",
    )
    parser.add_argument(
        "--inter-dim",
        type=int,
        dest="inter_dim",
        default=INTER_DIM,
        help="Expert intermediate dimension.",
    )
    parser.add_argument(
        "--block-m",
        type=int,
        dest="block_m",
        default=BLOCK_M,
        help="Sort block size M.",
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="FILE",
        help="Output CSV file path for results.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=300,
        help="Number of graph replays per timed rep.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warmup replays before each timed rep.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of timed reps; the best is reported.",
    )
    return parser.parse_args()


def main():
    run_benchmark(parse_args())


if __name__ == "__main__":
    main()
