# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the BF16 ``rope_norm_store_kv`` Triton kernel.

Sweeps over (batch x sequence length) shapes; reports kernel latency, achieved
bandwidth, and the reference latency for comparison. Picks a reasonable head
config that mirrors a Llama-3-70B-ish setup unless overridden via CLI.
"""

import argparse
import torch
import triton

from aiter.ops.triton.fusions.rope_norm_store_kv import rope_norm_store_kv
from op_tests.triton_tests.fusions.test_rope_norm_store_kv import (
    rope_norm_store_kv_reference,
    _make_inputs,
)


def _bytes_per_call(inp, is_prefill: bool) -> int:
    qkv = inp["qkv"]
    num_rows = qkv.shape[0]
    hidden = qkv.shape[1]
    num_kv = inp["num_kv_heads"]
    num_q = inp["num_q_heads"]
    qkd = inp["qk_head_dim"]
    vd = inp["v_head_dim"]
    bs = qkv.element_size()
    # Read qkv + write out_q + write per-row K/V to cache
    qkv_bytes = num_rows * hidden * bs
    out_q_bytes = num_rows * num_q * qkd * bs
    out_k_bytes = num_rows * num_kv * qkd * bs
    out_v_bytes = num_rows * num_kv * vd * bs
    # cos/sin gathers and norm weights are smaller; ignore.
    return qkv_bytes + out_q_bytes + out_k_bytes + out_v_bytes


def _run_once(inp, is_prefill: bool, policy: int):
    kc = inp["key_cache"].clone()
    vc = inp["value_cache"].clone()
    qnw = inp["q_norm_weight"] if policy != 0 else None
    knw = inp["k_norm_weight"] if policy != 0 else None
    rope_norm_store_kv(
        kc, vc, inp["qkv"], inp["cos_sin"],
        inp["num_seqlen_per_req"], inp["q_index"],
        inp["kvcache_indices"], is_prefill,
        q_norm_weight=qnw, k_norm_weight=knw,
        qk_norm_policy=policy,
    )


def _run_ref_once(inp, is_prefill: bool, policy: int):
    kc = inp["key_cache"].clone()
    vc = inp["value_cache"].clone()
    qnw = inp["q_norm_weight"] if policy != 0 else None
    knw = inp["k_norm_weight"] if policy != 0 else None
    rope_norm_store_kv_reference(
        kc, vc, inp["qkv"], inp["cos_sin"],
        inp["num_seqlen_per_req"], inp["q_index"],
        inp["kvcache_indices"], is_prefill,
        q_norm_weight=qnw, k_norm_weight=knw,
        qk_norm_policy=policy,
    )


def run_benchmark(args):
    mode = args.mode
    is_prefill = (mode == "prefill")
    policy = args.policy
    num_q_heads = args.qh
    num_kv_heads = args.kvh
    qk_head_dim = args.qkd
    v_head_dim = args.vd
    block_size = args.block_size

    if is_prefill:
        x_vals = [(b, s) for b in args.batch for s in args.seqlen]
    else:
        # decode: batch sweeps; seq stays "current total seq"
        x_vals = [(b, s) for b in args.batch for s in args.seqlen]

    metric = args.metric
    suffix = {"us": "us", "bw": "GB/s"}[metric]

    benchmark = triton.testing.Benchmark(
        x_names=["batch", "seqlen"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["triton", "torch_ref"] if args.include_ref else ["triton"],
        line_names=[f"Triton ({suffix})", f"PyTorch ref ({suffix})"]
            if args.include_ref else [f"Triton ({suffix})"],
        styles=[("blue", "-"), ("red", "-")],
        ylabel=suffix,
        plot_name=(f"rope_norm_store_kv_{mode}_policy{policy}"
                   f"_qh{num_q_heads}_kvh{num_kv_heads}"
                   f"_qkd{qk_head_dim}_vd{v_head_dim}_bs{block_size}"),
        args={},
    )

    @triton.testing.perf_report(benchmark)
    def bench(batch, seqlen, provider):
        seqlens = [seqlen] * batch
        inp = _make_inputs(
            seqlens, num_q_heads, num_kv_heads,
            qk_head_dim, v_head_dim, block_size,
            is_prefill=is_prefill,
        )
        fn = (lambda: _run_once(inp, is_prefill, policy)) if provider == "triton" \
            else (lambda: _run_ref_once(inp, is_prefill, policy))
        ms = triton.testing.do_bench(fn, warmup=10, rep=50)
        if metric == "us":
            return ms * 1e3
        # bandwidth (GB/s) - movement excludes the reference's host-driven copies
        n_bytes = _bytes_per_call(inp, is_prefill)
        return n_bytes / (ms * 1e-3) / 1e9

    bench.run(save_path=args.save_path, print_data=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark rope_norm_store_kv (BF16) Triton kernel."
    )
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--policy", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seqlen", type=int, nargs="+",
                        default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--qh", type=int, default=32)
    parser.add_argument("--kvh", type=int, default=8)
    parser.add_argument("--qkd", type=int, default=128)
    parser.add_argument("--vd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--include-ref", action="store_true",
                        help="Also benchmark the pure-PyTorch reference (very slow).")
    parser.add_argument("--metric", choices=["us", "bw"], default="us",
                        help="us: latency in microseconds; bw: achieved GB/s.")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
