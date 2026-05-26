# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the FP8 ``rope_norm_store_kv_fp8`` Triton kernel.

Sweeps over (batch, seqlen) shapes. Reports latency (us) or achieved bandwidth
(GB/s). Selectable quant policy and norm policy.
"""

import argparse
import torch
import triton

from aiter.ops.triton.fusions.rope_norm_store_kv_fp8 import rope_norm_store_kv_fp8
from op_tests.triton_tests.fusions.test_rope_norm_store_kv_fp8 import _make_inputs


def _bytes_per_call(inp, is_prefill: bool) -> int:
    qkv = inp["qkv"]
    num_rows = qkv.shape[0]
    hidden = qkv.shape[1]
    num_kv = inp["num_kv_heads"]
    num_q = inp["num_q_heads"]
    qkd = inp["qk_head_dim"]
    vd = inp["v_head_dim"]
    # qkv is bf16 (2B), outputs are fp8 (1B)
    in_b = qkv.element_size()
    out_b = 1  # fp8
    qkv_bytes = num_rows * hidden * in_b
    out_q_bytes = num_rows * num_q * qkd * out_b
    out_k_bytes = num_rows * num_kv * qkd * out_b
    out_v_bytes = num_rows * num_kv * vd * out_b
    return qkv_bytes + out_q_bytes + out_k_bytes + out_v_bytes


def _run_once(inp, is_prefill: bool, qk_norm_policy: int, quant_policy: int):
    qnw = inp["q_norm_weight"] if qk_norm_policy != 0 else None
    knw = inp["k_norm_weight"] if qk_norm_policy != 0 else None
    q_scale_inv = inp["q_scale_inv"] if quant_policy == 2 else None
    rope_norm_store_kv_fp8(
        inp["key_cache"], inp["value_cache"], inp["qkv"], inp["cos_sin"],
        inp["num_seqlen_per_req"], inp["q_index"], inp["kvcache_indices"],
        is_prefill,
        inp["k_scale"], inp["v_scale"],
        quant_policy=quant_policy,
        max_seqlens=inp["max_seqlens"],
        q_scale_inv=q_scale_inv,
        q_norm_weight=qnw, k_norm_weight=knw,
        qk_norm_policy=qk_norm_policy,
    )


def run_benchmark(args):
    mode = args.mode
    is_prefill = (mode == "prefill")
    quant_policy = args.quant_policy
    qk_norm_policy = args.norm_policy
    num_q_heads = args.qh
    num_kv_heads = args.kvh
    qk_head_dim = args.qkd
    v_head_dim = args.vd
    block_size = args.block_size

    metric = args.metric
    suffix = {"us": "us", "bw": "GB/s"}[metric]
    x_vals = [(b, s) for b in args.batch for s in args.seqlen]

    benchmark = triton.testing.Benchmark(
        x_names=["batch", "seqlen"],
        x_vals=x_vals,
        line_arg="provider",
        line_vals=["triton"],
        line_names=[f"Triton ({suffix})"],
        styles=[("blue", "-")],
        ylabel=suffix,
        plot_name=(
            f"rope_norm_store_kv_fp8_{mode}_qp{quant_policy}_np{qk_norm_policy}"
            f"_qh{num_q_heads}_kvh{num_kv_heads}_qkd{qk_head_dim}_vd{v_head_dim}_bs{block_size}"
        ),
        args={},
    )

    @triton.testing.perf_report(benchmark)
    def bench(batch, seqlen, provider):
        seqlens = [seqlen] * batch
        inp = _make_inputs(
            seqlens, num_q_heads, num_kv_heads,
            qk_head_dim, v_head_dim, block_size,
            is_prefill=is_prefill, quant_policy=quant_policy,
        )
        fn = lambda: _run_once(inp, is_prefill, qk_norm_policy, quant_policy)
        ms = triton.testing.do_bench(fn, warmup=10, rep=50)
        if metric == "us":
            return ms * 1e3
        return _bytes_per_call(inp, is_prefill) / (ms * 1e-3) / 1e9

    bench.run(save_path=args.save_path, print_data=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark rope_norm_store_kv_fp8 (FP8) Triton kernel."
    )
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--quant-policy", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument("--norm-policy", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--seqlen", type=int, nargs="+",
                        default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--qh", type=int, default=32)
    parser.add_argument("--kvh", type=int, default=8)
    parser.add_argument("--qkd", type=int, default=128)
    parser.add_argument("--vd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--metric", choices=["us", "bw"], default="us")
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
