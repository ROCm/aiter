#!/usr/bin/env python3
"""
Benchmark CK ck_moe_stage1 in isolation on DSR1 block-scale FP8 shapes.

stage1 = GEMM1 (FP8 1x128 block-scale) + SiLU into bf16 intermediate.

Usage:
  HIP_VISIBLE_DEVICES=0 python op_tests/bench_ck_stage1.py -m 16 128 1024
"""
import argparse
import torch
from einops import rearrange

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk
from aiter.fused_moe_bf16_asm import moe_sorting_ck
from aiter.ops.quant import pertoken_quant
from aiter.test_common import run_perftest

torch.set_default_device("cuda")


def block_quant_weight(w, sb_n=128, sb_k=128):
    tmp = rearrange(
        w.view(-1, w.shape[1] // sb_n, sb_n, w.shape[2] // sb_k, sb_k),
        "e n0 n1 k0 k1 -> e n0 k0 (n1 k1)",
    )
    q, s = pertoken_quant(tmp, quant_dtype=dtypes.fp8)
    q = rearrange(
        q.view(-1, w.shape[1] // sb_n, w.shape[2] // sb_k, sb_n, sb_k),
        "e n0 k0 n1 k1 -> e (n0 n1) (k0 k1)",
    )
    return q, s.squeeze(-1)


def bench_stage1(M, K, N, E, topk, dtype=torch.bfloat16, warmup=10, iters=50):
    # build inputs
    inp = torch.randn((M, K), dtype=dtype)
    w1 = torch.randn((E, N * 2, K), dtype=dtype) / 10
    w2 = torch.randn((E, K, N), dtype=dtype) / 10
    score = torch.randn((M, E), dtype=dtype)
    topk_w, topk_ids = fused_topk(inp, score, topk, True)

    w1_q, w1_s = block_quant_weight(w1)
    w2_q, w2_s = block_quant_weight(w2)
    a1_q, a1_s = pertoken_quant(inp.view(-1, K // 128, 128), quant_dtype=dtypes.fp8)
    a1_q = a1_q.view(-1, K)
    a1_s = a1_s.squeeze(-1)

    block_m = 32 if (M * topk) // E <= 128 else 64
    sorted_ids, sorted_w, sorted_eids, num_valid, _ = moe_sorting_ck(
        topk_ids, topk_w, E, K, dtype, block_m, None
    )

    a2 = torch.empty((M, topk, N), dtype=dtype, device="cuda")

    def _run():
        aiter.ck_moe_stage1(
            a1_q, w1_q, w2_q,
            sorted_ids, sorted_eids, num_valid,
            a2, topk,
            kernelName="",
            w1_scale=w1_s,
            a1_scale=a1_s,
            block_m=block_m,
            sorted_weights=None,
            quant_type=QuantType.per_1x128,
            activation=1,  # silu
            splitk=1,
        )
        return a2

    _, t_us = run_perftest(_run, num_warmup=warmup, num_iters=iters)

    Meff = M * topk
    flops = 2 * Meff * (2 * N) * K
    tflops = flops / (t_us * 1e-6) / 1e12
    # rough bytes for an M-bound view:
    #   activation read  = Meff*K  (fp8)
    #   weight read      = experts_touched * (2N) * K (fp8)  -- assume each routed token hits 1 expert
    #   output write     = Meff * N * 2 (bf16)
    return dict(M=M, us=t_us, tflops=tflops)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", nargs="*", type=int,
                   default=[1, 16, 32, 64, 128, 256, 1024, 2048])
    p.add_argument("-K", type=int, default=7168)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-E", type=int, default=256)
    p.add_argument("-topk", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    print(f"\nCK ck_moe_stage1  (FP8 1x128 blockscale + silu)  "
          f"K={args.K} N={args.N} E={args.E} topk={args.topk}")
    print("=" * 60)
    print(f"{'M':>6} | {'stage1 us':>11} | {'TFLOPS':>8}")
    print("-" * 60)

    for M in args.m:
        try:
            r = bench_stage1(M, args.K, args.N, args.E, args.topk,
                             warmup=args.warmup, iters=args.iters)
            print(f"{r['M']:6d} | {r['us']:11.2f} | {r['tflops']:8.1f}")
        except RuntimeError as e:
            print(f"{M:6d} | ERROR: {str(e)[:120]}")


if __name__ == "__main__":
    main()
