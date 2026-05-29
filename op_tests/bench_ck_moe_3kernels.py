#!/usr/bin/env python3
"""
Benchmark the 3 CK kernels that together equal one AITER ASM fused-MoE launch.

The CK 2-stage path is:
    1) ck_moe_stage1   -- GEMM1 (FP8 blockscale) + SiLU, write fp32/bf16 intermediate to HBM
    2) intermediate_quant   -- bf16/fp32 -> fp8 (1x128 block-scale) on the intermediate
    3) ck_moe_stage2   -- GEMM2 (FP8 blockscale) + splitk reduce -> bf16 output

(moe_sorting_ck is shared with the ASM path, so we omit it from the breakdown
 unless --include-sort is given.)

For each tile shape (DSR1 defaults), we time each kernel in isolation with
warmup + many iters, report mean us, GB/s for memory-bound stages, and
TFLOPS for the GEMM stages.
"""
import argparse
import torch
from einops import rearrange

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk
from aiter.fused_moe_bf16_asm import moe_sorting_ck
from aiter.ops.quant import get_hip_quant, pertoken_quant
from aiter.test_common import run_perftest

torch.set_default_device("cuda")


def block_quant_weight(w, sb_n=128, sb_k=128, qd=None):
    qd = qd or dtypes.fp8
    tmp = rearrange(
        w.view(-1, w.shape[1] // sb_n, sb_n, w.shape[2] // sb_k, sb_k),
        "e n0 n1 k0 k1 -> e n0 k0 (n1 k1)",
    )
    q, s = pertoken_quant(tmp, quant_dtype=qd)
    q = rearrange(
        q.view(-1, w.shape[1] // sb_n, w.shape[2] // sb_k, sb_n, sb_k),
        "e n0 k0 n1 k1 -> e (n0 n1) (k0 k1)",
    )
    return q, s.squeeze(-1)


def build_inputs(M, K, N, E, topk, dtype=torch.bfloat16):
    inp = torch.randn((M, K), dtype=dtype)
    w1 = torch.randn((E, N * 2, K), dtype=dtype) / 10  # gate-up packed
    w2 = torch.randn((E, K, N), dtype=dtype) / 10      # down
    score = torch.randn((M, E), dtype=dtype)
    topk_w, topk_ids = fused_topk(inp, score, topk, True)

    w1_q, w1_s = block_quant_weight(w1)
    w2_q, w2_s = block_quant_weight(w2)

    a1_q, a1_s = pertoken_quant(inp.view(-1, K // 128, 128), quant_dtype=dtypes.fp8)
    a1_q = a1_q.view(-1, K)
    a1_s = a1_s.squeeze(-1)

    return a1_q, w1_q, w1_s, w2_q, w2_s, topk_w, topk_ids, a1_s


def bench_three_ck(M, K, N, E, topk, dtype=torch.bfloat16, warmup=10, iters=50):
    a1_q, w1_q, w1_s, w2_q, w2_s, topk_w, topk_ids, a1_s = build_inputs(
        M, K, N, E, topk, dtype=dtype
    )

    block_size = 32 if (M * topk) // E <= 128 else 64
    sorted_ids, sorted_w, sorted_eids, num_valid, moe_buf = moe_sorting_ck(
        topk_ids, topk_w, E, K, dtype, block_size, None
    )

    # ----- stage1: GEMM1 + SiLU into intermediate (M, topk, N) -----
    a2 = torch.empty((M, topk, N), dtype=dtype, device="cuda")

    def _stage1():
        aiter.ck_moe_stage1(
            a1_q, w1_q, w2_q,
            sorted_ids, sorted_eids, num_valid,
            a2, topk,
            w1_s, a1_s, block_size, None, 1,  # ActOP=1 silu
        )
        return a2

    _, t_s1 = run_perftest(_stage1, num_warmup=warmup, num_iters=iters)

    # ----- midquant: bf16 intermediate -> fp8 (1x128 block-scale) -----
    qfn = get_hip_quant(QuantType.per_1x128)

    def _midquant():
        return qfn(a2, scale=None, quant_dtype=dtypes.fp8)

    _, t_qm = run_perftest(_midquant, num_warmup=warmup, num_iters=iters)
    a2q, a2s = _midquant()
    a2q = a2q.view(M, topk, -1)

    # ----- stage2: GEMM2 + splitk reduce -> bf16 output -----
    def _stage2():
        aiter.ck_moe_stage2(
            a2q, w1_q, w2_q,
            sorted_ids, sorted_eids, num_valid,
            moe_buf, topk,
            w2_s, a2s, block_size, sorted_w,
        )
        return moe_buf

    _, t_s2 = run_perftest(_stage2, num_warmup=warmup, num_iters=iters)

    # ----- analytic rates -----
    # Effective tokens processed: M_eff = M * topk
    Meff = M * topk
    # Stage1 GEMM:   [Meff, K] x [K, 2N]  but only valid_experts
    flops_s1 = 2 * Meff * (2 * N) * K
    # Stage2 GEMM:   [Meff, N] x [N, K]
    flops_s2 = 2 * Meff * K * N
    # midquant bytes:  read bf16 (Meff*N*2) + write fp8 (Meff*N) + scales
    bytes_qm = Meff * N * 2 + Meff * N + Meff * (N // 128) * 4

    tflops_s1 = flops_s1 / (t_s1 * 1e-6) / 1e12
    tflops_s2 = flops_s2 / (t_s2 * 1e-6) / 1e12
    gbs_qm = bytes_qm / (t_qm * 1e-6) / 1e9

    total = t_s1 + t_qm + t_s2

    return dict(
        M=M, stage1=t_s1, midquant=t_qm, stage2=t_s2, total=total,
        tflops_s1=tflops_s1, tflops_s2=tflops_s2, gbs_qm=gbs_qm,
    )


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

    print(f"\nCK 3-kernel breakdown : K={args.K} N={args.N} E={args.E} topk={args.topk}")
    print("=" * 100)
    print(f"{'M':>6} | {'stage1 us':>10} {'TFLOPS':>8} | "
          f"{'midq us':>9} {'GB/s':>7} | "
          f"{'stage2 us':>10} {'TFLOPS':>8} | {'sum us':>9}")
    print("-" * 100)

    rows = []
    for M in args.m:
        try:
            r = bench_three_ck(M, args.K, args.N, args.E, args.topk,
                               warmup=args.warmup, iters=args.iters)
            rows.append(r)
            print(f"{r['M']:6d} | {r['stage1']:10.2f} {r['tflops_s1']:8.1f} | "
                  f"{r['midquant']:9.2f} {r['gbs_qm']:7.1f} | "
                  f"{r['stage2']:10.2f} {r['tflops_s2']:8.1f} | {r['total']:9.2f}")
        except RuntimeError as e:
            print(f"{M:6d} | ERROR: {e}")
    print("=" * 100)

    if rows:
        # share-of-total summary
        print("\nShare of total time per stage:")
        print(f"{'M':>6} | {'stage1':>8} {'midquant':>9} {'stage2':>8}")
        print("-" * 40)
        for r in rows:
            sh1 = 100 * r["stage1"] / r["total"]
            shq = 100 * r["midquant"] / r["total"]
            sh2 = 100 * r["stage2"] / r["total"]
            print(f"{r['M']:6d} | {sh1:7.1f}% {shq:8.1f}% {sh2:7.1f}%")


if __name__ == "__main__":
    main()
