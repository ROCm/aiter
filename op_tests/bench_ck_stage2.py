#!/usr/bin/env python3
"""
Benchmark CK ck_moe_stage2 in isolation on DSR1 block-scale FP8 shapes.

stage2 = GEMM2 (FP8 1x128 block-scale) + splitk reduce -> bf16 output.

This is the kernel that corresponds to the symbol
  ck::kernel_moe_gemm_2lds<GridwiseMoeGemmBlockScale<RowMajor, ColumnMajor,
       Tuple<RowMajor>, RowMajor, ..., 256, 1, 128, 128, 64, 128, 128, ...,
       PipelineVersion::v3, ..., IsSplitK=true, MulRoutedWeight=false>, ...>
in module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_1x128_mulWeightStage2_splitk.so

Usage:
  HIP_VISIBLE_DEVICES=0 python op_tests/bench_ck_stage2.py -m 16 128 1024
"""
import argparse
import torch
from einops import rearrange

import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.fused_moe import fused_topk
from aiter.fused_moe_bf16_asm import moe_sorting_ck
from aiter.ops.quant import pertoken_quant, get_hip_quant
from aiter.test_common import run_perftest

torch.set_default_device("cuda")


def make_fp8_weight_directly(E, N, K, sb_n=128, sb_k=128):
    """Skip bf16 staging; create random FP8 weight + scales directly to save VRAM."""
    q = torch.randint(-127, 127, (E, N, K), dtype=torch.int8, device="cuda").view(
        dtypes.fp8
    )
    s = (torch.rand((E, N // sb_n, K // sb_k), dtype=torch.float32, device="cuda")
         * 0.01 + 0.001)
    return q, s


def bench_stage2(M, K, N, E, topk, dtype=torch.bfloat16, warmup=10, iters=50):
    inp = torch.randn((M, K), dtype=dtype)
    score = torch.randn((M, E), dtype=dtype)
    topk_w, topk_ids = fused_topk(inp, score, topk, True)

    w1_q, w1_s = make_fp8_weight_directly(E, N * 2, K)
    w2_q, w2_s = make_fp8_weight_directly(E, K, N)
    a1_q, a1_s = pertoken_quant(inp.view(-1, K // 128, 128), quant_dtype=dtypes.fp8)
    a1_q = a1_q.view(-1, K)
    a1_s = a1_s.squeeze(-1)
    del inp, score

    block_m = 32 if (M * topk) // E <= 128 else 64

    sorted_ids, sorted_w, sorted_eids, num_valid, moe_buf = moe_sorting_ck(
        topk_ids, topk_w, E, K, dtype, block_m, None
    )

    # We need a valid intermediate to feed into stage2.
    # Run stage1 once (untimed) to produce it, then quantize.
    a2 = torch.empty((M, topk, N), dtype=dtype, device="cuda")
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
        activation=1,
        splitk=1,
    )

    qfn = get_hip_quant(QuantType.per_1x128)
    a2q, a2s = qfn(a2, scale=None, quant_dtype=dtypes.fp8)
    a2q = a2q.view(M, topk, -1)

    def _run():
        aiter.ck_moe_stage2(
            a2q, w1_q, w2_q,
            sorted_ids, sorted_eids, num_valid,
            moe_buf, topk,
            kernelName="",
            w2_scale=w2_s,
            a2_scale=a2s,
            block_m=block_m,
            sorted_weights=sorted_w,
            quant_type=QuantType.per_1x128,
            activation=0,
            splitk=1,
        )
        return moe_buf

    _, t_us = run_perftest(_run, num_warmup=warmup, num_iters=iters)

    Meff = M * topk
    flops = 2 * Meff * K * N
    tflops = flops / (t_us * 1e-6) / 1e12
    return dict(M=M, us=t_us, tflops=tflops, block_m=block_m)


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

    print(f"\nCK ck_moe_stage2  (FP8 1x128 blockscale GEMM2 + splitk reduce)  "
          f"K={args.K} N={args.N} E={args.E} topk={args.topk}")
    print("=" * 70)
    print(f"{'M':>6} | {'block_m':>8} | {'stage2 us':>11} | {'TFLOPS':>8}")
    print("-" * 70)

    for M in args.m:
        try:
            r = bench_stage2(M, args.K, args.N, args.E, args.topk,
                             warmup=args.warmup, iters=args.iters)
            print(f"{r['M']:6d} | {r['block_m']:8d} | "
                  f"{r['us']:11.2f} | {r['tflops']:8.1f}")
        except RuntimeError as e:
            print(f"{M:6d} | ERROR: {str(e)[:120]}")


if __name__ == "__main__":
    main()
