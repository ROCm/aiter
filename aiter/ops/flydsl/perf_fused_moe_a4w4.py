# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Performance comparison: fused vs non-fused FlyDSL MoE A4W4 paths.

Benchmarks:
  1. Stage1 kernel only (non-fused bf16 vs fused fp4)
  2. Stage1 + quant overhead (non-fused stage1 + torch_quant vs fused stage1)
  3. Full e2e pipeline (non-fused vs fused through stage2)

Usage:
    python perf_fused_moe_a4w4.py
    python perf_fused_moe_a4w4.py -t 16 64 256
    python perf_fused_moe_a4w4.py --num-iters 20
"""

import argparse
import sys

import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.test_common import run_perftest
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

torch.set_default_device("cuda")

Q_TYPE = QuantType.per_1x32
Q_DTYPE_A = dtypes.fp4x2
TORCH_QUANT = aiter.get_torch_quant(Q_TYPE)


def setup_data(token, model_dim, inter_dim, E, topk, block_m, dtype=torch.bfloat16):
    """Prepare all tensors needed for benchmarking."""
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    inp = torch.randn((token, model_dim), dtype=dtype) / 10
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype) / 10
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype) / 10
    score = torch.randn((token, E), dtype=dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = TORCH_QUANT(w1, quant_dtype=Q_DTYPE_A)
    w2_qt, w2_scale = TORCH_QUANT(w2, quant_dtype=Q_DTYPE_A)
    w1_qt = w1_qt.view(E, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(E, model_dim, inter_dim // 2)
    a1_qt, a1_scale = TORCH_QUANT(inp, quant_dtype=Q_DTYPE_A)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m
    )

    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, E, False)

    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, block_size=block_m,
    )

    fused_out = torch.zeros((token, topk, inter_dim // 2), dtype=torch.uint8, device="cuda")

    return dict(
        token=token, model_dim=model_dim, inter_dim=inter_dim,
        E=E, topk=topk, block_m=block_m, dtype=dtype,
        a1_qt=a1_qt, a1_scale_sort=a1_scale_sort,
        w1_qt_shuf=w1_qt_shuf, w1_scale_shuf=w1_scale_shuf,
        w2_qt_shuf=w2_qt_shuf, w2_scale_shuf=w2_scale_shuf,
        sorted_ids=sorted_ids, sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids, num_valid_ids=num_valid_ids,
        fused_out=fused_out,
    )


# ---------------------------------------------------------------------------
# Benchmark target functions (explicit tensor args for run_perftest)
# ---------------------------------------------------------------------------

def fn_nonfused_stage1(
    a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
    w1_scale_shuf, a1_scale_sort,
    topk, block_m,
):
    return flydsl_moe_stage1(
        a=a1_qt, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
    )


def fn_fused_stage1(
    a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
    w1_scale_shuf, a1_scale_sort,
    fused_out,
    topk, block_m,
):
    return flydsl_moe_stage1(
        a=a1_qt, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=fused_out, topk=topk,
        tile_m=block_m, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
        fuse_fp4_quant=True, fuse_sort_scale=True,
    )


def fn_nonfused_stage1_plus_quant(
    a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
    w1_scale_shuf, a1_scale_sort,
    topk, block_m, token, inter_dim,
):
    s1_out = flydsl_moe_stage1(
        a=a1_qt, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
    )
    a2_qt, a2_scale_sort = fused_dynamic_mxfp4_quant_moe_sort(
        s1_out.view(-1, inter_dim),
        sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token, topk=topk, block_size=block_m,
    )
    return a2_qt.view(token, topk, -1), a2_scale_sort


def fn_fused_stage1_with_sort(
    a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
    w1_scale_shuf, a1_scale_sort,
    fused_out,
    topk, block_m,
):
    return flydsl_moe_stage1(
        a=a1_qt, w1=w1_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=fused_out, topk=topk,
        tile_m=block_m, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16",
        w1_scale=w1_scale_shuf, a1_scale=a1_scale_sort,
        fuse_fp4_quant=True, fuse_sort_scale=True,
    )


def fn_nonfused_e2e(
    a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
    w1_scale_shuf, a1_scale_sort,
    w2_qt_shuf, w2_scale_shuf, sorted_weights,
    topk, block_m, token, inter_dim,
):
    a2_qt, a2_scale_sort = fn_nonfused_stage1_plus_quant(
        a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        w1_scale_shuf, a1_scale_sort,
        topk, block_m, token, inter_dim,
    )
    return flydsl_moe_stage2(
        inter_states=a2_qt, w2=w2_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2_scale_shuf, a2_scale=a2_scale_sort,
        sorted_weights=sorted_weights,
    )


def fn_fused_e2e(
    a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
    w1_scale_shuf, a1_scale_sort,
    fused_out,
    w2_qt_shuf, w2_scale_shuf, sorted_weights,
    topk, block_m, token,
):
    result = fn_fused_stage1_with_sort(
        a1_qt, w1_qt_shuf, sorted_ids, sorted_expert_ids, num_valid_ids,
        w1_scale_shuf, a1_scale_sort,
        fused_out,
        topk, block_m,
    )
    fused_a2, fused_scale_sort = result
    return flydsl_moe_stage2(
        inter_states=fused_a2.view(token, topk, -1),
        w2=w2_qt_shuf,
        sorted_token_ids=sorted_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        topk=topk, tile_m=block_m, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2_scale_shuf, a2_scale=fused_scale_sort,
        sorted_weights=sorted_weights,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fused vs Non-fused MoE A4W4 perf comparison")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=256)
    parser.add_argument("-k", "--topk", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    args = parser.parse_args()

    from aiter.ops.flydsl.utils import is_flydsl_available
    if not is_flydsl_available():
        print("[SKIP] FlyDSL not available.")
        sys.exit(0)

    ni = args.num_iters
    nw = args.num_warmup
    results = []

    for token in args.tokens:
        torch.cuda.empty_cache()
        print(f"\n{'='*80}")
        print(f"  token={token}, model_dim={args.model_dim}, inter_dim={args.inter_dim}, "
              f"E={args.experts}, topk={args.topk}, block_m={args.block_m}")
        print(f"{'='*80}")

        d = setup_data(
            token, args.model_dim, args.inter_dim,
            args.experts, args.topk, args.block_m,
        )

        common_args = (
            d["a1_qt"], d["w1_qt_shuf"],
            d["sorted_ids"], d["sorted_expert_ids"], d["num_valid_ids"],
            d["w1_scale_shuf"], d["a1_scale_sort"],
        )
        topk = d["topk"]
        block_m = d["block_m"]
        inter_dim = d["inter_dim"]

        nr = 1  # num_rotate_args=1: no deep copies, kernel doesn't modify inputs

        # --- Bench 1: Stage1 kernel only ---
        print(f"\n  [1] Stage1 kernel only:", flush=True)
        print(f"      running non-fused stage1...", flush=True)
        _, us_nf_s1 = run_perftest(
            fn_nonfused_stage1,
            *common_args,
            topk, block_m,
            num_iters=ni, num_warmup=nw, num_rotate_args=nr,
        )
        print(f"      running fused stage1...", flush=True)
        _, us_f_s1 = run_perftest(
            fn_fused_stage1,
            *common_args,
            d["fused_out"],
            topk, block_m,
            num_iters=ni, num_warmup=nw, num_rotate_args=nr,
        )
        print(f"      non-fused stage1 (bf16 out):  {us_nf_s1:>8.1f} us")
        print(f"      fused stage1 (fp4+scale out): {us_f_s1:>8.1f} us")
        diff_pct = (us_f_s1 / us_nf_s1 - 1) * 100 if us_nf_s1 > 0 else 0
        print(f"      diff: {us_f_s1 - us_nf_s1:>+8.1f} us  ({diff_pct:>+.1f}%)")

        # --- Bench 2: Stage1 + quant pipeline ---
        print(f"\n  [2] Stage1 + quantization pipeline:", flush=True)
        print(f"      running non-fused stage1+quant...", flush=True)
        _, us_nf_sq = run_perftest(
            fn_nonfused_stage1_plus_quant,
            *common_args,
            topk, block_m, token, inter_dim,
            num_iters=ni, num_warmup=nw, num_rotate_args=nr,
        )
        print(f"      running fused stage1 (with sort)...", flush=True)
        _, us_f_sq = run_perftest(
            fn_fused_stage1_with_sort,
            *common_args,
            d["fused_out"],
            topk, block_m,
            num_iters=ni, num_warmup=nw, num_rotate_args=nr,
        )
        print(f"      non-fused (stage1 + fused_quant_sort):   {us_nf_sq:>8.1f} us")
        print(f"      fused    (fused_stage1 + fused_sort):    {us_f_sq:>8.1f} us")
        saving_sq = us_nf_sq - us_f_sq
        pct_sq = saving_sq / us_nf_sq * 100 if us_nf_sq > 0 else 0
        print(f"      saving: {saving_sq:>8.1f} us  ({pct_sq:.1f}%)")

        # --- Bench 3: Full e2e ---
        print(f"\n  [3] Full e2e (stage1 + quant + stage2):", flush=True)
        print(f"      running non-fused e2e...", flush=True)
        _, us_nf_e2e = run_perftest(
            fn_nonfused_e2e,
            *common_args,
            d["w2_qt_shuf"], d["w2_scale_shuf"], d["sorted_weights"],
            topk, block_m, token, inter_dim,
            num_iters=ni, num_warmup=nw, num_rotate_args=nr,
        )
        print(f"      running fused e2e...", flush=True)
        _, us_f_e2e = run_perftest(
            fn_fused_e2e,
            *common_args,
            d["fused_out"],
            d["w2_qt_shuf"], d["w2_scale_shuf"], d["sorted_weights"],
            topk, block_m, token,
            num_iters=ni, num_warmup=nw, num_rotate_args=nr,
        )
        print(f"      non-fused e2e: {us_nf_e2e:>8.1f} us")
        print(f"      fused e2e:     {us_f_e2e:>8.1f} us")
        saving_e2e = us_nf_e2e - us_f_e2e
        pct_e2e = saving_e2e / us_nf_e2e * 100 if us_nf_e2e > 0 else 0
        print(f"      saving: {saving_e2e:>8.1f} us  ({pct_e2e:.1f}%)")

        tflops_op = token * args.model_dim * inter_dim * 3 * topk * 2
        print(f"\n      TFLOPS (non-fused e2e): {tflops_op/us_nf_e2e/1e6:>.2f}")
        print(f"      TFLOPS (fused e2e):     {tflops_op/us_f_e2e/1e6:>.2f}")

        results.append({
            "token": token,
            "nf_s1_us": us_nf_s1, "f_s1_us": us_f_s1,
            "nf_sq_us": us_nf_sq, "f_sq_us": us_f_sq,
            "nf_e2e_us": us_nf_e2e, "f_e2e_us": us_f_e2e,
        })

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'token':>6s}  {'nf_s1':>8s}  {'f_s1':>8s}  {'nf_s1+q':>8s}  {'f_s1+s':>8s}  {'nf_e2e':>8s}  {'f_e2e':>8s}  {'e2e_save':>10s}")
    print(f"  {'':>6s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  {'(us / %)':>10s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    for r in results:
        save = r["nf_e2e_us"] - r["f_e2e_us"]
        pct = save / r["nf_e2e_us"] * 100 if r["nf_e2e_us"] > 0 else 0
        print(f"  {r['token']:>6d}  {r['nf_s1_us']:>8.1f}  {r['f_s1_us']:>8.1f}  "
              f"{r['nf_sq_us']:>8.1f}  {r['f_sq_us']:>8.1f}  "
              f"{r['nf_e2e_us']:>8.1f}  {r['f_e2e_us']:>8.1f}  "
              f"{save:>+6.1f}/{pct:>+.1f}%")


if __name__ == "__main__":
    main()
