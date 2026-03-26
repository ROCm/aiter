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
import os
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import aiter
from aiter import dtypes, QuantType, ActivationType
from aiter.test_common import run_perftest, checkAllclose
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2
from aiter.ops.shuffle import shuffle_weight, shuffle_weight_a16w4, shuffle_scale_a16w4
from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

torch.set_default_device("cuda")

# ---------------------------------------------------------------------------
# FP4 dequantization helpers (from mixed_moe_gemm_2stage reference)
# ---------------------------------------------------------------------------

_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    u8 = scale.view(torch.uint8)
    i32 = u8.to(torch.int32) << 23
    i32[u8 == 0] = 0x00400000
    i32[u8 == 0xFF] = 0x7F800001
    return i32.view(torch.float32)


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8)
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=x.device)
    return lut[x.long()]


def dequant_sorted_fp4(
    fp4_packed: torch.Tensor,
    sorted_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    *,
    token_num: int,
    out_dim: int,
    topk: int,
    tile_m: int,
) -> torch.Tensor:
    """Dequant fp4x2 packed data using e8m0 sorted scale → f32 in token order.

    fp4_packed: (token_num, topk, out_dim//2) uint8
    sorted_scale: flat e8m0 in sorted tiled layout
    sorted_ids: packed (token_id | slot_id<<24)
    """
    a_f32 = _mxfp4_to_f32(fp4_packed.view(token_num, topk, out_dim // 2))
    padded_rows = max(sorted_ids.numel(),
                      sorted_scale.numel() // (out_dim // 32))
    padded_rows = (padded_rows + tile_m - 1) // tile_m * tile_m
    scale_groups = out_dim // 32
    scale_f32 = _e8m0_to_f32(sorted_scale.view(torch.uint8)
                              [:padded_rows * scale_groups]
                              .view(padded_rows, scale_groups))
    out = torch.zeros((token_num, topk, scale_groups),
                      dtype=torch.float32, device=fp4_packed.device)
    rows = sorted_ids[:min(sorted_ids.numel(), padded_rows)].to(torch.int64)
    tok = rows & 0xFFFFFF
    slot = rows >> 24
    valid = (tok < token_num) & (slot < topk)
    for i in torch.nonzero(valid, as_tuple=False).flatten():
        t, s = int(tok[i]), int(slot[i])
        out[t, s] = scale_f32[int(i)]
    return (a_f32.view(token_num, topk, scale_groups, 32) *
            out.unsqueeze(-1)).view(token_num, topk, out_dim)


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
    # score = torch.randn((token, E), dtype=dtype)
    score = torch.zeros((token, E), dtype=dtype)
    start_col = 0
    end_col = topk
    for token_id in range(token):
        score[token_id, start_col:end_col] = 1.0
        start_col = end_col % E
        end_col = start_col + topk
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
        a1_qt=a1_qt, a1_scale=a1_scale, a1_scale_sort=a1_scale_sort,
        w1_qt=w1_qt, w1_scale=w1_scale,
        w1_qt_shuf=w1_qt_shuf, w1_scale_shuf=w1_scale_shuf,
        w2_qt=w2_qt, w2_scale=w2_scale,
        w2_qt_shuf=w2_qt_shuf, w2_scale_shuf=w2_scale_shuf,
        topk_weights=topk_weights, topk_ids=topk_ids,
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
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
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
        tile_m=block_m, tile_n=128, tile_k=256,
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
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
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
        tile_m=block_m, tile_n=128, tile_k=256,
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
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
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
        topk=topk, tile_m=block_m, tile_n=128, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2_scale_shuf, a2_scale=fused_scale_sort,
        sorted_weights=sorted_weights,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fused vs Non-fused MoE A4W4 perf comparison")
    parser.add_argument("-t", "--tokens", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--model-dim", type=int, default=7168)
    parser.add_argument("--inter-dim", type=int, default=256)
    parser.add_argument("-E", "--experts", type=int, default=256)
    parser.add_argument("-k", "--topk", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.01)
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

        # ===================================================================
        # Precision checks (use fresh fused_out to avoid stale buffer issues)
        # ===================================================================
        print(f"\n  [Precision checks]", flush=True)
        prec_fused_out = torch.zeros_like(d["fused_out"])
        prec_s1 = "N/A"
        fp4_pct = 0.0
        scale_pct = 0.0
        prec_sq = "N/A"

        # -- [P3] E2E: compute torch reference, compare nf & fused against it
        # Torch reference: stage1
        ref_s1 = torch_moe_stage1(
            d["a1_qt"], d["w1_qt"], d["w2_qt"],
            d["topk_weights"], d["topk_ids"],
            dtype=torch.bfloat16,
            activation=ActivationType.Silu,
            quant_type=Q_TYPE,
            a1_scale=d["a1_scale"],
            w1_scale=d["w1_scale"],
        )
        # Torch reference: quant stage1 output → stage2
        ref_s1_flat = ref_s1.view(-1, inter_dim)
        ref_a2_qt, ref_a2_scale = TORCH_QUANT(ref_s1_flat, quant_dtype=Q_DTYPE_A)
        ref_e2e = torch_moe_stage2(
            ref_a2_qt, d["w1_qt"], d["w2_qt"],
            d["topk_weights"], d["topk_ids"],
            dtype=torch.bfloat16,
            quant_type=Q_TYPE,
            w2_scale=d["w2_scale"],
            a2_scale=ref_a2_scale,
        )
        torch.cuda.synchronize()

        # Non-fused e2e output
        nf_e2e_out = fn_nonfused_e2e(
            *common_args,
            d["w2_qt_shuf"], d["w2_scale_shuf"], d["sorted_weights"],
            topk, block_m, token, inter_dim,
        )
        torch.cuda.synchronize()

        # Fused e2e output
        f_e2e_out = fn_fused_e2e(
            *common_args, prec_fused_out,
            d["w2_qt_shuf"], d["w2_scale_shuf"], d["sorted_weights"],
            topk, block_m, token,
        )
        torch.cuda.synchronize()

        # Compare: non-fused vs torch
        err_nf_vs_torch = checkAllclose(
            ref_e2e, nf_e2e_out,
            rtol=args.rtol, atol=args.atol,
            msg="      [P3 torch vs nf_e2e] ",
        )
        prec_nf = "PASS" if err_nf_vs_torch == 0 else (
            "WARN" if err_nf_vs_torch <= 0.05 else "FAIL")

        # Compare: fused vs torch
        err_f_vs_torch = checkAllclose(
            ref_e2e, f_e2e_out,
            rtol=args.rtol, atol=args.atol,
            msg="      [P3 torch vs f_e2e]  ",
        )
        prec_f = "PASS" if err_f_vs_torch == 0 else (
            "WARN" if err_f_vs_torch <= 0.05 else "FAIL")

        # Compare: non-fused vs fused
        err_nf_vs_f = checkAllclose(
            nf_e2e_out, f_e2e_out,
            rtol=args.rtol, atol=args.atol,
            msg="      [P3 nf vs fused]     ",
        )
        prec_nf_f = "PASS" if err_nf_vs_f == 0 else (
            "WARN" if err_nf_vs_f <= 0.05 else "FAIL")

        results.append({
            "token": token,
            "nf_s1_us": us_nf_s1, "f_s1_us": us_f_s1,
            "nf_sq_us": us_nf_sq, "f_sq_us": us_f_sq,
            "nf_e2e_us": us_nf_e2e, "f_e2e_us": us_f_e2e,
            "prec_nf": prec_nf, "prec_f": prec_f, "prec_nf_f": prec_nf_f,
        })

    # --- Summary table ---
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  {'token':>6s}  {'nf_s1':>8s}  {'f_s1':>8s}  "
          f"{'nf_s1+q':>8s}  {'f_s1+s':>8s}  "
          f"{'nf_e2e':>8s}  {'f_e2e':>8s}  {'save':>10s}  "
          f"{'nf/T':>4s}  {'f/T':>4s}  {'nf/f':>4s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  "
          f"{'-'*8}  {'-'*8}  {'-'*10}  "
          f"{'-'*4}  {'-'*4}  {'-'*4}")
    for r in results:
        save = r["nf_e2e_us"] - r["f_e2e_us"]
        pct = save / r["nf_e2e_us"] * 100 if r["nf_e2e_us"] > 0 else 0
        print(f"  {r['token']:>6d}  {r['nf_s1_us']:>8.1f}  {r['f_s1_us']:>8.1f}  "
              f"{r['nf_sq_us']:>8.1f}  {r['f_sq_us']:>8.1f}  "
              f"{r['nf_e2e_us']:>8.1f}  {r['f_e2e_us']:>8.1f}  "
              f"{save:>+6.1f}/{pct:>+.1f}%  "
              f"{r['prec_nf']:>4s}  {r['prec_f']:>4s}  {r['prec_nf_f']:>4s}")


if __name__ == "__main__":
    main()
