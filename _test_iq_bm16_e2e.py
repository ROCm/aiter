"""End-to-end BM=16 inline-quant path correctness.

  BM16 cheap sort (prologue=0, MB=16)
  -> torch a1_scale (validated make_preshuffle layout; perf later via triton)
  -> FlyDSL inline-quant gemm1 (tile_m=16, reads bf16) 
  -> FlyDSL gemm2 (atomic, sort_block_m=16)
Compare final output cosine vs the mxfp4 reference (fused_moe with mx_w).
"""
import sys
import torch
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

from bench_up_moe_v1 import KIMI, build_weights, build_inputs
from _test_iq_quantcheck import quant_like_kernel
from _test_iq_scalelayout import build_a1_scale_preshuffle


def _empty_bf16(d):
    return torch.empty((0,), dtype=dtypes.bf16, device=d)


def cosine(a, b):
    a = a.float().reshape(-1); b = b.float().reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def bm16_inline(hidden, fly_w, topk_ids, topk_weight):
    device = hidden.device
    w1u = fly_w["w1"].view(torch.uint8)
    w2u = fly_w["w2"].view(torch.uint8)
    w1_scale = fly_w["w1_scale"]; w2_scale = fly_w["w2_scale"]
    NE = w1u.shape[0]; D_HIDDEN = hidden.shape[1]; D_INTER = w1u.shape[1] // 2
    M = hidden.shape[0]; topk = topk_ids.shape[1]; K32 = D_HIDDEN // 32
    BM = 16

    active = min(NE, M * topk)
    cumsum_max = M * topk + active * (BM - 1)
    max_sorted = ((cumsum_max + BM - 1) // BM) * BM

    sorted_token_ids = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    sorted_expert_ids = torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32)
    cumsum_tensor = torch.empty((1,), device=device, dtype=dtypes.i32)
    reverse_sorted = torch.empty((M * topk,), device=device, dtype=dtypes.i32)
    sorted_weights = torch.empty((max_sorted,), device=device, dtype=dtypes.fp32)
    masked_m = torch.empty((NE,), device=device, dtype=dtypes.i32)
    m_indices = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    out_buf = torch.zeros((M, D_HIDDEN), dtype=dtypes.bf16, device=device)

    # cheap BM=16 single-CTA sort (prologue=0 = inline-quant sort variant)
    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids, topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights, masked_m=masked_m, m_indices=m_indices,
        bf16_zero_out=_empty_bf16(device), bf16_zero_workspace=_empty_bf16(device),
        M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM,
        prologue=0,
    )
    num_valid_ids = cumsum_tensor.repeat(2)

    # torch a1_scale (validated layout); e8m0 from hidden (validated quant)
    _, e8m0_tok = quant_like_kernel(hidden)          # [M, K32] uint8
    a1_scale_u32 = build_a1_scale_preshuffle(e8m0_tok, sorted_token_ids, max_sorted, K32, 1)
    a1_scale = a1_scale_u32.view(torch.uint8).view(dtypes.fp8_e8m0)

    inter = flydsl_moe_stage1(
        a=hidden, w1=w1u.view(dtypes.fp4x2),
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids, topk=topk,
        tile_m=16, tile_n=128, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="fp4",
        w1_scale=w1_scale.view(dtypes.fp8_e8m0),
        a1_scale=a1_scale, sorted_weights=None, use_async_copy=False,
        waves_per_eu=2, b_nt=2, gate_mode="separated", inline_quant=True,
    )
    inter_q, inter_scale = (inter if isinstance(inter, tuple) else (inter, None))

    flydsl_moe_stage2(
        inter_states=inter_q, w2=w2u.view(dtypes.fp4x2),
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids, topk=topk, out=out_buf,
        tile_m=16, tile_n=256, tile_k=256,
        a_dtype="fp4", b_dtype="fp4", out_dtype="bf16", mode="atomic",
        w2_scale=w2_scale.view(dtypes.fp8_e8m0), a2_scale=inter_scale,
        sorted_weights=sorted_weights, b_nt=2, sort_block_m=16, persist=None,
    )
    return out_buf


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    device = torch.device("cuda")
    shape = KIMI
    fly_w, mx_w = build_weights(shape, device)
    hidden, topk_ids, topk_weight = build_inputs(shape, M, device)

    ref = fused_moe(
        hidden, mx_w["w1"], mx_w["w2"], topk_weight, topk_ids,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        w1_scale=mx_w["w1_scale"], w2_scale=mx_w["w2_scale"],
    ).clone()
    torch.cuda.synchronize()

    out = bm16_inline(hidden, fly_w, topk_ids, topk_weight).clone()
    torch.cuda.synchronize()

    c = cosine(out, ref)
    print(f"M={M}  BM16-inline vs mxfp4 ref cosine: {c:.5f}")
    print(f"  out  norm={out.float().norm():.3f}  ref norm={ref.float().norm():.3f}")
    print("PASS" if c > 0.95 else "FAIL")


if __name__ == "__main__":
    main()
