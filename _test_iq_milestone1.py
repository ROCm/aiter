"""Milestone 1: validate FlyDSL gemm1 inline bf16->fp4 quant.

Compare two gemm1 runs that share the SAME w1 / w1_scale / a1_scale:
  * std : a = a_quant (fp4, from mxfp4 quant kernel), inline_quant=False
  * iq  : a = hidden  (bf16),                         inline_quant=True

If the inline quant reproduces a_quant exactly, the two gemm1 inputs are
bit-identical and inter_q / inter_scale must match.
"""
import sys

import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1
from aiter.utility.fp4_utils import moe_mxfp4_sort

from bench_up_moe_v1 import KIMI, build_weights, build_inputs


def _empty_bf16(device):
    return torch.empty((0,), dtype=dtypes.bf16, device=device)


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    device = torch.device("cuda")
    shape = KIMI
    fly_w, _mx_w = build_weights(shape, device)
    hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
    topk = topk_ids.shape[1]

    w1u = fly_w["w1"].view(torch.uint8)
    w1_scale = fly_w["w1_scale"]
    NE = w1u.shape[0]
    D_HIDDEN = hidden.shape[1]
    D_INTER = w1u.shape[1] // 2
    BM = 32

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
    a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((M, D_HIDDEN // 32), device=device, dtype=torch.uint8)
    out_buf = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)

    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids, topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights, masked_m=masked_m, m_indices=m_indices,
        bf16_zero_out=_empty_bf16(device), bf16_zero_workspace=_empty_bf16(device),
        M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM,
        prologue=1,
    )
    aiter.mxfp4_moe_quant(
        a_input=hidden, a_quant=a_quant, a_scale=a_scale,
        bf16_zero_out=out_buf, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
    )
    num_valid_ids = cumsum_tensor.repeat(2)
    a1 = a_quant.view(dtypes.fp4x2)
    a1_scale = moe_mxfp4_sort(
        a_scale.view(dtypes.fp8_e8m0).view(M, 1, -1),
        sorted_ids=sorted_token_ids, num_valid_ids=num_valid_ids,
        token_num=M, block_size=BM,
    )

    w1v = w1u.view(dtypes.fp4x2)
    w1sv = w1_scale.view(dtypes.fp8_e8m0)

    tile_m = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    from aiter.utility.fp4_utils import mxfp4_to_f32

    def run(a, tm, iq):
        r = flydsl_moe_stage1(
            a=a, w1=w1v, use_async_copy=False, inline_quant=iq,
            sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids, topk=topk,
            tile_m=tm, tile_n=128, tile_k=256,
            a_dtype="fp4", b_dtype="fp4", out_dtype="fp4",
            w1_scale=w1sv, a1_scale=a1_scale, sorted_weights=None,
            waves_per_eu=2, b_nt=2, gate_mode="separated",
            sort_block_m_override=32,
        )
        return r if isinstance(r, tuple) else (r, None)

    ref_q, _ = run(a1, 32, False)        # known-good reference (tile_m=32 std)
    std_q, _ = run(a1, tile_m, False)    # std at requested tile_m
    iq_q, _ = run(hidden, tile_m, True)  # inline at requested tile_m
    torch.cuda.synchronize()

    def cmp(name, q):
        m = (ref_q.view(torch.uint8).reshape(-1) == q.view(torch.uint8).reshape(-1)).float().mean().item()
        c = torch.nn.functional.cosine_similarity(
            mxfp4_to_f32(ref_q.view(dtypes.fp4x2)).reshape(-1),
            mxfp4_to_f32(q.view(dtypes.fp4x2)).reshape(-1), dim=0).item()
        print(f"  {name:18s} vs std@32: byte={m*100:6.2f}%  cos={c:.5f}")

    print(f"M={M} tile_m={tile_m}:")
    cmp(f"std@{tile_m}", std_q)
    cmp(f"inline@{tile_m}", iq_q)


if __name__ == "__main__":
    main()
