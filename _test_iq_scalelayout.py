"""Validate the decoded make_preshuffle A-scale layout in pure torch.

Build a1_scale [cdiv(sorted,32), cdiv(K32,8), 4, 16] uint32 from per-token e8m0
using the layout decoded from moe_mxfp4_sort's triton scatter kernel, then diff
byte-for-byte against moe_mxfp4_sort() (BM=32). 100% match => layout is correct
and BM-agnostic (usable for BM=16).
"""
import sys
import torch
import aiter
from aiter import dtypes
from aiter.utility.fp4_utils import moe_mxfp4_sort
from bench_up_moe_v1 import KIMI, build_inputs


def build_a1_scale_preshuffle(e8m0_token, sorted_token_ids, sorted_m, K32, topk):
    """e8m0_token: [M, K32] uint8. Returns uint32 [MB, NB, 4, 16] make_preshuffle."""
    device = e8m0_token.device
    M = e8m0_token.shape[0]
    raw = sorted_token_ids[:sorted_m].to(torch.int64)
    tid = raw & 0xFFFFFF
    if topk > 1:
        ridx = tid * topk + (raw >> 24)
    else:
        ridx = tid
    valid = tid < M
    ridx_safe = torch.where(valid, ridx, torch.zeros_like(ridx))
    gathered = e8m0_token[ridx_safe].to(torch.int64)  # [sorted_m, K32]
    gathered = torch.where(valid[:, None], gathered, torch.zeros_like(gathered))

    MB = (sorted_m + 31) // 32
    NB = (K32 + 7) // 8
    out = torch.zeros((MB, NB, 4, 16), dtype=torch.int64, device=device)

    rows = torch.arange(sorted_m, device=device)
    mb = rows // 32
    within = rows % 32
    r = within % 16
    m_half = within // 16  # [sorted_m]

    kg = torch.arange(K32, device=device)
    nb = kg // 8
    kg8 = kg % 8
    c = kg8 % 4
    n_half = kg8 // 4  # [K32]

    # broadcast (sorted_m, K32)
    MB_i = mb[:, None].expand(sorted_m, K32)
    NB_i = nb[None, :].expand(sorted_m, K32)
    C_i = c[None, :].expand(sorted_m, K32)
    R_i = r[:, None].expand(sorted_m, K32)
    BYTE = (m_half[:, None] + n_half[None, :] * 2)  # 0..3
    val = gathered << (BYTE * 8)

    flat_idx = ((MB_i * NB + NB_i) * 4 + C_i) * 16 + R_i
    out_flat = out.view(-1)
    out_flat.scatter_add_(0, flat_idx.reshape(-1), val.reshape(-1))
    return out.to(torch.uint32)


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    BM = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    device = torch.device("cuda")
    shape = KIMI
    hidden, topk_ids, topk_weight = build_inputs(shape, M, device)
    NE, D_HIDDEN, D_INTER = shape.NE, shape.H, shape.INTER
    topk = topk_ids.shape[1]
    K32 = D_HIDDEN // 32

    active = min(NE, M * topk)
    cumsum_max = M * topk + active * (BM - 1)
    max_sorted = ((cumsum_max + BM - 1) // BM) * BM

    def _e(d): return torch.empty((0,), dtype=dtypes.bf16, device=device)
    sorted_token_ids = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    sorted_expert_ids = torch.empty((max_sorted // BM,), device=device, dtype=dtypes.i32)
    cumsum_tensor = torch.empty((1,), device=device, dtype=dtypes.i32)
    reverse_sorted = torch.empty((M * topk,), device=device, dtype=dtypes.i32)
    sorted_weights = torch.empty((max_sorted,), device=device, dtype=dtypes.fp32)
    masked_m = torch.empty((NE,), device=device, dtype=dtypes.i32)
    m_indices = torch.empty((max_sorted,), device=device, dtype=dtypes.i32)
    a_quant = torch.empty((M, D_HIDDEN // 2), device=device, dtype=torch.uint8)
    a_scale = torch.empty((M, K32), device=device, dtype=torch.uint8)
    out_buf = torch.empty((M, D_HIDDEN), dtype=dtypes.bf16, device=device)

    aiter.mxfp4_moe_sort(
        topk_ids=topk_ids, topk_weight=topk_weight,
        sorted_token_ids=sorted_token_ids, sorted_expert_ids=sorted_expert_ids,
        cumsum_tensor=cumsum_tensor, reverse_sorted=reverse_sorted,
        sorted_weights=sorted_weights, masked_m=masked_m, m_indices=m_indices,
        bf16_zero_out=_e(0), bf16_zero_workspace=_e(0),
        M_logical=M, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, D_INTER=D_INTER, MB=BM,
        prologue=1,
    )
    aiter.mxfp4_moe_quant(
        a_input=hidden, a_quant=a_quant, a_scale=a_scale,
        bf16_zero_out=out_buf, NE=NE, TOPK=topk, D_HIDDEN=D_HIDDEN, MB=BM,
    )
    num_valid_ids = cumsum_tensor.repeat(2)

    ref = moe_mxfp4_sort(
        a_scale.view(dtypes.fp8_e8m0).view(M, 1, -1),
        sorted_ids=sorted_token_ids, num_valid_ids=num_valid_ids,
        token_num=M, block_size=BM,
    )
    # a_scale is per-token [M, K32]; gather by token id (scale topk dim = 1).
    mine = build_a1_scale_preshuffle(a_scale, sorted_token_ids, max_sorted, K32, 1)

    ref_b = ref.reshape(-1).view(torch.uint8)
    mine_b = mine.reshape(-1).view(torch.uint8)
    print(f"M={M} BM={BM}  ref shape={tuple(ref.shape)} mine shape={tuple(mine.shape)}")
    n = min(ref_b.numel(), mine_b.numel())
    match = (ref_b[:n] == mine_b[:n]).float().mean().item()
    print(f"  a1_scale byte match: {match*100:.3f}%  ({n} bytes)")
    # only valid rows matter
    cum = int(cumsum_tensor.item())
    print(f"  cumsum(valid sorted rows)={cum}")


if __name__ == "__main__":
    main()
