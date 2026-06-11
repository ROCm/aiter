#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""op-test for ``fused_qk_rope_concat_and_cache_mla_seg`` (DeepSeek V3.1 MLA).

Fused kernel that (NO RMSNorm; q/k are already post-projection):
  - q: nope quantized directly; pe RoPE'd then quantized -> q_out fp8
  - k: nope quantized directly; pe RoPE'd then quantized
       -> kv_cache fp8, segmented block layout
        block: [page_size x kv_lora (nope)][page_size x pe (rope)]
  - static per-tensor fp8 quant (q_scale, k_scale)
  - q_out head_dim padded (tail untouched)
"""

import argparse
import random

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

KV_LORA = 512
PE_DIM = 64
PAGE_SIZE = 64
HALF = PE_DIM // 2
_FP8 = dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8).max)


def _rope_ref(pe, cos, sin, pos, is_neox):
    """pe: [N, PE_DIM] (N = T or T*H). cos/sin: [max_pos, HALF]. pos: [N]."""
    N = pe.shape[0]
    c = cos[pos].float()  # [N, HALF]
    s = sin[pos].float()
    out = torch.empty_like(pe, dtype=torch.float32)
    pe = pe.float()
    if is_neox:
        lo = pe[:, :HALF]
        hi = pe[:, HALF:]
        out[:, :HALF] = lo * c - hi * s
        out[:, HALF:] = hi * c + lo * s
    else:
        even = pe[:, 0::2]
        odd = pe[:, 1::2]
        out[:, 0::2] = even * c - odd * s
        out[:, 1::2] = odd * c + even * s
    return out


def _ref(
    q_nope, q_pe, kv_c, k_pe, cos, sin, pos, slot_mapping,
    q_scale, k_scale, num_blocks, is_neox,
):
    T, H, _ = q_nope.shape
    q_inv = 1.0 / q_scale.item()
    k_inv = 1.0 / k_scale.item()

    # ---- q_out [T, H, 576] (compare only [:576]) ----
    q_nope_q = (q_nope.float() * q_inv).to(_FP8)
    qpe_pos = pos.view(T, 1).expand(T, H).reshape(-1)
    q_pe_roped = _rope_ref(q_pe.reshape(-1, PE_DIM), cos, sin, qpe_pos, is_neox)
    q_pe_q = (q_pe_roped * q_inv).to(_FP8).view(T, H, PE_DIM)
    q_out_ref = torch.cat([q_nope_q.view(T, H, KV_LORA), q_pe_q], dim=-1)  # [T,H,576]

    # ---- k: nope quant (no norm); pe RoPE + quant ----
    k_nope_q = (kv_c.float() * k_inv).to(_FP8)  # [T, 512]
    k_pe_roped = _rope_ref(k_pe, cos, sin, pos, is_neox)
    k_pe_q = (k_pe_roped * k_inv).to(_FP8)  # [T, 64]

    # ---- segmented kv_cache [num_blocks, 36864] ----
    block_stride = PAGE_SIZE * KV_LORA + PAGE_SIZE * PE_DIM
    kv_cache_ref = torch.zeros(num_blocks, block_stride, dtype=_FP8)
    blk = (slot_mapping // PAGE_SIZE).long()
    off = (slot_mapping % PAGE_SIZE).long()
    for i in range(T):
        if slot_mapping[i].item() < 0:
            continue
        b, o = blk[i].item(), off[i].item()
        nbase = o * KV_LORA
        rbase = PAGE_SIZE * KV_LORA + o * PE_DIM
        kv_cache_ref[b, nbase:nbase + KV_LORA] = k_nope_q[i]
        kv_cache_ref[b, rbase:rbase + PE_DIM] = k_pe_q[i]
    return q_out_ref, kv_cache_ref


@benchmark()
def test_fused_qk_rope_concat_cache_mla_seg(T, H, is_neox, q_out_dim=768):
    torch.manual_seed(0)
    dev = "cuda"
    num_blocks = (T + PAGE_SIZE - 1) // PAGE_SIZE + 1
    total_slots = num_blocks * PAGE_SIZE
    slot_lst = random.sample(range(total_slots), T)
    slot_mapping = torch.tensor(slot_lst, dtype=torch.int64, device=dev)

    q_nope = torch.randn(T, H, KV_LORA, dtype=dtypes.bf16, device=dev) * 0.1
    q_pe = torch.randn(T, H, PE_DIM, dtype=dtypes.bf16, device=dev) * 0.1
    kv_c = torch.randn(T, KV_LORA, dtype=dtypes.bf16, device=dev) * 0.1
    k_pe = torch.randn(T, PE_DIM, dtype=dtypes.bf16, device=dev) * 0.1

    max_pos = max(T, 64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, PE_DIM, 2, device=dev).float() / PE_DIM))
    freqs = torch.einsum("i,j->ij", torch.arange(max_pos, device=dev).float(), inv_freq)
    cos = freqs.cos().to(dtypes.bf16).contiguous()  # [max_pos, 32]
    sin = freqs.sin().to(dtypes.bf16).contiguous()
    pos = torch.randint(0, max_pos, (T,), dtype=torch.int64, device=dev)

    q_scale = torch.tensor(0.05, dtype=torch.float32, device=dev)
    k_scale = torch.tensor(0.05, dtype=torch.float32, device=dev)

    block_stride = PAGE_SIZE * KV_LORA + PAGE_SIZE * PE_DIM
    kv_cache = torch.zeros(num_blocks, block_stride, dtype=_FP8, device=dev)
    q_out = torch.zeros(T, H, q_out_dim, dtype=_FP8, device=dev)

    q_out_ref, kv_cache_ref = _ref(
        q_nope, q_pe, kv_c, k_pe, cos, sin, pos, slot_mapping,
        q_scale, k_scale, num_blocks, is_neox,
    )

    _, us = run_perftest(
        aiter.fused_qk_rope_concat_and_cache_mla_seg,
        q_nope, q_pe, kv_c, k_pe, kv_cache, q_out, slot_mapping,
        k_scale, q_scale, pos, cos, sin, is_neox,
    )

    # dequant compare
    q_got = q_out[:, :, : KV_LORA + PE_DIM].float() * q_scale.item()
    q_exp = q_out_ref.float() * q_scale.item()
    kv_got = kv_cache.float() * k_scale.item()
    kv_exp = kv_cache_ref.float() * k_scale.item()
    err_q = checkAllclose(q_exp, q_got, rtol=0.05, atol=0.05, msg="q_out")
    err_kv = checkAllclose(kv_exp, kv_got, rtol=0.05, atol=0.05, msg="kv_cache")
    return {"us": round(us, 3), "err_q": err_q, "err_kv": err_kv}


parser = argparse.ArgumentParser()
parser.add_argument("-T", type=int, nargs="*", default=[1, 16, 128, 256])
parser.add_argument("-H", type=int, nargs="*", default=[16, 128])
parser.add_argument("-n", "--is_neox", type=dtypes.str2bool, nargs="*",
                    default=[True, False])
args = parser.parse_args()

rows = []
for T in args.T:
    for H in args.H:
        for is_neox in args.is_neox:
            rows.append(test_fused_qk_rope_concat_cache_mla_seg(T, H, is_neox))
aiter.logger.info(
    "fused_qk_rope_concat_and_cache_mla_seg:\n%s",
    pd.DataFrame(rows).to_markdown(index=False),
)
