# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import torch
import aiter
from aiter import dtypes
import triton
import triton.language as tl
import functools
from .jit.utils.chip_info import get_cu_num


@triton.jit
def _fwd_kernel_stage2_asm(
    Mid_O,
    Mid_lse,
    O,
    qo_indptr,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_offs = tl.program_id(2)

    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_qo = cur_qo_start + cur_qo_offs
    if cur_qo > cur_qo_end:
        return
    cur_kv_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = (cur_qo * stride_mid_ob + cur_head * stride_mid_oh) * Lv + offs_d
    offs_logic = cur_qo * stride_mid_ob + cur_head * stride_mid_oh

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.maximum(mgc, tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS))
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os * Lv,
                mask=mask_d,
                other=0.0,
            )
            tlogic = tl.load(Mid_lse + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


@functools.lru_cache()
def get_meta_param(num_kv_splits, device, bs, nhead):
    if num_kv_splits is None:
        cu_num = get_cu_num()
        num_kv_splits = min(16, max(1, cu_num // bs))

    get_mgc = {16: 64, 128: 16}
    assert nhead in get_mgc, f"{nhead=} not supported"
    mgc = get_mgc[nhead]
    return num_kv_splits, mgc


def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1

    num_kv_splits, mgc = get_meta_param(num_kv_splits, device, bs, nhead)

    if nhead == 16:
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        assert (
            max_seqlen_q == 1
        ), f"Assertion: max_seqlen_q should be 1 when n_head=16, but got {max_seqlen_q}"
    elif nhead == 128:
        logits = (
            o.view((total_s, num_kv_splits, nhead, v_head_dim))
            if num_kv_splits == 1
            else torch.empty(
                (total_s, num_kv_splits, nhead, v_head_dim),
                dtype=dtypes.fp32,
                device=device,
            )
        )
    else:
        assert False, f"{nhead=} not supported"

    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
    )

    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        sm_scale,
        logits,
        attn_lse,
    )

    if num_kv_splits == 1 and nhead == 128:
        return logits.view(total_s, nhead, v_head_dim), attn_lse
    Lv = v_head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    grid = (bs, nhead, max_seqlen_q)
    extra_kargs = {"waves_per_eu": 4}
    _fwd_kernel_stage2_asm[grid](
        logits,
        attn_lse,
        o,
        qo_indptr,
        kv_indptr,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        mgc=mgc,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )
    return logits, attn_lse


def mla_prefill_fwd(
    q,  # [num_seqs, num_heads, head_size]
    kv_buffer,  # [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    o,  # [num_seqs, num_heads, v_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    bs, nhead, v_head_dim = o.shape

    num_kv_splits = 1

    logits = o.view(bs, num_kv_splits, nhead, v_head_dim)
    # logits = torch.empty(
    #     (bs, num_kv_splits, nhead, v_head_dim), dtype=dtypes.fp32, device=device
    # )
    attn_lse = torch.empty(
        (bs, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
    )

    aiter.mla_prefill_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q,
        sm_scale,
        logits,
        attn_lse,
    )

    # return logits.view(bs, nhead, v_head_dim).to(o.dtype), attn_lse
    return o.view(bs, nhead, v_head_dim), attn_lse
