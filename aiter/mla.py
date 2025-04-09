# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import torch
import aiter
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_stage2_asm(
    Mid_O,
    Mid_lse,
    O,
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

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) * Lv + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.maximum(mgc, tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS))
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

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
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
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

    if num_kv_splits is None:
        device_properties = torch.cuda.get_device_properties(device)
        cu_num = device_properties.multi_processor_count
        num_kv_splits = min(16, max(1, cu_num // bs))

    logits = torch.empty(
        (bs, num_kv_splits, nhead, v_head_dim), dtype=torch.float, device=device
    )
    attn_lse = torch.empty(
        (bs, num_kv_splits, nhead, 1), dtype=torch.float, device=device
    )

    aiter.mla_stage1_asm_fwd(
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        logits,
        attn_lse,
    )

    Lv = v_head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    grid = (bs, nhead)
    extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}
    _fwd_kernel_stage2_asm[grid](
        logits,
        attn_lse,
        o,
        kv_indptr,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        mgc=64,
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
    #     (bs, num_kv_splits, nhead, v_head_dim), dtype=torch.float, device=device
    # )
    attn_lse = torch.empty(
        (bs, num_kv_splits, nhead, 1), dtype=torch.float, device=device
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
