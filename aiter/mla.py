# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import torch
import aiter
from aiter import dtypes
import triton
import triton.language as tl
import functools
from aiter.jit.utils.chip_info import get_cu_num


@triton.jit
def _fwd_kernel_stage2_asm(
    Mid_O,
    Mid_lse,
    O,
    qo_indptr,
    kv_indptr,
    num_kv_splits_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAYBE_FINAL_OUT: tl.constexpr,
    BATCH_NUM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_split_start = tl.load(num_kv_splits_indptr + cur_batch)
    cur_split_end = tl.load(num_kv_splits_indptr + cur_batch + 1)
    num_max_kv_splits = tl.load(num_kv_splits_indptr + BATCH_NUM)
    cur_kv_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    offs_logic = cur_qo_start * stride_mid_ob + cur_head * stride_mid_oh
    offs_v = offs_logic * Lv + offs_d
    num_valid_kv_splits = tl.minimum(
        cur_split_end - cur_split_start, tl.cdiv(cur_kv_seq_len, mgc)
    )
    FINAL_OUT = MAYBE_FINAL_OUT and num_max_kv_splits == BATCH_NUM

    for cur_qo in range(cur_qo_start, cur_qo_end):
        if FINAL_OUT:
            input_ptr = Mid_O.to(tl.pointer_type(O.type.element_ty))
            out = tl.load(
                # input_ptr + offs_v + stride_mid_ob * Lv,
                input_ptr
                + Lv * (cur_qo * stride_mid_os + cur_head * stride_mid_oh)
                + offs_d,
                mask=mask_d,
                other=0.0,
            )
            tl.store(
                O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
                out,
                mask=mask_d,
            )
        else:
            e_sum = 0.0
            e_max = -float("inf")
            acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
            for split_kv_id in range(0, num_valid_kv_splits):
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
            offs_logic += stride_mid_ob
            offs_v += stride_mid_ob * Lv
            tl.store(
                O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
                acc / e_sum,
                mask=mask_d,
            )

@triton.jit
def _fwd_kernel_stage2_asm_skip_1(
    Mid_O,
    Mid_lse,
    O,
    qo_indptr,
    kv_indptr,
    num_kv_splits_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    MAYBE_FINAL_OUT: tl.constexpr,
    BATCH_NUM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_split_start = tl.load(num_kv_splits_indptr + cur_batch)
    cur_split_end = tl.load(num_kv_splits_indptr + cur_batch + 1)
    num_max_kv_splits = tl.load(num_kv_splits_indptr + BATCH_NUM)
    cur_kv_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    offs_logic = cur_qo_start * stride_mid_ob + cur_head * stride_mid_oh
    offs_v = offs_logic * Lv + offs_d
    num_valid_kv_splits = tl.minimum(
        cur_split_end - cur_split_start, tl.cdiv(cur_kv_seq_len, mgc)
    )
    FINAL_OUT = MAYBE_FINAL_OUT and num_valid_kv_splits == 1
    if FINAL_OUT:
        return

    for cur_qo in range(cur_qo_start, cur_qo_end):
        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
        for split_kv_id in range(0, num_valid_kv_splits):
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
        offs_logic += stride_mid_ob
        offs_v += stride_mid_ob * Lv
        tl.store(
            O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
            acc / e_sum,
            mask=mask_d,
        )

@functools.lru_cache(maxsize=1)
def get_kv_splits_indptr(num_kv_splits, bs, device):
    """
    Returns the kv_splits_indptr tensor for the given number of kv splits.
    """
    num_kv_splits = min(16, max(1, num_kv_splits))
    kv_splits_indptr = torch.arange(
        0, (bs + 1) * num_kv_splits, num_kv_splits, device=device, dtype=torch.int32
    )
    return kv_splits_indptr


@functools.lru_cache()
def get_meta_param_balanced(bs, kv_indptr, device):
    kv_seq_les = torch.tensor([kv_indptr[i + 1] - kv_indptr[i] for i in range(bs)], device = device)
    total_kv_pad = 0

    cu_num = int(get_cu_num()) * int(bs / 16 + 0.5)

    for i in range(bs):
        total_kv_pad += (kv_seq_les[i] + 16 - 1) // 16 * 16 

    split_size_pad = (total_kv_pad + cu_num - 1) // cu_num + 80 

    num_kv_splits_indptr = torch.empty_like(kv_indptr)
    num_kv_splits_indptr[0] = 0
    for i in range(bs):
        num_kv_splits = (kv_seq_les[i] + split_size_pad - 1) // split_size_pad 
        num_kv_splits_indptr[i + 1] = num_kv_splits_indptr[i] + num_kv_splits

    batch_split_table = torch.empty(
        (cu_num), dtype=torch.int32, device="cuda"
    )
    split_table = torch.empty(
        (cu_num), dtype=torch.int32, device="cuda"
    )

    b_idx = 0
    split_idx = 0
    cur_idx = 0
    num_kv_splits_indptr_fixed = torch.empty_like(num_kv_splits_indptr)

    fix_size = num_kv_splits_indptr[-1] - cu_num

    fixed_size = 0
    sign = 1 if fix_size > 0 else -1
    num_kv_splits_indptr_fixed[0] = 0
    if num_kv_splits_indptr[-1] != cu_num:
        if fix_size > 0:
            for i in range(1, bs + 1):
                if fixed_size != fix_size and kv_seq_les[i-1] > split_size_pad and kv_seq_les[i-1] % split_size_pad <= split_size_pad / 2:
                    fixed_size += sign
                num_kv_splits_indptr_fixed[i] = num_kv_splits_indptr[i] - fixed_size
                # print(i, num_kv_splits_indptr[i])
        else:
            for i in range(1, bs + 1):
                if fixed_size != fix_size and kv_seq_les[i-1] > 3 * split_size_pad and kv_seq_les[i-1] % split_size_pad > split_size_pad / 2:
                    fixed_size += sign
                num_kv_splits_indptr_fixed[i] = num_kv_splits_indptr[i] - fixed_size
                # print(i, num_kv_splits_indptr[i])
    else:
        num_kv_splits_indptr_fixed = num_kv_splits_indptr

    fixed_gap = fix_size - fixed_size

    end_dim = bs 
    while fixed_gap != 0:
        num_kv_splits_indptr_fixed[end_dim] -= fixed_gap
        if kv_seq_les[end_dim - 1] > 1:
            fixed_gap -= sign 
        end_dim -= 1

    for i in range(cu_num):
        if i < num_kv_splits_indptr_fixed[b_idx + 1]:
            batch_split_table[i] = b_idx
            split_table[i] = split_idx
        else:
            split_idx = 0 
            b_idx = b_idx + 1
            batch_split_table[i] = b_idx
            split_table[i] = split_idx
        split_idx += 1

    return num_kv_splits_indptr_fixed, batch_split_table, split_table, cu_num


@functools.lru_cache()
def get_meta_param(num_kv_splits, bs, total_kv, nhead, max_seqlen_q, device):
    if num_kv_splits is None:
        cu_num = get_cu_num()
        avg_kv = total_kv / bs
        overhead = 84.1
        tmp = [
            (
                bs
                * i
                / ((bs * i + cu_num - 1) // cu_num * cu_num)
                * avg_kv
                / (avg_kv + overhead * i),
                i,
            )
            for i in range(1, 17)
        ]
        num_kv_splits = sorted(tmp, key=lambda x: x[0], reverse=True)[0][1]
        # num_kv_splits = min(16, max(1, cu_num // bs))

    get_mgc = {16: 16, 128: 16}

    assert nhead in get_mgc, f"{nhead=} not supported"
    mgc = get_mgc[nhead]
    if max_seqlen_q == 1 and nhead == 16:
        mgc = 64
    return num_kv_splits, get_kv_splits_indptr(num_kv_splits, bs, device), mgc


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
    num_kv_splits=None,
    num_kv_splits_indptr=None,
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1
    total_kv = kv_indices.shape[0]

    if num_kv_splits_indptr is None:
        num_kv_splits, num_kv_splits_indptr, mgc = get_meta_param(
            None, bs, total_kv, nhead, max_seqlen_q, device
        )

    if nhead == 16 and max_seqlen_q == 1:
        # special case for 16 heads and max_seqlen_q == 1
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        MAYBE_FINAL_OUT = False
    elif nhead in [16, 128]:
        MAYBE_FINAL_OUT = True
        num_kv_splits = 16
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
    else:
        assert False, f"{nhead=} not supported"

    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
    )

    Lv = v_head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)

    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_kv_splits_indptr,
        max_seqlen_q,
        sm_scale,
        logits,
        attn_lse,
    )

    if num_kv_splits == 1 and not (max_seqlen_q == 1 and nhead == 16):
        return logits.view(total_s, nhead, v_head_dim), attn_lse

    grid = (bs, nhead)
    extra_kargs = {"waves_per_eu": 4}
    _fwd_kernel_stage2_asm[grid](
        logits,
        attn_lse,
        o,
        qo_indptr,
        kv_indptr,
        num_kv_splits_indptr,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        o.stride(0),
        o.stride(1),
        MAYBE_FINAL_OUT=MAYBE_FINAL_OUT,
        BATCH_NUM=bs,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        mgc=mgc,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )

    return logits, attn_lse


def mla_decode_fwd_balenced(
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
    num_kv_splits=16,
    num_kv_splits_indptr=None,
    batch_split_table=None,  # for experts only!!!
    split_table=None,
    q_rope=None,
    k_rope=None, 
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1
    total_kv = kv_indices.shape[0]

    if nhead == 16 and max_seqlen_q == 1:
        # special case for 16 heads and max_seqlen_q == 1
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        MAYBE_FINAL_OUT = False
    elif nhead in [16, 128]:
        MAYBE_FINAL_OUT = True
        num_kv_splits = 16
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
    else:
        assert False, f"{nhead=} not supported"

    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
    )

    Lv = v_head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)

    # q_nope = torch.empty_like(q[:, :, :512])
    # q_rope = torch.empty_like(q[:, :, 512:])
    # k_nope = torch.empty_like(kv_buffer[:, :, :, :512])
    # k_rope = torch.empty_like(kv_buffer[:, :, :, 512:])
    #
    # q_nope[:, :, :] = q[:, :, :512]
    # q_rope[:, :, :] = q[:, :, 512:]
    # k_nope[:, :, :, :] = kv_buffer[:, :, :, :512]
    # k_rope[:, :, :, :] = kv_buffer[:, :, :, 512:]

    if num_kv_splits_indptr is None:
        aiter.get_mla_metadata_impl(
            kv_indptr,
            num_kv_splits_indptr,
            batch_split_table,
            split_table,
        )

    aiter.flash_mla_fwd_inline_impl(
        q,
        kv_buffer,
        # q_nope,
        # k_nope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        num_kv_splits_indptr,
        max_seqlen_q,
        sm_scale,
        logits,
        attn_lse,
        q_rope,
        k_rope,
        batch_split_table,
        split_table,
        o,
        num_kv_splits_indptr[bs].item(),
    )

    grid = (bs, nhead)
    extra_kargs = {"waves_per_eu": 4}
    _fwd_kernel_stage2_asm_skip_1[grid](
        logits,
        attn_lse,
        o,
        qo_indptr,
        kv_indptr,
        num_kv_splits_indptr,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        o.stride(0),
        o.stride(1),
        MAYBE_FINAL_OUT=MAYBE_FINAL_OUT,
        BATCH_NUM=bs,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        mgc=16,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )

    return logits, attn_lse


def mla_decode_fwd_dispatch(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    varlen=False,
    logit_cap=0.0,
    num_kv_splits=None,
    num_kv_splits_indptr=None,
    batch_split_table=None,
    split_table=None,
    q_rope=None,
    k_rope=None, 
):
    if batch_split_table is None:
        return mla_decode_fwd(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_q,
            sm_scale,
            logit_cap,
            num_kv_splits,
        )
    else:
        return mla_decode_fwd_balenced(
            q,
            kv_buffer,
            o,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_q,
            sm_scale,
            logit_cap,
            num_kv_splits,
            num_kv_splits_indptr,
            batch_split_table,
            split_table,
            q_rope,
            k_rope, 
        )


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
