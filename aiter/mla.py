# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import torch
import aiter
from aiter import dtypes
import triton
import triton.language as tl
import functools


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


@functools.lru_cache()
def get_meta_param(num_kv_splits, kv_indptr, nhead, nhead_kv, max_seqlen_q):
    if num_kv_splits is None:
        (kv_splits_indptr, max_splits) = aiter.get_mla_metadata_v0(
            kv_indptr, nhead // nhead_kv, nhead_kv
        )
        num_kv_splits = max_splits.item()

    get_mgc = {16: 16, 128: 16}

    assert nhead in get_mgc, f"{nhead=} not supported"
    mgc = get_mgc[nhead]
    if max_seqlen_q == 1 and nhead == 16:
        mgc = 64
    return num_kv_splits, kv_splits_indptr, mgc


def mla_decode_fwd(
    q,
    kv_buffer,
    o,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    max_seqlen_q,
    q_scale=1.0,    # default 1.0 with fp32 dirctly cast to fp8
    kv_scale=1.0,   # default 1.0 with fp32 dirctly cast to fp8
    sm_scale=None,  # 1.0 / (qk_head_dim**0.5)
    logit_cap=0.0,
    num_kv_splits=None,  # for experts only!!!
    num_kv_splits_indptr=None,  # for experts only!!!
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1
    total_kv = kv_indices.shape[0]

    if num_kv_splits is None:
        num_kv_splits, num_kv_splits_indptr, mgc = get_meta_param(
            num_kv_splits, kv_indptr, nhead, nhead_kv, max_seqlen_q
        )
    else:
        assert (
            num_kv_splits_indptr is not None
        ), "num_kv_splits_indptr must be provided when num_kv_splits is specified"

    if nhead == 16 and max_seqlen_q == 1:
        # special case for 16 heads and max_seqlen_q == 1
        logits = torch.zeros(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        MAYBE_FINAL_OUT = False
    elif nhead in [16, 128]:
        MAYBE_FINAL_OUT = True
        num_kv_splits = 80
        logits = torch.zeros(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
    else:
        assert False, f"{nhead=} not supported"

    attn_lse = torch.zeros(
        (total_s, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
    )
    final_lse = torch.zeros((total_s, nhead), dtype=dtypes.fp32, device=device)
    # import pdb;pdb.set_trace()

    def ref_masked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        dtype,
        is_causal=True,
    ) -> torch.Tensor:
        attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
        if is_causal:
            s_q = query.shape[0]
            s_k = key.shape[0]
            attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
            temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)
            attn_weights += attn_bias
        lse = attn_weights.logsumexp(dim=-1)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
        return out.to(dtype), lse

    def torch_mla_extend(
        q,  # [total_q, nheads, headdim_q]
        kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype,
        is_causal=True,
    ):
        qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
        kvc = torch.index_select(kvc_cache, 0, kv_indices)



        kvs = torch.tensor_split(kvc, kv_indptr.tolist()[:])
        bs = qo_indptr.shape[0] - 1

        os = []
        lses = []
        for i in range(bs):
            kvc = kvs[1]
            q = qs[i]
            k = kvc
            v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
            o, lse = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=is_causal)
            os.append(o)
            lses.append(lse)
        o = torch.concat(os)
        lse = torch.concat(lses)
        return o, lse.transpose(0, 1)

    # kv_indptr[0] = 192
    # kv_indptr[1] = 384

    # out_ref, lse_ref = torch_mla_extend(
    #     q,
    #     kv_buffer.reshape(-1, 1, 576),
    #     qo_indptr,
    #     kv_indptr,
    #     kv_indices,
    #     sm_scale,
    #     512,
    #     64,
    #     is_causal=True,
    #     dtype=torch.bfloat16,
    # )

    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        # num_kv_splits_indptr,
        # None,
        # None,
        None,
        work_indptr,
        work_info_set,
        max_seqlen_q,
        sm_scale,
        logits,
        attn_lse,
        o,
    )
    # import pdb; pdb.set_trace()

    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        o,
        final_lse,
    )

    # if num_kv_splits == 1 and not (max_seqlen_q == 1 and nhead == 16):
    #     return logits.view(total_s, nhead, v_head_dim), attn_lse
    # Lv = v_head_dim
    # BLOCK_DV = triton.next_power_of_2(Lv)
    # grid = (bs, nhead)
    # extra_kargs = {"waves_per_eu": 4}
    # _fwd_kernel_stage2_asm[grid](
    #     logits,
    #     attn_lse,
    #     o,
    #     qo_indptr,
    #     kv_indptr,
    #     num_kv_splits_indptr,
    #     attn_lse.stride(0),
    #     attn_lse.stride(2),
    #     attn_lse.stride(1),
    #     o.stride(0),
    #     o.stride(1),
    #     MAYBE_FINAL_OUT=MAYBE_FINAL_OUT,
    #     BATCH_NUM=bs,
    #     BLOCK_DV=BLOCK_DV,
    #     Lv=Lv,
    #     mgc=mgc,
    #     num_warps=4,
    #     num_stages=2,
    #     **extra_kargs,
    # )
    return logits, final_lse


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
