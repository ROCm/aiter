# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# user interface

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num


@triton.jit
def _fwd_kernel_stage2_asm(
    Mid_O,
    Mid_lse,
    O,
    qo_indptr,
    kv_indptr,
    num_kv_splits_indptr,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
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


@functools.lru_cache(maxsize=1)
def get_meta_param(num_kv_splits, bs, total_kv, nhead, max_seqlen_q, dtype):
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

    get_mgc = {16: 16, 128: 16}

    get_block_n_fp8 = {
        16: 128,
        32: 128,
        48: 64,
        64: 64,
        128: 32,
        256: 32,
        384: 32,
        512: 32,
    }

    if dtype == dtypes.fp8:
        min_block_n = get_block_n_fp8[int(nhead * max_seqlen_q)]
        num_kv_splits = min(
            num_kv_splits, int(total_kv / bs + min_block_n - 1) // min_block_n
        )

    assert nhead in get_mgc, f"{nhead=} not supported"
    mgc = get_mgc[nhead]
    if max_seqlen_q == 1 and nhead == 16:
        mgc = 64

    num_kv_splits_indptr = torch.arange(
        0, (bs + 1) * num_kv_splits, num_kv_splits, dtype=torch.int, device="cuda"
    )

    return num_kv_splits, mgc, num_kv_splits_indptr


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
    num_kv_splits_indptr=None,  # for experts only!!!
    work_meta_data=None,
    work_indptr=None,
    work_info_set=None,
    reduce_indptr=None,
    reduce_final_map=None,
    reduce_partial_map=None,
    q_scale=None,
    kv_scale=None,
):
    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    num_page, page_size, nhead_kv, qk_head_dim = kv_buffer.shape
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    ori_total_s, ori_nhead, ori_v_head_dim = o.shape
    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1
    total_kv = kv_indices.shape[0]

    persistent_mode = work_meta_data is not None

    io_transformed = False

    if not persistent_mode:
        num_kv_splits, mgc, num_kv_splits_indptr = get_meta_param(
            num_kv_splits, bs, total_kv, nhead, max_seqlen_q, q.dtype
        )

        MAYBE_FINAL_OUT = True

        if nhead == 16 and max_seqlen_q == 1:
            MAYBE_FINAL_OUT = False

        logits = (
            o.view((total_s, num_kv_splits, nhead, v_head_dim))
            if (
                num_kv_splits == 1
                and (
                    q.dtype == dtypes.fp8
                    or (q.dtype == dtypes.bf16 and max_seqlen_q == 4)
                )
            )
            else torch.empty(
                (total_s, num_kv_splits, nhead, v_head_dim),
                dtype=dtypes.fp32,
                device=device,
            )
        )

        attn_lse = torch.empty(
            (total_s, num_kv_splits, nhead, 1), dtype=dtypes.fp32, device=device
        )
        final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)

        aiter.mla_decode_stage1_asm_fwd(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_kv_splits_indptr,
            None,
            None,
            None,
            max_seqlen_q,
            sm_scale,
            logits,
            attn_lse,
            o,
            q_scale,
            kv_scale,
        )

        if num_kv_splits == 1 and (
            q.dtype == dtypes.fp8 or (q.dtype == dtypes.bf16 and max_seqlen_q == 4)
        ):
            return logits.view(total_s, nhead, v_head_dim), attn_lse

        Lv = v_head_dim
        BLOCK_DV = triton.next_power_of_2(Lv)
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
    else:
        if num_kv_splits is None:
            num_kv_splits = get_cu_num()
        if nhead == 16 or (nhead == 128 and kv_buffer.dtype == dtypes.fp8):
            # Natively support cases
            pass
        elif nhead in range(32, 128 + 1, 16) and persistent_mode and max_seqlen_q == 1:
            # we use nhead=16 to simulate such cases by customized metadata
            # metadata also views qo's tensor as shape (total_s * (nhead // 16), 16, ...)
            total_s = ori_total_s * (ori_nhead // 16)
            nhead = 16
            q = q.view(total_s, nhead, -1)
            o = o.view(total_s, nhead, -1)
            io_transformed = True
        else:
            assert False, f"{nhead=} and {max_seqlen_q=} not supported"

        logits = torch.empty(
            (reduce_partial_map.size(0) * max_seqlen_q, 1, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * max_seqlen_q, 1, nhead, 1),
            dtype=dtypes.fp32,
            device=device,
        )
        final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)

        aiter.mla_decode_stage1_asm_fwd(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            num_kv_splits_indptr,
            work_meta_data,
            work_indptr,
            work_info_set,
            max_seqlen_q,
            sm_scale,
            logits,
            attn_lse,
            o,
            q_scale,
            kv_scale,
        )

        aiter.mla_reduce_v1(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            o,
            final_lse,
        )

    if io_transformed:
        if persistent_mode:
            logits = logits.view(-1, 1, ori_nhead, v_head_dim)
        else:
            logits = logits.view(ori_total_s, num_kv_splits, ori_nhead, v_head_dim)
        q = q.view(ori_total_s, ori_nhead, -1)
        o = o.view(ori_total_s, ori_nhead, -1)

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


def mla_ps_prefill_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: Optional[torch.Tensor],
    work_info_set: Optional[torch.Tensor],
    max_seqlen_q: int,
    is_causal: bool,
    reduce_indptr: Optional[torch.Tensor] = None,
    reduce_final_map: Optional[torch.Tensor] = None,
    reduce_partial_map: Optional[torch.Tensor] = None,
    softmax_scale: float = None,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> None:
    device = Q.device
    bs, nhead, v_head_dim = output.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / (v_head_dim**0.5)

    max_partial_loc = reduce_partial_map[:, 0].max().item()

    q_lens = reduce_final_map[:, 1] - reduce_final_map[:, 0]
    typical_tile_size = q_lens.max().item()

    padded_num_tokens = max_partial_loc + typical_tile_size

    available_tgs = work_indptr.size(0) - 1

    logits = torch.empty(
        (padded_num_tokens * available_tgs, nhead, v_head_dim),
        dtype=dtypes.fp32, device=device
    )
    attn_lse = torch.empty(
        (padded_num_tokens * available_tgs, nhead),
        dtype=dtypes.fp32, device=device
    )

    aiter.mla_ps_prefill_asm_fwd(
        Q,
        K,
        V,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        work_indptr,
        work_info_set,
        max_seqlen_q,
        softmax_scale,
        is_causal,
        logits,
        attn_lse,
        output,
        q_scale,
        k_scale,
        v_scale,
    )

    mla_prefill_reduce(
        logits,
        attn_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        output,
    )
    return output.view(bs, nhead, v_head_dim), attn_lse



def mla_prefill_reduce(
    partial_output: torch.Tensor,      # [padded_num_tokens * available_tgs, num_head_q, v_head_dim]
    partial_lse: torch.Tensor,         # [padded_num_tokens * available_tgs, num_head_q]
    reduce_indptr: torch.Tensor,       # [num_reduce_groups + 1], int32
    reduce_final_map: torch.Tensor,    # [num_reduce_groups, 4], int32
                                       # [qo_start, qo_end, q_head_start, q_head_end]
    reduce_partial_map: torch.Tensor,  # [num_partial_tiles, 3], int32
                                       # [partial_qo_loc, q_head_start, q_head_end]
    output: torch.Tensor,              # [total_tokens, num_head_q, v_head_dim], output buffer
) -> None:


    num_reduce_groups = reduce_indptr.shape[0] - 1
    device = partial_output.device
    dtype = partial_output.dtype

    for group_id in range(num_reduce_groups):
        start_idx = reduce_indptr[group_id].item()  # 0
        end_idx = reduce_indptr[group_id + 1].item() # 2
        num_partials = end_idx - start_idx

        if num_partials == 0:
            continue

        final_map = reduce_final_map[group_id]
        qo_start = final_map[0].item()
        qo_end = final_map[1].item()
        q_head_start = final_map[2].item()
        q_head_end = final_map[3].item()

        q_len = qo_end - qo_start
        num_heads = q_head_end - q_head_start
        v_head_dim = partial_output.shape[2]

        partial_indices = []
        partial_heads = []

        for partial_idx in range(start_idx, end_idx):
            partial_map = reduce_partial_map[partial_idx]
            partial_qo_loc = partial_map[0].item()
            partial_q_head_start = partial_map[1].item()
            partial_q_head_end = partial_map[2].item()

            partial_indices.append(partial_qo_loc)  # [4, 10]
            partial_heads.append((partial_q_head_start, partial_q_head_end))  # [(0, 1), (0, 1)]

        for head_offset in range(num_heads):
            head_idx = q_head_start + head_offset

            partial_lses = []
            partial_outputs = []

            for i, (partial_qo_loc, (ph_start, ph_end)) in enumerate(
                zip(partial_indices, partial_heads)  # [(4, 10), (0, 1), (0, 1)]
            ):
                if ph_start <= head_idx < ph_end:
                    # Extract LSE: [q_len]
                    lse = partial_lse[
                        partial_qo_loc : partial_qo_loc + q_len, head_idx
                    ]
                    partial_lses.append(lse)

                    # Extract output: [q_len, v_head_dim]
                    out = partial_output[
                        partial_qo_loc : partial_qo_loc + q_len, head_idx, :
                    ]
                    partial_outputs.append(out)

            if len(partial_lses) == 0:
                continue

            # Stack: [num_partials, q_len] and [num_partials, q_len, v_head_dim]
            partial_lses = torch.stack(partial_lses, dim=0)  # [K, q_len]
            partial_outputs = torch.stack(partial_outputs, dim=0)  # [K, q_len, D]

            max_lse = torch.max(partial_lses, dim=0)[0]  # [q_len]

            sum_exp = torch.sum(torch.exp(partial_lses - max_lse.unsqueeze(0)), dim=0)

            #  final_lse
            final_lse = max_lse + torch.log(sum_exp)

            scales = torch.exp(partial_lses - final_lse.unsqueeze(0)).unsqueeze(-1)

            final_output = torch.sum(partial_outputs * scales, dim=0)

            output[qo_start:qo_end, head_idx, :] = final_output
