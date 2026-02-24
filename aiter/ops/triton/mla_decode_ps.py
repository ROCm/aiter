# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Memory-efficient attention for mtp.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

from typing import Optional
import functools
import triton
import triton.language as tl
import torch
from aiter.ops.triton.activation import _tanh
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.pid_preprocessing import remap_xcd


# Need to 
@triton.jit
def _fwd_grouped_kernel_stage1_ps(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    Out,
    work_indptr,
    work_info_set,
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
    stride_ob,
    stride_oh,
    stride_os,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    batch: tl.constexpr,
    logit_cap: tl.constexpr,
    max_qo_len: tl.constexpr,
    nhead: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    cu_id = tl.program_id(0)

    work_start = tl.load(work_indptr + cu_id)
    work_end = tl.load(work_indptr + cu_id + 1)

    if work_end <= work_start:
        # tl.device_print("Program ID: ", cu_id)
        # tl.device_print("work_start ID: ", work_start)
        # tl.device_print("work_end ID: ", work_end)
        return

    for work_id in range(work_start, work_end):
        cur_batch = tl.load(work_info_set + work_id * 8)
        split_id = tl.load(work_info_set + work_id * 8 + 1)

        # q_start = tl.load(work_info_set + work_id * 8 + 2) # batch_start
        # q_end = tl.load(work_info_set + work_id * 8 + 3) # batch_end

        split_kv_start = tl.load(work_info_set + work_id * 8 + 4)
        split_kv_end = tl.load(work_info_set + work_id * 8 + 5)

        token_to_batch_end = tl.load(work_info_set + work_id * 8 + 6)

        num_q_head_blk = tl.cdiv(q_head_num, BLOCK_H)

        if BLOCK_H < kv_group_num:
            VALID_BLOCK_H: tl.constexpr = BLOCK_H
        else:
            VALID_BLOCK_H: tl.constexpr = kv_group_num

        cur_head = tl.arange(0, BLOCK_H)
        mask_h = cur_head < VALID_BLOCK_H
        mask_h = mask_h & (cur_head < q_head_num)

        offs_c = tl.arange(0, BLOCK_C)
        offs_qk_r = tl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

        off_q_pe = (
            cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]
        )
        offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

        mask_c = offs_c < kv_lora_rank
        mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)

        # cur_batch_kv = cur_batch # // (max_qo_len // 2)
        # cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch_kv)
        # cur_batch_seq_len = tl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

        q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
        q_pe = tl.load(
            Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0
        )

        e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
        e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_C], dtype=tl.float32)

        indices = tl.arange(0, BLOCK_H)
        if token_to_batch_end == 0:

            quarter = BLOCK_H // max_qo_len
            seq_m_extand = max_qo_len - 1 - (indices % BLOCK_H) // quarter
            # seq_m_extand = tl.where(indices < nhead, 1, 0)
        else:
            seq_m_extand = tl.where(indices, 0, 0)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                # kv_indices + cur_batch_kv_start_idx + offs_n,
                kv_indices + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )

            offs_buf_kv = kv_loc[None, :] * stride_buf_kbs + offs_c[:, None]
            offs_buf_k_pe = kv_loc[None, :] * stride_buf_kbs + offs_qk_r[:, None]

            k_pe = tl.load(
                K_Buffer + offs_buf_k_pe,
                mask=(offs_n[None, :] < split_kv_end) & (mask_qk_r[:, None]),
                other=0.0,
            )  # positional embedding part of keys

            # (16, 64) x (64, 32)
            # dot product of rope parts
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))

            kv = tl.load(
                K_Buffer + offs_buf_kv,
                mask=(offs_n[None, :] < split_kv_end) & (mask_c[:, None]),
                other=0.0,
            )  # the shared latent tensor for keys and values

            # (16, 512) x (512, 32)
            # dot product of nope parts
            qk += tl.dot(q, kv)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            qk = tl.where(
                # mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
                ((seq_m_extand[:, None] + offs_n[None, :]) < split_kv_end), qk, float("-inf")
            )

            # offs_buf_v = kv_loc[:, None] * stride_buf_vbs + offs_c[None, :]
            # v = tl.load(
            #     V_buffer + offs_buf_v,
            #     mask=(offs_n[:, None] < split_kv_end) & (mask_c[None, :]),
            #     other=0.0,
            # )
            # v = tl.transpose(kv)

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            # (16, 32) x (32, 512)
            acc += tl.dot(p.to(kv.dtype), kv.T)
            # acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        if split_id == -1:
            # tl.device_print("Program ID: ", stride_ob)
            offs_o = (
                cur_batch * stride_ob
                + cur_head[:, None] * stride_oh
                + offs_c[None, :]
            )
            # tl.device_print("Program ID: ", cur_batch * stride_ob)

            tl.store(
                Out + offs_o,
                acc / e_sum[:, None],
                mask=(mask_h[:, None]) & (mask_c[None, :]),
            )
        else:
            offs_mid_o = (
                split_id * stride_mid_ob // max_qo_len
                + cur_head[:, None] * stride_mid_oh
                + offs_c[None, :]
            )

            tl.store(
                Att_Out + offs_mid_o,
                acc / e_sum[:, None],
                mask=(mask_h[:, None]) & (mask_c[None, :]),
            )

            offs_mid_lse = (
                split_id * stride_mid_lse_b // max_qo_len
                + cur_head * stride_mid_lse_h
            )

            tl.store(
                Att_Lse + offs_mid_lse,
                e_max + tl.log(e_sum),
                mask=mask_h,
            )


def decode_grouped_att_m_fwd_ps(
    q,               # [B, Sq, hq, 576]
    k_buffer,        # [Pages, hk, 576]
    v_buffer,
    att_out,
    att_lse,
    out,
    work_indptr,
    work_info_set,
    kv_lora_rank,  # c
    kv_indptr,
    kv_indices,
    sm_scale,
    logit_cap,
    mtp,
    config,
):
    qk_rope_head_dim = k_buffer.shape[-1] - kv_lora_rank
    # batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    # batch = batch*(mtp + 1)

    config["BLOCK_C"] = triton.next_power_of_2(kv_lora_rank)
    config["BLOCK_R"] = triton.next_power_of_2(qk_rope_head_dim)
    # print(batch, head_num, kv_group_num)

    config["BLOCK_H"] = ((kv_group_num + 15) // 16) * 16

    grid = (80,)
    # print(config["BLOCK_H"])

    _fwd_grouped_kernel_stage1_ps[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        out,
        work_indptr,
        work_info_set,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        v_buffer.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        att_out.stride(-3),
        att_out.stride(-2),
        att_out.stride(-1),
        att_lse.stride(-3),
        att_lse.stride(-2),
        att_lse.stride(-1),
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        batch=batch,
        logit_cap=logit_cap,
        max_qo_len=mtp + 1,
        nhead=head_num // (mtp + 1),
        **config,
    )
