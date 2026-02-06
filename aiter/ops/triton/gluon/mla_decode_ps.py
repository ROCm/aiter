# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Memory-efficient attention for decoding.
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

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
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
    kv_lora_rank: gl.constexpr,
    qk_rope_head_dim: gl.constexpr,
    kv_group_num: gl.constexpr,
    q_head_num: gl.constexpr,
    batch: gl.constexpr,
    logit_cap: gl.constexpr,
    max_qo_len: gl.constexpr,
    nhead: gl.constexpr,
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    cache_modifier: gl.constexpr,
):
    cu_id = gl.program_id(0)

    work_start = gl.load(work_indptr + cu_id)
    work_end = gl.load(work_indptr + cu_id + 1)

    if work_end <= work_start:
        return

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        # size_per_thread=[1, 128],  # 64 * 576
        # threads_per_warp=[16, 4],
        # warps_per_cta=[4, 1],
        # order=[1, 0],
        size_per_thread=[4, 8],  # 64 * 512
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    blocked_q_rope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 64 in
        size_per_thread=[1, 4],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    blocked_ld_k_nope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        # size_per_thread=[8, 4],  # 64 * 576
        # threads_per_warp=[2, 32],
        # warps_per_cta=[1, 4],
        # order=[0, 1],
        size_per_thread=[4, 8],  # 64 * 512
        threads_per_warp=[4, 16],
        warps_per_cta=[1, 4],
        order=[1, 0],
        # size_per_thread=[32, 1],
        # threads_per_warp=[4, 16],
        # warps_per_cta=[4, 1],
        # order=[0, 1],
    )

    blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        # size_per_thread=[1, 4],
        # threads_per_warp=[16, 4],
        # warps_per_cta=[1, 4],
        # order=[0, 1],
        size_per_thread=[1, 4],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
        # size_per_thread=[8, 1],
        # threads_per_warp=[4, 16],
        # warps_per_cta=[4, 1],
        # order=[0, 1],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=4, max_phase=8, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=4, max_phase=8, order=[0, 1]
    )

    smem_k_nope = gl.allocate_shared_memory(
        # K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
        K_Buffer.type.element_ty, [kv_lora_rank, BLOCK_N], layout=shared_k
    )
    smem_k_rope = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k
    )

    smem_q_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
    )

    smem_v = gl.allocate_shared_memory(
        # K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
    )

    smem_p = gl.allocate_shared_memory(
        # K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_k
    )



    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[1, 4]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[1, 4]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=4
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=4
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=4
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=4
    )


    for work_id in range(work_start, work_end):
        cur_batch = gl.load(work_info_set + work_id * 8)
        split_id = gl.load(work_info_set + work_id * 8 + 1)

        # q_start = gl.load(work_info_set + work_id * 8 + 2) # batch_start
        # q_end = gl.load(work_info_set + work_id * 8 + 3) # batch_end

        split_kv_start = gl.load(work_info_set + work_id * 8 + 4)
        split_kv_end = gl.load(work_info_set + work_id * 8 + 5)

        token_to_batch_end = gl.load(work_info_set + work_id * 8 + 6)

        num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

        if BLOCK_H < kv_group_num:
            VALID_BLOCK_H: gl.constexpr = BLOCK_H
        else:
            VALID_BLOCK_H: gl.constexpr = kv_group_num

        cur_head = gl.arange(
            0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
        )
        mask_h = cur_head < VALID_BLOCK_H
        mask_h = mask_h & (cur_head < q_head_num)

        cur_head_pe = gl.arange(
            0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_rope_mk)
        )
        mask_h_pe = cur_head_pe < VALID_BLOCK_H
        mask_h_pe = mask_h_pe & (cur_head_pe < q_head_num)

        offs_c = gl.arange(
            0, BLOCK_C, layout=gl.SliceLayout(0, blocked_q_nope_mk)
        )

        offs_om = gl.arange(
            0, BLOCK_H, layout=gl.SliceLayout(1, mfma_layout_kv)
        )
        mask_om = offs_om < VALID_BLOCK_H
        mask_om = mask_om & (offs_om < q_head_num)
        offs_on = gl.arange(
            0, BLOCK_C, layout=gl.SliceLayout(0, mfma_layout_kv)
        )
        mask_on = offs_on < kv_lora_rank

        offs_qk_r = gl.arange(
            kv_lora_rank,
            kv_lora_rank + BLOCK_R,
            layout=gl.SliceLayout(0, blocked_q_rope_mk)
        )  # to get the k_pe

        offs_k_c = gl.arange(
            0, BLOCK_C, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
        )
        offs_k_r = gl.arange(
            kv_lora_rank,
            kv_lora_rank + BLOCK_R,
            layout=gl.SliceLayout(1, blocked_ld_k_rope_kn)
        )  # to get the k_pe


        off_q_pe = (
            cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
        )
        offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

        mask_c = offs_c < kv_lora_rank
        mask_k_c = offs_k_c < kv_lora_rank
        mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
        mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

        # cur_batch_kv = cur_batch # // (max_qo_len // 2)
        # cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
        # cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

        q = gl.amd.cdna3.buffer_load(
            ptr=Q,
            offsets=offs_q,
            mask=(mask_h[:, None]) & (mask_c[None, :]),
            cache=cache_modifier,
        )
        smem_q_nope.store(q)
        q = smem_q_nope.load(layout=dot_q_layout)
        q_pe = gl.amd.cdna3.buffer_load(
            ptr=Q,
            offsets=off_q_pe,
            mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
            cache=cache_modifier,
        )
        smem_q_rope.store(q_pe)
        q_pe = smem_q_rope.load(layout=dot_q_layout)

        acc = gl.zeros([BLOCK_H, BLOCK_C], dtype=gl.float32,
            layout=mfma_layout_kv,
        )
        # acc_zeros = gl.zeros([BLOCK_H, BLOCK_C], dtype=gl.float32,
        #     layout=mfma_layout_kv,
        # )
        zeros = gl.zeros(
            (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
        )
        e_max = gl.zeros_like(gl.max(zeros, 1), dtype=gl.float32) - float("inf")
        e_sum = gl.zeros_like(e_max, dtype=gl.float32)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            # start_n = split_kv_start
            offs_n = start_n + gl.arange(0, BLOCK_N,
                layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
            )
            offs_n_pe = start_n + gl.arange(0, BLOCK_N,
                layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
            )
            kv_loc = gl.load(
                kv_indices + start_n + cur_head,
                # mask=cur_head < split_kv_end,
                # other=0.0,
            )
            kv_pe_loc = gl.load(
                kv_indices + start_n + cur_head_pe,
                # mask= cur_head_pe < split_kv_end,
                # other=0.0,
            )


            offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_c[None, :]
            offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_qk_r[None, :]

            k_pe = gl.amd.cdna3.buffer_load(
                ptr=K_Buffer,
                offsets=offs_buf_k_pe,
                # offsets=off_q_pe,
                # mask=(offs_n_pe[None, :] < split_kv_end) & (mask_k_r[:, None]),
                # mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
                # cache=cache_modifier,
            )  # positional embedding part of keys

            # if cur_batch == 1:
            #     gl.device_print("offs_buf_k_pe", offs_buf_k_pe)
                # gl.device_print("cur_k_pe", k_pe)


            smem_k_rope.store(k_pe.T)
            cur_k_pe = smem_k_rope.load(layout=dot_k_layout)


            # (16, 64) x (64, 32)
            # dot product of rope parts
            # qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))
            qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, zeros)

            kv = gl.amd.cdna3.buffer_load(
                ptr=K_Buffer,
                offsets=offs_buf_kv,
                # offsets=offs_q,
                # mask=(mask_h[:, None]) & (mask_c[None, :]),
                # mask=(offs_n[None, :] < split_kv_end) & (mask_k_c[:, None]),
                cache=cache_modifier,
            )  # the shared latent tensor for keys and values
            smem_v.store(kv)
            smem_k_nope.store(kv.T)
            cur_kv = smem_k_nope.load(layout=dot_k_layout)

            # (16, 512) x (512, 32)
            # dot product of nope parts
            # qk = tl.dot(q, kv)
            qk = gl.amd.cdna3.mfma(q, cur_kv, qk)

            # smem_v.store(cur_kv)

            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * _tanh(qk / logit_cap)

            # qk = gl.where(
            #     mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            # )

            cur_v = smem_v.load(layout=dot_v_layout)

            n_e_max = gl.maximum(gl.max(qk, 1), e_max)
            re_scale = gl.exp(e_max - n_e_max)
            p = gl.exp(qk - n_e_max[:, None])

            # gl.device_print("p", p)

            acc = acc * re_scale[:, None]


            smem_p.store(p.to(cur_v.dtype))
            e_sum = e_sum * re_scale + gl.sum(p, 1)
            cur_p = smem_p.load(layout=dot_p_layout)
            e_max = n_e_max

            # gl.device_print("e_max", e_max)
            # gl.device_print("e_sum", e_sum)

            # if start_n  == 0:
            acc = gl.amd.cdna3.mfma(cur_p, cur_v, acc)
            # acc += gl.amd.cdna3.mfma(cur_p, cur_v, acc_zeros)


        # ======= Epilogue ========
        if split_id == -1:
            offs_o = (
                cur_batch * stride_ob
                + offs_om[:, None] * stride_oh
                + offs_on[None, :]
            )

            acc = acc / e_sum[:, None]
            o = acc.to(Out.type.element_ty)
            gl.amd.cdna3.buffer_store(
                stored_value=o,
                ptr=Out,
                offsets=offs_o,
                mask=(mask_om[:, None]) & (mask_on[None, :]),
            )
        else:
            offs_mid_o = (
                split_id * stride_mid_ob
                + offs_om[:, None] * stride_mid_oh
                + offs_on[None, :]
            )

            gl.amd.cdna3.buffer_store(
                # stored_value=acc / e_sum[:, None],
                stored_value=acc,
                ptr=Att_Out,
                offsets=offs_mid_o,
                # mask=(mask_om[:, None]) & (mask_on[None, :]),
            )

            offs_mid_lse = (
                split_id * stride_mid_lse_b
                + offs_om * stride_mid_lse_h
            )

            gl.amd.cdna3.buffer_store(
                stored_value=e_max + gl.log(e_sum),
                ptr=Att_Lse,
                offsets=offs_mid_lse,
                # mask=mask_om,
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
