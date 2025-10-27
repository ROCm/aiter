# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# Copyright (C) 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

from typing import Optional
import functools
import json
import triton
import triton.language as tl
import torch
from aiter.ops.triton.activation import _tanh
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.pid_preprocessing import remap_xcd

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Need to 
@gluon.jit
def _fwd_grouped_kernel_stage1_n16_db(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
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
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

    pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
    pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS

    cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch


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

    smem_k_rope_1 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k
    )
    # smem_k_rope_2 = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [qk_rope_head_dim, 16], layout=shared_k
    # )

    smem_q_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
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

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
    )
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
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

    cur_batch_kv = cur_batch # // (max_qo_len // 2)
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

    q = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=offs_q,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
        # cache=cache_modifier,
    )
    q_pe = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=off_q_pe,
        mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
        # cache=cache_modifier,
    )
    smem_q_nope.store(q)
    q = smem_q_nope.load(layout=dot_q_layout)
    smem_q_rope.store(q_pe)
    q_pe = smem_q_rope.load(layout=dot_q_layout)


    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    smem_q_nope._keep_alive()
    smem_q_rope._keep_alive()

    smem_v = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
    )

    # smem_k_nope_1 = gl.allocate_shared_memory(
    #     # K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
    #     K_Buffer.type.element_ty, [kv_lora_rank, 16], layout=shared_k
    # )
    #
    # smem_k_nope_2 = gl.allocate_shared_memory(
    #     # K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
    #     K_Buffer.type.element_ty, [kv_lora_rank, 16], layout=shared_k
    # )
    smem_k = gl.allocate_shared_memory(
        # K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
        K_Buffer.type.element_ty, [2, kv_lora_rank, BLOCK_N], layout=shared_k
    )

    acc = gl.zeros([BLOCK_H, BLOCK_C], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    # zeros = gl.zeros(
    #     (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    # )
    zeros = gl.zeros(
        (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    )
    e_max = gl.zeros_like(gl.max(zeros, 1), dtype=gl.float32) - float("inf")
    e_sum = gl.zeros_like(e_max, dtype=gl.float32)


    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_head + cur_batch_kv_start_idx,
        # mask=cur_head < split_kv_end,
        # other=0.0,
    )
    kv_pe_loc = gl.load(
        kv_indices + split_kv_start + cur_head_pe + cur_batch_kv_start_idx,
        # mask= cur_head_pe < split_kv_end,
        # other=0.0,
    )

    # if cur_batch == 1:
    #     gl.device_print("kv_pe_loc", kv_pe_loc)

    offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_c[None, :]
    offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_qk_r[None, :]

    sk_0 = smem_k.index(0)

    kv = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv,
        # offsets=offs_q,
        # mask=(mask_h[:, none]) & (mask_c[none, :]),
        # mask=(offs_n[none, :] < split_kv_end) & (mask_k_c[:, none]),
        # cache=cache_modifier,
    )  # the shared latent tensor for keys and values

    k_pe = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
        # mask=(offs_n_pe[none, :] < split_kv_end) & (mask_k_r[:, none]),
        # mask=(mask_h_pe[:, none]) & (mask_qk_r[none, :]),
        # cache=cache_modifier,
    )  # positional embedding part of keys

    sk_0.store(kv.T)

    ld_stage = 0
    st_stage = 1

    for start_n in range(split_kv_start, split_kv_end - BLOCK_N, BLOCK_N):
        # start_n = split_kv_start
        # offs_n = start_n + gl.arange(0, BLOCK_N,
        #     layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
        # )
        # offs_n_pe = start_n + gl.arange(0, BLOCK_N,
        #     layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
        # )
        smem_k_rope_1.store(k_pe.T)

        smem_v.store(kv)

        kv_loc = gl.load(
            kv_indices + start_n + cur_head + cur_batch_kv_start_idx,
            # mask=cur_head < split_kv_end,
            # other=0.0,
        )

        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_head_pe + cur_batch_kv_start_idx,
            # mask= cur_head_pe < split_kv_end,
            # other=0.0,
        )

        # if cur_batch == 1:
        #     gl.device_print("kv_pe_loc", kv_pe_loc)

        offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_c[None, :]
        offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_qk_r[None, :]

        sk_load = smem_k.index(ld_stage)
        sk_store = smem_k.index(st_stage)

        cur_k_pe = smem_k_rope_1.load(layout=dot_k_layout)

        kv = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv,
            # offsets=offs_q,
            # mask=(mask_h[:, none]) & (mask_c[none, :]),
            # mask=(offs_n[none, :] < split_kv_end) & (mask_k_c[:, none]),
            # cache=cache_modifier,
        )  # the shared latent tensor for keys and values

        cur_kv = sk_load.load(layout=dot_k_layout)

        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
            # mask=(offs_n_pe[none, :] < split_kv_end) & (mask_k_r[:, none]),
            # mask=(mask_h_pe[:, none]) & (mask_qk_r[none, :]),
            # cache=cache_modifier,
        )  # positional embedding part of keys

        kv_loc = gl.load(
            kv_indices + start_n + cur_head + cur_batch_kv_start_idx + BLOCK_N,
            # mask=cur_head < split_kv_end,
            # other=0.0,
        )
        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_head_pe + cur_batch_kv_start_idx + BLOCK_N,
            # mask= cur_head_pe < split_kv_end,
            # other=0.0,
        )
        offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_c[None, :]
        offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_qk_r[None, :]



        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, zeros)

        qk = gl.amd.cdna3.mfma(q, cur_kv, qk)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)

        # qk = gl.where(
        #     mask_h[:, none] & (offs_n[none, :] < split_kv_end), qk, float("-inf")
        # )
        sk_store.store(kv.T)

        cur_v = smem_v.load(layout=dot_v_layout)

        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp(e_max - n_e_max)
        p = gl.exp(qk - n_e_max[:, None])

        acc = acc * re_scale[:, None]

        smem_p.store(p.to(cur_v.dtype))
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max

        acc = gl.amd.cdna3.mfma(cur_p, cur_v, acc)

        ld_stage ^= 1
        st_stage ^= 1

    smem_v.store(kv)

    if True:
        cur_k_pe = smem_k_rope_1.load(layout=dot_k_layout)

        sk_load = smem_k.index(ld_stage)

        cur_kv = sk_load.load(layout=dot_k_layout)


        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, zeros)

        qk = gl.amd.cdna3.mfma(q, cur_kv, qk)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)

        # qk = gl.where(
        #     mask_h[:, none] & (offs_n[none, :] < split_kv_end), qk, float("-inf")
        # )

        cur_v = smem_v.load(layout=dot_v_layout)

        n_e_max = gl.maximum(gl.max(qk, 1), e_max)
        re_scale = gl.exp(e_max - n_e_max)
        p = gl.exp(qk - n_e_max[:, None])

        acc = acc * re_scale[:, None]

        smem_p.store(p.to(cur_v.dtype))
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max

        acc = gl.amd.cdna3.mfma(cur_p, cur_v, acc)

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + offs_om[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_on[None, :]
    )

    e_sum = 1 / e_sum

    gl.amd.cdna3.buffer_store(
        stored_value=acc * e_sum[:, None],
        ptr=Att_Out,
        offsets=offs_mid_o,
        # mask=(mask_om[:, none]) & (mask_on[none, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    gl.amd.cdna3.buffer_store(
        stored_value=e_max + gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        # mask=mask_om,
    )


@gluon.jit
def _fwd_grouped_kernel_stage1_n32(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
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
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

    pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
    pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS

    log2e: gl.constexpr = 1.4426950408889634

    cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch

    cur_batch_kv = cur_batch # // (max_qo_len // 2)
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        size_per_thread=[1, 8],  # 16 * 512
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blocked_q_rope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 64 in
        size_per_thread=[1, 4],  # 64 * 576
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_ld_k_nope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[4, 8],  # 64 * 512
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )
    blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[2, 8],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 2],
        order=[1, 0],
    )


    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )

    smem_q0_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q1_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
    )

    smem_p = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_q
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 2]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 2]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=4
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=4
    )

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
    )
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_N = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
    )

    cur_N_pe = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_rope_kn)
    )

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

    # offs_k_c = gl.arange(
    #     0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    # )
    offs_k_c = gl.arange(
        0, 256, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    )
    offs_k_r = gl.arange(
        kv_lora_rank,
        kv_lora_rank + BLOCK_R,
        layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
    )  # to get the k_pe

    off_q_pe = (
        cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_k_c = offs_k_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
    mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    q = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=offs_q,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )
    q_pe = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=off_q_pe,
        mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
    )
    kv_pe_loc = gl.load(
        kv_indices + split_kv_start + cur_N_pe + cur_batch_kv_start_idx,
    )

    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_batch_kv_start_idx + cur_N,
    )

    q = gl.reshape(q, [BLOCK_H, 2, BLOCK_C // 2])
    q = gl.permute(q, (0, 2, 1))
    q0, q1 = gl.split(q)

    smem_q0_nope.store(q0)
    smem_q1_nope.store(q1)
    q0 = smem_q0_nope.load(layout=dot_q_layout)
    q1 = smem_q1_nope.load(layout=dot_q_layout)

    smem_q_rope.store(q_pe)
    q_pe = smem_q_rope.load(layout=dot_q_layout)

    smem_q0_nope._keep_alive()
    smem_q1_nope._keep_alive()
    smem_q_rope._keep_alive()

    smem_v = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_k
    )
    smem_k1 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )
    smem_k2 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )

    acc1 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    acc2 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    zeros = gl.zeros(
        (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    )
    offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]
    offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]

    kv1 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv,
    )  # the shared latent tensor for keys and values
    kv2 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv + 256,
    )  # the shared latent tensor for keys and values

    smem_k1.store(kv1.T)
    smem_k2.store(kv2.T)

    k_pe = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
    )  # positional embedding part of keys

    # e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout = gl.SliceLayout(1, mfma_layout_qk)) - float("inf")
    # e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout = gl.SliceLayout(1, mfma_layout_qk))
    e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
    e_max = e_sum - float("inf")

    cur_k1 = smem_k1.load(layout=dot_k_layout)
    cur_k2 = smem_k2.load(layout=dot_k_layout)

    cur_k_pe = gl.convert_layout(k_pe.T, layout=dot_k_layout)

    kv_pe_loc = gl.load(
        kv_indices + split_kv_start + cur_N_pe + cur_batch_kv_start_idx + BLOCK_N,
    )
    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_batch_kv_start_idx + BLOCK_N + cur_N,
    )

    gl.amd.cdna3.sched_barrier(0x0)

    # for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    for start_n in range(split_kv_start, split_kv_end - BLOCK_N, BLOCK_N):
        smem_v.store(kv1)
        cur_v = smem_v.load(layout)

        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_N_pe + cur_batch_kv_start_idx + BLOCK_N * 2,
        )
        kv_loc = gl.load(
            kv_indices + start_n + cur_batch_kv_start_idx + BLOCK_N * 2 + cur_N,
        )

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, zeros)
        qk = gl.amd.cdna3.mfma(q0, cur_k1, qk)
        qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

        qk *= sm_scale
        # cur_v = gl.convert_layout(cur_v, layout=dot_v_layout)

        # kv_trans = gl.reshape(kv, [BLOCK_N, 2, BLOCK_C // 2])
        # kv_trans = gl.permute(kv_trans, (0, 2, 1))
        # kv1, kv2 = gl.split(kv_trans)
        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))
        acc1 = acc1 * re_scale[:, None]
        acc2 = acc2 * re_scale[:, None]
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max
        smem_k.store(kv1.T)

        acc1 = gl.amd.cdna3.mfma(cur_p, cur_v1, acc1)
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_v2, acc2)

        cur_k = smem_k.load(layout=dot_k_layout)
        cur_v1 = gl.convert_layout(kv1, layout=dot_v_layout)
        cur_v2 = gl.convert_layout(kv2, layout=dot_v_layout)



    # # gl.amd.cdna3.sched_barrier(0x0)
    # # #
    # # # for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    # if True:
    #     cur_k_pe = gl.convert_layout(k_pe.T, layout=dot_k_layout)
    #
    #     # cur_k    = gl.convert_layout(kv.T, layout=dot_k_layout)
    #     # cur_k = smem_k.load(layout=dot_k_layout)
    #
    #     smem_k.store(kv2.T)
    #     qk = gl.amd.cdna3.mfma(q0, cur_k, zeros)
    #
    #     qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)
    #     # cur_k    = gl.convert_layout(kv.T, layout=dot_k_layout)
    #     # smem_v.store(kv)
    #
    #     cur_k = smem_k.load(layout=dot_k_layout)
    #
    #     qk = gl.amd.cdna3.mfma(q1, cur_k, qk)
    #
    #     qk *= sm_scale
    #
    #     # if logit_cap > 0:
    #     #     qk = logit_cap * _tanh(qk / logit_cap)
    #
    #     n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
    #     # cur_v1 = gl.convert_layout(kv1, layout=dot_v_layout)
    #     # cur_v2 = gl.convert_layout(kv2, layout=dot_v_layout)
    #
    #     n_e_max = gl.maximum(n_e_max, e_max)
    #     re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
    #     p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
    #     acc1 = acc1 * re_scale[:, None]
    #     acc2 = acc2 * re_scale[:, None]
    #
    #     smem_p.store(p.to(q0.dtype))
    #     e_sum = e_sum * re_scale + gl.sum(p, 1)
    #     cur_p = smem_p.load(layout=dot_p_layout)
    #     e_max = n_e_max
    #
    #     acc1 = gl.amd.cdna3.mfma(cur_p, cur_v1, acc1)
    #     acc2 = gl.amd.cdna3.mfma(cur_p, cur_v2, acc2)

    smem_k._keep_alive()
    # smem_v._keep_alive()

    acc = gl.join(acc1, acc2)
    acc  = gl.permute(acc, (0, 2, 1))
    acc = gl.reshape(acc, (BLOCK_H, BLOCK_C))
    acc = gl.convert_layout(acc, mfma_layout_kv)

    smem_o = gl.allocate_shared_memory(
        Att_Out.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )

    e_sum = 1 / e_sum
    smem_o.store(acc * e_sum[:, None])
    # smem_o.store(acc)
    cur_o = smem_o.load(layout=blocked_q_nope_mk)

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_c[None, :]
    )
    # offs_mid_o = (
    #     cur_batch * stride_mid_ob
    #     + offs_om[:, None] * stride_mid_oh
    #     + split_kv_id * stride_mid_os
    #     + offs_on[None, :]
    # )

    gl.amd.cdna3.buffer_store(
        stored_value=cur_o,
        # stored_value=acc / e_sum[:, None],
        ptr=Att_Out,
        offsets=offs_mid_o,
        # mask=(mask_om[:, none]) & (mask_on[none, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        # mask=mask_om,
    )


@gluon.jit
def _fwd_grouped_kernel_stage1_n32_prefetch_k(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
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
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

    pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
    pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS

    log2e: gl.constexpr = 1.4426950408889634

    cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch

    cur_batch_kv = cur_batch # // (max_qo_len // 2)
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        # size_per_thread=[1, 128],  # 64 * 576
        # threads_per_warp=[16, 4],
        # warps_per_cta=[4, 1],
        # order=[1, 0],
        size_per_thread=[4, 8],  # 64 * 512
        threads_per_warp=[8, 8],
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
        size_per_thread=[4, 8],  # 64 * 512
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    # blocked_k: gl.constexpr = gl.DistributedLinearLayout( # 128x256
    #     reg_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,1,0,0), (0,8,0,0), (4,0,0,0), (8,0,0,0)), # 16 x 8
    #     lane_bases=((0,0,1,0), (0,0,2,0), (0,0,4,0), (0,0,8,0), (0,2,0,0), (0,4,0,0)), # 64
    #     warp_bases=((1,0,0,0), (2,0,0,0)), # 4
    #     block_bases=[], # 8
    #     shape=[64, 512],
    # )

    blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        # size_per_thread=[1, 4],
        # threads_per_warp=[16, 4],
        # warps_per_cta=[1, 4],
        # order=[0, 1],
        size_per_thread=[1, 8],  # 64 * 576
        threads_per_warp=[32, 2],
        warps_per_cta=[1, 4],
        order=[1, 0],
        # size_per_thread=[8, 1],
        # threads_per_warp=[4, 16],
        # warps_per_cta=[4, 1],
        # order=[0, 1],
    )
    # mfma_layout_a_linear: gl.constexpr = gl.DistributedLinearLayout( # 64x128
    #     reg_bases=((0,1), (0,2), (0,4), (0,8), (16,0), (32,0), (0,64)),
    #     lane_bases=((1,0), (2,0), (4,0), (8,0), (0,16), (0,32)),
    #     warp_bases=((0,0), (0,0)),
    #     block_bases=[],
    #     shape=[64, 128],
    # )
    # mfma_layout_b_linear: gl.constexpr = gl.DistributedLinearLayout( # 128x256
    #     reg_bases=((1,0), (2,0), (4,0), (8,0), (0,64), (0,128), (64,0)),
    #     lane_bases=((0,1), (0,2), (0,4), (0,8), (16,0), (32,0)),
    #     warp_bases=((0,16), (0,32)),
    #     block_bases=[],
    #     shape=[128, 256],
    # )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )

    # smem_k_rope_2 = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [qk_rope_head_dim, 16], layout=shared_k
    # )

    smem_q0_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q1_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
    )

    smem_p = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_q
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=4
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=4
    )

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
    )
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_N = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
    )

    cur_N_pe = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_rope_kn)
    )

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

    # offs_k_c = gl.arange(
    #     0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    # )
    offs_k_c = gl.arange(
        0, 256, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    )
    offs_k_r = gl.arange(
        kv_lora_rank,
        kv_lora_rank + BLOCK_R,
        layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
    )  # to get the k_pe

    off_q_pe = (
        cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_k_c = offs_k_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
    mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    q = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=offs_q,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )
    q_pe = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=off_q_pe,
        mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
    )
    kv_pe_loc = gl.load(
        kv_indices + split_kv_start + cur_N_pe + cur_batch_kv_start_idx,
    )

    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_batch_kv_start_idx + cur_N,
    )

    q = gl.reshape(q, [BLOCK_H, 2, BLOCK_C // 2])
    q = gl.permute(q, (0, 2, 1))
    q0, q1 = gl.split(q)

    smem_q0_nope.store(q0)
    smem_q1_nope.store(q1)
    q0 = smem_q0_nope.load(layout=dot_q_layout)
    q1 = smem_q1_nope.load(layout=dot_q_layout)

    smem_q_rope.store(q_pe)
    q_pe = smem_q_rope.load(layout=dot_q_layout)

    smem_q0_nope._keep_alive()
    smem_q1_nope._keep_alive()
    smem_q_rope._keep_alive()

    # smem_v = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_k
    # )
    smem_k_rope = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k
    )

    acc1 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    acc2 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )

    zeros = gl.zeros(
        (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    )

    offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]
    offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]

    kv1 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv,
    )  # the shared latent tensor for keys and values
    kv2 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv + 256,
    )  # the shared latent tensor for keys and values

    smem_kv1 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )
    smem_kv2 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )

    smem_kv1.store(kv1.T)
    smem_kv2.store(kv2.T)

    k_pe = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
    )  # positional embedding part of keys

    e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
    e_max = e_sum - float("inf")

    # cur_k1 = gl.convert_layout(kv1.T, layout=dot_k_layout)
    # cur_k2 = gl.convert_layout(kv2.T, layout=dot_k_layout)

    cur_k1 = smem_kv1.load(layout=dot_k_layout)
    cur_k2 = smem_kv2.load(layout=dot_k_layout)

    smem_k_rope.store(k_pe.T)
    gl.amd.cdna3.sched_barrier(0x0)
    split_kv_start += BLOCK_N

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):

        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_N_pe + cur_batch_kv_start_idx,
        )
        kv_loc = gl.load(
            kv_indices + start_n + cur_batch_kv_start_idx + cur_N,
        )

        cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

        qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
        qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

        # smem_kv1 = smem_kv1.permute([1, 0])
        # smem_kv2 = smem_kv2.permute([1, 0])
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)

        smem_kv1.store(kv1)
        smem_kv2.store(kv2)

        offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]
        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
        )  # positional embedding part of keys

        offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]

        kv1 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv,
        )
        kv2 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv + 256,
        )

        cur_k1 = smem_kv1.load(layout=dot_v_layout)
        cur_k2 = smem_kv2.load(layout=dot_v_layout)

        # smem_kv1._keep_alive()
        # smem_kv2._keep_alive()

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk *= sm_scale

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))

        # smem_kv1 = smem_kv1.permute([1, 0])
        # smem_kv2 = smem_kv2.permute([1, 0])
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)

        smem_kv1.store(kv1.T)
        smem_kv2.store(kv2.T)
        smem_k_rope.store(k_pe.T)

        acc1 = acc1 * re_scale[:, None]
        acc2 = acc2 * re_scale[:, None]
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max

        acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)

        cur_k1 = smem_kv1.load(layout=dot_k_layout)
        cur_k2 = smem_kv2.load(layout=dot_k_layout)

    if True:
        # smem_kv1 = smem_kv1.permute([1, 0])
        # smem_kv2 = smem_kv2.permute([1, 0])
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)

        smem_kv1.store(kv1)
        smem_kv2.store(kv2)

        qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
        qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

        cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        cur_k1 = smem_kv1.load(layout=dot_v_layout)
        cur_k2 = smem_kv2.load(layout=dot_v_layout)

        qk *= sm_scale

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))

        acc1 = acc1 * re_scale[:, None]
        acc2 = acc2 * re_scale[:, None]
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max

        acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)

    smem_kv1._keep_alive()
    smem_kv2._keep_alive()

    acc = gl.join(acc1, acc2)
    acc  = gl.permute(acc, (0, 2, 1))
    acc = gl.reshape(acc, (BLOCK_H, BLOCK_C))
    acc = gl.convert_layout(acc, mfma_layout_kv)

    smem_o = gl.allocate_shared_memory(
        Att_Out.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )

    e_sum = 1 / e_sum
    smem_o.store(acc * e_sum[:, None])
    # smem_o.store(acc)
    cur_o = smem_o.load(layout=blocked_q_nope_mk)

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_c[None, :]
    )
    # offs_mid_o = (
    #     cur_batch * stride_mid_ob
    #     + offs_om[:, None] * stride_mid_oh
    #     + split_kv_id * stride_mid_os
    #     + offs_on[None, :]
    # )

    gl.amd.cdna3.buffer_store(
        stored_value=cur_o,
        # stored_value=acc / e_sum[:, None],
        ptr=Att_Out,
        offsets=offs_mid_o,
        # mask=(mask_om[:, none]) & (mask_on[none, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        # mask=mask_om,
    )



@gluon.jit
def _fwd_grouped_kernel_stage1_n64(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
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
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

    pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
    pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS

    log2e: gl.constexpr = 1.4426950408889634

    cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch

    cur_batch_kv = cur_batch # // (max_qo_len // 2)
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

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
        size_per_thread=[16, 8],  # 64 * 512
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
        size_per_thread=[4, 8],  # 64 * 576
        threads_per_warp=[32, 2],
        warps_per_cta=[1, 4],
        order=[1, 0],
        # size_per_thread=[8, 1],
        # threads_per_warp=[4, 16],
        # warps_per_cta=[4, 1],
        # order=[0, 1],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )

    # smem_k_rope_2 = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [qk_rope_head_dim, 16], layout=shared_k
    # )

    smem_q0_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q1_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
    )

    smem_p = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_q
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=4
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=4
    )

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
    )
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_N = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
    )

    cur_N_pe = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_rope_kn)
    )

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
        0, BLOCK_C, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    )
    offs_k_r = gl.arange(
        kv_lora_rank,
        kv_lora_rank + BLOCK_R,
        layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
    )  # to get the k_pe

    off_q_pe = (
        cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_k_c = offs_k_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
    mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    q = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=offs_q,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )
    q_pe = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=off_q_pe,
        mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
    )
    kv_pe_loc = gl.load(
        kv_indices + split_kv_start + cur_N_pe + cur_batch_kv_start_idx,
    )

    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_N + cur_batch_kv_start_idx,
    )

    q = gl.reshape(q, [BLOCK_H, BLOCK_C // 2, 2])
    q0, q1 = gl.split(q)

    smem_q0_nope.store(q0)
    smem_q1_nope.store(q1)
    q0 = smem_q0_nope.load(layout=dot_q_layout)
    q1 = smem_q1_nope.load(layout=dot_q_layout)

    smem_q_rope.store(q_pe)
    q_pe = smem_q_rope.load(layout=dot_q_layout)

    smem_q0_nope._keep_alive()
    smem_q1_nope._keep_alive()
    smem_q_rope._keep_alive()

    # smem_v = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
    # )
    smem_k = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )
    # smem_k_rope = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k
    # )

    # acc = gl.zeros([BLOCK_H, BLOCK_C], dtype=gl.float32,
    #     layout=mfma_layout_kv,
    # )
    acc1 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    acc2 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    # zeros = gl.zeros(
    #     (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    # )
    zeros = gl.zeros(
        (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    )
    # e_max = gl.zeros_like(gl.max(zeros, 1), dtype=gl.float32) - float("inf")
    # e_sum = gl.zeros_like(e_max, dtype=gl.float32)
    offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]
    offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]

    kv = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv,
    )  # the shared latent tensor for keys and values

    kv_trans = gl.reshape(kv, [BLOCK_N, BLOCK_C // 2, 2])
    kv1, kv2 = gl.split(kv_trans)

    smem_k.store(kv1.T)

    k_pe = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
    )  # positional embedding part of keys

    # e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout = gl.SliceLayout(1, mfma_layout_qk)) - float("inf")
    # e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout = gl.SliceLayout(1, mfma_layout_qk))
    e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
    e_max = e_sum - float("inf")

    gl.amd.cdna3.sched_barrier(0x0)

    # for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    for start_n in range(split_kv_start, split_kv_end - BLOCK_N, BLOCK_N):
        cur_k = smem_k.load(layout=dot_k_layout)
        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_N_pe + cur_batch_kv_start_idx + BLOCK_N,
        )
        kv_loc = gl.load(
            kv_indices + start_n + cur_N + cur_batch_kv_start_idx + BLOCK_N,
        )
        smem_k.store(kv2.T)

        cur_k_pe = gl.convert_layout(k_pe.T, layout=dot_k_layout)
        qk = gl.amd.cdna3.mfma(q0, cur_k, zeros)

        cur_v1 = gl.convert_layout(kv1, layout=dot_v_layout)

        cur_k = smem_k.load(layout=dot_k_layout)
        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk = gl.amd.cdna3.mfma(q1, cur_k, qk)

        cur_v2 = gl.convert_layout(kv2, layout=dot_v_layout)

        offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]

        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
        )  # positional embedding part of keys
        offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]

        kv = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv,
        )  # the shared latent tensor for keys and values

        qk *= sm_scale
        # cur_v = gl.convert_layout(cur_v, layout=dot_v_layout)

        kv_trans = gl.reshape(kv, [BLOCK_N, BLOCK_C // 2, 2])
        kv1, kv2 = gl.split(kv_trans)
        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(kv.dtype))
        acc1 = acc1 * re_scale[:, None]
        acc2 = acc2 * re_scale[:, None]
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max
        smem_k.store(kv1.T)

        acc1 = gl.amd.cdna3.mfma(cur_p, cur_v1, acc1)
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_v2, acc2)
    #
    # # for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    if True:
        cur_k_pe = gl.convert_layout(k_pe.T, layout=dot_k_layout)

        # cur_k    = gl.convert_layout(kv.T, layout=dot_k_layout)
        cur_k = smem_k.load(layout=dot_k_layout)

        smem_k.store(kv2.T)

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, zeros)
        # cur_k    = gl.convert_layout(kv.T, layout=dot_k_layout)
        # smem_v.store(kv)

        qk = gl.amd.cdna3.mfma(q0, cur_k, qk)
        cur_k = smem_k.load(layout=dot_k_layout)

        qk = gl.amd.cdna3.mfma(q1, cur_k, qk)

        qk *= sm_scale

        # if logit_cap > 0:
        #     qk = logit_cap * _tanh(qk / logit_cap)

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
        cur_v1 = gl.convert_layout(kv1, layout=dot_v_layout)
        cur_v2 = gl.convert_layout(kv2, layout=dot_v_layout)

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        acc1 = acc1 * re_scale[:, None]
        acc2 = acc2 * re_scale[:, None]

        smem_p.store(p.to(kv.dtype))
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max

        acc1 = gl.amd.cdna3.mfma(cur_p, cur_v1, acc1)
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_v2, acc2)

    smem_k._keep_alive()
    # smem_v._keep_alive()

    acc = gl.join(acc1, acc2)
    acc = gl.reshape(acc, (BLOCK_H, BLOCK_C))
    acc = gl.convert_layout(acc, mfma_layout_kv)

    smem_o = gl.allocate_shared_memory(
        Att_Out.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )

    e_sum = 1 / e_sum
    smem_o.store(acc * e_sum[:, None])
    # smem_o.store(acc)
    cur_o = smem_o.load(layout=blocked_q_nope_mk)

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_c[None, :]
    )
    # offs_mid_o = (
    #     cur_batch * stride_mid_ob
    #     + offs_om[:, None] * stride_mid_oh
    #     + split_kv_id * stride_mid_os
    #     + offs_on[None, :]
    # )

    gl.amd.cdna3.buffer_store(
        stored_value=cur_o,
        # stored_value=acc / e_sum[:, None],
        ptr=Att_Out,
        offsets=offs_mid_o,
        # mask=(mask_om[:, none]) & (mask_on[none, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        # mask=mask_om,
    )

@gluon.jit
def _fwd_grouped_kernel_stage1_n16(
    Q,  # Holds [Q_NOPE; Q_PE], b x h x (d+r)
    K_Buffer,  # Holds [KV; K_PE], b*s x (c+r)
    V_buffer,  # Holds [KV], b*s x (c)
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,  # b x h x NUM_KV_SPLITS x (kv_lora_rank)
    Att_Lse,  # b x h x NUM_KV_SPLITS x (1)
    stride_qb,
    stride_qh,
    stride_buf_kbs,
    stride_buf_vbs,
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
    BLOCK_C: gl.constexpr,
    BLOCK_R: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_H: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
):
    pid = gl.program_id(0)
    num_q_head_blk = gl.cdiv(q_head_num, BLOCK_H)

    pid_head_kv_split = pid % (num_q_head_blk * NUM_KV_SPLITS)
    pid_head_kv_split = remap_xcd(pid_head_kv_split, (num_q_head_blk * NUM_KV_SPLITS))

    cur_head_id = pid_head_kv_split % num_q_head_blk
    split_kv_id = (pid_head_kv_split // num_q_head_blk) % NUM_KV_SPLITS

    log2e: gl.constexpr = 1.4426950408889634

    cur_batch = (pid // (num_q_head_blk * NUM_KV_SPLITS)) % batch

    cur_batch_kv = cur_batch # // (max_qo_len // 2)
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        # size_per_thread=[1, 128],  # 64 * 576
        # threads_per_warp=[16, 4],
        # warps_per_cta=[4, 1],
        # order=[1, 0],
        size_per_thread=[4, 8],  # 64 * 512
        threads_per_warp=[8, 8],
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
        threads_per_warp=[8, 8],
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
        size_per_thread=[1, 8],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
        # size_per_thread=[8, 1],
        # threads_per_warp=[4, 16],
        # warps_per_cta=[4, 1],
        # order=[0, 1],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )

    # smem_k_rope_2 = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [qk_rope_head_dim, 16], layout=shared_k
    # )

    smem_q0_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q1_nope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, kv_lora_rank // 2], layout=shared_q
    )
    smem_q_rope = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, qk_rope_head_dim], layout=shared_q
    )

    smem_p = gl.allocate_shared_memory(
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_q
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=4
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=4
    )

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: gl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: gl.constexpr = kv_group_num

    cur_head = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_nope_mk)
    )
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    cur_N = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
    )

    cur_N_pe = cur_head_id * VALID_BLOCK_H + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_rope_kn)
    )

    cur_head_pe = gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(1, blocked_q_rope_mk)
    )
    mask_h_pe = cur_head_pe < VALID_BLOCK_H
    mask_h_pe = mask_h_pe & (cur_head_pe < q_head_num)

    offs_c = gl.arange(
        0, 512, layout=gl.SliceLayout(0, blocked_q_nope_mk)
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
        0, 256, layout=gl.SliceLayout(0, blocked_ld_k_nope_kn)
    )
    offs_k_r = gl.arange(
        kv_lora_rank,
        kv_lora_rank + BLOCK_R,
        layout=gl.SliceLayout(0, blocked_ld_k_rope_kn)
    )  # to get the k_pe

    off_q_pe = (
        cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_k_c = offs_k_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
    mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

    kv_len_per_split = gl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    q = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=offs_q,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )
    q_pe = gl.amd.cdna3.buffer_load(
        ptr=Q,
        offsets=off_q_pe,
        mask=(mask_h_pe[:, None]) & (mask_qk_r[None, :]),
    )
    kv_pe_loc = gl.load(
        kv_indices + split_kv_start + cur_N_pe + cur_batch_kv_start_idx,
    )

    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_N + cur_batch_kv_start_idx,
    )

    q = gl.reshape(q, [BLOCK_H, 2, BLOCK_C // 2])
    q = gl.permute(q, (0, 2, 1))
    q0, q1 = gl.split(q)

    smem_q0_nope.store(q0)
    smem_q1_nope.store(q1)
    q0 = smem_q0_nope.load(layout=dot_q_layout)
    q1 = smem_q1_nope.load(layout=dot_q_layout)

    smem_q_rope.store(q_pe)
    q_pe = smem_q_rope.load(layout=dot_q_layout)

    smem_q0_nope._keep_alive()
    smem_q1_nope._keep_alive()
    smem_q_rope._keep_alive()

    # smem_v = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank], layout=shared_k
    # )
    smem_k = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )
    # smem_k_rope = gl.allocate_shared_memory(
    #     K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k
    # )

    acc1 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    acc2 = gl.zeros([BLOCK_H, BLOCK_C // 2], dtype=gl.float32,
        layout=mfma_layout_kv,
    )
    # zeros = gl.zeros(
    #     (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    # )
    zeros = gl.zeros(
        (BLOCK_H, BLOCK_N), dtype=gl.float32, layout=mfma_layout_qk,
    )
    # e_max = gl.zeros_like(gl.max(zeros, 1), dtype=gl.float32) - float("inf")
    # e_sum = gl.zeros_like(e_max, dtype=gl.float32)
    offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]
    offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]

    kv1 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv,
    )  # the shared latent tensor for keys and values
    kv2 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv + 256,
    )  # the shared latent tensor for keys and values

    # kv_trans = gl.reshape(kv, [BLOCK_N, 2, BLOCK_C // 2])
    # kv_trans = gl.permute(kv_trans, (0, 2, 1))
    # kv1, kv2 = gl.split(kv_trans)

    smem_k.store(kv1.T)

    k_pe = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
    )  # positional embedding part of keys

    # e_max = gl.zeros([BLOCK_H], dtype=gl.float32, layout = gl.SliceLayout(1, mfma_layout_qk)) - float("inf")
    # e_sum = gl.zeros([BLOCK_H], dtype=gl.float32, layout = gl.SliceLayout(1, mfma_layout_qk))
    e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
    e_max = e_sum - float("inf")

    gl.amd.cdna3.sched_barrier(0x0)

    # for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    for start_n in range(split_kv_start, split_kv_end - BLOCK_N, BLOCK_N):
        cur_k = smem_k.load(layout=dot_k_layout)
        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_N_pe + cur_batch_kv_start_idx + BLOCK_N,
        )
        kv_loc = gl.load(
            kv_indices + start_n + cur_N + cur_batch_kv_start_idx + BLOCK_N,
        )
        smem_k.store(kv2.T)

        cur_k_pe = gl.convert_layout(k_pe.T, layout=dot_k_layout)
        qk = gl.amd.cdna3.mfma(q0, cur_k, zeros)


        cur_k = smem_k.load(layout=dot_k_layout)
        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk = gl.amd.cdna3.mfma(q1, cur_k, qk)

        cur_v1 = gl.convert_layout(kv1, layout=dot_v_layout)
        cur_v2 = gl.convert_layout(kv2, layout=dot_v_layout)

        offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]

        offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]

        kv1 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv,
        )  # the shared latent tensor for keys and values
        kv2 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv + 256,
        )  # the shared latent tensor for keys and values

        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
        )  # positional embedding part of keys


        qk *= sm_scale
        # cur_v = gl.convert_layout(cur_v, layout=dot_v_layout)

        # kv_trans = gl.reshape(kv, [BLOCK_N, 2, BLOCK_C // 2])
        # kv_trans = gl.permute(kv_trans, (0, 2, 1))
        # kv1, kv2 = gl.split(kv_trans)
        if logit_cap > 0:
            qk = logit_cap * _tanh(qk / logit_cap)

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(cur_k.dtype))
        acc1 = acc1 * re_scale[:, None]
        acc2 = acc2 * re_scale[:, None]
        e_sum = e_sum * re_scale + gl.sum(p, 1)
        cur_p = smem_p.load(layout=dot_p_layout)
        e_max = n_e_max
        smem_k.store(kv1.T)

        # acc1 = gl.amd.cdna3.mfma(cur_p, cur_v1, acc1)
        # acc2 = gl.amd.cdna3.mfma(cur_p, cur_v2, acc2)
    #
    # # for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
    # if True:
    #     cur_k_pe = gl.convert_layout(k_pe.T, layout=dot_k_layout)
    #
    #     # cur_k    = gl.convert_layout(kv.T, layout=dot_k_layout)
    #     cur_k = smem_k.load(layout=dot_k_layout)
    #
    #     smem_k.store(kv2.T)
    #
    #     qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, zeros)
    #     # cur_k    = gl.convert_layout(kv.T, layout=dot_k_layout)
    #     # smem_v.store(kv)
    #
    #     qk = gl.amd.cdna3.mfma(q0, cur_k, qk)
    #     cur_k = smem_k.load(layout=dot_k_layout)
    #
    #     qk = gl.amd.cdna3.mfma(q1, cur_k, qk)
    #
    #     qk *= sm_scale
    #
    #     # if logit_cap > 0:
    #     #     qk = logit_cap * _tanh(qk / logit_cap)
    #
    #     n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
    #     # cur_v1 = gl.convert_layout(kv1, layout=dot_v_layout)
    #     # cur_v2 = gl.convert_layout(kv2, layout=dot_v_layout)
    #
    #     n_e_max = gl.maximum(n_e_max, e_max)
    #     re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
    #     p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
    #     acc1 = acc1 * re_scale[:, None]
    #     acc2 = acc2 * re_scale[:, None]
    #
    #     smem_p.store(p.to(cur_k.dtype))
    #     e_sum = e_sum * re_scale + gl.sum(p, 1)
    #     cur_p = smem_p.load(layout=dot_p_layout)
    #     e_max = n_e_max
    #
    #     # acc1 = gl.amd.cdna3.mfma(cur_p, cur_v1, acc1)
    #     # acc2 = gl.amd.cdna3.mfma(cur_p, cur_v2, acc2)

    smem_k._keep_alive()
    # smem_v._keep_alive()

    acc = gl.join(acc1, acc2)
    acc  = gl.permute(acc, (0, 2, 1))
    acc = gl.reshape(acc, (BLOCK_H, BLOCK_C))
    acc = gl.convert_layout(acc, mfma_layout_kv)

    smem_o = gl.allocate_shared_memory(
        Att_Out.type.element_ty, [BLOCK_H, kv_lora_rank], layout=shared_q
    )

    e_sum = 1 / e_sum
    smem_o.store(acc * e_sum[:, None])
    # smem_o.store(acc)
    cur_o = smem_o.load(layout=blocked_q_nope_mk)

    offs_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head[:, None] * stride_mid_oh
        + split_kv_id * stride_mid_os
        + offs_c[None, :]
    )
    # offs_mid_o = (
    #     cur_batch * stride_mid_ob
    #     + offs_om[:, None] * stride_mid_oh
    #     + split_kv_id * stride_mid_os
    #     + offs_on[None, :]
    # )

    gl.amd.cdna3.buffer_store(
        stored_value=cur_o,
        # stored_value=acc / e_sum[:, None],
        ptr=Att_Out,
        offsets=offs_mid_o,
        # mask=(mask_om[:, none]) & (mask_on[none, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        # mask=mask_om,
    )

def _decode_grouped_att_m_fwd(
    q,               # [b, sq, hq, 576]
    k_buffer,        # [pages, hk, 576]
    v_buffer,
    att_out,
    att_lse,
    kv_lora_rank,  # c
    kv_indptr,
    kv_indices,
    num_kv_splits,
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

    # print(config["BLOCK_H"])

    config["NUM_KV_SPLITS"] = num_kv_splits
    grid = (
        triton.cdiv(head_num, min(config["BLOCK_H"], kv_group_num))
        * batch
        * config["NUM_KV_SPLITS"],
    )
    # print(q.shape, grid)

    _fwd_grouped_kernel_stage1_n32_prefetch_k[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        v_buffer.stride(0),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_lse.stride(0),
        att_lse.stride(1),
        att_lse.stride(2),
        kv_lora_rank,
        qk_rope_head_dim,
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        batch=batch,
        logit_cap=logit_cap,
        max_qo_len=mtp + 1,
        **config,
    )


@triton.jit
def _fwd_kernel_stage2(
    Att_Out,
    Att_Lse,
    O,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    batch: tl.constexpr,
    head_num: tl.constexpr,
    max_qo_len: tl.constexpr,
):
    pid = tl.program_id(0)

    pid = remap_xcd(pid, batch * head_num)
    cur_batch = pid % batch
    cur_head = (pid // batch) % head_num

    cur_batch_kv = cur_batch // max_qo_len
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch_kv + 1) - tl.load(
        kv_indptr + cur_batch_kv
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_lse_b + cur_head * stride_mid_lse_h

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Att_Out + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Att_Lse + offs_logic + split_kv_id * stride_mid_lse_s)
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


def _decode_softmax_reducev_fwd(
    att_out,
    att_lse,
    q,
    o,
    v_buffer,
    kv_indptr,
    num_kv_splits,
    mtp,
    config,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    config["BLOCK_DV"] = triton.next_power_of_2(Lv)

    config["NUM_KV_SPLITS"] = num_kv_splits

    grid = (batch * head_num,)
    _fwd_kernel_stage2[grid](
        att_out,
        att_lse,
        o,
        kv_indptr,
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        att_lse.stride(0),
        att_lse.stride(1),
        att_lse.stride(2),
        o.stride(0) * (mtp + 1),
        o.stride(1),
        Lv=Lv,
        head_num=head_num,
        batch=batch,
        max_qo_len=mtp + 1,
        **config,
    )


@functools.lru_cache(maxsize=1024)
def _get_config():
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MLA_DECODE_ROPE-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict


def decode_attention_fwd_grouped(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_lora_rank: int,
    attn_logits: torch.Tensor,
    attn_lse: torch.Tensor,
    num_kv_splits: int,
    sm_scale: float,
    logit_cap: Optional[float] = 0.0,
    mtp: Optional[int] = 1,
    config: Optional[dict[str, any]] = None,
):
    """
    Implements deepseek decode attention with grouped query attention and rotary positional encoding

    parameters:
    q: Query Tensor
    k_buffer: Key Cache Tensor
    v_buffer: Value Cache Tensor
    o: Output tensor containing the result of decode. Allocated by the caller
    kv_indptr:
    kv_indices:
    kv_lora_rank:
    attn_logits:
    num_kv_splits:
    sm_scale
    logit_cap:

    Returns:
    o: output Tensor

    """
    if config is None:
        config = _get_config()

    _decode_grouped_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        attn_lse,
        kv_lora_rank,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
        mtp,
        config["fwd_grouped_kernel_stage1_rope"],
    )
    _decode_softmax_reducev_fwd(
        attn_logits,
        attn_lse,
        q,
        o,
        v_buffer,
        kv_indptr,
        num_kv_splits,
        mtp,
        config["fwd_kernel_stage2"],
    )

