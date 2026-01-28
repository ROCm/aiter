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


@gluon.jit
def _fwd_grouped_kernel_stage1_n16x2_prefetch_k(
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
        size_per_thread=[2, 8],  # 64 * 512
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

    blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[2, 4],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
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
        operand_index=0, parent=mfma_layout_kv, k_width=16
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=16
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

        smem_kv1.store(tl.trans(kv1).T)
        smem_kv2.store(tl.trans(kv2).T)

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

        # smem_kv1._keep_alive()
        # smem_kv2._keep_alive()

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk *= sm_scale

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))
        cur_k1 = smem_kv1.load(layout=dot_v_layout)
        cur_k2 = smem_kv2.load(layout=dot_v_layout)

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

    # smem_kv1 = smem_kv1.permute([1, 0])
    # smem_kv2 = smem_kv2.permute([1, 0])
    smem_kv1 = smem_kv1._reinterpret(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
    smem_kv2 = smem_kv2._reinterpret(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)

    # smem_kv1.store(kv1)
    # smem_kv2.store(kv2)
    smem_kv1.store(tl.trans(kv1).T)
    smem_kv2.store(tl.trans(kv2).T)

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
    smem_p.store(p.to(Q.dtype.element_ty))

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
def _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64(
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
    stride_buf_kh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_lse_b,
    stride_mid_lse_h,
    stride_mid_lse_s,
    stride_b_block_table,
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
    PAGE_BLOCK_SIZE: gl.constexpr,
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
    cur_batch_kv_end_idx = gl.load(kv_indptr + cur_batch_kv + 1)

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        size_per_thread=[1, 16],  # 64 * 512
        threads_per_warp=[16, 4],
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
        size_per_thread=[4, 16],  # 64 * 512
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[1, 16],  # 64 * 576
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_k_rope: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_p: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[0, 1]
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
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_p
    )

    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )

    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_qk, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_qk, k_width=16
    )

    dot_p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout_kv, k_width=16
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=16
    )

    kv_itt_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1, 0], [2, 0], [0, 1], [0, 2], [0, 4], [0, 8]],
                lane_bases=[[0, 16], [0, 32], [4, 0], [8, 0], [16, 0], [32, 0]],
                warp_bases=[[0, 64], [0, 128]],
                block_bases=[],
                shape=[64, 256],
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

    cur_N = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
    )

    cur_N_pe = gl.arange(
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

    cur_batch_seq_len = cur_batch_kv_end_idx - cur_batch_kv_start_idx
    cur_batch_block_nums = gl.cdiv(cur_batch_seq_len, PAGE_BLOCK_SIZE)

    off_q_pe = (
        cur_batch * stride_qb + cur_head_pe[:, None] * stride_qh + offs_qk_r[None, :]
    )
    offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

    mask_c = offs_c < kv_lora_rank
    mask_k_c = offs_k_c < kv_lora_rank
    mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)
    mask_k_r = offs_k_r < (kv_lora_rank + qk_rope_head_dim)

    blocks_per_split = gl.cdiv(cur_batch_block_nums, NUM_KV_SPLITS)
    split_kv_start = blocks_per_split * split_kv_id
    
    # if pid == 4:
    #     tl.device_print("split_kv_start   ", split_kv_start)
    #     tl.device_print("cur_batch_seq_len   ", cur_batch_seq_len)

    if split_kv_start * PAGE_BLOCK_SIZE > cur_batch_seq_len:
        return

    split_kv_end = gl.minimum(split_kv_start + blocks_per_split, cur_batch_block_nums)

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
    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_batch * stride_b_block_table,
    )

    # if pid == 4:
    #     tl.device_print("kv_loc ", kv_loc)

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
        K_Buffer.type.element_ty, [qk_rope_head_dim, BLOCK_N], layout=shared_k_rope
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
    k_id = kv_loc * PAGE_BLOCK_SIZE + cur_N
    mask_k_id = split_kv_start * PAGE_BLOCK_SIZE + cur_N
    mask_k = mask_k_id < cur_batch_seq_len
    offs_buf_kv = (k_id[:, None]) * stride_buf_kh + offs_k_c[None, :]
    # if pid == 2:
    #     tl.device_print("k_id  ", mask_k_id)
    #     tl.device_print("mask_k   ", mask_k)

    kv1 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv,
        mask=mask_k[:, None] & mask_k_c[None, :]
    )  # the shared latent tensor for keys and values

    kv2 = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_kv + 256,
        mask=mask_k[:, None] & mask_k_c[None, :]
    )  # the shared latent tensor for keys and values

    # if pid == 1:
    #     tl.device_print("kv2 before ", kv2.to(gl.float32))

    smem_kv1 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )
    smem_kv2 = gl.allocate_shared_memory(
        K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k
    )

    gl.amd.cdna3.sched_barrier(0x0)

    smem_kv1.store(kv1.T)
    smem_kv2.store(kv2.T)

    k_id_pe = kv_loc * PAGE_BLOCK_SIZE + cur_N_pe
    offs_buf_k_pe = k_id_pe[:, None] * stride_buf_kh + offs_k_r[None, :]
    mask_k_id = split_kv_start * PAGE_BLOCK_SIZE + cur_N_pe
    mask_k_pe = mask_k_id < cur_batch_seq_len 

    # if pid == 2:
    #     tl.device_print("cur_batch_kv_end_idx before ", cur_batch_kv_end_idx)

    k_pe = gl.amd.cdna3.buffer_load(
        ptr=K_Buffer,
        offsets=offs_buf_k_pe,
        mask=mask_k_pe[:, None] & mask_k_r[None, :]
    )  # positional embedding part of keys

    # if pid == 2:
    #     tl.device_print("k_pe before ", k_pe)

    e_sum = gl.arange(0, BLOCK_H, layout = gl.SliceLayout(1, mfma_layout_qk)).to(gl.float32) * float(0)
    e_max = e_sum - float("inf")

    # cur_k1 = gl.convert_layout(kv1.T, layout=dot_k_layout)
    # cur_k2 = gl.convert_layout(kv2.T, layout=dot_k_layout)

    cur_k1 = smem_kv1.load(layout=dot_k_layout)
    cur_k2 = smem_kv2.load(layout=dot_k_layout)

    gl.amd.cdna3.sched_barrier(0x0)
    smem_kv1 = smem_kv1._reinterpret(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
    kv1_transpose = gl.convert_layout(kv1, kv_itt_layout)
    gl.amd.cdna3.sched_barrier(0x0)

    smem_kv1.store(kv1_transpose)
    smem_kv2 = smem_kv2._reinterpret(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
    kv2_transpose = gl.convert_layout(kv2, kv_itt_layout)
    gl.amd.cdna3.sched_barrier(0x0)

    smem_kv2.store(kv2_transpose)

    smem_k_rope.store(k_pe.T)
    gl.amd.cdna3.sched_barrier(0x0)
    split_kv_start += 1 

    mask_qk_h = gl.arange(0, BLOCK_H, gl.SliceLayout(1, mfma_layout_qk))
    # if pid == 1:
    #     tl.device_print("split_kv_start  ", split_kv_start)
    #     tl.device_print("split_kv_end  ", split_kv_end)

    for start_n in range(split_kv_start, split_kv_end):
        kv_loc = gl.load(
            kv_indices + start_n + cur_batch * stride_b_block_table,
        )

        cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

        gl.amd.cdna3.sched_barrier(0x0)
        k_id = kv_loc * PAGE_BLOCK_SIZE + cur_N
        offs_buf_kv = k_id[:, None] * stride_buf_kh + offs_k_c[None, :]
        mask_k_id = start_n * PAGE_BLOCK_SIZE + cur_N
        mask_k = mask_k_id < cur_batch_seq_len
        gl.amd.cdna3.sched_barrier(0x0)

        qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
        kv1 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv,
            mask=mask_k[:, None] & mask_k_c[None, :]
        )

        gl.amd.cdna3.sched_barrier(0x0)

        qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)
        kv2 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv + 256,
            mask=mask_k[:, None] & mask_k_c[None, :]
        )

        # smem_kv1 = smem_kv1.permute([1, 0])
        # smem_kv2 = smem_kv2.permute([1, 0])

        cur_k1 = smem_kv1.load(layout=dot_v_layout)
        cur_k2 = smem_kv2.load(layout=dot_v_layout)

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk *= sm_scale
        # if pid == 0:
        #     tl.device_print("qk before ", qk)
        # mask_qk_n = (start_n - 1) * PAGE_BLOCK_SIZE + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout_qk))
        # qk = tl.where(
        #     (mask_qk_n[None, :] < cur_batch_seq_len) & (mask_qk_h[:, None] >= 0), qk, float("-inf")
        # )
        # if pid == 0:
        #     tl.device_print("qk after ", qk)
        

        k_id_pe = kv_loc * PAGE_BLOCK_SIZE + cur_N_pe
        offs_buf_k_pe = k_id_pe[:, None] * stride_buf_kh + offs_k_r[None, :]
        mask_k_id = start_n * PAGE_BLOCK_SIZE + cur_N_pe
        mask_k_pe = mask_k_id < cur_batch_seq_len

        gl.amd.cdna3.sched_barrier(0x0)
        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
            mask=mask_k_pe[:, None] & mask_k_r[None, :]
        )  # positional embedding part of keys


        # smem_kv1._keep_alive()
        # smem_kv2._keep_alive()

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))
        n_e_max = gl.maximum(n_e_max, e_max)

        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))
        gl.amd.cdna3.sched_barrier(0x0)

        cur_p = smem_p.load(layout=dot_p_layout)
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)

        smem_kv1.store(kv1.T)
        acc1 = acc1 * re_scale[:, None]
        acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
        e_sum = e_sum * re_scale + gl.sum(p, 1)

        smem_kv2.store(kv2.T)
        acc2 = acc2 * re_scale[:, None]
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)
        smem_k_rope.store(k_pe.T)
        e_max = n_e_max

        cur_k1 = smem_kv1.load(layout=dot_k_layout)
        kv1_transpose = gl.convert_layout(kv1, kv_itt_layout)
        gl.amd.cdna3.sched_barrier(0x0)
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)

        smem_kv1.store(kv1_transpose)
        cur_k2 = smem_kv2.load(layout=dot_k_layout)

        kv2_transpose = gl.convert_layout(kv2, kv_itt_layout)
        gl.amd.cdna3.sched_barrier(0x0)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
        smem_kv2.store(kv2_transpose)

    # smem_kv1 = smem_kv1._reinterpret(
    #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
    # smem_kv2 = smem_kv2._reinterpret(
    #     K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
    #
    # # kv1_transpose = tl.trans(kv1)
    # # smem_kv1.store(kv1_transpose.T)
    # # kv2_transpose = tl.trans(kv2)
    # # smem_kv2.store(kv2_transpose.T)

    qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
    qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

    cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

    qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

    # if pid == 5:
    #     tl.device_print("qk", qk)

    cur_k1 = smem_kv1.load(layout=dot_v_layout)
    cur_k2 = smem_kv2.load(layout=dot_v_layout)

    qk *= sm_scale
    # if pid == 1:
    #     tl.device_print("qk before ", (split_kv_end - 1) * PAGE_BLOCK_SIZE)
    mask_qk_n = (split_kv_end - 1) * PAGE_BLOCK_SIZE + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout_qk))

    qk = tl.where(
        (mask_qk_n[None, :] < cur_batch_seq_len) & (mask_qk_h[:, None] >= 0), qk, float("-inf")
    )
    # if pid == 1:
    #     tl.device_print("qk after", qk)

    n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

    n_e_max = gl.maximum(n_e_max, e_max)
    re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
    p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)

    # if pid == 5:
    #     tl.device_print("p", p)

    smem_p.store(p.to(Q.dtype.element_ty))

    acc1 = acc1 * re_scale[:, None]
    acc2 = acc2 * re_scale[:, None]
    e_sum = e_sum * re_scale + gl.sum(p, 1)
    gl.amd.cdna3.sched_barrier(0x0)
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

    # if pid == 5:
    #     tl.device_print("acc", acc)

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
        ptr=Att_Out,
        offsets=offs_mid_o,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    mask_lse = offs_om < (cur_head_id + 1) * VALID_BLOCK_H
    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        mask=mask_lse,
    )


@gluon.jit
def _fwd_grouped_kernel_stage1_n16x4_prefetch_k(
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
        size_per_thread=[1, 16],  # 64 * 512
        threads_per_warp=[16, 4],
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
        size_per_thread=[4, 16],  # 64 * 512
        threads_per_warp=[16, 4],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    blocked_ld_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[1, 16],  # 64 * 576
        threads_per_warp=[64, 1],
        warps_per_cta=[1, 4],
        order=[1, 0],
    )

    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=8, order=[1, 0]
    )
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[0, 1]
    )
    shared_p: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[0, 1]
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
        Q.type.element_ty, [BLOCK_H, BLOCK_N], layout=shared_p
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
        operand_index=0, parent=mfma_layout_kv, k_width=16
    )
    dot_v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout_kv, k_width=16
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

    cur_N = gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(1, blocked_ld_k_nope_kn)
    )

    cur_N_pe = gl.arange(
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
        kv_indices + split_kv_start + cur_batch_kv_start_idx + cur_N_pe,
    )

    kv_loc = gl.load(
        kv_indices + split_kv_start + cur_batch_kv_start_idx + cur_N,
    )
    # if pid == 5:
    #     tl.device_print("kv_loc", kv_loc)

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
        kv_loc = gl.load(
            kv_indices + start_n + cur_batch_kv_start_idx + cur_N,
        )
        kv_pe_loc = gl.load(
            kv_indices + start_n + cur_N_pe + cur_batch_kv_start_idx,
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


        kv1 = tl.trans(kv1)
        smem_kv1.store(kv1.T)
        kv2 = tl.trans(kv2)
        smem_kv2.store(kv2.T)

        offs_buf_kv = kv_loc[:, None] * stride_buf_kbs + offs_k_c[None, :]

        kv1 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv,
        )

        cur_k1 = smem_kv1.load(layout=dot_v_layout)
        cur_k2 = smem_kv2.load(layout=dot_v_layout)

        qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

        qk *= sm_scale

        kv2 = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_kv + 256,
        )

        offs_buf_k_pe = kv_pe_loc[:, None] * stride_buf_kbs + offs_k_r[None, :]
        k_pe = gl.amd.cdna3.buffer_load(
            ptr=K_Buffer,
            offsets=offs_buf_k_pe,
        )  # positional embedding part of keys


        # smem_kv1._keep_alive()
        # smem_kv2._keep_alive()

        n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))

        n_e_max = gl.maximum(n_e_max, e_max)
        re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
        p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)
        smem_p.store(p.to(q0.dtype))

        cur_p = smem_p.load(layout=dot_p_layout)
        smem_kv1 = smem_kv1._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)
        smem_kv2 = smem_kv2._reinterpret(
            K_Buffer.type.element_ty, [kv_lora_rank // 2, BLOCK_N], layout=shared_k)

        smem_kv1.store(kv1.T)
        acc1 = acc1 * re_scale[:, None]
        acc1 = gl.amd.cdna3.mfma(cur_p, cur_k1, acc1)
        e_sum = e_sum * re_scale + gl.sum(p, 1)

        smem_kv2.store(kv2.T)
        acc2 = acc2 * re_scale[:, None]
        acc2 = gl.amd.cdna3.mfma(cur_p, cur_k2, acc2)
        smem_k_rope.store(k_pe.T)
        e_max = n_e_max


        cur_k1 = smem_kv1.load(layout=dot_k_layout)
        cur_k2 = smem_kv2.load(layout=dot_k_layout)

    # smem_kv1 = smem_kv1.permute([1, 0])
    # smem_kv2 = smem_kv2.permute([1, 0])
    smem_kv1 = smem_kv1._reinterpret(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)
    smem_kv2 = smem_kv2._reinterpret(
        K_Buffer.type.element_ty, [BLOCK_N, kv_lora_rank // 2], layout=shared_v)

    kv1 = tl.trans(kv1)
    smem_kv1.store(kv1.T)
    kv2 = tl.trans(kv2)
    smem_kv2.store(kv2.T)

    qk = gl.amd.cdna3.mfma(q0, cur_k1, zeros)
    qk = gl.amd.cdna3.mfma(q1, cur_k2, qk)

    cur_k_pe = smem_k_rope.load(layout=dot_k_layout)

    qk = gl.amd.cdna3.mfma(q_pe, cur_k_pe, qk)

    # if pid == 5:
    #     tl.device_print("qk", qk)

    cur_k1 = smem_kv1.load(layout=dot_v_layout)
    cur_k2 = smem_kv2.load(layout=dot_v_layout)

    qk *= sm_scale

    n_e_max = gl.convert_layout(gl.max(qk, 1), gl.SliceLayout(1, mfma_layout_qk))


    n_e_max = gl.maximum(n_e_max, e_max)
    re_scale = tl.math.exp2((e_max - n_e_max) * log2e)
    p = tl.math.exp2((qk - n_e_max[:, None]) * log2e)

    # if pid == 5:
    #     tl.device_print("p", p)

    smem_p.store(p.to(Q.dtype.element_ty))

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

    # if pid == 5:
    #     tl.device_print("acc", acc)

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
        ptr=Att_Out,
        offsets=offs_mid_o,
        mask=(mask_h[:, None]) & (mask_c[None, :]),
    )

    offs_mid_lse = (
        cur_batch * stride_mid_lse_b
        + offs_om * stride_mid_lse_h
        + split_kv_id * stride_mid_lse_s 
    )

    mask_lse = offs_om < (cur_head_id + 1) * VALID_BLOCK_H
    gl.amd.cdna3.buffer_store(
        stored_value=e_max - gl.log(e_sum),
        ptr=Att_Lse,
        offsets=offs_mid_lse,
        mask=mask_lse,
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
    block_tables,
    num_kv_splits,
    sm_scale,
    logit_cap,
    mtp,
    config,
):
    qk_rope_head_dim = q.shape[-1] - kv_lora_rank
    # batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]
    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]
    page_block_size = k_buffer.shape[1]

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

    
    if page_block_size == 1:
        _fwd_grouped_kernel_stage1_n16x4_prefetch_k[grid](
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
    elif page_block_size == 64:
        # import pdb; pdb.set_trace()
        config["PAGE_BLOCK_SIZE"] = page_block_size
        _fwd_grouped_kernel_stage1_n16x4_prefetch_k_paged_64[grid](
            q,
            k_buffer,
            v_buffer,
            sm_scale,
            kv_indptr,
            block_tables,
            att_out,
            att_lse,
            q.stride(0),
            q.stride(1),
            k_buffer.stride(-3),
            k_buffer.stride(-2),
            att_out.stride(0),
            att_out.stride(1),
            att_out.stride(2),
            att_lse.stride(0),
            att_lse.stride(1),
            att_lse.stride(2),
            block_tables.stride(0),
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
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
    block_tables: torch.Tensor,
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
        block_tables,
        num_kv_splits,
        sm_scale,
        logit_cap,
        mtp,
        config["fwd_grouped_kernel_stage1_rope_fp8"],
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


