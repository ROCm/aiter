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
):
    cu_id = gl.program_id(0)

    work_start = gl.load(work_indptr + cu_id)
    work_end = gl.load(work_indptr + cu_id + 1)

    if work_end <= work_start:
        return

    blocked_q_nope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 512 in
        size_per_thread=[1, 128],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    blocked_q_rope_mk: gl.constexpr = gl.BlockedLayout(  # max 64 * 64 in
        size_per_thread=[1, 128],  # 64 * 576
        threads_per_warp=[16, 4],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    blocked_k_nope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[16, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )
    blocked_k_rope_kn: gl.constexpr = gl.BlockedLayout(  # max 16 * 576 for per wave
        size_per_thread=[16, 4],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, 4],
        order=[0, 1],
    )

    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    mfma_layout_qk: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True, warps_per_cta=[2, 2]
    )
    mfma_layout_kv: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16], transposed=True, warps_per_cta=[2, 2]
    )
    dot_q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
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

        cur_head = gl.arange(0, BLOCK_H)
        mask_h = cur_head < VALID_BLOCK_H
        mask_h = mask_h & (cur_head < q_head_num)

        offs_c = gl.arange(0, BLOCK_C)
        offs_qk_r = gl.arange(kv_lora_rank, kv_lora_rank + BLOCK_R)  # to get the k_pe

        off_q_pe = (
            cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_qk_r[None, :]
        )
        offs_q = cur_batch * stride_qb + cur_head[:, None] * stride_qh + offs_c[None, :]

        mask_c = offs_c < kv_lora_rank
        mask_qk_r = offs_qk_r < (kv_lora_rank + qk_rope_head_dim)

        # cur_batch_kv = cur_batch # // (max_qo_len // 2)
        # cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch_kv)
        # cur_batch_seq_len = gl.load(kv_indptr + cur_batch_kv + 1) - cur_batch_kv_start_idx

        q = gl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_c[None, :]), other=0.0)
        q_pe = gl.load(
            Q + off_q_pe, mask=(mask_h[:, None]) & (mask_qk_r[None, :]), other=0.0
        )

        e_max = gl.zeros([BLOCK_H], dtype=gl.float32) - float("inf")
        e_sum = gl.zeros([BLOCK_H], dtype=gl.float32)
        acc = gl.zeros([BLOCK_H, BLOCK_C], dtype=gl.float32)


