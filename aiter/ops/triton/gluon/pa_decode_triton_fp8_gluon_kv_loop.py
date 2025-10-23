# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional
import tempfile
import subprocess
import os

import triton
import triton.language as tl
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import compile
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import torch
import aiter
from aiter.test_common import perftest


TEST_NUM_ITERS = 101

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

# _SEQ_PARTITION_SIZE = 256
# _SEQ_PARTITION_SIZE = 128
# _SEQ_PARTITION_SIZE = 1024
# _SEQ_PARTITION_SIZE = 512
_SEQ_PARTITION_SIZE = 256


@gluon.jit
def pa_decode_v2_gluon_big_blk_fp8(
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    softmax_scale,
    q_scale,            # [num_seqs, num_kv_heads * query_grp_sz, 1]
    k_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
    v_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_nh,
    stride_v_hz,
    stride_bt_s,
    q_scale_stride0,
    kv_scale_stride0,
    kv_scale_stride1,
    Q_SEQ_LEN: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ_ORI: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    KV_16B_ELE_NUM: tl.constexpr,
    Q_QUANT_MODE: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TRANS_V: tl.constexpr,          # [num_blks, num_kv_heads, kv_blk_sz/x, head_sz, x]
    IS_CAUSAL: tl.constexpr,
):
    """
    #TODO: Add Doc
    """
    gl.static_assert(Q_SEQ_LEN <= 4, "Q_SEQ_LEN={}, Do not support Q_SEQ_LEN > 4".format(Q_SEQ_LEN))
    gl.static_assert(QUERY_GRP_SZ_POW2 <= 64, "QUERY_GRP_SZ_POW2={}, Do not support QUERY_GRP_SZ_POW2 > 64".format(QUERY_GRP_SZ_POW2))
    gl.static_assert(SEQ_PARTITION_SZ == 256, "SEQ_PARTITION_SZ={}, Only support SEQ_PARTITION_SZ == 256".format(SEQ_PARTITION_SZ))
    gl.static_assert(KV_BLK_SZ == 1024, "KV_BLK_SZ={}, Only support KV_BLK_SZ == 1024".format(KV_BLK_SZ))
    gl.static_assert(q_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(k_cache_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(v_cache_ptr.dtype.element_ty == gl.float8e4b8)
    if Q_QUANT_MODE >= 0:
        gl.static_assert(q_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(k_scale.dtype.element_ty == gl.float32)
        gl.static_assert(v_scale.dtype.element_ty == gl.float32)

    # # 0 for per_tensor quant
    # # 1 for per_token quant
    # # Q_QUANT_MODE: gl.constexpr = 0
    # Q_QUANT_MODE: gl.constexpr = 1
    # # KV_QUANT_MODE: gl.constexpr = 0
    # KV_QUANT_MODE: gl.constexpr = 1

    log2e: gl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: gl.constexpr = KV_16B_ELE_NUM
    K_HEAD_SZ_POW2_SPLIT: gl.constexpr = HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD
    # # KV_COMPUTE_BLOCK_SIZE: gl.constexpr = 128
    # KV_COMPUTE_BLOCK_SIZE: gl.constexpr = 256

    # ==================== Layout Definitions ====================
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta   =[4, 1],
        order           =[1, 0],
    )

    # # [K_HEAD_SZ_POW2_SPLIT, KV_COMPUTE_BLOCK_SIZE, CONTIGUOUS_KV_ELEMS_16B_LOAD] x fp8
    # blocked_k2: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases=((0, 0, 1), (0, 0, 2), (0, 0, 4), (0, 0, 8), (4, 0, 0), (0, 64, 0), (0, 128, 0)),
    #     lane_bases=((0, 1, 0), (0, 2, 0), (0, 4, 0), (0, 8, 0), (1, 0, 0), (2, 0, 0)),
    #     warp_bases=((0, 16, 0), (0, 32, 0)),
    #     block_bases=[],
    #     shape=[K_HEAD_SZ_POW2_SPLIT, KV_COMPUTE_BLOCK_SIZE, CONTIGUOUS_KV_ELEMS_16B_LOAD],
    # )
    # [K_HEAD_SZ_POW2_SPLIT, KV_COMPUTE_BLOCK_SIZE, CONTIGUOUS_KV_ELEMS_16B_LOAD] x fp8
    blocked_k2: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 1, 16],
        threads_per_warp=[4, 16, 1],
        warps_per_cta   =[1, 4, 1],
        order           =[2, 1, 0],
    )
    blocked_k: gl.constexpr = blocked_k2

    # transposed: indicates the result tensor is transposed so that each thread holds consecutive elements
    # in the same row instead of column, which is good for chained dot and global write.
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    if QUERY_GRP_SZ_POW2 == 16:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (0,128))
    elif QUERY_GRP_SZ_POW2 == 32:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (16,0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (0,128), (16,0))
    elif QUERY_GRP_SZ_POW2 == 64:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (16,0), (32,0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (0,128), (16,0), (32,0))

    qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=((1,0), (2,0), (4,0), (8,0), (0,4), (0,8)),
        warp_bases=((0,16), (0,32)),
        block_bases=[],
        shape=[QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE],
    )

    if TRANS_V:
        # [KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMS_16B_LOAD, HEAD_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
        blocked_v_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread =[1, 1, 16],
            threads_per_warp=[4, 16, 1],
            warps_per_cta   =[1, 4, 1],
            order           =[2, 1, 0],
        )
        v_dim0_offs = gl.arange(0, KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
        v_dim1_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_v_layout)))
        v_dim2_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_v_layout)))
    else:
        # # [HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE]
        # blocked_v_layout: gl.constexpr = gl.DistributedLinearLayout(
        #     # reg_bases=((0, 1), (0, 2), (0, 4), (0, 8), (64, 0), (0, 64), (0, 128)),
        #     reg_bases=((0, 1), (0, 2), (0, 4), (0, 8), (0, 64), (0, 128), (64, 0)),
        #     lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 16), (0, 32)),
        #     warp_bases=((16, 0), (32, 0)),
        #     block_bases=[],
        #     shape=[HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE],
        # )
        # [HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE]
        blocked_v_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread =[1, 16],
            threads_per_warp=[16, 4],
            warps_per_cta   =[4, 1],
            order           =[1, 0],
        )
        v_dim0_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(1, blocked_v_layout))
        v_dim1_offs = gl.arange(0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, blocked_v_layout))

    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    pv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # blocked_q dim0
    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    # blocked_q dim1
    head_sz_layout: gl.constexpr = gl.SliceLayout(0, blocked_q)

    # blocked_k dim0
    head_sz_div_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blocked_k))
    # blocked_k dim1
    blk_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_k))
    # blocked_k dim2
    contiguous_kv_elems_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_k))

    q_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=query_grp_sz_layout)
    head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=head_sz_layout)

    kv_scale_col_offs = gl.arange(0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout))

    head_sz_div_offs = gl.arange(0, K_HEAD_SZ_POW2_SPLIT, layout=head_sz_div_layout)
    blk_offs = gl.arange(0, KV_COMPUTE_BLOCK_SIZE, layout=blk_layout)
    contiguous_kv_elems_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=contiguous_kv_elems_layout)

    qk_row_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_linear_layout))

    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    seq_part_idx = gl.program_id(2)
    page_offset = 0
    if seq_part_idx % 4 == 1:
        page_offset = 1 * SEQ_PARTITION_SZ
    elif seq_part_idx % 4 == 2:
        page_offset = 2 * SEQ_PARTITION_SZ
    elif seq_part_idx % 4 == 3:
        page_offset = 3 * SEQ_PARTITION_SZ

    q_offs_base = (
        seq_idx * Q_SEQ_LEN * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs_base, mask=q_mask)

    if Q_QUANT_MODE == 0:
        q_scale_val = tl.load(q_scale)
    elif Q_QUANT_MODE == 1:
        q_scale_offs = seq_idx * Q_SEQ_LEN * q_scale_stride0 + kv_head_idx * QUERY_GRP_SZ + qk_row_offs[:, None]
        # [QUERY_GRP_SZ_POW2, 1]
        q_scale_val = gl.amd.cdna3.buffer_load(
            ptr=q_scale, offsets=q_scale_offs, mask=qk_row_offs[:, None] < QUERY_GRP_SZ
        )

    m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_linear_layout))
    m_l_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + m_l_base_offs
    )
    m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ
    o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, pv_mfma_layout))
    o_head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, pv_mfma_layout))
    o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (o_head_sz_offs[None, :] < HEAD_SZ)
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + o_grp_offs[:, None] * stride_logits_g
        + o_head_sz_offs[None, :]
    )

    m_i = m_l_base_offs.to(gl.float32) * float(0.0) - float("inf")
    l_i = m_l_base_offs.to(gl.float32) * float(0.0)
    acc0 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=gl.float32, layout=pv_mfma_layout)


    kv_seq_len = gl.load(seq_lens_ptr + seq_idx)
    kv_seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if kv_seq_start_idx >= kv_seq_len:
        return
    KV_COMPUTE_BLOCK_NUM: gl.constexpr = SEQ_PARTITION_SZ // KV_COMPUTE_BLOCK_SIZE

    for kv_idx in range(KV_COMPUTE_BLOCK_NUM):
        kv_sub_seq_start_idx = kv_seq_start_idx + kv_idx * KV_COMPUTE_BLOCK_SIZE
        blk_tb_id = kv_sub_seq_start_idx // KV_BLK_SZ
        page_offset_i = page_offset + kv_idx * KV_COMPUTE_BLOCK_SIZE

        qk_col_offs = kv_sub_seq_start_idx + gl.arange(0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout))

        # load alibi slopes[QUERY_GRP_SZ_POW2]
        if alibi_slopes is None:
            alibi_slope = gl.zeros([QUERY_GRP_SZ_POW2], dtype=gl.float32)
        else:
            alibi_slope = gl.amd.cdna3.buffer_load(ptr=alibi_slopes + kv_head_idx * QUERY_GRP_SZ, offsets=qk_row_offs, mask=qk_row_offs < QUERY_GRP_SZ)

        blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
        kv_page_id = tl.load(blk_tables_start_ptr + blk_tb_id)
        # tl.static_print(kv_page_id.type)

        # k_blk_offs[K_HEAD_SZ_POW2_SPLIT, KV_COMPUTE_BLOCK_SIZE, CONTIGUOUS_KV_ELEMS_16B_LOAD]
        k_blk_offs = (
            kv_page_id * stride_k_b
            + kv_head_idx * stride_k_nh
            + head_sz_div_offs[:, None, None] * stride_k_hz
            + (page_offset_i + blk_offs)[None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
            + contiguous_kv_elems_offs[None, None, :]
        )
        k = gl.amd.cdna3.buffer_load(ptr=k_cache_ptr, offsets=k_blk_offs)
        if KV_QUANT_MODE >= 0:
            # 0 for per_tensor quant
            # 1 for per_token quant
            if KV_QUANT_MODE == 0:
                k_scale_val = tl.load(k_scale)
                v_scale_val = tl.load(v_scale)
            elif KV_QUANT_MODE == 1:
                # [KV_COMPUTE_BLOCK_SIZE]
                k_scale_offs = kv_page_id * kv_scale_stride0 + kv_head_idx * kv_scale_stride1 + page_offset_i + kv_scale_col_offs
                k_scale_val = gl.amd.cdna3.buffer_load(ptr=k_scale, offsets=k_scale_offs)
                v_scale_val = gl.amd.cdna3.buffer_load(ptr=v_scale, offsets=k_scale_offs)

        k = gl.permute(k, [0, 2, 1])
        k = gl.reshape(k, [HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE])

        if TRANS_V:
            # [KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMS_16B_LOAD, HEAD_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
            v_blk_offs = (
                kv_page_id * stride_v_b
                + kv_head_idx * stride_v_nh
                + (page_offset_i // CONTIGUOUS_KV_ELEMS_16B_LOAD + v_dim0_offs)[:, None, None] * stride_v_hz
                + v_dim1_offs[None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
                + v_dim2_offs[None, None, :]
            )
            # v[KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMS_16B_LOAD, HEAD_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
            v = gl.amd.cdna3.buffer_load(ptr=v_cache_ptr, offsets=v_blk_offs)
            # from v[KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMS_16B_LOAD, HEAD_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
            # to   v[KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMS_16B_LOAD, CONTIGUOUS_KV_ELEMS_16B_LOAD, HEAD_SZ_POW2]
            v = gl.permute(v, [0, 2, 1])
            v = gl.reshape(v, [KV_COMPUTE_BLOCK_SIZE, HEAD_SZ_POW2])
        else:
            # v_blk_offs[HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE]
            v_blk_offs = (
                kv_page_id * stride_v_b
                + kv_head_idx * stride_v_nh
                + v_dim0_offs[:, None] * stride_v_hz
                + (page_offset_i + v_dim1_offs)[None, :]
            )
            # [HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE]
            v = gl.amd.cdna3.buffer_load(ptr=v_cache_ptr, offsets=v_blk_offs)
            # [HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE] --> [KV_COMPUTE_BLOCK_SIZE, HEAD_SZ_POW2]
            v = gl.permute(v, [1, 0])

        accumulator = gl.zeros((QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE), dtype=gl.float32, layout=qk_mfma_layout)


        qc = gl.convert_layout(q, layout=qk_lhs_layout)
        kc = gl.convert_layout(k, layout=qk_rhs_layout)
        qk = gl.amd.cdna3.mfma(qc, kc, accumulator)
        qk = gl.reshape(qk, [QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE])

        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # [KV_COMPUTE_BLOCK_SIZE] --> [1, KV_COMPUTE_BLOCK_SIZE]
                k_scale_val = k_scale_val[None, :]
            if Q_QUANT_MODE >= 0:
                qk_scale_val = softmax_scale * q_scale_val * k_scale_val
            else:
                qk_scale_val = softmax_scale * k_scale_val
        else:
            if Q_QUANT_MODE >= 0:
                qk_scale_val = softmax_scale * q_scale_val
            else:
                qk_scale_val = softmax_scale

        qk = qk_scale_val * qk

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (qk_col_offs - kv_seq_len + 1)[None, :]).to(gl.float32)

        qk_bound_mask = (qk_row_offs[:, None] < QUERY_GRP_SZ)
        if IS_CAUSAL:
            # causal_mask = qk_col_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
            seq_m_extand = Q_SEQ_LEN - 1 - qk_row_offs // QUERY_GRP_SZ_ORI
            causal_mask = seq_m_extand[:, None] + qk_col_offs[None, :] < kv_seq_len
        else:
            causal_mask = qk_col_offs[None, :] < kv_seq_len
        qk_bound_mask = qk_bound_mask & causal_mask
        # if [0, SEQ_PARTITION_SZ) are all -inf, the result will be nan
        # so, we use -1e37 other than -inf
        # qk = tl.where(qk_bound_mask, qk, float("-inf"))
        qk = tl.where(qk_bound_mask, qk, float(-1e38))

        curr_m = gl.max(qk, axis=1)
        m_i_new = gl.maximum(m_i, curr_m)
        # acc_scale = tl.math.exp2(m_i - m_i_new)
        acc_scale = tl.math.exp2((m_i - m_i_new) * log2e)

        # p[QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE]
        # p = tl.math.exp2((qk - m_i_new[:, None]))
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        l_i = acc_scale * l_i + gl.sum(p, axis=1)

        if v.dtype.is_fp8():
            # 1 for per_token quant
            if KV_QUANT_MODE == 1:
                # vs_1d = gl.reshape(v_scale_val, [KV_COMPUTE_BLOCK_SIZE])
                vs_max = gl.max(v_scale_val, axis=0)
                v_scale_val = v_scale_val * float(FP8_MAX) / vs_max
                p = v_scale_val[None, :] * p
                p_scale = vs_max / float(FP8_MAX)
        # p_scale = 1.0

        if CONTIGUOUS_KV_ELEMS_16B_LOAD == 16:
            # kv element is 1 bytes(fp8)
            p = p.to(v_cache_ptr.dtype.element_ty)
        else:
            # kv element is 2 bytes(fp16, bf16)
            p = p.to(COMPUTE_TYPE)

        pc = gl.convert_layout(p, layout=pv_lhs_layout)
        vc = gl.convert_layout(v, layout=pv_rhs_layout)

        # acc_scale = l_i * 0 + alpha  # Workaround some compiler bug
        acc_scale = gl.convert_layout(acc_scale[:, None], layout=pv_mfma_layout)
        acc0 *= acc_scale
        # acc0 = gl.amd.cdna3.mfma(pc, vc, acc0)
        # acc0 = p_scale * acc0
        accumulator1 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=gl.float32, layout=pv_mfma_layout)
        acc = gl.amd.cdna3.mfma(pc, vc, accumulator1)
        acc0 += p_scale * acc

        m_i = m_i_new

    exp_sum = 1.0 / l_i
    exp_sum_cvt = gl.convert_layout(exp_sum[:, None], layout=pv_mfma_layout)
    acc0 = acc0 * exp_sum_cvt
    acc0 = acc0.to(COMPUTE_TYPE)
    gl.amd.cdna3.buffer_store(stored_value=m_i, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=l_i, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=acc0, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)


# @triton.autotune(
#     configs=[
#         triton.Config({'matrix_instr_nonkdim' : dim, 'waves_per_eu' : wa}, num_stages=s, num_warps=w) \
#         for s in [1, 2, 3, 4, 5, 6, 7, 8] \
#         for w in [4] \
#         for wa in [1, 2, 3, 4] \
#         for dim in [16] \
#     ],
#     key = ['Q_SEQ_LEN', 'QUERY_GRP_SZ_POW2', 'KV_BLK_SZ'],
# )
@gluon.jit
def pa_decode_v2_gluon_fp8(
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    softmax_scale,
    q_scale,            # [num_seqs, num_kv_heads * query_grp_sz, 1]
    k_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
    v_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_nh,
    stride_v_hz,
    stride_bt_s,
    q_scale_stride0,
    kv_scale_stride0,
    kv_scale_stride1,
    Q_SEQ_LEN: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ_ORI: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    KV_16B_ELE_NUM: tl.constexpr,
    Q_QUANT_MODE: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TRANS_V: tl.constexpr,          # [num_blks, num_kv_heads, kv_blk_sz/x, head_sz, x]
    IS_CAUSAL: tl.constexpr,
):
    """
    Gluon version of paged attention decode kernel with FP8/BF16 support
    """
    gl.static_assert(Q_SEQ_LEN <= 4, "Q_SEQ_LEN={}, Do not support Q_SEQ_LEN > 4".format(Q_SEQ_LEN))
    gl.static_assert(QUERY_GRP_SZ_POW2 <= 64, "QUERY_GRP_SZ_POW2={}, Do not support QUERY_GRP_SZ_POW2 > 64".format(QUERY_GRP_SZ_POW2))
    gl.static_assert(KV_BLK_SZ == 16 or KV_BLK_SZ == 64, "KV_BLK_SZ={}, Only support KV_BLK_SZ in [16, 64]".format(KV_BLK_SZ))
    gl.static_assert(q_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(k_cache_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(v_cache_ptr.dtype.element_ty == gl.float8e4b8)
    if Q_QUANT_MODE >= 0:
        gl.static_assert(q_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(k_scale.dtype.element_ty == gl.float32)
        gl.static_assert(v_scale.dtype.element_ty == gl.float32)
    # gl.static_print(k_cache_ptr.dtype)
    # tl.static_print(k_cache_ptr.dtype.element_ty)
    # tl.static_print(gl.float8e4b8)
    # tl.static_print(k_cache_ptr.dtype.element_ty == gl.float8e4b8)
    # tl.static_print(q_scale.dtype.element_ty == gl.float32)

    # # 0 for per_tensor quant
    # # 1 for per_token quant
    # # Q_QUANT_MODE: gl.constexpr = 0
    # Q_QUANT_MODE: gl.constexpr = 1
    # # KV_QUANT_MODE: gl.constexpr = 0
    # KV_QUANT_MODE: gl.constexpr = 1

    # # KV_COMPUTE_BLOCK_SIZE: gl.constexpr = 128
    # KV_COMPUTE_BLOCK_SIZE: gl.constexpr = 256

    log2e: gl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: gl.constexpr = KV_16B_ELE_NUM
    K_HEAD_SZ_POW2_SPLIT: gl.constexpr = HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD
    MAX_NUM_KV_BLKS: gl.constexpr = KV_COMPUTE_BLOCK_SIZE // KV_BLK_SZ

    # ==================== Layout Definitions ====================
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta   =[4, 1],
        order           =[1, 0],
    )
    # MAX_NUM_KV_BLKS x K_HEAD_SZ_POW2_SPLIT x KV_BLK_SZ x CONTIGUOUS_KV_ELEMS_16B_LOAD
    # 16 x 16 x 16 x 8 x fp16
    blocked_k1: gl.constexpr = gl.DistributedLinearLayout( # fp16
        reg_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,0,1,0), (0,1,0,0), (0,8,0,0), (8,0,0,0)), # 16 x 8
        lane_bases=((0,0,2,0), (0,0,4,0), (0,0,8,0), (1,0,0,0), (0,2,0,0), (0,4,0,0)), # 64
        warp_bases=((2,0,0,0), (4,0,0,0)), # 4
        block_bases=[], # 8
        shape=[MAX_NUM_KV_BLKS, 16, 16, 8],
    )
    # 16 x 8 x 16 x 16 x fp8
    # blocked_k2: gl.constexpr = gl.DistributedLinearLayout( # fp8
    #     reg_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,0,0,8), (0,0,1,0), (0,4,0,0), (8,0,0,0)), # 16 x 8
    #     lane_bases=((0,0,2,0), (0,0,4,0), (0,0,8,0), (1,0,0,0), (0,1,0,0), (0,2,0,0)), # 64
    #     warp_bases=((2,0,0,0), (4,0,0,0)), # 4
    #     block_bases=[], # 8
    #     shape=[MAX_NUM_KV_BLKS, 8, 16, 16],
    # )
    # blocked_k2: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread =[1, 1, 2, 16],
    #     threads_per_warp=[2, 4, 8, 1],
    #     warps_per_cta   =[4, 1, 1, 1],
    #     order           =[3, 2, 0, 1],
    # )
    blocked_k2: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 1, 1, 16],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta   =[4, 1, 1, 1],
        order           =[3, 2, 1, 0],
    )
    blocked_k: gl.constexpr = blocked_k1 if CONTIGUOUS_KV_ELEMS_16B_LOAD == 8 else blocked_k2

    # transposed: indicates the result tensor is transposed so that each thread holds consecutive elements
    # in the same row instead of column, which is good for chained dot and global write.
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )
    # qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases=((0,1), (0,2), (0,4), (0,128)), # 16 x 8
    #     lane_bases=((1,0), (2,0), (4,0), (8,0), (0,8), (0,16)), # 64
    #     warp_bases=((0,32), (0,64)), # 4
    #     block_bases=[], # 8
    #     shape=[16, KV_COMPUTE_BLOCK_SIZE],
    # )

    if QUERY_GRP_SZ_POW2 == 16:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (0,128))
    elif QUERY_GRP_SZ_POW2 == 32:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (16,0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (0,128), (16,0))
    elif QUERY_GRP_SZ_POW2 == 64:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (16,0), (32,0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            reg_bases: gl.constexpr = ((0,1), (0,2), (0,64), (0,128), (16,0), (32,0))

    qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=((1,0), (2,0), (4,0), (8,0), (0,4), (0,8)),
        warp_bases=((0,16), (0,32)),
        block_bases=[],
        shape=[QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE],
    )
    # # Layout mismatch with qk_linear_layout
    # qk_linear_layout: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread =[1, 4],
    #     threads_per_warp=[16, 4],
    #     warps_per_cta   =[1, 4],
    #     order           =[0, 1],
    # )

    if TRANS_V:
        # [MAX_NUM_KV_BLKS, 1, 128, 16]
        blocked_v_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread =[1, 1, 1, 16],
            threads_per_warp=[4, 1, 16, 1],
            warps_per_cta   =[1, 1, 4, 1],
            order           =[3, 2, 1, 0],
        )
        v_dim1_offs = gl.arange(0, KV_BLK_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_v_layout))))
        v_dim2_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_v_layout))))
        v_dim3_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout))))
    else:
        # blocked_v_layout: gl.constexpr = gl.DistributedLinearLayout( # 256x128
        #     reg_bases=((0,0,1), (0,0,2), (0,0,4), (0,0,8), (4,0,0), (8,0,0), (0,64,0)), # 16 x 8
        #     lane_bases=((0,1,0), (0,2,0), (0,4,0), (0,8,0), (1,0,0), (2,0,0)), # 64
        #     warp_bases=((0,16,0), (0,32,0)), # 4
        #     block_bases=[], # 8
        #     shape=[MAX_NUM_KV_BLKS, 128, 16],
        # )
        # [MAX_NUM_KV_BLKS, 128, 16]
        blocked_v_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread =[1, 1, 16],
            threads_per_warp=[4, 16, 1],
            warps_per_cta   =[1, 4, 1],
            order           =[2, 1, 0],
        )
        v_dim1_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_v_layout)))
        v_dim2_offs = gl.arange(0, KV_BLK_SZ, layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_v_layout)))

    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    pv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # blocked_q dim0
    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    # blocked_q dim1
    head_sz_layout: gl.constexpr = gl.SliceLayout(0, blocked_q)

    # blocked_k dim0
    blk_id_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim1
    head_sz_div_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim2
    blk_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim3
    contiguous_kv_elems_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_k)))

    q_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=query_grp_sz_layout)
    head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=head_sz_layout)
    head_sz_div_offs = gl.arange(0, K_HEAD_SZ_POW2_SPLIT, layout=head_sz_div_layout)
    blk_offs = gl.arange(0, KV_BLK_SZ, layout=blk_layout)
    contiguous_kv_elems_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=contiguous_kv_elems_layout)
    qk_row_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_linear_layout))

    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    seq_part_idx = gl.program_id(2)

    q_offs_base = (
        seq_idx * Q_SEQ_LEN * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs_base, mask=q_mask)

    if Q_QUANT_MODE == 0:
        q_scale_val = tl.load(q_scale)
    elif Q_QUANT_MODE == 1:
        q_scale_offs = seq_idx * Q_SEQ_LEN * q_scale_stride0 + kv_head_idx * QUERY_GRP_SZ + qk_row_offs[:, None]
        # [QUERY_GRP_SZ_POW2, 1]
        q_scale_val = gl.amd.cdna3.buffer_load(
            ptr=q_scale, offsets=q_scale_offs, mask=qk_row_offs[:, None] < QUERY_GRP_SZ
        )

    m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_linear_layout))
    m_l_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + m_l_base_offs
    )
    m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ
    o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, pv_mfma_layout))
    o_head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, pv_mfma_layout))
    o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (o_head_sz_offs[None, :] < HEAD_SZ)
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + o_grp_offs[:, None] * stride_logits_g
        + o_head_sz_offs[None, :]
    )

    # m_i = gl.zeros((QUERY_GRP_SZ_POW2,), dtype=gl.float32, layout=gl.SliceLayout(1, qk_linear_layout))
    # m_i = gl.zeros((QUERY_GRP_SZ_POW2,), dtype=gl.float32)
    # m_i = gl.convert_layout(m_i, layout=gl.SliceLayout(1, qk_linear_layout))
    m_i = m_l_base_offs.to(gl.float32) * float(0.0) - float("inf")
    l_i = m_l_base_offs.to(gl.float32) * float(0.0)
    # acc0 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=COMPUTE_TYPE, layout=pv_mfma_layout)
    acc0 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=gl.float32, layout=pv_mfma_layout)


    kv_seq_len = gl.load(seq_lens_ptr + seq_idx)
    kv_seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if kv_seq_start_idx >= kv_seq_len:
        return
    KV_COMPUTE_BLOCK_NUM: gl.constexpr = SEQ_PARTITION_SZ // KV_COMPUTE_BLOCK_SIZE
    # seq_end_idx = gl.minimum(kv_seq_start_idx + SEQ_PARTITION_SZ, kv_seq_len)
    SEQ_PARTITION_NUM_KV_BLKS: gl.constexpr = SEQ_PARTITION_SZ // KV_BLK_SZ

    for kv_idx in range(KV_COMPUTE_BLOCK_NUM):
        kv_sub_seq_start_idx = kv_seq_start_idx + kv_idx * KV_COMPUTE_BLOCK_SIZE
        kv_sub_seq_end_idx = gl.minimum(kv_sub_seq_start_idx + KV_COMPUTE_BLOCK_SIZE, kv_seq_len)

        num_kv_blks = gl.cdiv(kv_sub_seq_end_idx - kv_sub_seq_start_idx, KV_BLK_SZ)
        kv_blk_start = seq_part_idx * SEQ_PARTITION_NUM_KV_BLKS + kv_idx * MAX_NUM_KV_BLKS
        qk_col_offs = kv_blk_start * KV_BLK_SZ + gl.arange(0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout))

        # load alibi slopes[QUERY_GRP_SZ_POW2]
        if alibi_slopes is None:
            alibi_slope = gl.zeros([QUERY_GRP_SZ_POW2], dtype=gl.float32)
        else:
            alibi_slope = gl.amd.cdna3.buffer_load(ptr=alibi_slopes + kv_head_idx * QUERY_GRP_SZ, offsets=qk_row_offs, mask=qk_row_offs < QUERY_GRP_SZ)

        # load all kv blocks in one time
        blk_ids = gl.arange(0, MAX_NUM_KV_BLKS, layout=blk_id_layout)
        masked_blk_ids = gl.where(blk_ids < num_kv_blks, blk_ids, 0)
        blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
        kv_blk_nums = gl.amd.cdna3.buffer_load(ptr=blk_tables_start_ptr + kv_blk_start, offsets=masked_blk_ids)

        # k_blk_offs[MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ, CONTIGUOUS_KV_ELEMS_16B_LOAD]
        k_blk_offs = (
            kv_blk_nums[:, None, None, None] * stride_k_b
            + kv_head_idx * stride_k_nh
            + head_sz_div_offs[None, :, None, None] * stride_k_hz
            + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
            + contiguous_kv_elems_offs[None, None, None, :]
        )
        k = gl.amd.cdna3.buffer_load(ptr=k_cache_ptr, offsets=k_blk_offs)
        if KV_QUANT_MODE >= 0:
            # 0 for per_tensor quant
            # 1 for per_token quant
            if KV_QUANT_MODE == 0:
                k_scale_val = tl.load(k_scale)
                v_scale_val = tl.load(v_scale)
            elif KV_QUANT_MODE == 1:
                # [MAX_NUM_KV_BLKS, 1, KV_BLK_SZ, 1]
                k_scale_offs = kv_blk_nums[:, None, None, None] * kv_scale_stride0 + kv_head_idx * kv_scale_stride1 + blk_offs[None, None, :, None]
                k_scale_offs = gl.reshape(k_scale_offs, [KV_COMPUTE_BLOCK_SIZE])
                k_scale_offs = gl.convert_layout(k_scale_offs, layout=gl.SliceLayout(0, qk_linear_layout))
                # [KV_COMPUTE_BLOCK_SIZE]
                k_scale_val = gl.amd.cdna3.buffer_load(ptr=k_scale, offsets=k_scale_offs)
                v_scale_val = gl.amd.cdna3.buffer_load(ptr=v_scale, offsets=k_scale_offs)

        k = gl.permute(k, [1, 3, 0, 2])
        k = gl.reshape(k, [HEAD_SZ_POW2, KV_COMPUTE_BLOCK_SIZE])


        if TRANS_V:
            kv_blk_nums2 = gl.convert_layout(kv_blk_nums, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_v_layout))))
            v_blk_offs = (
                kv_blk_nums2[:, None, None, None] * stride_v_b
                + kv_head_idx * stride_v_nh
                + v_dim1_offs[None, :, None, None] * stride_v_hz
                + v_dim2_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
                + v_dim3_offs[None, None, None, :]
            )
            # v[MAX_NUM_KV_BLKS, BLK_DIV_16B, HEAD_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
            v = gl.amd.cdna3.buffer_load(ptr=v_cache_ptr, offsets=v_blk_offs)
            # from v[MAX_NUM_KV_BLKS, BLK_DIV_16B, HEAD_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
            # to   v[MAX_NUM_KV_BLKS, BLK_DIV_16B, CONTIGUOUS_KV_ELEMS_16B_LOAD, HEAD_SZ_POW2]
            v = gl.permute(v, [0, 1, 3, 2])
            v = gl.reshape(v, [KV_COMPUTE_BLOCK_SIZE, HEAD_SZ_POW2])
        else:
            kv_blk_nums2 = gl.convert_layout(kv_blk_nums, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
            # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ]
            v_blk_offs = (
                kv_blk_nums2[:, None, None] * stride_v_b
                + kv_head_idx * stride_v_nh
                + v_dim1_offs[None, :, None] * stride_v_hz
                + v_dim2_offs[None, None, :]
            )
            # v[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ]
            v = gl.amd.cdna3.buffer_load(ptr=v_cache_ptr, offsets=v_blk_offs)
            # [MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ] --> [MAX_NUM_KV_BLKS, KV_BLK_SZ, HEAD_SZ_POW2]
            v = gl.permute(v, [0, 2, 1])
            # v[KV_COMPUTE_BLOCK_SIZE, HEAD_SZ_POW2]
            v = gl.reshape(v, [KV_COMPUTE_BLOCK_SIZE, HEAD_SZ_POW2])


        # accumulator0 = gl.zeros((QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE // 2), dtype=gl.float32, layout=qk_mfma_layout)
        # accumulator1 = gl.zeros((QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE // 2), dtype=gl.float32, layout=qk_mfma_layout)
        accumulator = gl.zeros((QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE), dtype=gl.float32, layout=qk_mfma_layout)


        # qc = gl.convert_layout(q, layout=qk_lhs_layout)
        # kc0 = gl.convert_layout(kt0, layout=qk_rhs_layout)
        # kc1 = gl.convert_layout(kt1, layout=qk_rhs_layout)
        # qk0 = gl.amd.cdna3.mfma(qc, kc0, accumulator0)
        # qk1 = gl.amd.cdna3.mfma(qc, kc1, accumulator1)
        # qk = gl.join(qk0, qk1)
        # qk = gl.reshape(qk, [QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE])
        qc = gl.convert_layout(q, layout=qk_lhs_layout)
        kc = gl.convert_layout(k, layout=qk_rhs_layout)
        qk = gl.amd.cdna3.mfma(qc, kc, accumulator)
        qk = gl.reshape(qk, [QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE])

        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # [KV_COMPUTE_BLOCK_SIZE] --> [1, KV_COMPUTE_BLOCK_SIZE]
                k_scale_val = k_scale_val[None, :]
            if Q_QUANT_MODE >= 0:
                qk_scale_val = softmax_scale * q_scale_val * k_scale_val
            else:
                qk_scale_val = softmax_scale * k_scale_val
        else:
            if Q_QUANT_MODE >= 0:
                qk_scale_val = softmax_scale * q_scale_val
            else:
                qk_scale_val = softmax_scale

        qk = qk_scale_val * qk

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (qk_col_offs - kv_seq_len + 1)[None, :]).to(gl.float32)

        qk_bound_mask = (qk_row_offs[:, None] < QUERY_GRP_SZ)
        if IS_CAUSAL:
            # causal_mask = qk_col_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
            seq_m_extand = Q_SEQ_LEN - 1 - qk_row_offs // QUERY_GRP_SZ_ORI
            causal_mask = seq_m_extand[:, None] + qk_col_offs[None, :] < kv_seq_len
        else:
            causal_mask = qk_col_offs[None, :] < kv_seq_len
        qk_bound_mask = qk_bound_mask & causal_mask
        # if [0, SEQ_PARTITION_SZ) are all -inf, the result will be nan
        # so, we use -1e37 other than -inf
        # qk = tl.where(qk_bound_mask, qk, float("-inf"))
        qk = tl.where(qk_bound_mask, qk, float(-1e38))

        curr_m = gl.max(qk, axis=1)
        m_i_new = gl.maximum(m_i, curr_m)
        # acc_scale = tl.math.exp2(m_i - m_i_new)
        acc_scale = tl.math.exp2((m_i - m_i_new) * log2e)

        # p[QUERY_GRP_SZ_POW2, KV_COMPUTE_BLOCK_SIZE]
        # p = tl.math.exp2((qk - m_i_new[:, None]))
        p = tl.math.exp2((qk - m_i_new[:, None]) * log2e)
        l_i = acc_scale * l_i + gl.sum(p, axis=1)

        if v.dtype.is_fp8():
            # 1 for per_token quant
            if KV_QUANT_MODE == 1:
                vs_max = gl.max(v_scale_val, axis=0)
                v_scale_val = v_scale_val * float(FP8_MAX) / vs_max
                p = v_scale_val[None, :] * p
                p_scale = vs_max / float(FP8_MAX)

        if CONTIGUOUS_KV_ELEMS_16B_LOAD == 16:
            # kv element is 1 bytes(fp8)
            p = p.to(v_cache_ptr.dtype.element_ty)
        else:
            # kv element is 2 bytes(fp16, bf16)
            p = p.to(COMPUTE_TYPE)

        pc = gl.convert_layout(p, layout=pv_lhs_layout)
        vc = gl.convert_layout(v, layout=pv_rhs_layout)

        # acc_scale = l_i * 0 + alpha  # Workaround some compiler bug
        acc_scale = gl.convert_layout(acc_scale[:, None], layout=pv_mfma_layout)
        # p_scale = gl.convert_layout(p_scale[:, None], layout=pv_mfma_layout)
        acc0 *= acc_scale
        # acc0 = gl.amd.cdna3.mfma(pc, vc, acc0)
        # acc0 = p_scale * acc0
        accumulator1 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=gl.float32, layout=pv_mfma_layout)
        acc = gl.amd.cdna3.mfma(pc, vc, accumulator1)

        # 0 for per_tensor quant
        # 1 for per_token quant
        if KV_QUANT_MODE == 0:
            acc0 += v_scale_val * acc
        elif KV_QUANT_MODE == 1:
            acc0 += p_scale * acc

        m_i = m_i_new

    exp_sum = 1.0 / l_i
    exp_sum_cvt = gl.convert_layout(exp_sum[:, None], layout=pv_mfma_layout)
    acc0 = acc0 * exp_sum_cvt
    acc0 = acc0.to(COMPUTE_TYPE)
    gl.amd.cdna3.buffer_store(stored_value=m_i, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=l_i, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=acc0, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)


@gluon.jit
def paged_attn_decode_v2_reduce_gluon(
    out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
    exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    seq_lens_ptr,   # [num_seqs]
    sink_ptr,       # [num_query_heads]
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
    USE_SINKS: tl.constexpr
):
    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    num_query_heads = gl.num_programs(1) * QUERY_GRP_SZ
    seq_len = gl.load(seq_lens_ptr + seq_idx)
    num_partitions = gl.cdiv(seq_len, SEQ_PARTITION_SZ)
    if MAX_NUM_SEQ_PARTITIONS_POW2 >= 256:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread =[1, 2, 4],
            threads_per_warp=[4, 4, 4],
            warps_per_cta   =[4, 1, 1],
            order           =[2, 1, 0],
        )
    else:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread =[4, 1, 2],
            threads_per_warp=[4, 4, 4],
            warps_per_cta   =[1, 1, 4],
            order           =[2, 1, 0],
        )
    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_layout))
    head_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_layout))
    seq_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, blocked_layout))

    part_offs = gl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2, layout=seq_layout)
    q_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=query_grp_sz_layout)
    head_offs = gl.arange(0, HEAD_SZ_POW2, layout=head_sz_layout)

    # get global max logit
    exp_sums_offs = (
        seq_idx * stride_exp_sums_s
        + kv_head_idx * stride_exp_sums_h
        + part_offs[:, None] * stride_exp_sums_p
        + q_grp_offs[None, :]
    )
    exp_sums_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )

    # max_logits: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    max_logits = gl.amd.cdna3.buffer_load(
        # ptr=max_logits_ptr, offsets=exp_sums_offs, mask=exp_sums_mask, other=float("-inf")
        ptr=max_logits_ptr, offsets=exp_sums_offs, mask=exp_sums_mask
    )
    # max_logit: [QUERY_GRP_SZ_POW2]
    ml = tl.max(max_logits, axis=0)

    # Rescale the exp sums and compute the global sum
    # exp_sums: [MAX_NUM_SEQ_PARTITIONS, QUERY_GRP_SZ_POW2]
    # exp_sums = gl.amd.cdna3.buffer_load(ptr=exp_sums_ptr, offsets=exp_sums_offs, mask=exp_sums_mask, other=0.0)
    exp_sums = gl.amd.cdna3.buffer_load(ptr=exp_sums_ptr, offsets=exp_sums_offs, mask=exp_sums_mask)
    exp_sums *= tl.exp(max_logits - ml[None, :])

    # exp_sum: [QUERY_GRP_SZ_POW2]
    exp_sum = tl.sum(exp_sums, axis=0)
    if USE_SINKS:
        M = gl.load(
            sink_ptr + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs),
            mask=(kv_head_idx * QUERY_GRP_SZ + q_grp_offs) < num_query_heads,
        )
        exp_sum += tl.math.exp(M - ml)
    # p: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    p = exp_sums / exp_sum[None, :]
    p = tl.reshape(p, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2, 1))

    # logits_offset
    logits_offset = (
        seq_idx * stride_logits_s
        + kv_head_idx * stride_logits_h
        + part_offs[:, None, None] * stride_logits_p
        + q_grp_offs[None, :, None] * stride_logits_g
        + head_offs[None, None, :]
    )
    # load logits
    logits_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )
    logits = gl.amd.cdna3.buffer_load(
        # ptr=logits_ptrs, offsets=logits_offset, mask=logits_mask[:, :, None], other=0.0
        ptr=logits_ptrs, offsets=logits_offset, mask=logits_mask[:, :, None]
    )

    # out: [QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    out = tl.sum((logits * gl.convert_layout(p, layout=blocked_layout)).to(tl.float32), axis=0, keep_dims=True).to(out_ptr.dtype.element_ty)
    # store output
    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[None, :, None]) * stride_o_h
        + head_offs[None, None, :]
    )
    # gl.static_print(out_offs)
    gl.amd.cdna3.buffer_store(
        stored_value=out,
        ptr=out_ptr,
        offsets=out_offs,
        mask=(q_grp_offs[None, :, None] < QUERY_GRP_SZ) & (head_offs[None, None, :] < HEAD_SZ),
    )


@triton.jit
def _paged_attn_decode_v2_w_dot_reduce_kernel(
    out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
    exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    seq_lens_ptr,   # [num_seqs]
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    """
    #TODO: Add Doc
    """

    log2e: gl.constexpr = 1.4426950408889634
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    kv_seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_partitions = tl.cdiv(kv_seq_len, SEQ_PARTITION_SZ)

    part_offs = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    head_offs = tl.arange(0, HEAD_SZ_POW2)

    # get global max logit
    exp_sums_offs = (
        seq_idx * stride_exp_sums_s
        + kv_head_idx * stride_exp_sums_h
        + part_offs[:, None] * stride_exp_sums_p
        + q_grp_offs[None, :]
    )
    exp_sums_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )

    # max_logits: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    max_logits = tl.load(
        max_logits_ptr + exp_sums_offs, mask=exp_sums_mask, other=float("-inf")
    )
    # max_logit: [QUERY_GRP_SZ_POW2]
    ml = tl.max(max_logits, axis=0)

    # Rescale the exp sums and compute the global sum
    # exp_sums: [MAX_NUM_SEQ_PARTITIONS, QUERY_GRP_SZ_POW2]
    exp_sums = tl.load(exp_sums_ptr + exp_sums_offs, mask=exp_sums_mask)
    exp_sums *= tl.exp(max_logits - ml[None, :])
    # exp_sums *= tl.exp2(max_logits - ml[None, :])
    # exp_sums *= tl.exp2((max_logits - ml[None, :]) * log2e)

    # exp_sum: [QUERY_GRP_SZ_POW2]
    exp_sum = tl.sum(exp_sums, axis=0)

    # p: [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2]
    p = exp_sums / exp_sum[None, :]
    p = tl.reshape(p, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GRP_SZ_POW2, 1))

    # logits_offset
    logits_offset = (
        seq_idx * stride_logits_s
        + kv_head_idx * stride_logits_h
        + part_offs[:, None, None] * stride_logits_p
        + q_grp_offs[None, :, None] * stride_logits_g
        + head_offs[None, None, :]
    )
    # load logits
    logits_mask = (part_offs[:, None] < num_partitions) & (
        q_grp_offs[None, :] < QUERY_GRP_SZ
    )
    logits = tl.load(
        logits_ptrs + logits_offset, mask=logits_mask[:, :, None]
    )

    # out: [QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    out = tl.sum((logits * p).to(tl.float32), axis=0).to(out_ptr.dtype.element_ty)

    # store output
    out_offs = (
        seq_idx * stride_o_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_o_h
        + head_offs[None, :]
    )
    tl.store(
        out_ptr + out_offs,
        out,
        mask=(q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_offs[None, :] < HEAD_SZ),
    )


# Compile ttgir with triton
def compile_ttgir_with_triton(ttgir_content: str):
    # Read ttgir string from file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttgir', delete=False) as f:
        f.write(ttgir_content)
        ttgir_path = f.name

    # Compile ttgir string to ttgir
    try:
        compiled_artifact = compile(ttgir_path)
        return compiled_artifact
    finally:
        os.unlink(ttgir_path)


@perftest()
# @perftest(num_iters=TEST_NUM_ITERS)
def _paged_attn_decode_v2_w_dot_kernel_reshape_wrapper(
    grid,
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    softmax_scale,
    q_scale,            # [num_seqs, num_kv_heads * query_grp_sz, 1]
    k_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
    v_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_hz,
    stride_v_bz,
    stride_bt_s,
    q_scale_stride0,
    kv_scale_stride0,
    kv_scale_stride1,
    kv_type,
    Q_SEQ_LEN,
    COMPUTE_TYPE,
    HEAD_SZ,
    HEAD_SZ_POW2,
    QUERY_GRP_SZ_ORI,
    QUERY_GRP_SZ,
    QUERY_GRP_SZ_POW2,
    KV_BLK_SZ,
    KV_BLK_SZ_POW2,
    SEQ_PARTITION_SZ,
    KV_16B_ELE_NUM,
    Q_QUANT_MODE,
    KV_QUANT_MODE,
    FP8_MAX,
    TRANS_V,
    IS_CAUSAL,
):
    # import pdb
    # pdb.set_trace()

    # if 1:
    if 0:
        # ttgir_file_path = os.path.join(os.path.dirname(__file__), "./ttgir/pa_noloop.ttgir")
        ttgir_file_path = os.path.join(os.path.dirname(__file__), "./thread_trace/triton_gen_asm/pa_decode_v2_fp8/pa_decode_v2_fp8.ttgir")
        # ttgir_file_path = os.path.join(os.path.dirname(__file__), "./thread_trace/triton_gen_asm/pa_decode_v2_fp8_rtn/pa_decode_v2_fp8.ttgir")
        with open(ttgir_file_path, 'r') as f:
            ttgir_content = f.read()
        try:
            compiled_kernel = compile_ttgir_with_triton(ttgir_content)
            compiled_kernel[grid](
            exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
            q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
            k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
            v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
            blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
            seq_lens_ptr,       # [num_seqs]
            softmax_scale,
            q_scale,            # [num_seqs, num_kv_heads * query_grp_sz, 1]
            k_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
            v_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
            stride_max_logits_s,
            stride_max_logits_nh,
            stride_max_logits_p,
            stride_logits_s,
            stride_logits_nh,
            stride_logits_p,
            stride_logits_g,
            stride_q_s,
            stride_q_nh,
            stride_k_b,
            stride_k_nh,
            stride_k_hz,
            stride_k_bz,
            stride_v_b,
            stride_v_hz,
            stride_v_bz,
            stride_bt_s,
            q_scale_stride0,
            kv_scale_stride0,
            kv_scale_stride1,
        )
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        KV_COMPUTE_BLOCK_SIZE = 256
        waves_per_eu = 1
        pa_decode_kernel = pa_decode_v2_gluon_fp8
        if KV_BLK_SZ > SEQ_PARTITION_SZ:
            pa_decode_kernel = pa_decode_v2_gluon_big_blk_fp8
            if TRANS_V:
                # for better perf
                KV_COMPUTE_BLOCK_SIZE = 128
        else:
            if QUERY_GRP_SZ_POW2 == 64:
                waves_per_eu = 3
            else:
                waves_per_eu = 4

        pa_decode_kernel[grid](
            exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
            q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
            k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
            v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
            blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
            seq_lens_ptr,       # [num_seqs]
            softmax_scale,
            q_scale,            # [num_seqs, num_kv_heads * query_grp_sz, 1]
            k_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
            v_scale,            # [num_blks, num_kv_heads, kv_blk_sz, 1]
            alibi_slopes,
            stride_max_logits_s,
            stride_max_logits_nh,
            stride_max_logits_p,
            stride_logits_s,
            stride_logits_nh,
            stride_logits_p,
            stride_logits_g,
            stride_q_s,
            stride_q_nh,
            stride_k_b,
            stride_k_nh,
            stride_k_hz,
            stride_k_bz,
            stride_v_b,
            stride_v_hz,
            stride_v_bz,
            stride_bt_s,
            q_scale_stride0,
            kv_scale_stride0,
            kv_scale_stride1,
            Q_SEQ_LEN=Q_SEQ_LEN,
            COMPUTE_TYPE=COMPUTE_TYPE,
            HEAD_SZ=HEAD_SZ,
            HEAD_SZ_POW2=HEAD_SZ_POW2,
            QUERY_GRP_SZ_ORI=QUERY_GRP_SZ_ORI,
            QUERY_GRP_SZ=QUERY_GRP_SZ,
            QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
            KV_BLK_SZ=KV_BLK_SZ,
            KV_BLK_SZ_POW2=KV_BLK_SZ_POW2,
            SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
            KV_16B_ELE_NUM=KV_16B_ELE_NUM,
            Q_QUANT_MODE=Q_QUANT_MODE,
            KV_QUANT_MODE=KV_QUANT_MODE,
            KV_COMPUTE_BLOCK_SIZE=KV_COMPUTE_BLOCK_SIZE,
            FP8_MAX=FP8_MAX,
            TRANS_V=TRANS_V,
            IS_CAUSAL=IS_CAUSAL,

            waves_per_eu=waves_per_eu,
            # waves_per_eu=1,
            num_stages=1,
            # waves_per_eu=3,
            # num_stages=1,
            # waves_per_eu=4,
            # num_stages=1,
        )


@perftest()
def _paged_attn_decode_v2_w_dot_reduce_kernel_wrapper(
    grid,
    out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
    exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    seq_lens_ptr,   # [num_seqs]
    stride_o_s,
    stride_o_h,
    stride_exp_sums_s,
    stride_exp_sums_h,
    stride_exp_sums_p,
    stride_logits_s,
    stride_logits_h,
    stride_logits_p,
    stride_logits_g,
    HEAD_SZ,
    HEAD_SZ_POW2,
    QUERY_GRP_SZ,
    QUERY_GRP_SZ_POW2,
    SEQ_PARTITION_SZ,
    MAX_NUM_SEQ_PARTITIONS,
    MAX_NUM_SEQ_PARTITIONS_POW2,
):
    # REDUCE_USE_GLUON = True
    REDUCE_USE_GLUON = False
    if REDUCE_USE_GLUON:
        paged_attn_decode_v2_reduce_gluon[grid](
            out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
            exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
            seq_lens_ptr,   # [num_seqs]
            None,
            stride_o_s,
            stride_o_h,
            stride_exp_sums_s,
            stride_exp_sums_h,
            stride_exp_sums_p,
            stride_logits_s,
            stride_logits_h,
            stride_logits_p,
            stride_logits_g,
            HEAD_SZ=HEAD_SZ,
            HEAD_SZ_POW2=HEAD_SZ_POW2,
            QUERY_GRP_SZ=QUERY_GRP_SZ,
            QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
            SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
            MAX_NUM_SEQ_PARTITIONS=MAX_NUM_SEQ_PARTITIONS,
            MAX_NUM_SEQ_PARTITIONS_POW2=MAX_NUM_SEQ_PARTITIONS_POW2,
            USE_SINKS=False,
        )
    else:
        _paged_attn_decode_v2_w_dot_reduce_kernel[grid](
            out_ptr,        # [num_seqs, num_kv_heads, q_grp_sz, head_sz]
            exp_sums_ptr,   # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            max_logits_ptr, # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
            logits_ptrs,    # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
            seq_lens_ptr,   # [num_seqs]
            stride_o_s,
            stride_o_h,
            stride_exp_sums_s,
            stride_exp_sums_h,
            stride_exp_sums_p,
            stride_logits_s,
            stride_logits_h,
            stride_logits_p,
            stride_logits_g,
            HEAD_SZ=HEAD_SZ,
            HEAD_SZ_POW2=HEAD_SZ_POW2,
            QUERY_GRP_SZ=QUERY_GRP_SZ,
            QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
            SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
            MAX_NUM_SEQ_PARTITIONS=MAX_NUM_SEQ_PARTITIONS,
            MAX_NUM_SEQ_PARTITIONS_POW2=MAX_NUM_SEQ_PARTITIONS_POW2,
        )


def paged_attention_decode(
    output: torch.Tensor,       # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, block_size/x, head_size, x]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    q_scale: torch.Tensor,      # [num_seqs, num_kv_heads * query_grp_sz, 1]
    k_scale: torch.Tensor,      # [num_blks, num_kv_heads, kv_blk_sz, 1]
    v_scale: torch.Tensor,      # [num_blks, num_kv_heads, kv_blk_sz, 1]
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
) -> None:
    """
    #TODO: Add Doc
    """
    batch_size = block_tables.shape[0]
    num_seqs = query.shape[0]
    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    q_seq_len = num_seqs // batch_size
    max_num_partitions = int((max_seq_len + _SEQ_PARTITION_SIZE - 1) // _SEQ_PARTITION_SIZE)
    num_q_heads = query.shape[1]
    head_sz = query.shape[-1]
    kv_blk_sz = key_cache.shape[-2]
    query_grp_sz = num_q_heads // num_kv_heads
    equi_query_grp_sz = q_seq_len * query_grp_sz
    equi_query_grp_sz_pow2 = triton.next_power_of_2(equi_query_grp_sz)
    kv_blk_sz_pow2 = triton.next_power_of_2(kv_blk_sz)
    head_sz_pow2 = triton.next_power_of_2(head_sz)
    is_causal = q_seq_len > 1
    # is_causal = False
    num_seqs = batch_size
    kv_16b_ele_num = 16 // key_cache.dtype.itemsize

    grid = (num_seqs, num_kv_heads, max_num_partitions)
    shape_info = (num_seqs, num_kv_heads, max_num_partitions, equi_query_grp_sz)
    max_logits = torch.zeros(shape_info, dtype=torch.float32, device=output.device)
    exp_sums = torch.zeros(shape_info, dtype=torch.float32, device=output.device)
    # tmp_output = torch.empty(
    tmp_output = torch.zeros(
        # *shape_info, head_sz, dtype=torch.float8_e4m3fnuz, device=output.device
        *shape_info, head_sz, dtype=output.dtype, device=output.device
    )

    if equi_query_grp_sz <= 16:
        equi_query_grp_sz_pow2 = 16
    else:
        equi_query_grp_sz_pow2 = triton.next_power_of_2(equi_query_grp_sz)
    assert equi_query_grp_sz_pow2 <= 64, f"equi_query_grp_sz_pow2={equi_query_grp_sz_pow2}"

    q_scale_stride0 = 0
    k_scale_stride0 = 0
    k_scale_stride1 = 0
    q_quant_mode = -1
    kv_quant_mode = -1
    if q_scale is not None:
        if len(q_scale.shape) == 0:
            q_quant_mode = 0
        else:
            if len(q_scale.shape) != 3:
                raise RuntimeError(f"This condition must be met: len(q_scale.shape) == 3, but q_scale.shape is {q_scale.shape}")
            q_quant_mode = 1
            q_scale_stride0 = q_scale.stride(0)

    if k_scale is not None and v_scale is not None:
        if len(k_scale.shape) == 0:
            kv_quant_mode = 0
        else:
            if len(k_scale.shape) != 4:
                raise RuntimeError(f"This condition must be met: len(k_scale.shape) == 4, but k_scale.shape is {k_scale.shape}")
            kv_quant_mode = 1
            k_scale_stride0 = k_scale.stride(0)
            k_scale_stride1 = k_scale.stride(1)
        if k_scale.shape != v_scale.shape:
            raise RuntimeError(f"This condition must be met: k_scale.shape == v_scale.shape, but k_scale.shape is {k_scale.shape}, v_scale.shape is {v_scale.shape}")

    trans_v = False
    if len(value_cache.shape) == 5:
        trans_v = True
    elif len(value_cache.shape) == 4:
        trans_v = False
    else:
        raise RuntimeError(f"Do not support such value_cache shape:{value_cache.shape}")

    fp8_max = 1.0
    if value_cache.dtype == aiter.dtypes.fp8:
        fp8_max = torch.finfo(aiter.dtypes.fp8).max

    output = output.reshape(batch_size, q_seq_len, num_kv_heads, query_grp_sz, head_sz)
    output = output.transpose(1, 2).reshape(batch_size, num_kv_heads * q_seq_len * query_grp_sz, head_sz).contiguous()

    # qt = query.reshape(batch_size, q_seq_len, num_kv_heads, query_grp_sz, head_sz)
    # qt = qt.transpose(1, 2).reshape(batch_size, num_kv_heads * q_seq_len * query_grp_sz, head_sz).contiguous()

    # qst = q_scale.reshape(batch_size, q_seq_len, num_kv_heads, query_grp_sz)
    # qst = qst.transpose(1, 2).reshape(batch_size, num_kv_heads * q_seq_len * query_grp_sz).contiguous()

    # print(f"grid={grid}")
    # print(f"shape_info={shape_info}")
    # print(f"query.shape={query.shape}")
    # print(f"q_scale.shape={q_scale.shape}")
    # print(f"k_scale.shape={k_scale.shape}")
    # print(f"v_scale.shape={v_scale.shape}")
    # print(f"key_cache.shape={key_cache.shape}")
    # print(f"value_cache.shape={value_cache.shape}")
    # print(f"output.shape={output.shape}")
    # print(f"tmp_output.shape={tmp_output.shape}")
    # print(f"block_tables.shape={block_tables.shape}")
    # print(f"query.dtype={query.dtype}")
    # print(f"key_cache.dtype={key_cache.dtype}")
    # print(f"value_cache.dtype={value_cache.dtype}")
    # print(f"output.dtype={output.dtype}")
    # print(f"tmp_output.dtype={tmp_output.dtype}")
    # print(f"block_tables.dtype={block_tables.dtype}")
    # print(f"value_cache.stride()={value_cache.stride()}")
    # print(f"query.stride()={query.stride()}")
    # print(f"q_scale.stride()={q_scale.stride()}")
    # print(f"k_scale.stride()={k_scale.stride()}")
    # print(f"v_scale.stride()={v_scale.stride()}")
    # print(f"tmp_output.stride()={tmp_output.stride()}")
    # input_config = dict(
    #     q_seq_len=q_seq_len,
    #     kv_type=compute_type,
    #     COMPUTE_TYPE=compute_type,
    #     HEAD_SZ=head_sz,
    #     HEAD_SZ_POW2=head_sz_pow2,
    #     QUERY_GRP_SZ_ORI=query_grp_sz,
    #     QUERY_GRP_SZ=equi_query_grp_sz,
    #     QUERY_GRP_SZ_POW2=equi_query_grp_sz_pow2,
    #     KV_BLK_SZ=kv_blk_sz,
    #     KV_BLK_SZ_POW2=kv_blk_sz_pow2,
    #     SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
    # )
    # print(input_config)


    _, decode_time = _paged_attn_decode_v2_w_dot_kernel_reshape_wrapper(
        grid,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        # qt,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        attn_scale,
        q_scale,
        # qst,
        k_scale,
        v_scale,
        alibi_slopes,
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        tmp_output.stride(0),
        tmp_output.stride(1),
        tmp_output.stride(2),
        tmp_output.stride(3),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        q_scale_stride0,
        k_scale_stride0,
        k_scale_stride1,
        kv_type=compute_type,
        Q_SEQ_LEN=q_seq_len,
        COMPUTE_TYPE=compute_type,
        HEAD_SZ=head_sz,
        HEAD_SZ_POW2=head_sz_pow2,
        QUERY_GRP_SZ_ORI=query_grp_sz,
        QUERY_GRP_SZ=equi_query_grp_sz,
        QUERY_GRP_SZ_POW2=equi_query_grp_sz_pow2,
        KV_BLK_SZ=kv_blk_sz,
        KV_BLK_SZ_POW2=kv_blk_sz_pow2,
        SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        KV_16B_ELE_NUM=kv_16b_ele_num,
        Q_QUANT_MODE=q_quant_mode,
        KV_QUANT_MODE=kv_quant_mode,
        FP8_MAX=fp8_max,
        TRANS_V=trans_v,
        IS_CAUSAL=is_causal,
    )

    grid = (num_seqs, num_kv_heads, 1)
    _, reduce_time = _paged_attn_decode_v2_w_dot_reduce_kernel_wrapper(
        grid,
        output,
        exp_sums,
        max_logits,
        tmp_output,
        seq_lens,
        output.stride(0),
        output.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        tmp_output.stride(0),
        tmp_output.stride(1),
        tmp_output.stride(2),
        tmp_output.stride(3),
        HEAD_SZ=head_sz,
        HEAD_SZ_POW2=head_sz_pow2,
        QUERY_GRP_SZ=equi_query_grp_sz,
        QUERY_GRP_SZ_POW2=equi_query_grp_sz_pow2,
        SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
        MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
    )
    output = output.reshape(batch_size, num_kv_heads, q_seq_len, query_grp_sz, head_sz)
    output = output.transpose(1, 2).reshape(batch_size * q_seq_len, num_kv_heads * query_grp_sz, head_sz).contiguous()

    tmp_output_nan_cnt = torch.isnan(tmp_output).sum()
    output_nan_cnt = torch.isnan(output).sum()
    # print(f"tmp_output_nan_cnt={tmp_output_nan_cnt}")
    # print(f"output_nan_cnt={output_nan_cnt}")

    # decode_time = 0
    # reduce_time = 0

    return {'triton_decode': decode_time,
            'triton_reduce': reduce_time,
            'tmp_output': tmp_output,
            'exp_sums': exp_sums,
            'max_logits': max_logits,
            'triton': decode_time + reduce_time}
