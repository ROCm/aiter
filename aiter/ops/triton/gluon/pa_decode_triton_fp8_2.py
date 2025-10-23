# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional

import triton
import triton.language as tl
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import compile
import torch
from aiter.test_common import perftest
import tempfile
import subprocess
import os


TEST_NUM_ITERS = 101

# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQ_PARTITION_SIZE = 256  # HIP


@triton.jit
def pa_decode_v2_big_blk_fp8_inner_one_q(
    q,
    k,
    k_scale_val,
    v,
    q_grp_offs,
    blk_seq_offs,
    kv_seq_len,
    alibi_slope,
    softmax_scale,
    log2e,
    max_logits_ptr,
    exp_sums_ptr,
    logits_ptr,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    seq_idx,
    kv_head_idx,
    seq_part_idx,
    k_cache_ptr,
    v_cache_ptr,
    COMPUTE_TYPE: tl.constexpr,
    QID: tl.constexpr,
    QUERY_GRP_SZ: tl.constexpr,
    HEAD_SZ: tl.constexpr,
    HEAD_SZ_POW2: tl.constexpr,
    Q_SEQ_LEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    if k_cache_ptr.dtype.element_ty.is_fp8():
        q = q.to(k_cache_ptr.dtype.element_ty)
    else:
        q = (q.to(tl.float32) * softmax_scale).to(COMPUTE_TYPE)

    # qk[QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
    qk = tl.dot(q, k, out_dtype=tl.float32)
    if k_cache_ptr.dtype.element_ty.is_fp8():
        qk = k_scale_val * qk

    qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ)
    if IS_CAUSAL:
        causal_mask = blk_seq_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
    else:
        causal_mask = blk_seq_offs[None, :] < kv_seq_len
    qk_bound_mask = qk_bound_mask & causal_mask

    if alibi_slope is not None:
        qk += (alibi_slope[:, None] * (blk_seq_offs - kv_seq_len + 1)[None, :]).to(
            tl.float32
        )
    qk = tl.where(qk_bound_mask, qk, float("-inf"))

    max_logit_new = tl.max(qk, axis=1)
    # p[QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    exp_sum = tl.sum(p, axis=1)

    if v_cache_ptr.dtype.element_ty.is_fp8():
        p = p.to(v_cache_ptr.dtype.element_ty)
    else:
        p = p.to(COMPUTE_TYPE)

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs + QID * QUERY_GRP_SZ, max_logit_new, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs + QID * QUERY_GRP_SZ, exp_sum, mask=m_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    acc = tl.dot(p, v, out_dtype=tl.float32)
    acc = acc / exp_sum[:, None]
    acc = acc.to(COMPUTE_TYPE)

    # end up computation
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    tl.store(logits_ptr + logits_offs + QID * QUERY_GRP_SZ * stride_logits_g, acc, mask=q_mask)


@triton.jit
def pa_decode_v2_big_blk_fp8(
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
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    KV_16B_ELE_NUM: tl.constexpr,
    TRANS_V: tl.constexpr,          # [num_blks, num_kv_heads, kv_blk_sz/x, head_sz, x]
    IS_CAUSAL: tl.constexpr,
):
    """
    #TODO: Add Doc
    """
    tl.static_assert(Q_SEQ_LEN <= 4, "Q_SEQ_LEN={}, Do not support Q_SEQ_LEN > 4".format(Q_SEQ_LEN))

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)
    # if seq_idx == 0 and kv_head_idx == 0 and seq_part_idx == 0:
    #     print('Q_SEQ_LEN=', Q_SEQ_LEN)

    log2e: tl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: tl.constexpr = KV_16B_ELE_NUM

    kv_seq_len = tl.load(seq_lens_ptr + seq_idx)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= kv_seq_len:
        return

    # seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, kv_seq_len)
    # MAX_NUM_KV_BLKS: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    V_BLK_SZ_DIV_NUM: tl.constexpr = SEQ_PARTITION_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD
    # num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    pn_blk_offs = tl.arange(0, SEQ_PARTITION_SZ)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    contiguous_kv_elems_offs = tl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD)
    v_blk_sz_div_offs = tl.arange(0, V_BLK_SZ_DIV_NUM)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load all kv blocks in one time
    kv_seq_start = seq_part_idx * SEQ_PARTITION_SZ
    blk_tb_id = kv_seq_start // KV_BLK_SZ
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    page_id = tl.load(blk_tables_start_ptr + blk_tb_id)
    page_offset = kv_seq_start % KV_BLK_SZ

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * Q_SEQ_LEN * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q0 = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)

    q_scale_offs = None
    if q0.dtype.is_fp8():
        # [QUERY_GRP_SZ_POW2]
        q_scale_offs = seq_idx * Q_SEQ_LEN * q_scale_stride0 + kv_head_idx * QUERY_GRP_SZ + q_grp_offs
        q_scale_val = tl.load(q_scale + q_scale_offs, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
        # q_scale_val = tl.load(q_scale + q_scale_offs)
        q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
        q0 = q_scale_val * q0.to(tl.float32)

    q1 = q0
    q2 = q0
    q3 = q0
    if Q_SEQ_LEN >= 2:
        qid = 1
        q1 = tl.load(q_ptr + q_offs + qid * stride_q_s, mask=q_mask, other=0.0)
        if q1.dtype.is_fp8():
            # [QUERY_GRP_SZ_POW2]
            q_scale_val = tl.load(q_scale + q_scale_offs + qid * q_scale_stride0, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
            q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
            q1 = q_scale_val * q1.to(tl.float32)
    elif Q_SEQ_LEN >= 3:
        qid = 2
        q2 = tl.load(q_ptr + q_offs + qid * stride_q_s, mask=q_mask, other=0.0)
        if q2.dtype.is_fp8():
            # [QUERY_GRP_SZ_POW2]
            q_scale_val = tl.load(q_scale + q_scale_offs + qid * q_scale_stride0, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
            q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
            q2 = q_scale_val * q2.to(tl.float32)
    elif Q_SEQ_LEN >= 4:
        qid = 3
        q3 = tl.load(q_ptr + q_offs + qid * stride_q_s, mask=q_mask, other=0.0)
        if q3.dtype.is_fp8():
            # [QUERY_GRP_SZ_POW2]
            q_scale_val = tl.load(q_scale + q_scale_offs + qid * q_scale_stride0, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
            q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
            q3 = q_scale_val * q3.to(tl.float32)


    # k_blk_offs[HEAD_SZ_POW2/x, SEQ_PARTITION_SZ, x]
    k_blk_offs = (
        page_id * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[:, None, None] * stride_k_hz
        + (page_offset + pn_blk_offs)[None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, :]
    )
    # blk_seq_offs[SEQ_PARTITION_SZ]
    blk_seq_offs = kv_seq_start + pn_blk_offs

    # k_blk_offs[HEAD_SZ_POW2/x, SEQ_PARTITION_SZ, x]
    # k = tl.load(k_cache_ptr + k_blk_offs, mask=blk_seq_offs[None, :, None] < kv_seq_len, other=0.0)
    k_temp = tl.load(k_cache_ptr + k_blk_offs)
    # [HEAD_SZ_POW2/x, SEQ_PARTITION_SZ, x]
    k_temp = tl.permute(k_temp, [0, 2, 1])
    k_temp = tl.reshape(k_temp, [HEAD_SZ_POW2, SEQ_PARTITION_SZ])

    k_scale_val = k_scale
    v_scale_val = v_scale
    if k_temp.dtype.is_fp8():
    # if tl.is_tensor(k_scale):
        # [SEQ_PARTITION_SZ]
        kv_scale_offs = page_id * kv_scale_stride0 + kv_head_idx * kv_scale_stride1 + page_offset + pn_blk_offs
        # k_scale_val = tl.load(k_scale + kv_scale_offs, mask=blk_seq_offs < kv_seq_len, other=0.0)
        # v_scale_val = tl.load(v_scale + kv_scale_offs, mask=blk_seq_offs < kv_seq_len, other=0.0)
        k_scale_val = tl.load(k_scale + kv_scale_offs)
        v_scale_val = tl.load(v_scale + kv_scale_offs)
        # k_scale_val = tl.zeros((SEQ_PARTITION_SZ,), dtype=tl.float32)
        # v_scale_val = tl.zeros((SEQ_PARTITION_SZ,), dtype=tl.float32)

        # [QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
        k_scale_val = tl.broadcast_to(k_scale_val[None, :], QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ)
        # v_scale_val = tl.broadcast_to(v_scale_val[None, :], QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ)
        v_scale_val = tl.broadcast_to(v_scale_val[:, None], SEQ_PARTITION_SZ, HEAD_SZ_POW2)
        k_scale_val = softmax_scale * k_scale_val
    if k_temp.dtype.is_fp8():
        k = k_temp
    else:
        k = k_temp.to(COMPUTE_TYPE)

    v = None
    if TRANS_V:
        # [num_blocks, num_kv_heads, block_size/x, head_size, x]
        # if seq_idx == 0 and kv_head_idx == 0 and seq_part_idx == 0:
        #     print('CONTIGUOUS_KV_ELEMS_16B_LOAD=', CONTIGUOUS_KV_ELEMS_16B_LOAD)

        # v_blk_offs[V_BLK_SZ_DIV_NUM, HEAD_SZ_POW2, x]
        page_offset_div_num = page_offset // CONTIGUOUS_KV_ELEMS_16B_LOAD
        v_blk_offs = (
            page_id * stride_v_b
            + kv_head_idx * stride_v_nh
            + (page_offset_div_num + v_blk_sz_div_offs)[:, None, None] * stride_v_hz
            + head_sz_offs[None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
            + contiguous_kv_elems_offs[None, None, :]
        )

        v = tl.load(v_cache_ptr + v_blk_offs)
        # [V_BLK_SZ_DIV_NUM, HEAD_SZ_POW2, x] --> [V_BLK_SZ_DIV_NUM, x, HEAD_SZ_POW2]
        v = tl.permute(v, [0, 2, 1])
        v = tl.reshape(v, [SEQ_PARTITION_SZ, HEAD_SZ_POW2])
    else:
        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
        # [HEAD_SZ_POW2, SEQ_PARTITION_SZ]
        v_blk_offs = (
            page_id * stride_v_b
            + kv_head_idx * stride_v_nh
            + head_sz_offs[:, None] * stride_v_hz
            + (page_offset + pn_blk_offs)[None, :]
        )
        v = tl.load(v_cache_ptr + v_blk_offs)
        # [SEQ_PARTITION_SZ, HEAD_SZ_POW2]
        v = tl.permute(v, [1, 0])

    if v.dtype.is_fp8():
        v = v_scale_val * v.to(tl.float32)
        v = v.to(v_cache_ptr.dtype.element_ty)
    else:
        v = v.to(COMPUTE_TYPE)


    # QID = 0
    # pa_decode_v2_big_blk_fp8_inner_one_q(
    #     q0,
    #     k,
    #     k_scale_val,
    #     v,
    #     q_grp_offs,
    #     blk_seq_offs,
    #     kv_seq_len,
    #     alibi_slope,
    #     softmax_scale,
    #     log2e,
    #     max_logits_ptr,
    #     exp_sums_ptr,
    #     logits_ptr,
    #     stride_max_logits_s,
    #     stride_max_logits_nh,
    #     stride_max_logits_p,
    #     stride_logits_s,
    #     stride_logits_nh,
    #     stride_logits_p,
    #     stride_logits_g,
    #     seq_idx,
    #     kv_head_idx,
    #     seq_part_idx,
    #     k_cache_ptr,
    #     v_cache_ptr,
    #     COMPUTE_TYPE,
    #     QID,
    #     QUERY_GRP_SZ,
    #     HEAD_SZ,
    #     HEAD_SZ_POW2,
    #     Q_SEQ_LEN,
    #     IS_CAUSAL,
    # )


    QID = 0
    q = q0
    if k_temp.dtype.is_fp8():
        q = q.to(k_cache_ptr.dtype.element_ty)
    else:
        q = (q.to(tl.float32) * softmax_scale).to(COMPUTE_TYPE)

    # qk[QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
    qk = tl.dot(q, k, out_dtype=tl.float32)
    if k_temp.dtype.is_fp8():
        qk = k_scale_val * qk

    qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ)
    if IS_CAUSAL:
        causal_mask = blk_seq_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
    else:
        causal_mask = blk_seq_offs[None, :] < kv_seq_len
    qk_bound_mask = qk_bound_mask & causal_mask
    # qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < kv_seq_len)

    if alibi_slopes is not None:
        qk += (alibi_slope[:, None] * (blk_seq_offs - kv_seq_len + 1)[None, :]).to(
            tl.float32
        )

    # if [0, SEQ_PARTITION_SZ) are all -inf, the result will be nan
    # so, we use -1e37 other than -inf
    # qk = tl.where(qk_bound_mask, qk, float("-inf"))
    qk = tl.where(qk_bound_mask, qk, float(-1e37))

    max_logit_new = tl.max(qk, axis=1)
    # p[QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    exp_sum = tl.sum(p, axis=1)

    if v.dtype.is_fp8():
        # p = v_scale_val * p
        p = p.to(v_cache_ptr.dtype.element_ty)
    else:
        p = p.to(COMPUTE_TYPE)

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs + QID * QUERY_GRP_SZ, max_logit_new, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs + QID * QUERY_GRP_SZ, exp_sum, mask=m_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    acc = tl.dot(p, v, out_dtype=tl.float32)
    acc = acc / exp_sum[:, None]
    acc = acc.to(COMPUTE_TYPE)

    # end up computation
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )
    tl.store(logits_ptr + logits_offs + QID * QUERY_GRP_SZ * stride_logits_g, acc, mask=q_mask)


    if Q_SEQ_LEN >= 2:
        QID = 1
        q = q1
        if k_temp.dtype.is_fp8():
            q = q.to(k_cache_ptr.dtype.element_ty)
        else:
            q = (q.to(tl.float32) * softmax_scale).to(COMPUTE_TYPE)

        # qk[QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
        qk = tl.dot(q, k, out_dtype=tl.float32)
        if k_temp.dtype.is_fp8():
            qk = k_scale_val * qk

        qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ)
        if IS_CAUSAL:
            causal_mask = blk_seq_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
        else:
            causal_mask = blk_seq_offs[None, :] < kv_seq_len
        qk_bound_mask = qk_bound_mask & causal_mask

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_offs - kv_seq_len + 1)[None, :]).to(
                tl.float32
            )
        qk = tl.where(qk_bound_mask, qk, float(-1e37))

        max_logit_new = tl.max(qk, axis=1)
        # p[QUERY_GRP_SZ_POW2, SEQ_PARTITION_SZ]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        exp_sum = tl.sum(p, axis=1)

        if v.dtype.is_fp8():
            p = p.to(v_cache_ptr.dtype.element_ty)
        else:
            p = p.to(COMPUTE_TYPE)

        max_logits_offs = (
            seq_idx * stride_max_logits_s
            + kv_head_idx * stride_max_logits_nh
            + seq_part_idx * stride_max_logits_p
            + q_grp_offs
        )
        m_grp_mask = q_grp_offs < QUERY_GRP_SZ
        tl.store(max_logits_ptr + max_logits_offs + QID * QUERY_GRP_SZ, max_logit_new, mask=m_grp_mask)
        tl.store(exp_sums_ptr + max_logits_offs + QID * QUERY_GRP_SZ, exp_sum, mask=m_grp_mask)

        # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
        acc = tl.dot(p, v, out_dtype=tl.float32)
        acc = acc / exp_sum[:, None]
        acc = acc.to(COMPUTE_TYPE)

        logits_offs = seq_idx * stride_logits_s
        logits_offs += kv_head_idx * stride_logits_nh
        logits_offs += (
            seq_part_idx * stride_logits_p
            + q_grp_offs[:, None] * stride_logits_g
            + head_sz_offs[None, :]
        )
        tl.store(logits_ptr + logits_offs + QID * QUERY_GRP_SZ * stride_logits_g, acc, mask=q_mask)
    elif Q_SEQ_LEN >= 3:
        QID = 2
    elif Q_SEQ_LEN >= 4:
        QID = 3


@triton.jit
def pa_decode_v2_fp8(
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
    QUERY_GRP_SZ: tl.constexpr,
    QUERY_GRP_SZ_POW2: tl.constexpr,
    KV_BLK_SZ: tl.constexpr,
    KV_BLK_SZ_POW2: tl.constexpr,
    SEQ_PARTITION_SZ: tl.constexpr,
    KV_16B_ELE_NUM: tl.constexpr,
    TRANS_V: tl.constexpr,          # [num_blks, num_kv_heads, kv_blk_sz/x, head_sz, x]
    IS_CAUSAL: tl.constexpr,
):
    """
    #TODO: Add Doc
    """
    tl.static_assert(Q_SEQ_LEN <= 4, "Q_SEQ_LEN={}, Do not support Q_SEQ_LEN > 4".format(Q_SEQ_LEN))

    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_part_idx = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: tl.constexpr = KV_16B_ELE_NUM

    kv_seq_len = tl.load(seq_lens_ptr + seq_idx)
    seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    if seq_start_idx >= kv_seq_len:
        return

    seq_end_idx = tl.minimum(seq_start_idx + SEQ_PARTITION_SZ, kv_seq_len)
    MAX_NUM_KV_BLKS: tl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ
    num_kv_blks = tl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)

    blk_offs = tl.arange(0, KV_BLK_SZ_POW2)
    head_sz_offs = tl.arange(0, HEAD_SZ_POW2)
    head_sz_div_offs = tl.arange(0, HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
    q_grp_offs = tl.arange(0, QUERY_GRP_SZ_POW2)
    contiguous_kv_elems_offs = tl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD)

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = tl.zeros([QUERY_GRP_SZ_POW2], dtype=tl.float32)
    else:
        alibi_slope = tl.load(
            alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
            mask=q_grp_offs < QUERY_GRP_SZ,
            other=0.0,
        )

    # load all kv blocks in one time
    blk_ids = tl.arange(0, MAX_NUM_KV_BLKS)
    masked_blk_ids = tl.where(blk_ids < num_kv_blks, blk_ids, 0)
    kv_blk_start = seq_part_idx * MAX_NUM_KV_BLKS
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    kv_blk_nums = tl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)


    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * Q_SEQ_LEN * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    q0 = tl.load(q_ptr + q_offs, mask=q_mask, other=0.0)

    q_scale_offs = seq_idx * Q_SEQ_LEN * q_scale_stride0 + kv_head_idx * QUERY_GRP_SZ + q_grp_offs
    if q0.dtype.is_fp8():
        # [QUERY_GRP_SZ_POW2]
        q_scale_val = tl.load(q_scale + q_scale_offs, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
        # q_scale_val = tl.load(q_scale + q_scale_offs)
        q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
        q0 = q_scale_val * q0.to(tl.float32)

    q1 = q0
    q2 = q0
    q3 = q0
    if Q_SEQ_LEN >= 2:
        qid = 1
        q1 = tl.load(q_ptr + q_offs + qid * stride_q_s, mask=q_mask, other=0.0)
        if q1.dtype.is_fp8():
            # [QUERY_GRP_SZ_POW2]
            q_scale_val = tl.load(q_scale + q_scale_offs + qid * q_scale_stride0, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
            q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
            q1 = q_scale_val * q1.to(tl.float32)
    elif Q_SEQ_LEN >= 3:
        qid = 2
        q2 = tl.load(q_ptr + q_offs + qid * stride_q_s, mask=q_mask, other=0.0)
        if q2.dtype.is_fp8():
            # [QUERY_GRP_SZ_POW2]
            q_scale_val = tl.load(q_scale + q_scale_offs + qid * q_scale_stride0, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
            q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
            q2 = q_scale_val * q2.to(tl.float32)
    elif Q_SEQ_LEN >= 4:
        qid = 3
        q3 = tl.load(q_ptr + q_offs + qid * stride_q_s, mask=q_mask, other=0.0)
        if q3.dtype.is_fp8():
            # [QUERY_GRP_SZ_POW2]
            q_scale_val = tl.load(q_scale + q_scale_offs + qid * q_scale_stride0, mask=q_grp_offs < QUERY_GRP_SZ, other=0.0)
            q_scale_val = tl.broadcast_to(q_scale_val[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
            q3 = q_scale_val * q3.to(tl.float32)


    # k_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    k_blk_offs = (
        kv_blk_nums[:, None, None, None] * stride_k_b
        + kv_head_idx * stride_k_nh
        + head_sz_div_offs[None, :, None, None] * stride_k_hz
        + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
        + contiguous_kv_elems_offs[None, None, None, :]
    )
    # blk_seq_offs[MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2]
    blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ  # blk_ids: [MAX_NUM_KV_BLKS]
                    + blk_offs[None, :]) # blk_offs: [KV_BLK_SZ_POW2]
    # k_mask = (
    #     (blk_seq_offs[:, None, :, None] < kv_seq_len) &
    #     (blk_offs[None, None, :, None] < KV_BLK_SZ) &
    #     (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
    # )

    # k[MAX_NUM_KV_BLKS, HEAD_SZ_POW2/x, KV_BLK_SZ_POW2, x]
    # k = tl.load(k_cache_ptr + k_blk_offs, mask=blk_seq_offs[:, None, :, None] < kv_seq_len, other=0.0)
    k_temp = tl.load(k_cache_ptr + k_blk_offs)
    # k = k_0.to(tl.float32) * k_scale if k_0.dtype.is_fp8() else k_0
    # k[HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    k_temp = tl.permute(k_temp, [1, 3, 0, 2]) # [HEAD_SZ_POW2/x, x, MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2]
    k_temp = tl.reshape(k_temp, [HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])

    if k_temp.dtype.is_fp8():
        # q = q.to(tl.float8e4nv)
        # q = q.to(tl.float8e4b8)
        k = k_temp
    else:
        k = k_temp.to(COMPUTE_TYPE)

    blk_seq_flatten_offs = tl.reshape(blk_seq_offs, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
    k_scale_val = k_scale
    v_scale_val = v_scale
    if k.dtype.is_fp8():
    # if tl.is_tensor(k_scale):
        # [MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2]
        kv_scale_offs = kv_blk_nums[:, None] * kv_scale_stride0 + kv_head_idx * kv_scale_stride1 + blk_offs[None, :]
        # k_scale_val = tl.load(k_scale + kv_scale_offs, mask=blk_seq_offs < kv_seq_len, other=0.0)
        # v_scale_val = tl.load(v_scale + kv_scale_offs, mask=blk_seq_offs < kv_seq_len, other=0.0)
        k_scale_val = tl.load(k_scale + kv_scale_offs)
        v_scale_val = tl.load(v_scale + kv_scale_offs)
        # k_scale_val = tl.zeros((MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2), dtype=tl.float32)
        # v_scale_val = tl.zeros((MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2), dtype=tl.float32)

        k_scale_val = tl.reshape(k_scale_val, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
        v_scale_val = tl.reshape(v_scale_val, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
        # [QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
        k_scale_val = tl.broadcast_to(k_scale_val[None, :], QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2)
        # v_scale_val = tl.broadcast_to(v_scale_val[None, :], QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2)
        v_scale_val = tl.broadcast_to(v_scale_val[:, None], MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2)
        k_scale_val = softmax_scale * k_scale_val


    v = None
    if TRANS_V:
        # if seq_idx == 0 and kv_head_idx == 0 and seq_part_idx == 0:
        #     print('CONTIGUOUS_KV_ELEMS_16B_LOAD=', CONTIGUOUS_KV_ELEMS_16B_LOAD)
        blk_sz_div_offs = tl.arange(0, KV_BLK_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD)
        # [MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2/x, HEAD_SZ_POW2, x]
        v_blk_offs = (
            kv_blk_nums[:, None, None, None] * stride_v_b
            + kv_head_idx * stride_v_nh
            + blk_sz_div_offs[None, :, None, None] * stride_v_hz
            + head_sz_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
            + contiguous_kv_elems_offs[None, None, None, :]
        )
        v = tl.load(v_cache_ptr + v_blk_offs)
        # [MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2/x, HEAD_SZ_POW2, x] --> [MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2/x, x, HEAD_SZ_POW2]
        # v = tl.permute(v, [0, 1, 3, 2])
        # v = tl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2])
        v = tl.permute(v, [0, 2, 1, 3])
        v = tl.reshape(v, [MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2])
        v = tl.permute(v, [0, 2, 1])
        v = tl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2])
    else:
        # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        v_blk_offs = (
            kv_blk_nums[:, None, None] * stride_v_b
            + kv_head_idx * stride_v_nh
            + head_sz_offs[None, :, None] * stride_v_hz
            + blk_offs[None, None, :]
        )
        v_mask = (
            (blk_seq_offs[:, None, :] < kv_seq_len) &
            (blk_offs[None, None, :] < KV_BLK_SZ) &
            (head_sz_offs[None, :, None] < HEAD_SZ)
        )
        # v[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
        # v = tl.load(v_cache_ptr + v_blk_offs, mask=v_mask, other=0.0)
        v = tl.load(v_cache_ptr + v_blk_offs)
        # v[MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2]
        v = tl.permute(v, [0, 2, 1])
        v = tl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2])        

    if v.dtype.is_fp8():
        v = v_scale_val * v.to(tl.float32)
        v = v.to(v_cache_ptr.dtype.element_ty)
    else:
        v = v.to(COMPUTE_TYPE)


    QID = 0
    q = q0
    if k_temp.dtype.is_fp8():
        q = q.to(k_cache_ptr.dtype.element_ty)
    else:
        q = (q.to(tl.float32) * softmax_scale).to(COMPUTE_TYPE)

    # qk[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    qk = tl.dot(q, k, out_dtype=tl.float32)
    if k_temp.dtype.is_fp8():
        qk = k_scale_val * qk

    qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ)
    if IS_CAUSAL:
        causal_mask = blk_seq_flatten_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
    else:
        causal_mask = blk_seq_flatten_offs[None, :] < kv_seq_len
    qk_bound_mask = qk_bound_mask & causal_mask
    # qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_offs[None, :] < kv_seq_len)

    if alibi_slopes is not None:
        qk += (alibi_slope[:, None] * (blk_seq_flatten_offs - kv_seq_len + 1)[None, :]).to(
            tl.float32
        )

    # if [0, SEQ_PARTITION_SZ) are all -inf, the result will be nan
    # so, we use -1e37 other than -inf
    # qk = tl.where(qk_bound_mask, qk, float("-inf"))
    qk = tl.where(qk_bound_mask, qk, float(-1e37))

    max_logit_new = tl.max(qk, axis=1)
    # p[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
    exp_sum = tl.sum(p, axis=1)

    if v.dtype.is_fp8():
        # p = v_scale_val * p
        p = p.to(v_cache_ptr.dtype.element_ty)
    else:
        p = p.to(COMPUTE_TYPE)

    max_logits_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + q_grp_offs
    )
    m_grp_mask = q_grp_offs < QUERY_GRP_SZ
    tl.store(max_logits_ptr + max_logits_offs + QID * QUERY_GRP_SZ, max_logit_new, mask=m_grp_mask)
    tl.store(exp_sums_ptr + max_logits_offs + QID * QUERY_GRP_SZ, exp_sum, mask=m_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    acc = tl.dot(p, v, out_dtype=tl.float32)
    acc = acc / exp_sum[:, None]
    acc = acc.to(COMPUTE_TYPE)

    # end up computation
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + q_grp_offs[:, None] * stride_logits_g
        + head_sz_offs[None, :]
    )
    tl.store(logits_ptr + logits_offs + QID * QUERY_GRP_SZ * stride_logits_g, acc, mask=q_mask)


    if Q_SEQ_LEN >= 2:
        QID = 1
        q = q1
        if k_temp.dtype.is_fp8():
            q = q.to(k_cache_ptr.dtype.element_ty)
        else:
            q = (q.to(tl.float32) * softmax_scale).to(COMPUTE_TYPE)

        # qk[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
        qk = tl.dot(q, k, out_dtype=tl.float32)
        if k_temp.dtype.is_fp8():
            qk = k_scale_val * qk

        qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ)
        if IS_CAUSAL:
            causal_mask = blk_seq_flatten_offs[None, :] < kv_seq_len - (Q_SEQ_LEN - 1 - QID)
        else:
            causal_mask = blk_seq_flatten_offs[None, :] < kv_seq_len
        qk_bound_mask = qk_bound_mask & causal_mask
        # qk_bound_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_flatten_offs[None, :] < kv_seq_len)

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (blk_seq_flatten_offs - kv_seq_len + 1)[None, :]).to(
                tl.float32
            )

        # if [0, SEQ_PARTITION_SZ) are all -inf, the result will be nan
        # so, we use -1e37 other than -inf
        qk = tl.where(qk_bound_mask, qk, float(-1e37))

        max_logit_new = tl.max(qk, axis=1)
        # p[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        exp_sum = tl.sum(p, axis=1)

        if v.dtype.is_fp8():
            # p = v_scale_val * p
            p = p.to(v_cache_ptr.dtype.element_ty)
        else:
            p = p.to(COMPUTE_TYPE)

        max_logits_offs = (
            seq_idx * stride_max_logits_s
            + kv_head_idx * stride_max_logits_nh
            + seq_part_idx * stride_max_logits_p
            + q_grp_offs
        )
        m_grp_mask = q_grp_offs < QUERY_GRP_SZ
        tl.store(max_logits_ptr + max_logits_offs + QID * QUERY_GRP_SZ, max_logit_new, mask=m_grp_mask)
        tl.store(exp_sums_ptr + max_logits_offs + QID * QUERY_GRP_SZ, exp_sum, mask=m_grp_mask)

        # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
        acc = tl.dot(p, v, out_dtype=tl.float32)
        acc = acc / exp_sum[:, None]
        acc = acc.to(COMPUTE_TYPE)

        logits_offs = seq_idx * stride_logits_s
        logits_offs += kv_head_idx * stride_logits_nh
        logits_offs += (
            seq_part_idx * stride_logits_p
            + q_grp_offs[:, None] * stride_logits_g
            + head_sz_offs[None, :]
        )
        tl.store(logits_ptr + logits_offs + QID * QUERY_GRP_SZ * stride_logits_g, acc, mask=q_mask)
    elif Q_SEQ_LEN >= 3:
        QID = 2
    elif Q_SEQ_LEN >= 4:
        QID = 3


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
    exp_sums = tl.load(exp_sums_ptr + exp_sums_offs, mask=exp_sums_mask, other=0.0)
    exp_sums *= tl.exp(max_logits - ml[None, :])

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
        logits_ptrs + logits_offset, mask=logits_mask[:, :, None], other=0.0
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
    QUERY_GRP_SZ,
    QUERY_GRP_SZ_POW2,
    KV_BLK_SZ,
    KV_BLK_SZ_POW2,
    SEQ_PARTITION_SZ,
    KV_16B_ELE_NUM,
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
        pa_decode_kernel = pa_decode_v2_fp8
        if KV_BLK_SZ > SEQ_PARTITION_SZ:
            pa_decode_kernel = pa_decode_v2_big_blk_fp8

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
            QUERY_GRP_SZ=QUERY_GRP_SZ,
            QUERY_GRP_SZ_POW2=QUERY_GRP_SZ_POW2,
            KV_BLK_SZ=KV_BLK_SZ,
            KV_BLK_SZ_POW2=KV_BLK_SZ_POW2,
            SEQ_PARTITION_SZ=SEQ_PARTITION_SZ,
            KV_16B_ELE_NUM=KV_16B_ELE_NUM,
            TRANS_V=TRANS_V,
            IS_CAUSAL=IS_CAUSAL,
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
    shape_info = (num_seqs, num_kv_heads, max_num_partitions, q_seq_len * query_grp_sz)
    max_logits = torch.zeros(shape_info, dtype=torch.float32, device=output.device)
    exp_sums = torch.zeros(shape_info, dtype=torch.float32, device=output.device)
    # tmp_output = torch.empty(
    tmp_output = torch.zeros(
        # (*shape_info, 256), dtype=output.dtype, device=output.device
        *shape_info, head_sz, dtype=output.dtype, device=output.device
    )

    if query_grp_sz <= 16:
        query_grp_sz_pow2 = 16
    else:
        query_grp_sz_pow2 = triton.next_power_of_2(query_grp_sz)
    if equi_query_grp_sz <= 16:
        equi_query_grp_sz_pow2 = 16
    else:
        equi_query_grp_sz_pow2 = triton.next_power_of_2(equi_query_grp_sz)
    trans_v = None
    if len(value_cache.shape) == 5:
        trans_v = True
    elif len(value_cache.shape) == 4:
        trans_v = False
    else:
        raise RuntimeError(f"Do not support such value_cache shape:{value_cache.shape}")

    output = output.reshape(batch_size, q_seq_len, num_kv_heads, query_grp_sz, head_sz)
    output = output.transpose(1, 2).reshape(batch_size, num_kv_heads * q_seq_len * query_grp_sz, head_sz).contiguous()

    # print(f"shape_info={shape_info}")
    # print(f"query.shape={query.shape}")
    # print(f"q_scale.shape={q_scale.shape}")
    # print(f"key_cache.shape={key_cache.shape}")
    # print(f"value_cache.shape={value_cache.shape}")
    # print(f"output.shape={output.shape}")
    # print(f"tmp_output.shape={tmp_output.shape}")
    # print(f"block_tables.shape={block_tables.shape}")
    # print(f"query.dtype={query.dtype}")
    # print(f"key_cache.dtype={key_cache.dtype}")
    # print(f"value_cache.dtype={value_cache.dtype}")
    # print(f"output.dtype={output.dtype}")
    # print(f"block_tables.dtype={block_tables.dtype}")
    # print(f"value_cache.stride()={value_cache.stride()}")
    # print(f"query.stride()={query.stride()}")
    # print(f"q_scale.stride()={q_scale.stride()}")
    # print(f"tmp_output.stride()={tmp_output.stride()}")
    # input_config = dict(
    #     q_seq_len=q_seq_len,
    #     kv_type=compute_type,
    #     COMPUTE_TYPE=compute_type,
    #     HEAD_SZ=head_sz,
    #     HEAD_SZ_POW2=head_sz_pow2,
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
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        attn_scale,
        q_scale,
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
        q_scale.stride(0),
        k_scale.stride(0),
        k_scale.stride(1),
        kv_type=compute_type,
        Q_SEQ_LEN=q_seq_len,
        COMPUTE_TYPE=compute_type,
        HEAD_SZ=head_sz,
        HEAD_SZ_POW2=head_sz_pow2,
        QUERY_GRP_SZ=query_grp_sz,
        QUERY_GRP_SZ_POW2=query_grp_sz_pow2,
        KV_BLK_SZ=kv_blk_sz,
        KV_BLK_SZ_POW2=kv_blk_sz_pow2,
        SEQ_PARTITION_SZ=_SEQ_PARTITION_SIZE,
        KV_16B_ELE_NUM=kv_16b_ele_num,
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
