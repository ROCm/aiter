# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mixed-precision SAGE attention forward kernel.

The HP LUT (kv_block_indices, lut_start, lut_count) selects which KV blocks are
processed in HIGH precision (INT8 + per-block scale) per (batch, head, q_block).
The LP LUT contains the complement (MXFP4 + per-row, per-32-group E8M0 scale).

Each precision is split into two specialised inner loops:

* ``_inner_hp_int8_nomask`` / ``_inner_lp_mxfp4_nomask`` -- iterate all but the
  last block in their respective LUT. No boundary masking; this is the fast
  path. Both prefetch the next block index ahead of time so the LUT load
  latency overlaps with the per-iteration GEMM.
* ``_inner_hp_int8_mask`` / ``_inner_lp_mxfp4_mask`` -- process the LAST block
  of the LUT (1 iteration) with boundary masking to handle the
  ``seqlen_k % BLOCK_N != 0`` tail.

Both precisions contribute to the same online-softmax accumulator. Online
softmax is associative w.r.t. block order so the final output is identical
to the per-iteration branched form.
"""

import triton
import triton.language as tl


@triton.jit
def _compute_n_extra(seqlen_k, BLOCK_N: tl.constexpr):
    if seqlen_k < BLOCK_N:
        n_extra = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra = seqlen_k % BLOCK_N
    else:
        n_extra = 0
    return n_extra


# =====================================================================
# HP inner loop (INT8) -- fast path, no masking.
# =====================================================================
@triton.jit
def _inner_hp_int8_nomask(
    acc,
    l_i,
    m_i,
    q_int8,
    q_descale_int8,
    k_int8_base,
    kd_int8_base,
    v_base,
    bias_base,
    stride_k_int8_n,
    stride_k_int8_d,
    stride_kd_int8_blk,
    stride_vk,
    stride_vd,
    stride_bn,
    block_indices,
    lut_start_val,
    n_blocks,
    offs_d_k_int8,
    offs_d_v,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    # Prefetch first block index.
    if n_blocks > 0:
        next_b = tl.load(block_indices + lut_start_val).to(tl.int32)
    else:
        next_b = tl.full([], 0, dtype=tl.int32)

    for i in range(0, n_blocks):
        start_b = next_b
        # Issue prefetch for next iteration BEFORE the heavy compute so the
        # LUT load overlaps with this iteration's GEMM.
        ip1 = i + 1
        if ip1 < n_blocks:
            next_b = tl.load(block_indices + lut_start_val + ip1).to(tl.int32)

        start_n = start_b * BLOCK_N

        k_ptrs = (
            k_int8_base
            + offs_d_k_int8[:, None] * stride_k_int8_d
            + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_k_int8_n
        )
        if PADDED_HEAD_QK:
            k_tile = tl.load(
                k_ptrs,
                mask=offs_d_k_int8[:, None] < ACTUAL_BLOCK_DMODEL_QK,
                other=0,
            )
        else:
            k_tile = tl.load(k_ptrs)

        k_descale = tl.load(kd_int8_base + start_b * stride_kd_int8_blk)

        v_ptrs = (
            v_base
            + (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_vk
            + offs_d_v[None, :] * stride_vd
        )
        if PADDED_HEAD_V:
            v = tl.load(
                v_ptrs,
                mask=offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V,
                other=0.0,
            )
        else:
            v = tl.load(v_ptrs)

        # Raw int32 -> fp32 GEMM result. We keep it un-descaled so that the
        # descale multiply can be fused with the softmax shift below.
        qk_raw = tl.dot(q_int8, k_tile).to(ACC_TYPE)
        scale = q_descale_int8 * k_descale

        if USE_BIAS:
            bias_ptrs = bias_base + start_n * stride_bn
            bias_v = tl.load(bias_ptrs)
            # ``qk_raw * scale + bias`` already lowers to a single FMA per
            # element. The subsequent ``qk - m_ij`` stays separate because
            # bias breaks the simple max-of-scaled equivalence.
            qk = qk_raw * scale + bias_v[None, :]
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"),
                float("-inf"),
                qk - m_ij[:, None],
            )
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            # Fused descale + softmax shift: ``scale`` is a positive scalar
            # so ``max(qk_raw * scale) == max(qk_raw) * scale``. Doing the
            # max in raw fp32 means we only multiply by ``scale`` on the
            # tiny [BLOCK_M] reduction result, and ``qk_raw * scale -
            # m_ij`` then lowers to a single v_fma_f32 per element instead
            # of a full elementwise mul followed by a full elementwise sub.
            m_ij = tl.maximum(m_i, tl.max(qk_raw, 1) * scale)
            q_shifted = qk_raw * scale - m_ij[:, None]
            m_diff = m_i - m_ij

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


# =====================================================================
# HP inner loop (INT8) -- processes a single masked block.
# =====================================================================
@triton.jit
def _inner_hp_int8_mask(
    acc,
    l_i,
    m_i,
    q_int8,
    q_descale_int8,
    k_int8_base,
    kd_int8_base,
    v_base,
    bias_base,
    stride_k_int8_n,
    stride_k_int8_d,
    stride_kd_int8_blk,
    stride_vk,
    stride_vd,
    stride_bn,
    seqlen_k,
    block_indices,
    lut_start_val,
    offs_n,
    offs_d_k_int8,
    offs_d_v,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    start_b = tl.load(block_indices + lut_start_val).to(tl.int32)
    start_n = start_b * BLOCK_N
    kv_offs_n = start_n + tl.arange(0, BLOCK_N)

    k_ptrs = (
        k_int8_base
        + offs_d_k_int8[:, None] * stride_k_int8_d
        + kv_offs_n[None, :] * stride_k_int8_n
    )
    k_mask = kv_offs_n[None, :] < seqlen_k
    if PADDED_HEAD_QK:
        k_mask = k_mask & (offs_d_k_int8[:, None] < ACTUAL_BLOCK_DMODEL_QK)
    k_tile = tl.load(k_ptrs, mask=k_mask, other=0)

    k_descale = tl.load(kd_int8_base + start_b * stride_kd_int8_blk)

    v_ptrs = v_base + kv_offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vd
    v_mask = kv_offs_n[:, None] < seqlen_k
    if PADDED_HEAD_V:
        v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)

    # Compute raw int32 GEMM, mask invalid columns to -inf, then defer the
    # descale so it can be fused with the softmax shift below.
    qk_raw = tl.dot(q_int8, k_tile).to(ACC_TYPE)
    qk_raw = tl.where(kv_offs_n[None, :] < seqlen_k, qk_raw, float("-inf"))
    scale = q_descale_int8 * k_descale

    if USE_BIAS:
        bias_ptrs = bias_base + start_n * stride_bn
        bias_v = tl.load(bias_ptrs, mask=kv_offs_n < seqlen_k, other=0.0)
        qk = qk_raw * scale + bias_v[None, :]
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )
    else:
        # Same fusion as the nomask path: ``qk_raw * scale - m_ij`` lowers
        # to a v_fma_f32 per element. ``-inf * scale`` is still ``-inf``
        # so masked columns continue to suppress contribution; the where
        # below handles the m_ij == -inf row case (otherwise -inf - -inf
        # would produce NaN).
        m_ij = tl.maximum(m_i, tl.max(qk_raw, 1) * scale)
        q_shifted_fma = qk_raw * scale - m_ij[:, None]
        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), q_shifted_fma
        )
    m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)

    p = tl.math.exp2(q_shifted)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_diff)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


# =====================================================================
# LP inner loop (MXFP4) -- fast path, no masking.
# =====================================================================
@triton.jit
def _inner_lp_mxfp4_nomask(
    acc,
    l_i,
    m_i,
    q_fp4,
    q_descale_fp4,
    k_fp4_base,
    kd_fp4_base,
    v_base,
    bias_base,
    stride_k_fp4_n,
    stride_k_fp4_d,
    stride_kd_fp4_n,
    stride_vk,
    stride_vd,
    stride_bn,
    block_indices,
    lut_start_val,
    n_blocks,
    offs_d_k_fp4,
    offs_d_v,
    offs_d_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
):
    if n_blocks > 0:
        next_b = tl.load(block_indices + lut_start_val).to(tl.int32)
    else:
        next_b = tl.full([], 0, dtype=tl.int32)

    for i in range(0, n_blocks):
        start_b = next_b
        ip1 = i + 1
        if ip1 < n_blocks:
            next_b = tl.load(block_indices + lut_start_val + ip1).to(tl.int32)

        start_n = start_b * BLOCK_N

        k_ptrs = (
            k_fp4_base
            + offs_d_k_fp4[:, None] * stride_k_fp4_d
            + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_k_fp4_n
        )
        if PADDED_HEAD_QK:
            k_tile = tl.load(
                k_ptrs,
                mask=offs_d_k_fp4[:, None] < (ACTUAL_BLOCK_DMODEL_QK // 2),
                other=0,
            )
        else:
            k_tile = tl.load(k_ptrs)

        kd_ptrs = (
            kd_fp4_base
            + (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_kd_fp4_n
            + offs_d_scale[None, :]
        )
        k_descale = tl.load(kd_ptrs)

        v_ptrs = (
            v_base
            + (start_n + tl.arange(0, BLOCK_N))[:, None] * stride_vk
            + offs_d_v[None, :] * stride_vd
        )
        if PADDED_HEAD_V:
            v = tl.load(
                v_ptrs,
                mask=offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V,
                other=0.0,
            )
        else:
            v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC_TYPE)
        qk = tl.dot_scaled(
            q_fp4,
            q_descale_fp4,
            Q_DTYPE_STR,
            k_tile,
            k_descale,
            K_DTYPE_STR,
            fast_math=True,
            acc=qk,
        )

        if USE_BIAS:
            bias_ptrs = bias_base + start_n * stride_bn
            bias_v = tl.load(bias_ptrs)
            qk = qk + bias_v[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        if USE_BIAS:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"),
                float("-inf"),
                qk - m_ij[:, None],
            )
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            q_shifted = qk - m_ij[:, None]
            m_diff = m_i - m_ij

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


# =====================================================================
# LP inner loop (MXFP4) -- processes a single masked block.
# =====================================================================
@triton.jit
def _inner_lp_mxfp4_mask(
    acc,
    l_i,
    m_i,
    q_fp4,
    q_descale_fp4,
    k_fp4_base,
    kd_fp4_base,
    v_base,
    bias_base,
    stride_k_fp4_n,
    stride_k_fp4_d,
    stride_kd_fp4_n,
    stride_vk,
    stride_vd,
    stride_bn,
    seqlen_k,
    block_indices,
    lut_start_val,
    offs_n,
    offs_d_k_fp4,
    offs_d_v,
    offs_d_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ACC_TYPE: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
):
    start_b = tl.load(block_indices + lut_start_val).to(tl.int32)
    start_n = start_b * BLOCK_N
    kv_offs_n = start_n + tl.arange(0, BLOCK_N)

    k_ptrs = (
        k_fp4_base
        + offs_d_k_fp4[:, None] * stride_k_fp4_d
        + kv_offs_n[None, :] * stride_k_fp4_n
    )
    k_mask = kv_offs_n[None, :] < seqlen_k
    if PADDED_HEAD_QK:
        k_mask = k_mask & (offs_d_k_fp4[:, None] < (ACTUAL_BLOCK_DMODEL_QK // 2))
    k_tile = tl.load(k_ptrs, mask=k_mask, other=0)

    kd_ptrs = (
        kd_fp4_base
        + kv_offs_n[:, None] * stride_kd_fp4_n
        + offs_d_scale[None, :]
    )
    k_descale = tl.load(kd_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0)

    v_ptrs = v_base + kv_offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vd
    v_mask = kv_offs_n[:, None] < seqlen_k
    if PADDED_HEAD_V:
        v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)

    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC_TYPE)
    qk = tl.where(kv_offs_n[None, :] < seqlen_k, qk, float("-inf"))
    qk = tl.dot_scaled(
        q_fp4,
        q_descale_fp4,
        Q_DTYPE_STR,
        k_tile,
        k_descale,
        K_DTYPE_STR,
        fast_math=True,
        acc=qk,
    )

    if USE_BIAS:
        bias_ptrs = bias_base + start_n * stride_bn
        bias_v = tl.load(bias_ptrs, mask=kv_offs_n < seqlen_k, other=0.0)
        qk = qk + bias_v[None, :]

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    q_shifted = tl.where(
        m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
    )
    m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)

    p = tl.math.exp2(q_shifted)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_diff)
    acc = acc * alpha[:, None]
    l_i = l_i * alpha + l_ij
    m_i = m_ij
    acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    return acc, l_i, m_i


# =====================================================================
# Top-level kernel
# =====================================================================
@triton.jit
def sage_fwd_mxfp4_mixed(
    Q_INT8,
    Q_FP4,
    K_INT8,
    K_FP4,
    V,
    Q_DESCALE_INT8,
    Q_DESCALE_FP4,
    K_DESCALE_INT8,
    K_DESCALE_FP4,
    V_DESCALE,
    BIAS,
    OUT,
    # Q strides
    stride_q_int8_z,
    stride_q_int8_h,
    stride_q_int8_m,
    stride_q_int8_d,
    stride_q_fp4_z,
    stride_q_fp4_h,
    stride_q_fp4_m,
    stride_q_fp4_d,
    # K strides
    stride_k_int8_z,
    stride_k_int8_h,
    stride_k_int8_n,
    stride_k_int8_d,
    stride_k_fp4_z,
    stride_k_fp4_h,
    stride_k_fp4_n,
    stride_k_fp4_d,
    # V strides
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vd,
    # Q descale strides
    stride_qd_int8_z,
    stride_qd_int8_h,
    stride_qd_int8_blk,
    stride_qd_fp4_z,
    stride_qd_fp4_h,
    stride_qd_fp4_m,
    # K descale strides
    stride_kd_int8_z,
    stride_kd_int8_h,
    stride_kd_int8_blk,
    stride_kd_fp4_z,
    stride_kd_fp4_h,
    stride_kd_fp4_n,
    # V descale strides
    stride_vd_z,
    stride_vd_h,
    # Bias strides
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    # Output strides
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    # HP LUT
    hp_kv_block_indices,
    hp_lut_start,
    hp_lut_count,
    # LP LUT (complement)
    lp_kv_block_indices,
    lp_lut_start,
    lp_lut_count,
    # Constants
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32
    ACC_TYPE: tl.constexpr = tl.float32
    Q_DTYPE_STR: tl.constexpr = "e2m1"
    K_DTYPE_STR: tl.constexpr = "e2m1"

    start_m = tl.program_id(0).to(tl.int64)
    off_h_q = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)
    off_h_k = off_h_q // (HQ // HK)

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    seqlen_q = MAX_SEQLENS_Q
    seqlen_k = MAX_SEQLENS_K

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_q_int8 = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_q_fp4 = tl.arange(0, BLOCK_DMODEL_QK // 2)
    offs_d_k_int8 = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_k_fp4 = tl.arange(0, BLOCK_DMODEL_QK // 2)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_scale = tl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP)

    # LUT bookkeeping
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    lut_idx = off_z * (HQ * num_q_blocks) + off_h_q * num_q_blocks + start_m
    n_hp = tl.load(hp_lut_count + lut_idx)
    n_lp = tl.load(lp_lut_count + lut_idx)
    hp_lut_start_val = tl.load(hp_lut_start + lut_idx)
    lp_lut_start_val = tl.load(lp_lut_start + lut_idx)

    # ----- Q tensor pointers -----
    q_int8_ptrs = (
        Q_INT8
        + off_z * stride_q_int8_z
        + off_h_q * stride_q_int8_h
        + offs_m[:, None] * stride_q_int8_m
        + offs_d_q_int8[None, :] * stride_q_int8_d
    )
    q_fp4_ptrs = (
        Q_FP4
        + off_z * stride_q_fp4_z
        + off_h_q * stride_q_fp4_h
        + offs_m[:, None] * stride_q_fp4_m
        + offs_d_q_fp4[None, :] * stride_q_fp4_d
    )

    q_row_mask = offs_m[:, None] < seqlen_q

    # ----- Online softmax state -----
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACC_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACC_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACC_TYPE)

    k_int8_base = K_INT8 + off_z * stride_k_int8_z + off_h_k * stride_k_int8_h
    k_fp4_base = K_FP4 + off_z * stride_k_fp4_z + off_h_k * stride_k_fp4_h
    kd_int8_base = (
        K_DESCALE_INT8 + off_z * stride_kd_int8_z + off_h_k * stride_kd_int8_h
    )
    kd_fp4_base = K_DESCALE_FP4 + off_z * stride_kd_fp4_z + off_h_k * stride_kd_fp4_h
    v_base = V + off_z * stride_vz + off_h_k * stride_vh

    if USE_BIAS:
        bias_base = (
            BIAS
            + off_z * stride_bz
            + off_h_q * stride_bh
            + start_m * stride_bm
            + tl.cast(offs_n, tl.int64) * stride_bn
        )
    else:
        bias_base = BIAS

    # =====================================================================
    # HP path (INT8): nomask for n_hp-1 blocks, then mask for the last one.
    # =====================================================================
    if n_hp > 0:
        if PADDED_HEAD_QK:
            q_int8 = tl.load(
                q_int8_ptrs,
                mask=q_row_mask & (offs_d_q_int8[None, :] < ACTUAL_BLOCK_DMODEL_QK),
                other=0,
            )
        else:
            q_int8 = tl.load(q_int8_ptrs, mask=q_row_mask, other=0)

        q_descale_int8 = tl.load(
            Q_DESCALE_INT8
            + off_z * stride_qd_int8_z
            + off_h_q * stride_qd_int8_h
            + start_m * stride_qd_int8_blk
        )

        acc, l_i, m_i = _inner_hp_int8_nomask(
            acc,
            l_i,
            m_i,
            q_int8,
            q_descale_int8,
            k_int8_base,
            kd_int8_base,
            v_base,
            bias_base,
            stride_k_int8_n,
            stride_k_int8_d,
            stride_kd_int8_blk,
            stride_vk,
            stride_vd,
            stride_bn,
            hp_kv_block_indices,
            hp_lut_start_val,
            n_hp - 1,
            offs_d_k_int8,
            offs_d_v,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_BIAS,
            ACC_TYPE,
        )

        # Last HP block -- always masked.
        invalid_q_rows = offs_m >= seqlen_q
        m_i = tl.where(invalid_q_rows, float("-inf"), m_i)
        l_i = tl.where(invalid_q_rows, 1.0, l_i)
        acc = tl.where(invalid_q_rows[:, None], 0.0, acc)
        acc, l_i, m_i = _inner_hp_int8_mask(
            acc,
            l_i,
            m_i,
            q_int8,
            q_descale_int8,
            k_int8_base,
            kd_int8_base,
            v_base,
            bias_base,
            stride_k_int8_n,
            stride_k_int8_d,
            stride_kd_int8_blk,
            stride_vk,
            stride_vd,
            stride_bn,
            seqlen_k,
            hp_kv_block_indices,
            hp_lut_start_val + (n_hp - 1),
            offs_n,
            offs_d_k_int8,
            offs_d_v,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_BIAS,
            ACC_TYPE,
        )

    # =====================================================================
    # LP path (MXFP4): nomask for n_lp-1, then mask for the last one.
    # =====================================================================
    if n_lp > 0:
        if PADDED_HEAD_QK:
            q_fp4 = tl.load(
                q_fp4_ptrs,
                mask=q_row_mask
                & (offs_d_q_fp4[None, :] < (ACTUAL_BLOCK_DMODEL_QK // 2)),
                other=0,
            )
        else:
            q_fp4 = tl.load(q_fp4_ptrs, mask=q_row_mask, other=0)

        q_descale_fp4_ptrs = (
            Q_DESCALE_FP4
            + off_z * stride_qd_fp4_z
            + off_h_q * stride_qd_fp4_h
            + offs_m[:, None] * stride_qd_fp4_m
            + offs_d_scale[None, :]
        )
        q_descale_fp4 = tl.load(q_descale_fp4_ptrs, mask=q_row_mask, other=0)

        acc, l_i, m_i = _inner_lp_mxfp4_nomask(
            acc,
            l_i,
            m_i,
            q_fp4,
            q_descale_fp4,
            k_fp4_base,
            kd_fp4_base,
            v_base,
            bias_base,
            stride_k_fp4_n,
            stride_k_fp4_d,
            stride_kd_fp4_n,
            stride_vk,
            stride_vd,
            stride_bn,
            lp_kv_block_indices,
            lp_lut_start_val,
            n_lp - 1,
            offs_d_k_fp4,
            offs_d_v,
            offs_d_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_BIAS,
            ACC_TYPE,
            Q_DTYPE_STR,
            K_DTYPE_STR,
        )

        invalid_q_rows = offs_m >= seqlen_q
        m_i = tl.where(invalid_q_rows, float("-inf"), m_i)
        l_i = tl.where(invalid_q_rows, 1.0, l_i)
        acc = tl.where(invalid_q_rows[:, None], 0.0, acc)
        acc, l_i, m_i = _inner_lp_mxfp4_mask(
            acc,
            l_i,
            m_i,
            q_fp4,
            q_descale_fp4,
            k_fp4_base,
            kd_fp4_base,
            v_base,
            bias_base,
            stride_k_fp4_n,
            stride_k_fp4_d,
            stride_kd_fp4_n,
            stride_vk,
            stride_vd,
            stride_bn,
            seqlen_k,
            lp_kv_block_indices,
            lp_lut_start_val + (n_lp - 1),
            offs_n,
            offs_d_k_fp4,
            offs_d_v,
            offs_d_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_BIAS,
            ACC_TYPE,
            Q_DTYPE_STR,
            K_DTYPE_STR,
        )

    # =====================================================================
    # Epilogue
    # =====================================================================
    invalid_mask = m_i == float("-inf")
    l_i_safe = tl.where(invalid_mask, 1.0, l_i)
    l_recip = 1 / l_i_safe[:, None]

    v_descale_ptr = V_DESCALE + off_z * stride_vd_z + off_h_k * stride_vd_h + offs_d_v
    v_descale = tl.load(
        v_descale_ptr, mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V, other=0.0
    )
    acc = acc * l_recip * v_descale
    acc = tl.where(invalid_mask[:, None], 0.0, acc)

    o_ptrs = (
        OUT
        + off_z * stride_oz
        + off_h_q * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d_v[None, :] * stride_od
    )
    o_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_V:
        o_mask = o_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
    tl.store(o_ptrs, acc.to(OUT.dtype.element_ty), mask=o_mask)
