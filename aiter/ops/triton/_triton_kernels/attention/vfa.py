# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
VFA (Vector Relieved Flash Attention) m-init estimator kernel.

Computes a per-row running-max estimate by dotting Q against the REAL int8 K
rows of the K blocks listed in ``Block_Idx`` and taking the per-row max.
Every evaluated block contributes its true block max, so the estimate is a
lower bound on the true rowmax that tightens as more blocks are listed -- no
safety margin needed. Cost is O(N_q * N_SAMPLES * BLOCK_N * D).

Feeding the resulting ``M_Init`` into ``sage_fwd`` with USE_PRECOMPUTED_MAX
lets the hot loop drop the online-softmax rowmax reduction and the acc rescale
(the four softmax vector ops), running ``p = exp2(qk - m_i)`` with ``m_i`` held
frozen.  See the host wrapper ``compute_m_proxy_topn`` in
``aiter/ops/triton/attention/block_sparse.py`` for the block-selection strategy
that builds ``Block_Idx``.

``Block_Idx`` is indexed with full [B, H_Q, num_q_blocks, N_SAMPLES] strides so
a single kernel serves both the shared-table and per-row selection modes:
  * per-(q-block) guided top-k -> pass the real 4D strides;
  * one shared sampled set     -> pass a 1D [N_SAMPLES] table with the
    batch/head/q-block strides set to 0, broadcasting it to every program.
"""

import triton
import triton.language as tl


@triton.jit
def _sage_vfa_m_blockidx_kernel(
    Q,                  # int8 query tensor
    K,                  # int8 key tensor
    Q_Descale,          # fp32 [B, H_Q, num_q_blocks]
    K_Descale,          # fp32 [B, H_K, num_k_blocks]
    Block_Idx,          # int32 [B, H_Q, num_q_blocks, N_SAMPLES]
    M_Init,             # fp32 [B, H_Q, num_q_blocks, BLOCK_M] output
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_qsz, stride_qsh, stride_qsblk,
    stride_ksz, stride_ksh, stride_ksblk,
    stride_biz, stride_bih, stride_biqblk, stride_bis,
    stride_mz, stride_mh, stride_mblk, stride_mr,
    SEQLEN_Q,
    SEQLEN_K,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    N_SAMPLES: tl.constexpr,
):
    start_m = tl.program_id(0).to(tl.int64)
    off_h_q = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_QK)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h_q * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < SEQLEN_Q
    if PADDED_HEAD_QK:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL_QK)
    q = tl.load(q_ptrs, mask=q_mask, other=0)

    q_descale = tl.load(
        Q_Descale + off_z * stride_qsz + off_h_q * stride_qsh + start_m * stride_qsblk
    )

    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    k_base_ptrs = k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn
    k_descale_off = K_Descale + off_z * stride_ksz + off_h_k * stride_ksh

    bi_off = (
        Block_Idx
        + off_z * stride_biz
        + off_h_q * stride_bih
        + start_m * stride_biqblk
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    for s in range(0, N_SAMPLES):
        jb = tl.load(bi_off + s * stride_bis).to(tl.int64)
        start_n = jb * BLOCK_N
        k_ptrs = k_base_ptrs + start_n * stride_kn

        col = start_n + offs_n
        col_mask = col < SEQLEN_K
        if PADDED_HEAD_QK:
            k_mask = (offs_d[:, None] < ACTUAL_BLOCK_DMODEL_QK) & col_mask[None, :]
        else:
            k_mask = tl.broadcast_to(col_mask[None, :], (BLOCK_DMODEL_QK, BLOCK_N))
        k = tl.load(k_ptrs, mask=k_mask, other=0)

        k_descale = tl.load(k_descale_off + jb * stride_ksblk)
        qk = tl.dot(q, k).to(tl.float32) * (q_descale * k_descale)
        qk = tl.where(col_mask[None, :], qk, float("-inf"))
        m_i = tl.maximum(m_i, tl.max(qk, axis=1))

    m_ptrs = (
        M_Init
        + off_z * stride_mz
        + off_h_q * stride_mh
        + start_m * stride_mblk
        + tl.arange(0, BLOCK_M) * stride_mr
    )
    tl.store(m_ptrs, m_i, mask=offs_m < SEQLEN_Q)
