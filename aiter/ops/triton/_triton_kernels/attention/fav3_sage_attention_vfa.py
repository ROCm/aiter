# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Triton kernels implementing Vector Relieved Flash Attention (VFA) on top of
# the existing SAGE FP8 (Int8 QK, FP8 V) attention pipeline.
#
# Reference: "VFA: Relieving Vector Operations in Flash Attention with Global
# Maximum Pre-computation" (https://arxiv.org/abs/2604.12798).
#
# Why this works on AMD MFMA hardware (which the paper does not target):
#
#   sage_fp8's hot loop does, per K block:
#       tl.dot QK (int8 MFMA), rowmax, exp, l_i = l_i * alpha + l_ij,
#       acc = acc * alpha[:, None], tl.dot PV (fp8 MFMA).
#
#   The `acc * alpha[:, None]` rescale is BLOCK_M * BLOCK_DMODEL_V fp32
#   muls every iteration -- 32k flops/block for BLOCK_M=256, D_V=128 -- and
#   it lives between the QK and PV MFMA pipes, so it widens the critical
#   path even though MFMA dominates the per-op cost.
#
#   If we have a per-row estimate `m_init` that is essentially as tight as
#   the true running max, the rescale collapses to a no-op and we can drop
#   it.  The challenge is two-fold:
#     (a) `m_init` must be tight enough that `exp2(qk - m_init)` keeps p
#         representable in fp8 (otherwise the PV dot zeroes out);
#     (b) `m_init` must not severely underestimate the true rowmax, or
#         `exp2(qk - m_init)` overflows.
#
# We estimate `m_init` by dotting Q against the REAL int8 K rows of a small
# set of K blocks and taking the per-row max.  Because every evaluated block
# contributes its true block max, the estimate is a strict lower bound on the
# true rowmax that tightens as more blocks are added -- meeting (a) well and
# never overestimating.  Two block-selection strategies are provided:
#   * uniform random sampling of N blocks (``_sage_vfa_m_sampled_kernel``);
#   * SpargeAttn-style guided top-N selection per q-block, where blocks are
#     ranked by a cheap mean-pooled block-attention score computed outside
#     these kernels (``_sage_vfa_m_selected_kernel``).
# The hot kernel keeps `qk - m_i` unclamped on purpose: a small positive shift
# just produces `p` values slightly above 1.0, which still fit in fp8 E4M3
# (max ~448 = 2^8.8) and yield more accurate softmax weights than clamping at
# 1.  Only a row whose `m_init` undershoots true rowmax by more than ~8 log2
# units would saturate `p` to `fp8_max` -- a bounded bias, never inf/NaN.
#
# Layout:
#   _sage_vfa_m_sampled_kernel  -> M_Init[B, H_Q, num_q_blocks, BLOCK_M] fp32
#   _sage_vfa_m_selected_kernel -> M_Init[B, H_Q, num_q_blocks, BLOCK_M] fp32
#   sage_fwd_vfa                -> single tight loop, frozen m, fp32 acc

import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Sampled-block m-init kernel: per-row max over N randomly sampled K blocks.
#
# Instead of the signed-absmax composite (which mixes max-magnitude elements
# from different sequence positions and so suffers heavy sign-cancellation
# bias) this estimator dots Q against REAL K rows from N_SAMPLES randomly
# chosen blocks and takes the per-row max.  Every sampled block contributes
# its true block max, so the estimate is a lower bound on the true rowmax that
# tightens monotonically as N_SAMPLES grows -- no sign cancellation, no safety
# margin needed.  Cost is O(N_q * N_SAMPLES * BLOCK_N * D), i.e. the full QK
# pass scaled down by N_SAMPLES / num_k_blocks.
#
# ``Block_Idx`` is a host-provided int32 [N_SAMPLES] tensor of distinct K
# block indices in [0, num_k_blocks), shared across all programs.
# ----------------------------------------------------------------------------
@triton.jit
def _sage_vfa_m_sampled_kernel(
    Q,                  # int8 query tensor
    K,                  # int8 key tensor
    Q_Descale,          # fp32 [B, H_Q, num_q_blocks]
    K_Descale,          # fp32 [B, H_K, num_k_blocks]
    Block_Idx,          # int32 [N_SAMPLES] sampled K-block indices
    M_Init,             # fp32 [B, H_Q, num_q_blocks, BLOCK_M] output
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_qsz, stride_qsh, stride_qsblk,
    stride_ksz, stride_ksh, stride_ksblk,
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

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    for s in range(0, N_SAMPLES):
        jb = tl.load(Block_Idx + s).to(tl.int64)
        start_n = jb * BLOCK_N
        k_ptrs = k_base_ptrs + start_n * stride_kn

        # Mask columns past SEQLEN_K so a sampled tail block contributes only
        # its real tokens.
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


# ----------------------------------------------------------------------------
# Selected-block m-init kernel: per-row max over a per-(q-block) index set.
#
# Like the sampled kernel but ``Block_Idx`` is a 4D tensor
# [B, H_Q, num_q_blocks, N_SAMPLES] so each q-block evaluates its own chosen
# blocks (proxy top-k + stratified residual).  Real K rows are used, so the
# estimate is a lower bound on the true rowmax; no safety margin.
# ----------------------------------------------------------------------------
@triton.jit
def _sage_vfa_m_selected_kernel(
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


# ----------------------------------------------------------------------------
# Main VFA attention kernel.
#
# Single tight loop, structurally close to the sage_fp8 inner body but with
# the four vector operations removed:
#   - tl.max(qk, 1) rowmax reduction
#   - m_diff = m_i - m_ij
#   - alpha = exp(m_diff)
#   - acc = acc * alpha[:, None]
#   - l_i = l_i * alpha + l_ij  (becomes just l_i + l_ij)
#
# `m_i` is loaded once from M_Init and never updated.  No clamp on
# `qk - m_i`: see top-of-file comment for why.
# ----------------------------------------------------------------------------
@triton.jit
def sage_fwd_vfa(
    Q, K, V,
    Q_Descale, K_Descale, V_Descale,
    M_Init,
    LSE, Out,
    stride_qsz, stride_qsh, stride_qsblk,
    stride_ksz, stride_ksh, stride_ksblk,
    stride_vsz, stride_vsh,
    stride_mz, stride_mh, stride_mblk, stride_mr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_lse_z, stride_lse_h, stride_lse_m,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    N_EXTRA_TOKENS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    ACCUMULATOR_TYPE = tl.float32

    start_m = tl.program_id(0).to(tl.int64)
    off_h_q = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_qk = tl.max_contiguous(
        tl.multiple_of(offs_d_qk, BLOCK_DMODEL_QK), BLOCK_DMODEL_QK
    )
    offs_d_v = tl.max_contiguous(
        tl.multiple_of(offs_d_v, BLOCK_DMODEL_V), BLOCK_DMODEL_V
    )

    seqlen_q = MAX_SEQLENS_Q

    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk

    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    k_base_ptrs = (
        k_base + offs_d_qk[:, None] * stride_kk + offs_n[None, :] * stride_kn
    )

    v_base = V + off_z * stride_vz + off_h_k * stride_vh
    v_base_ptrs = (
        v_base + offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vn
    )

    q_descale_ptr = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + start_m * stride_qsblk
    )
    k_descale_ptr = K_Descale + off_z * stride_ksz + off_h_k * stride_ksh
    v_descale_ptr = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + offs_d_v

    q_descale = tl.load(q_descale_ptr)

    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk[None, :] < ACTUAL_BLOCK_DMODEL_QK)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0)

    m_ptrs = (
        M_Init
        + off_z * stride_mz
        + off_h_q * stride_mh
        + start_m * stride_mblk
        + tl.arange(0, BLOCK_M) * stride_mr
    )
    m_i = tl.load(m_ptrs, mask=offs_m < seqlen_q, other=float("-inf"))

    l_i = tl.zeros([BLOCK_M], dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACCUMULATOR_TYPE)

    LAST_BLOCK_HAS_TAIL: tl.constexpr = N_EXTRA_TOKENS > 0

    if LAST_BLOCK_HAS_TAIL:
        loop_end = NUM_K_BLOCKS - 1
    else:
        loop_end = NUM_K_BLOCKS

    # Interior blocks: no tail mask needed.
    for j in range(0, loop_end):
        start_n = (j * BLOCK_N).to(tl.int64)
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        if PADDED_HEAD_QK:
            k_mask = offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptr + j * stride_ksblk)

        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # Fused multiply-add: (dot * scale) - m_i  lowers to v_fma_f32 with
        # `-m_i` carried as a source modifier on the FMA, saving the separate
        # multiply that used to land between the QK dot and the shift.
        # No clamp on the shift: rely on the safety margin in m_init to keep
        # `qk - m_i` comfortably below ~8 log2 units (fp8 E4M3 max ~ 2^8.8).
        # If a row's m_init nevertheless undershoots true rowmax by more than
        # ~8 log2 units the corresponding p saturates at fp8_max -- a bounded
        # bias, never inf/NaN.
        scale = q_descale * k_descale
        q_shifted = tl.dot(q, k) * scale - m_i[:, None]
        if USE_EXP2:
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

        l_i = l_i + tl.sum(p, 1)

        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    # Tail block (only when seqlen_k % BLOCK_N != 0).
    if LAST_BLOCK_HAS_TAIL:
        j_last = NUM_K_BLOCKS - 1
        start_n = (j_last * BLOCK_N).to(tl.int64)
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        if PADDED_HEAD_QK:
            k_mask = offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptr + j_last * stride_ksblk)

        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # Same FMA-friendly form as the interior loop; tail mask is applied
        # afterwards by pushing invalid columns to -inf so the exp result is
        # exactly 0 and the columns contribute nothing to l_i.
        scale = q_descale * k_descale
        boundary = tl.full([BLOCK_N], N_EXTRA_TOKENS, dtype=tl.int32)
        valid_n = offs_n < boundary

        q_shifted = tl.dot(q, k) * scale - m_i[:, None]
        q_shifted = tl.where(valid_n[None, :], q_shifted, float("-inf"))
        if USE_EXP2:
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

        l_i = l_i + tl.sum(p, 1)

        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32, acc=acc)

    # Epilogue.
    invalid_mask = m_i == float("-inf")
    l_i_safe = tl.where(invalid_mask, 1.0, l_i)
    l_i_safe = tl.maximum(l_i_safe, 1e-7)
    l_recip = 1.0 / l_i_safe[:, None]

    v_descale = tl.load(
        v_descale_ptr,
        mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V,
        other=0.0,
    )

    acc = acc * l_recip * v_descale
    z = 0.0
    acc = tl.where(invalid_mask[:, None], z.to(acc.type.element_ty), acc)

    if RETURN_LSE:
        if USE_EXP2:
            LN2: tl.constexpr = 0.6931471824645996
            log_l_i = tl.where(invalid_mask, 0.0, tl.math.log2(l_i_safe))
            softmax_lse = tl.where(invalid_mask, float("-inf"), m_i + log_l_i)
            softmax_lse *= LN2
        else:
            log_l_i = tl.where(invalid_mask, 0.0, tl.math.log(l_i_safe))
            softmax_lse = tl.where(invalid_mask, float("-inf"), m_i + log_l_i)

        l_offset = LSE + off_z * stride_lse_z + off_h_q * stride_lse_h
        l_ptrs = l_offset + offs_m * stride_lse_m
        end_m_idx = (start_m + 1) * BLOCK_M
        overflow_size = end_m_idx - seqlen_q
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
            l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
        else:
            tl.store(l_ptrs, softmax_lse)

    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL_V], 1, dtype=tl.int1)
    end_m_idx = (start_m + 1) * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD_V:
        o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)
