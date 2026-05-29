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
# VFA's signed-absmax estimate
#   m_init[i] = max_j (Q[i] @ sabsmax_K[j]) * q_descale * k_descale[j]
# meets (a) very well in practice -- on real attention payloads we measured
# gaps of |m_init - true_max| < 1 log2 unit on average.  It does NOT meet
# (b) exactly (~30% of rows underestimate by < 0.5 log2 unit on real data).
# We close that gap with a small additive safety margin (in log2 units) on
# top of `m_init` in the precompute kernel.  The hot kernel keeps
# `qk - m_i` unclamped on purpose: a small positive shift just produces `p`
# values slightly above 1.0, which still fit in fp8 E4M3 (max ~448 = 2^8.8)
# and yield more accurate softmax weights than clamping at 1.  Only a row
# whose `m_init` undershoots true rowmax by more than ~8 log2 units would
# saturate `p` to `fp8_max` -- a bounded bias, never inf/NaN.
#
# Layout:
#   _sage_k_sabsmax_kernel    -> K_Repr[B, H_K, num_k_blocks, D]   int8 (signed)
#   _sage_vfa_m_init_kernel   -> M_Init[B, H_Q, num_q_blocks, BLOCK_M] fp32
#   sage_fwd_vfa              -> single tight loop, frozen m, fp32 acc

import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# K-block signed-absmax kernel.
#
# For each (block, dim), picks the K element with maximum |K| and preserves
# its sign.  Storing the sign lets the m-init dot exploit dim-wise sign
# alignment between Q and K, which is what makes the bound tight on real
# attention payloads where Q and K co-vary.
# ----------------------------------------------------------------------------
@triton.jit
def _sage_k_sabsmax_kernel(
    K_Int8,             # [B, S, H_K, D] (bshd) or [B, H_K, S, D] (bhsd) int8
    K_Repr,             # [B, H_K, num_k_blocks, D] int8
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_rz, stride_rh, stride_rblk, stride_rd,
    SEQLEN_K,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    off_b = tl.program_id(0).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)
    off_blk = tl.program_id(2).to(tl.int64)

    offs_n = off_blk * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    n_mask = offs_n < SEQLEN_K

    k_ptr = (
        K_Int8
        + off_b * stride_kz
        + off_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    # Promote to int32 to avoid -128 corner case when taking abs.
    k = tl.load(k_ptr, mask=n_mask[:, None], other=0).to(tl.int32)
    k_abs = tl.abs(k)
    # Mask padding rows so they cannot win the per-dim absmax.
    k_abs_masked = tl.where(n_mask[:, None], k_abs, -1)
    max_abs_per_d = tl.max(k_abs_masked, axis=0)  # [D]

    # Pick the signed value whose |.| equals the column max.  Ties resolve to
    # the positive choice via tl.max (which is fine -- both have the same |.|).
    is_max = k_abs_masked == max_abs_per_d[None, :]
    signed = tl.where(is_max, k, 0)
    repr_val = tl.max(signed, axis=0)
    repr_val = tl.where(max_abs_per_d <= 0, 0, repr_val).to(tl.int8)

    r_ptr = (
        K_Repr
        + off_b * stride_rz
        + off_h * stride_rh
        + off_blk * stride_rblk
        + offs_d * stride_rd
    )
    tl.store(r_ptr, repr_val)


# ----------------------------------------------------------------------------
# m-init kernel: per-row running-max estimate.
#
#   m_init[i] = SAFETY + max_j  Q[i] @ sabsmax_K[j]  *  q_descale[i_blk] *
#                                                       k_descale[j]
#
# `SAFETY` is a small additive bias in log2 units that absorbs the cases
# where the sabsmax representative happens to be on the wrong side of the
# true block max for some row.  The hot kernel also clamps `qk - m_init` to
# <= 0 before exp2 so a leftover undershoot is bounded in its effect on the
# softmax weights rather than producing inf/NaN.
# ----------------------------------------------------------------------------
@triton.jit
def _sage_vfa_m_init_kernel(
    Q,                  # int8 query tensor
    Q_Descale,          # fp32 per-Q-block descale
    K_Repr,             # int8 [B, H_K, num_k_blocks, D]  signed absmax
    K_Descale,          # fp32 [B, H_K, num_k_blocks]
    M_Init,             # fp32 [B, H_Q, num_q_blocks, BLOCK_M] output
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_qsz, stride_qsh, stride_qsblk,
    stride_krz, stride_krh, stride_krblk, stride_krd,
    stride_ksz, stride_ksh, stride_ksblk,
    stride_mz, stride_mh, stride_mblk, stride_mr,
    SEQLEN_Q,
    NUM_K_BLOCKS,
    SAFETY,             # fp32 additive bias in log2 units
    HQ: tl.constexpr,
    HK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K_REPR: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
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
    offs_d = tl.arange(0, BLOCK_DMODEL_QK)

    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h_q * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q_descale_ptr = (
        Q_Descale + off_z * stride_qsz + off_h_q * stride_qsh + start_m * stride_qsblk
    )
    k_repr_off = K_Repr + off_z * stride_krz + off_h_k * stride_krh
    k_descale_off = K_Descale + off_z * stride_ksz + off_h_k * stride_ksh

    q_mask = offs_m[:, None] < SEQLEN_Q
    if PADDED_HEAD_QK:
        q_mask = q_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL_QK)
    q = tl.load(q_ptrs, mask=q_mask, other=0)
    q_descale = tl.load(q_descale_ptr)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    for j_start in range(0, NUM_K_BLOCKS, BLOCK_K_REPR):
        j_offs = j_start + tl.arange(0, BLOCK_K_REPR)
        j_mask = j_offs < NUM_K_BLOCKS

        kr_ptrs = (
            k_repr_off
            + j_offs[None, :] * stride_krblk
            + offs_d[:, None] * stride_krd
        )
        if PADDED_HEAD_QK:
            kr_load_mask = (offs_d[:, None] < ACTUAL_BLOCK_DMODEL_QK) & j_mask[None, :]
        else:
            kr_load_mask = tl.broadcast_to(
                j_mask[None, :], (BLOCK_DMODEL_QK, BLOCK_K_REPR)
            )
        kr = tl.load(kr_ptrs, mask=kr_load_mask, other=0)

        kd_ptrs = k_descale_off + j_offs * stride_ksblk
        kd = tl.load(kd_ptrs, mask=j_mask, other=0.0)

        # Signed int8 dot then per-block descale.
        approx = tl.dot(q, kr).to(tl.float32) * (q_descale * kd[None, :])
        approx = tl.where(j_mask[None, :], approx, float("-inf"))
        m_i = tl.maximum(m_i, tl.max(approx, axis=1))

    # Additive safety margin.  Mask out the -inf for empty rows.
    valid = m_i != float("-inf")
    m_i = tl.where(valid, m_i + SAFETY, m_i)

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
