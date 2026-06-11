# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Lloyd-Max attention kernel for Sage v2.

Replaces tl.dot_scaled (hardwired e2m1 on MI355) with a software codebook
lookup + BF16 matmul.  This enables the Lloyd-Max quantization approach that
reduces attention output error by ~17% over e2m1 MXFP4.

Input format (vs sage_fwd_mxfp4):
  Q/K  → uint8 packed 4-bit Lloyd-Max indices, shape (B, S, H, D//2)
  Q/K norms → float16 per-vector scale, shape (B, S, H)
         For Q: norm = ||Q_rot|| * log2(e) / head_dim   (sm_scale absorbed in)
         For K: norm = ||K_rot|| / sqrt(head_dim)
  codebook → float32 (16,) sorted Lloyd-Max centroids (for d=128, 4-bit)
  V/V_Descale → unchanged (fp8 + per-channel float32)

Math:
  q_dequant[i,j_lo] = codebook[q_lo_idx[i,j]] * q_norm[i]
  k_dequant[j_lo,k] = codebook[k_lo_idx[j,k]] * k_norm[k]
  qk[i,k] = q_lo_dequant @ k_lo_dequant + q_hi_dequant @ k_hi_dequant
  p = exp2(qk)  -- correct because q_norm absorbs log2(e)/sqrt(D)
"""

import triton
import triton.language as tl


# ── Codebook lookup ──────────────────────────────────────────────────────────

@triton.jit
def _lm_lookup(idx, c0, c1, c2, c3, c4, c5, c6, c7,
                    c8, c9, c10, c11, c12, c13, c14, c15):
    """16-level lookup using compile-time unrolled where-ops. idx: uint8 0-15."""
    r = tl.where(idx == 1,  c1,  c0)
    r = tl.where(idx == 2,  c2,  r)
    r = tl.where(idx == 3,  c3,  r)
    r = tl.where(idx == 4,  c4,  r)
    r = tl.where(idx == 5,  c5,  r)
    r = tl.where(idx == 6,  c6,  r)
    r = tl.where(idx == 7,  c7,  r)
    r = tl.where(idx == 8,  c8,  r)
    r = tl.where(idx == 9,  c9,  r)
    r = tl.where(idx == 10, c10, r)
    r = tl.where(idx == 11, c11, r)
    r = tl.where(idx == 12, c12, r)
    r = tl.where(idx == 13, c13, r)
    r = tl.where(idx == 14, c14, r)
    r = tl.where(idx == 15, c15, r)
    return r


# ── Inner K-loop: dense, no mask ─────────────────────────────────────────────

@triton.jit
def _sage_fwd_no_mask_lloymax(
    acc, l_i, m_i,
    q_lo, q_hi,          # (BLOCK_M, HALF_D) float32 — pre-dequantized Q halves
    k_base_ptrs,         # pointer to K_packed at (batch, head) offset
    kn_base_ptrs,        # pointer to K_norms row: K_norms[(batch*HK+head), 0]
    v_base_ptrs,
    stride_kn, stride_vk,
    seqlen_k, seqlen_q, offs_m, offs_d_half, offs_d_v,
    block_min, block_max,
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HALF_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr, ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    for start_n in range(block_min, block_max, BLOCK_N):
        kv_offs_n = start_n + offs_n

        # Load K_packed (HALF_D, BLOCK_N) transposed layout
        k_ptrs = k_base_ptrs + start_n * stride_kn + offs_d_half[:, None] + offs_n[None, :] * stride_kn
        k_packed = tl.load(k_ptrs)                                           # (HALF_D, BLOCK_N) uint8
        # K_norms is (B*HK, S_k): row already at kn_base_ptrs, col = start_n+offs_n (stride 1)
        k_norms = tl.load(kn_base_ptrs + start_n + offs_n).to(tl.float32)  # (BLOCK_N,)

        # Unpack and dequantize K
        k_lo_idx = k_packed & 0xF
        k_hi_idx = (k_packed >> 4) & 0xF
        k_lo = _lm_lookup(k_lo_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15) * k_norms[None, :]
        k_hi = _lm_lookup(k_hi_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15) * k_norms[None, :]

        if PRE_LOAD_V:
            v_ptrs = v_base_ptrs + (start_n + offs_n[:, None]) * stride_vk + offs_d_v[None, :]
            if PADDED_HEAD_V:
                v = tl.load(v_ptrs, mask=offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # QK via two half-dimension matmuls
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC_TYPE)
        qk = tl.dot(q_lo.to(tl.bfloat16), k_lo.to(tl.bfloat16), out_dtype=ACC_TYPE, acc=qk)
        qk = tl.dot(q_hi.to(tl.bfloat16), k_hi.to(tl.bfloat16), out_dtype=ACC_TYPE, acc=qk)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = qk - m_ij[:, None]
        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v_ptrs = v_base_ptrs + (start_n + offs_n[:, None]) * stride_vk + offs_d_v[None, :]
            if PADDED_HEAD_V:
                v = tl.load(v_ptrs, mask=offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V, other=0.0)
            else:
                v = tl.load(v_ptrs)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=ACC_TYPE, acc=acc)

    return acc, l_i, m_i


# ── Inner K-loop: dense, with mask (causal or padding) ───────────────────────

@triton.jit
def _sage_fwd_mask_lloymax(
    acc, l_i, m_i,
    q_lo, q_hi,
    k_base_ptrs, kn_base_ptrs, v_base_ptrs,
    stride_kn, stride_vk,
    seqlen_k, seqlen_q, offs_m, offs_n, offs_d_half, offs_d_v,
    block_min, block_max, n_extra_tokens,
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HALF_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr, ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for start_n in range(block_min, block_max, BLOCK_N):
        kv_offs_n = start_n + offs_n

        k_ptrs = k_base_ptrs + start_n * stride_kn + offs_d_half[:, None] + offs_n[None, :] * stride_kn
        k_mask = kv_offs_n[None, :] < seqlen_k
        k_packed = tl.load(k_ptrs, mask=k_mask, other=0)
        k_norms = tl.load(kn_base_ptrs + start_n + offs_n,
                          mask=kv_offs_n < seqlen_k, other=1.0).to(tl.float32)

        k_lo_idx = k_packed & 0xF
        k_hi_idx = (k_packed >> 4) & 0xF
        k_lo = _lm_lookup(k_lo_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15) * k_norms[None, :]
        k_hi = _lm_lookup(k_hi_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15) * k_norms[None, :]

        if PRE_LOAD_V:
            v_ptrs = v_base_ptrs + (start_n + offs_n[:, None]) * stride_vk + offs_d_v[None, :]
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC_TYPE)
        # Mask padding before accumulating -inf
        boundary_mask = kv_offs_n[None, :] < seqlen_k
        qk = tl.where(boundary_mask, qk, float("-inf"))
        qk = tl.dot(q_lo.to(tl.bfloat16), k_lo.to(tl.bfloat16), out_dtype=ACC_TYPE, acc=qk)
        qk = tl.dot(q_hi.to(tl.bfloat16), k_hi.to(tl.bfloat16), out_dtype=ACC_TYPE, acc=qk)

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (kv_offs_n - seqlen_delta_qk)[None, :],
                qk, float("-inf"),
            )

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v_ptrs = v_base_ptrs + (start_n + offs_n[:, None]) * stride_vk + offs_d_v[None, :]
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=ACC_TYPE, acc=acc)

    return acc, l_i, m_i


# ── Inner K-loop: block-sparse (Sparge) ──────────────────────────────────────

@triton.jit
def _sage_fwd_blocksparse_lloymax(
    acc, l_i, m_i,
    q_lo, q_hi,
    k_base_ptrs, kn_base_ptrs, v_base_ptrs,
    stride_kn, stride_vk,
    seqlen_k, seqlen_q, offs_m, offs_n, offs_d_half, offs_d_v,
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
    kv_block_indices, lut_start_val, n_blocks,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HALF_D: tl.constexpr, BLOCK_DV: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr, ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACC_TYPE: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for i in range(n_blocks):
        block_id = tl.load(kv_block_indices + lut_start_val + i)
        start_n = block_id * BLOCK_N
        kv_offs_n = start_n + offs_n

        k_ptrs = k_base_ptrs + start_n * stride_kn + offs_d_half[:, None] + offs_n[None, :] * stride_kn
        k_mask = kv_offs_n[None, :] < seqlen_k
        k_packed = tl.load(k_ptrs, mask=k_mask, other=0)
        k_norms = tl.load(kn_base_ptrs + start_n + offs_n,
                          mask=kv_offs_n < seqlen_k, other=1.0).to(tl.float32)

        k_lo_idx = k_packed & 0xF
        k_hi_idx = (k_packed >> 4) & 0xF
        k_lo = _lm_lookup(k_lo_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15) * k_norms[None, :]
        k_hi = _lm_lookup(k_hi_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                          c8, c9, c10, c11, c12, c13, c14, c15) * k_norms[None, :]

        if PRE_LOAD_V:
            v_ptrs = v_base_ptrs + (start_n + offs_n[:, None]) * stride_vk + offs_d_v[None, :]
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACC_TYPE)
        boundary_mask = kv_offs_n[None, :] < seqlen_k
        qk = tl.where(boundary_mask, qk, float("-inf"))
        qk = tl.dot(q_lo.to(tl.bfloat16), k_lo.to(tl.bfloat16), out_dtype=ACC_TYPE, acc=qk)
        qk = tl.dot(q_hi.to(tl.bfloat16), k_hi.to(tl.bfloat16), out_dtype=ACC_TYPE, acc=qk)

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (kv_offs_n - seqlen_delta_qk)[None, :],
                qk, float("-inf"),
            )

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        q_shifted = tl.where(m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None])
        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            v_ptrs = v_base_ptrs + (start_n + offs_n[:, None]) * stride_vk + offs_d_v[None, :]
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc = tl.dot(p.to(v.type.element_ty), v, out_dtype=ACC_TYPE, acc=acc)

    return acc, l_i, m_i


# ── Main kernel entry ─────────────────────────────────────────────────────────

@triton.jit
def sage_fwd_lloymax(
    # Q/K: packed uint8 indices + float16 per-vector norms
    Q_packed, Q_norms,
    K_packed, K_norms,
    # V: fp8 + per-channel float32 descale (unchanged from Sage v2)
    V, V_Descale,
    # Lloyd-Max codebook (16 float32 centroids)
    codebook,
    # Output
    Out,
    # Q strides (packed Q: B,S,H,D//2)
    stride_qz, stride_qh, stride_qm,
    # Q_norms is (B*HQ, S_q): row = off_z*HQ+off_h_q, stride-1 along S — no extra strides
    # K strides (packed K: B,S,H,D//2)
    stride_kz, stride_kh, stride_kn,
    # K_norms is (B*HK, S_k): row = off_z*HK+off_h_k, stride-1 along S — no extra strides
    # V strides
    stride_vz, stride_vh, stride_vk,
    # V_Descale strides
    stride_vsz, stride_vsh,
    # Out strides
    stride_oz, stride_oh, stride_om,
    # Optional bias (delta_s for Q-smoothing)
    bias,
    stride_bz, stride_bh, stride_bm, stride_bn,
    # LUT for block-sparse
    kv_block_indices, lut_start, lut_count,
    # Sequence info
    seqlen_q, seqlen_k,
    # GQA
    HQ: tl.constexpr, HK: tl.constexpr,
    # Head dims
    HALF_D: tl.constexpr,          # D // 2 (packed elements per vector)
    BLOCK_DV: tl.constexpr,        # D_v (V head dim, padded to power of 2)
    ACTUAL_DV: tl.constexpr,       # actual V head dim
    # Tuning
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    # Mode flags
    IS_CAUSAL: tl.constexpr,
    USE_BLOCK_SPARSE: tl.constexpr,
    USE_BIAS: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
):
    ACC_TYPE: tl.constexpr = tl.float32

    start_m  = tl.program_id(0).to(tl.int64)
    off_h_q  = tl.program_id(1).to(tl.int64)
    off_z    = tl.program_id(2).to(tl.int64)
    off_h_k  = off_h_q // (HQ // HK)

    offs_m     = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n     = tl.arange(0, BLOCK_N)
    offs_d_half = tl.arange(0, HALF_D)
    offs_d_v    = tl.arange(0, BLOCK_DV)

    # ── Load codebook into scalar registers (16 individual loads) ─────────────
    # Triton doesn't support integer indexing on a loaded block (cb[0] fails),
    # so we load each centroid as a separate scalar.
    c0  = tl.load(codebook +  0).to(tl.float32)
    c1  = tl.load(codebook +  1).to(tl.float32)
    c2  = tl.load(codebook +  2).to(tl.float32)
    c3  = tl.load(codebook +  3).to(tl.float32)
    c4  = tl.load(codebook +  4).to(tl.float32)
    c5  = tl.load(codebook +  5).to(tl.float32)
    c6  = tl.load(codebook +  6).to(tl.float32)
    c7  = tl.load(codebook +  7).to(tl.float32)
    c8  = tl.load(codebook +  8).to(tl.float32)
    c9  = tl.load(codebook +  9).to(tl.float32)
    c10 = tl.load(codebook + 10).to(tl.float32)
    c11 = tl.load(codebook + 11).to(tl.float32)
    c12 = tl.load(codebook + 12).to(tl.float32)
    c13 = tl.load(codebook + 13).to(tl.float32)
    c14 = tl.load(codebook + 14).to(tl.float32)
    c15 = tl.load(codebook + 15).to(tl.float32)

    # ── Q pointers ────────────────────────────────────────────────────────────
    # Q_packed: (B, S, H, D//2) — stride along D-half is 1 (contiguous last dim)
    q_ptrs = (Q_packed
              + off_z * stride_qz
              + off_h_q * stride_qh
              + offs_m[:, None] * stride_qm
              + offs_d_half[None, :])
    # Q_norms: (B*H_q, S_q) — row = off_z*HQ + off_h_q, col = offs_m (stride 1)
    qn_row   = off_z * HQ + off_h_q
    qn_ptrs  = Q_norms + qn_row * seqlen_q + offs_m

    # ── Load Q block and dequantize (once, before K loop) ────────────────────
    q_mask = offs_m[:, None] < seqlen_q
    q_packed = tl.load(q_ptrs, mask=q_mask, other=0)                # (BLOCK_M, HALF_D) uint8
    q_norms  = tl.load(qn_ptrs, mask=offs_m < seqlen_q, other=0.0).to(tl.float32)  # (BLOCK_M,)

    q_lo_idx = q_packed & 0xF
    q_hi_idx = (q_packed >> 4) & 0xF
    q_lo = _lm_lookup(q_lo_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                      c8, c9, c10, c11, c12, c13, c14, c15) * q_norms[:, None]
    q_hi = _lm_lookup(q_hi_idx, c0, c1, c2, c3, c4, c5, c6, c7,
                      c8, c9, c10, c11, c12, c13, c14, c15) * q_norms[:, None]

    # ── K / V / V_descale base pointers ───────────────────────────────────────
    # K_packed: (B, S, H, D//2) — loaded transposed (D//2, BLOCK_N) in inner loop
    k_base = (K_packed
               + off_z * stride_kz
               + off_h_k * stride_kh)
    # K_norms: (B*H_k, S_k) — row = off_z*HK + off_h_k, col = start_n+offs_n (stride 1)
    kn_row  = off_z * HK + off_h_k
    kn_base = K_norms + kn_row * seqlen_k
    v_base  = (V
               + off_z * stride_vz
               + off_h_k * stride_vh)
    vd_ptr  = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + offs_d_v

    # ── Accumulator init ──────────────────────────────────────────────────────
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACC_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACC_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=ACC_TYPE)

    # ── Determine block ranges ────────────────────────────────────────────────
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra = (BLOCK_N - seqlen_k % BLOCK_N) % BLOCK_N

    if not USE_BLOCK_SPARSE:
        if IS_CAUSAL:
            q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
            k_max = tl.minimum(q_end + (seqlen_k - seqlen_q), seqlen_k - 1)
            last_vis_block = k_max // BLOCK_N
            n_vis = tl.minimum(last_vis_block + 1, total_k_blocks)
            n_masked = tl.minimum(BLOCK_M // BLOCK_N + 1, n_vis)
            n_full = n_vis - n_masked
            b_full_max = n_full * BLOCK_N
            b_mask_max = n_vis * BLOCK_N

            if n_full > 0:
                acc, l_i, m_i = _sage_fwd_no_mask_lloymax(
                    acc, l_i, m_i, q_lo, q_hi,
                    k_base, kn_base, v_base,
                    stride_kn, stride_vk,
                    seqlen_k, seqlen_q, offs_m, offs_d_half, offs_d_v,
                    0, b_full_max,
                    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,
                    BLOCK_M, BLOCK_N, HALF_D, BLOCK_DV, PRE_LOAD_V,
                    PADDED_HEAD_V, ACTUAL_DV, ACC_TYPE,
                )

            if n_masked > 0:
                acc, l_i, m_i = _sage_fwd_mask_lloymax(
                    acc, l_i, m_i, q_lo, q_hi,
                    k_base, kn_base, v_base,
                    stride_kn, stride_vk,
                    seqlen_k, seqlen_q, offs_m, offs_n, offs_d_half, offs_d_v,
                    b_full_max, b_mask_max, n_extra,
                    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,
                    IS_CAUSAL,
                    BLOCK_M, BLOCK_N, HALF_D, BLOCK_DV, PRE_LOAD_V,
                    PADDED_HEAD_V, ACTUAL_DV, ACC_TYPE,
                )
        else:
            # Non-causal: full blocks + possibly one padded block
            if n_extra == 0:
                acc, l_i, m_i = _sage_fwd_no_mask_lloymax(
                    acc, l_i, m_i, q_lo, q_hi,
                    k_base, kn_base, v_base,
                    stride_kn, stride_vk,
                    seqlen_k, seqlen_q, offs_m, offs_d_half, offs_d_v,
                    0, seqlen_k,
                    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,
                    BLOCK_M, BLOCK_N, HALF_D, BLOCK_DV, PRE_LOAD_V,
                    PADDED_HEAD_V, ACTUAL_DV, ACC_TYPE,
                )
            else:
                full_end = (total_k_blocks - 1) * BLOCK_N
                if full_end > 0:
                    acc, l_i, m_i = _sage_fwd_no_mask_lloymax(
                        acc, l_i, m_i, q_lo, q_hi,
                        k_base, kn_base, v_base,
                        stride_kn, stride_vk,
                        seqlen_k, seqlen_q, offs_m, offs_d_half, offs_d_v,
                        0, full_end,
                        c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,
                        BLOCK_M, BLOCK_N, HALF_D, BLOCK_DV, PRE_LOAD_V,
                        PADDED_HEAD_V, ACTUAL_DV, ACC_TYPE,
                    )
                acc, l_i, m_i = _sage_fwd_mask_lloymax(
                    acc, l_i, m_i, q_lo, q_hi,
                    k_base, kn_base, v_base,
                    stride_kn, stride_vk,
                    seqlen_k, seqlen_q, offs_m, offs_n, offs_d_half, offs_d_v,
                    full_end, seqlen_k, n_extra,
                    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,
                    False,  # not causal, just padding mask
                    BLOCK_M, BLOCK_N, HALF_D, BLOCK_DV, PRE_LOAD_V,
                    PADDED_HEAD_V, ACTUAL_DV, ACC_TYPE,
                )
    else:
        # Block-sparse (Sparge) path
        lut_idx = off_z * HQ * tl.cdiv(seqlen_q, BLOCK_M) + off_h_q * tl.cdiv(seqlen_q, BLOCK_M) + start_m
        lut_start_val = tl.load(lut_start + lut_idx)
        n_blocks_val  = tl.load(lut_count + lut_idx)
        # Last block might need mask — treat all sparse blocks as masked for safety
        acc, l_i, m_i = _sage_fwd_blocksparse_lloymax(
            acc, l_i, m_i, q_lo, q_hi,
            k_base, kn_base, v_base,
            stride_kn, stride_vk,
            seqlen_k, seqlen_q, offs_m, offs_n, offs_d_half, offs_d_v,
            c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,
            kv_block_indices, lut_start_val, n_blocks_val,
            IS_CAUSAL,
            BLOCK_M, BLOCK_N, HALF_D, BLOCK_DV, PRE_LOAD_V,
            PADDED_HEAD_V, ACTUAL_DV, ACC_TYPE,
        )

    # ── Epilogue: normalise + apply V descale ─────────────────────────────────
    l_recip = 1.0 / tl.where(m_i == float("-inf"), 1.0, l_i)[:, None]
    v_descale = tl.load(vd_ptr, mask=offs_d_v < ACTUAL_DV, other=0.0)
    acc = acc * l_recip * v_descale[None, :]

    # ── Store output ──────────────────────────────────────────────────────────
    o_ptrs = (Out
              + off_z * stride_oz
              + off_h_q * stride_oh
              + offs_m[:, None] * stride_om
              + offs_d_v[None, :])
    o_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_V:
        o_mask &= offs_d_v[None, :] < ACTUAL_DV
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_mask)
