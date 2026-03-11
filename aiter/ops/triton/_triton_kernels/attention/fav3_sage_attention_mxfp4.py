import functools

import torch
import triton
import triton.language as tl
import aiter


from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    map_dims,
)


@triton.jit
def compute_padding_info(seqlen_k, BLOCK_N: tl.constexpr):
    """Calculate padding information for the last K block."""
    # check if we will need to do masking due either BLOCK_N being bigger than seqlen_k or seqlen_k not being a factor of BLOCK_N
    # n_extra_tokens = 10 % 4 = 2
    # This means the last K block has 2 valid tokens and 2 padding positions
    # K blocks visualization:
    #         Block 0         Block 1         Block 2 (last)
    #         K0 K1 K2 K3    K4 K5 K6 K7     K8 K9 ?? ??
    #         ↑---------↑    ↑---------↑     ↑---↑ ↑---↑
    #         full block     full block      valid  pad
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0
    return n_extra_tokens


@triton.jit
def compute_block_masking(
    seqlen_k,
    seqlen_q,
    start_m,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Classify K blocks for attention computation with sliding window support.

    Returns:
        - n_front_skip_blocks: Blocks completely before the window
        - n_front_masked_blocks: Blocks partially overlapping window front
        - n_full_blocks: Blocks completely inside the window
        - n_back_masked_blocks: Blocks partially overlapping window back
        - n_extra_tokens: Padding tokens in last K block
    """

    # common
    # q_start = start_m * BLOCK_M
    q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
    diag = seqlen_k - seqlen_q
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)

    if IS_CAUSAL:
        # ========== CAUSAL MODE: Classify K Blocks ==========
        # Calculate causal boundary for this Q block
        #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
        # Q0-Q3:   [ 1  0  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q0
        #          [ 1  1  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q1
        #          [ 1  1  1  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q2
        #          [ 1  1  1  1] [ 1  1  0  0] [ 0  0 -- --]  ← Q3
        #                            ↑ can see up to K5
        #
        # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  0] [ 0  0 -- --]  ← Q4
        #          [ 1  1  1  1] [ 1  1  1  1] [ 0  0 -- --]  ← Q5
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  0 -- --]  ← Q6
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -- --]  ← Q7

        # ------------------------------------------------------------
        # 1. figure out, in tokens, the right-most K position
        #    this Q-block may attend to
        # ------------------------------------------------------------
        k_max_token = q_end + diag  # last visible K index

        # this Q-block is entirely above the diagonal ⇒ nothing to do
        if k_max_token < 0:
            return 0, 0, 0, 0, n_extra_tokens

        k_max_token = tl.minimum(k_max_token, seqlen_k - 1)

        # ------------------------------------------------------------
        # 2. translate token indices into K-block indices
        # ------------------------------------------------------------
        last_visible_k_block = k_max_token // BLOCK_N
        n_visible_k_blocks = tl.minimum(last_visible_k_block + 1, total_k_blocks)

        # ------------------------------------------------------------
        # 3. classify those visible blocks
        #    – we *never* skip or mask blocks in front, because causal
        #      attention always starts at K0
        #    – the back side can require several masked blocks:
        #         • intersection of the causal diagonal with K-grid
        #           (at most  ⌈BLOCK_M / BLOCK_N⌉ blocks)
        #         • plus one extra block if this Q-block stops in the
        #           middle of a K-block or the last K-block is padded
        # ------------------------------------------------------------
        padded_last_k = n_extra_tokens != 0
        is_modulo_mn = (not padded_last_k) & (seqlen_q % BLOCK_M == 0)

        n_back_masked_blocks = BLOCK_M // BLOCK_N + tl.where(is_modulo_mn, 0, 1)
        n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_visible_k_blocks)

        n_front_skip_blocks = 0  # causal never skips the left side
        n_front_masked_blocks = 0  # ditto
        n_full_blocks = n_visible_k_blocks - n_back_masked_blocks
    else:
        # ========== NON-CAUSAL MODE ==========
        # Without causal mask, all positions can attend to all positions
        # Only need to handle the padding in the last block
        #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
        # Q0-Q3:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #
        # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
        #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]

        n_front_skip_blocks = 0  # never skips the left side
        n_front_masked_blocks = 0  # ditto
        if n_extra_tokens != 0:
            n_back_masked_blocks = 1  # Last block needs padding mask
            n_full_blocks = total_k_blocks - 1
        else:
            n_back_masked_blocks = 0  # All blocks are aligned
            n_full_blocks = total_k_blocks

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    )


@triton.jit
def _sage_fwd_no_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    for start_n in range(block_min, block_max, BLOCK_N):
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Refactored K Load
        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptrs)

        if PRE_LOAD_V:
            # Refactored V Load
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        if USE_BIAS:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
            )
        else:
            q_shifted = qk - m_ij[:, None]

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        if USE_BIAS:
            m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        else:
            m_diff = m_i - m_ij

        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]

        if not PRE_LOAD_V:
            # Refactored V Load (Lazy)
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    n_extra_tokens,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    ACCUMULATOR_TYPE: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    seqlen_delta_qk = seqlen_k - seqlen_q
    for start_n in range(block_min, block_max, BLOCK_N):
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)

        # Refactored K Load with mandatory boundary check + optional padding check
        k_mask = kv_offs_n[None, :] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask &= offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK

        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )

        if PRE_LOAD_V:
            # Refactored V Load
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            mask = (start_n + offs_n[None, :]) < seqlen_k
            qk = tl.where(mask, qk, float("-inf"))

        qk = tl.dot_scaled(
            q, q_descale, Q_DTYPE_STR, k, k_descale, K_DTYPE_STR, fast_math=True, acc=qk
        )

        if IS_CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= (start_n + offs_n - seqlen_delta_qk)[None, :],
                qk,
                float("-inf"),
            )

        if USE_BIAS:
            bias_mask = kv_offs_n < seqlen_k
            bias = tl.load(
                bias_base_ptrs + start_n * stride_bn, mask=bias_mask, other=0.0
            )
            qk += bias[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )

        p = tl.math.exp2(q_shifted)
        l_ij = tl.sum(p, 1)

        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        alpha = tl.math.exp2(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            # Refactored V Load (Lazy)
            v_mask = kv_offs_n[:, None] < seqlen_k
            if PADDED_HEAD_V:
                v_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def sage_fwd_mxfp4(
    Q,
    K,
    V,
    bias,
    Q_Descale,
    K_Descale,
    V_Descale,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_vsz,
    stride_vsh,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_oz,
    stride_oh,
    stride_om,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    cu_seqlens_q,
    cu_seqlens_k,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
):
    # Constants
    Q_HEAD_DIV: tl.constexpr = 2 if Q_DTYPE_STR == "e2m1" else 1
    K_HEAD_DIV: tl.constexpr = 2 if K_DTYPE_STR == "e2m1" else 1
    SCALE_GROUP: tl.constexpr = 32
    ACC_TYPE: tl.constexpr = tl.float32
    # b*h*s*d can grow to be larger than int32 max, so turn to int64
    start_m, off_h_q, off_z = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1).to(tl.int64),
        tl.program_id(2).to(tl.int64),
    )
    off_h_k = off_h_q // (HQ // HK)

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_DMODEL_QK // Q_HEAD_DIV)
    offs_d_k = tl.arange(0, BLOCK_DMODEL_QK // K_HEAD_DIV)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_scale = tl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP)

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        seqlen_q = tl.load(cu_seqlens_q + off_z + 1) - q_start
        k_start = tl.load(cu_seqlens_k + off_z)
        seqlen_k = tl.load(cu_seqlens_k + off_z + 1) - k_start
        if start_m * BLOCK_M >= seqlen_q:
            return
    else:
        q_start, k_start = 0, 0
        seqlen_q, seqlen_k = MAX_SEQLENS_Q, MAX_SEQLENS_K

    # Masking logic
    mask_info = compute_block_masking(
        seqlen_k, seqlen_q, start_m.to(tl.int32), IS_CAUSAL, BLOCK_M, BLOCK_N
    )  # need to turn start_m to int32 for consistent return values
    n_front_skip, n_front_masked, n_full, n_back_masked, n_extra = mask_info

    if (n_front_masked + n_full + n_back_masked) == 0:
        o_ptr = (
            Out
            + off_z * stride_oz
            + off_h_q * stride_oh
            + (q_start + offs_m[:, None]) * stride_om
            + offs_d_v[None, :]
        )
        tl.store(
            o_ptr,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.dtype.element_ty),
            mask=(offs_m[:, None] < seqlen_q),
        )
        return

    # Pointers
    q_ptrs = (
        Q
        + off_z * stride_qz
        + off_h_q * stride_qh
        + (q_start + offs_m[:, None]) * stride_qm
        + offs_d_q[None, :]
    )
    k_ptrs = (
        K
        + off_z * stride_kz
        + off_h_k * stride_kh
        + (k_start + offs_n[None, :]) * stride_kn
        + offs_d_k[:, None]
    )
    v_ptrs = (
        V
        + off_z * stride_vz
        + off_h_k * stride_vh
        + (k_start + offs_n[:, None]) * stride_vk
        + offs_d_v[None, :]
    )

    qd_ptrs = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + (q_start + offs_m[:, None]) * stride_qsm
        + offs_d_scale[None, :]
    )
    kd_ptrs = (
        K_Descale
        + off_z * stride_ksz
        + off_h_k * stride_ksh
        + (k_start + offs_n[:, None]) * stride_ksn
        + offs_d_scale[None, :]
    )
    vd_ptr = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + offs_d_v

    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q), other=0.0)
    q_descale = tl.load(qd_ptrs, mask=(offs_m[:, None] < seqlen_q), other=0.0)

    # Bias is delta s
    bias_ptrs = (
        (
            bias
            + off_z * stride_bz
            + off_h_q * stride_bh
            + start_m * stride_bm
            + tl.cast(offs_n, tl.int64) * stride_bn
        )
        if USE_BIAS
        else None
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACC_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACC_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACC_TYPE)

    if n_full > 0:
        b_min = (n_front_skip + n_front_masked) * BLOCK_N
        b_max = b_min + n_full * BLOCK_N
        acc, l_i, m_i = _sage_fwd_no_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_d_k,
            offs_d_v,
            b_min,
            b_max,
            q_descale,
            kd_ptrs,
            stride_ksn,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            Q_DTYPE_STR,
            K_DTYPE_STR,
            ACC_TYPE,
            USE_BIAS,
        )

    if n_back_masked > 0:
        b_min = (n_front_skip + n_front_masked + n_full) * BLOCK_N
        b_max = b_min + n_back_masked * BLOCK_N
        acc, l_i, m_i = _sage_fwd_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_n,
            offs_d_k,
            offs_d_v,
            b_min,
            b_max,
            n_extra,
            q_descale,
            kd_ptrs,
            stride_ksn,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            Q_DTYPE_STR,
            K_DTYPE_STR,
            ACC_TYPE,
            USE_BIAS,
        )

    # Epilogue
    l_recip = 1 / tl.where(m_i == float("-inf"), 1.0, l_i)[:, None]
    v_descale = tl.load(vd_ptr, mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V, other=0.0)
    acc = acc * l_recip * v_descale

    o_ptr = (
        Out
        + off_z * stride_oz
        + off_h_q * stride_oh
        + (q_start + offs_m[:, None]) * stride_om
        + offs_d_v[None, :]
    )
    o_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_V:
        o_mask &= offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
    tl.store(o_ptr, acc.to(Out.dtype.element_ty), mask=o_mask)
