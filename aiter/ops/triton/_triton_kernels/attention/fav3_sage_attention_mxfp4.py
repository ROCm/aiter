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

        alpha = tl.math.exp2(m_i - m_ij)
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

        if IS_CAUSAL:
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
            )
        else:
            q_shifted = qk - m_ij[:, None]

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

    start_m, off_h_q, off_z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
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
        seqlen_k, seqlen_q, start_m, IS_CAUSAL, BLOCK_M, BLOCK_N
    )
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
            + offs_n.to(tl.int64) * stride_bn
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


@triton.jit
def _compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
):
    """
    Compute MX quantization with RNE (Round to Nearest Even) rounding for the scale.

    RNE is applied when converting max_abs to E8M0 format (nearest power of 2).
    This is equivalent to computing: scale = 2^(clip(floor(log2(RNE(max_abs(x)))), -127, 127) - 2)
    where RNE rounds to the nearest power of 2, with ties going to even exponent.
    """
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(
        valid_src_mask, abs_tensor, -1.0
    )  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)

    # RNE (Round to Nearest Even) rounding when converting max_abs to E8M0 format
    # E8M0 stores only exponent (no mantissa), so we round to nearest power of 2
    # Extract exponent and mantissa from float32
    max_val_bits = max_val.to(tl.uint32, bitcast=True)
    exponent = (max_val_bits >> 23) & 0xFF
    mantissa = max_val_bits & 0x7FFFFF

    # RNE to nearest power of 2:
    # For value 2^n * (1 + m/2^23), the threshold is at m = 0.5 * 2^23 = 0x400000
    # - If mantissa < 0x400000: round to 2^n (keep exponent)
    # - If mantissa > 0x400000: round to 2^(n+1) (increment exponent)
    # - If mantissa == 0x400000: tie case, round to even exponent (RNE)

    # Determine if we should round up
    should_round_up = (mantissa > 0x400000) | (
        (mantissa == 0x400000) & ((exponent & 1) == 1)
    )

    rounded_exponent = tl.where(should_round_up, exponent + 1, exponent)

    # Subtract 2 from exponent (divide by 4) to get final scale exponent
    # Clamp to valid E8M0 range [-127, 127] (exponent 0-254 in biased representation)
    scale_exponent = rounded_exponent - 2
    scale_exponent = tl.maximum(scale_exponent, 0)
    scale_exponent = tl.minimum(scale_exponent, 254)

    # Construct the scale as a power of 2
    dequant_scale_exponent = (scale_exponent << 23) & 0x7F800000
    dequant_scale = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale == 0, 0, 1.0 / dequant_scale)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    # Extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)

    # Convert the tensors to the mx format
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = quant_tensor & 0x7FFFFF

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(
            E8_BIAS, exponents + 1, sanitize_overflow=False
        )
        mantissas = tl.where(
            exponents < E8_BIAS,
            (0x400000 | (mantissas >> 1)) >> adjusted_exponents,
            mantissas,
        )

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


def sage_quant_mxfp4(
    q,
    k,
    v,
    BLOCK_M,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    q_smoothing=False,
    layout="bshd",
):

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")

    # padded_head_dim = max(16, 1 << (head_dim - 1).bit_length())
    sm_scale = head_dim**-0.5

    q_fp4, q_scale, k_fp4, k_scale, delta_s = smooth_rotate_downcast_qk(
        q,
        k,
        BLOCK_SIZE_M=BLOCK_M,
        hadamard_rotation=hadamard_rotation,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    FP8_TYPE = aiter.dtypes.fp8
    FP8_MAX = torch.finfo(FP8_TYPE).max
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    BLOCK_K = 128
    K_NUM_BLKS = (kv_len + BLOCK_K - 1) // BLOCK_K

    # Apply K tensor smoothing following SageAttention approach
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)

    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLOCK_K,
        num_stages=3,
        num_warps=8,
    )

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


@triton.jit
def sage_quant_v_kernel(
    V_Input,
    V_Output,
    V_Scale,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vsz,
    stride_vsh,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    SEQLEN_K,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    # V
    off_blk, off_h, off_b = pid_grid_3d(pid, K_NUM_BLKS, K_HEAD, BATCH)
    offs_kn = off_blk * BLK_K + offs_blk_k

    v_offs = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_kn[:, None] * stride_kn
        + offs_d[None, :]
    )

    v_input_ptrs = V_Input + v_offs
    v_output_ptrs = V_Output + v_offs

    # just apply the per channel v_scales that have been computed outside
    v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
    v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
    v = v.to(tl.float32)
    v_scales = tl.load(v_scale_ptrs)
    v_quant = v / v_scales
    v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
    tl.store(v_output_ptrs, v_quant, mask=offs_kn[:, None] < SEQLEN_K)


@triton.jit
def _rotate_quantize_qk_kernel(
    Q,
    Q_q,
    Q_descale,
    Q_mean,
    K,
    K_q,
    K_descale,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    batch,
    heads_q,
    heads_k,
    seqlen_q,
    seqlen_k,
    d_model,
    q_smoothing: tl.constexpr,
    hadamard_rotation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,  # rotation block size
    D: tl.constexpr,  # D is 128
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    q_pids = batch * heads_q * tl.cdiv(seqlen_q, BLOCK_M)
    pid = tl.program_id(0)
    is_q_pid = pid < q_pids

    if is_q_pid:
        pid_b = pid % batch
        pid_h = pid // batch % heads_q
        pid_m = pid // (batch * heads_q)
    else:  # is k pid
        pid -= q_pids
        pid_b = pid % batch
        pid_h = pid // batch % heads_k
        pid_m = pid // (batch * heads_k)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    # set pointers to either Q or K tensor, descale, quantized output
    # Q block shape: [BLOCK_M, D]
    if is_q_pid:
        tensor_offset = Q + (
            pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        descale_offset = (
            Q_descale
            + (pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm)
            // SCALE_GROUP_SIZE
        )  # we group 32 values together for quantization
        # Store rotated and quantized Q
        quant_tensor_offset = (
            Q_q
            + (pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qm) // 2
        )
        seqlen = seqlen_q
    else:
        tensor_offset = K + (
            pid_b * stride_kb
            + pid_h * stride_kh
            + offs_m[:, None] * stride_km
            + offs_d[None, :] * stride_kd
        )
        descale_offset = (
            K_descale
            + (pid_b * stride_kb + pid_h * stride_kh + offs_m[:, None] * stride_km)
            // SCALE_GROUP_SIZE
        )  # we group 32 values together for quantization

        quant_tensor_offset = (
            K_q
            + (pid_b * stride_kb + pid_h * stride_kh + offs_m[:, None] * stride_km) // 2
        )
        seqlen = seqlen_k

    qk_ptr = tensor_offset
    qk_descale_ptr = descale_offset + offs_ds[None, :]
    qk_quant_ptr = quant_tensor_offset + offs_dq[None, :]

    qk_tile = tl.load(
        qk_ptr, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < d_model), other=0.0
    )  # (BLOCK_M, D)
    original_dtype = qk_tile.dtype

    if is_q_pid:
        if q_smoothing:
            ACTUAL_BLOCK_M = tl.minimum(BLOCK_M, seqlen - pid_m * BLOCK_M)
            m_row_mean = (
                tl.sum(qk_tile, axis=0) / ACTUAL_BLOCK_M
            )  # Sum over BLOCK_M -> shape [D]
            qk_tile -= m_row_mean[None, :]
            qk_tile = qk_tile.to(original_dtype)
            mean_ptr = (
                Q_mean
                + pid_b * stride_mb
                + pid_h * stride_mh
                + pid_m * stride_mm
                + offs_d * stride_md
            )
            tl.store(mean_ptr, m_row_mean * sm_scale)

    if hadamard_rotation:
        r_ptr = (
            R
            + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
            + tl.arange(0, BLOCK_R)[None, :]
        )
        r_mat = tl.load(r_ptr)  # BLOCK_R x BLOCK_R

        shape0: tl.constexpr = BLOCK_M * D // BLOCK_R

        # Rotate: Q_rot = Q @ R
        qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
        qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))
    else:
        qk_rot_tile = qk_tile.to(tl.float32)

    if is_q_pid:
        qk_rot_tile *= sm_scale

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale(
        qk_rot_tile, offs_m[:, None] < seqlen, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen))

    tl.store(
        qk_quant_ptr,
        qk_quant_tile,
        mask=(offs_m[:, None] < seqlen),
    )


@triton.jit
def _compute_delta_s_kernel(
    Q_mean,
    K_rot,
    Delta_S,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sm,
    stride_sn,
    n_heads_k,
    n_heads_q,
    seq_k,
    d_model,
    BLOCK_N: tl.constexpr,  # Number of K-tokens to process
):
    pid_bh = tl.program_id(0)
    pid_m_q = tl.program_id(1)  # The Q-block index
    pid_n_k = tl.program_id(2)  # The K-block index

    pid_hq = pid_bh % n_heads_q
    pid_h = pid_hq // (n_heads_q // n_heads_k)
    pid_b = pid_bh // n_heads_q

    offs_n = pid_n_k * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate dot product across the whole d_model
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over d_model in steps of 32 (our block_size)
    for d_offset in range(0, d_model, 32):
        offs_d = d_offset + tl.arange(0, 32)

        # Load Q_mean segment: [32]
        qm_ptr = (
            Q_mean
            + pid_b * stride_mb
            + pid_hq * stride_mh
            + pid_m_q * stride_mm
            + offs_d * stride_md
        )
        qm_val = tl.load(qm_ptr)

        # Load K_rot segment: [BLOCK_N, 32]
        kn_ptr = (
            K_rot
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        kn_val = tl.load(kn_ptr, mask=offs_n[:, None] < seq_k, other=0.0)

        # Compute dot product for this d-segment
        acc += tl.sum(qm_val[None, :] * kn_val, axis=1)

    # Store to Delta_S [B, H, Q_BLKS, seq_k]
    s_ptr = (
        Delta_S
        + pid_b * stride_sb
        + pid_hq * stride_sh
        + pid_m_q * stride_sm
        + offs_n * stride_sn
    )
    tl.store(s_ptr, acc, mask=offs_n < seq_k)


@functools.lru_cache(maxsize=16)
def create_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    """
    Returns a Hadamard matrix of size block_size x block_size. Remember to normalize with sqrt(block_size) for it to be orthogonal.
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half

    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # remember to divide by sqrt(block_size) to get orthogonal matrix
    return H


def smooth_rotate_downcast_qk(
    q,
    k,
    BLOCK_SIZE_M,
    hadamard_rotation=False,
    R=None,
    BLOCK_R=None,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
):
    if hadamard_rotation:
        if R is None:
            assert (
                BLOCK_R is not None
            ), "if using hadamard rotation, BLOCK_R (size of the hadamard matrix) must be provided."
            R = create_hadamard_matrix(BLOCK_R, q.device) / (BLOCK_R**0.5)
        else:
            BLOCK_R = R.shape[-1]

    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)

    Q_q = torch.empty((*q.shape[:-1], d // 2), dtype=torch.uint8, device=q.device)
    Q_descale = torch.empty(
        (*q.shape[:-1], d // 32), dtype=torch.uint8, device=q.device
    )
    K_q = torch.empty((*k.shape[:-1], d // 2), dtype=torch.uint8, device=k.device)
    K_descale = torch.empty(
        (*k.shape[:-1], d // 32), dtype=torch.uint8, device=k.device
    )

    grid = (b * (h_q * Q_NUM_BLKS + h_k * K_NUM_BLKS),)
    _rotate_quantize_qk_kernel[grid](
        q,
        Q_q,
        Q_descale,
        q_mean,
        k,
        K_q,
        K_descale,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        b,
        h_q,
        h_k,
        s_q,
        s_k,
        d,
        q_smoothing=q_smoothing,
        hadamard_rotation=hadamard_rotation,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_R=BLOCK_R,
        D=d,
    )

    if q_smoothing:
        # 3. Compute Smoothing Delta S
        # Grid: Each Q-block x Each K-block
        grid_delta = (b * h_q, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            k,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_k,
            h_q,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_q, Q_descale, K_q, K_descale, delta_s


def return_static_random_hadamard(device):
    return torch.tensor(
        [
            [
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
            ],
            [
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
            ],
            [
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                1,
            ],
            [
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
            ],
            [
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            [
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
            ],
            [
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
            ],
            [
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
            ],
            [
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
            ],
            [
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
            ],
            [
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
            ],
            [
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
            ],
            [
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
            ],
            [
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
            ],
            [
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
            ],
            [
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
            ],
            [
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
            ],
            [
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
            ],
            [
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
            ],
            [
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
            ],
            [
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
            ],
            [
                1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
            ],
            [
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
            ],
            [
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            [
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
            ],
            [
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
            ],
            [
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
            ],
            [
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
            ],
            [
                -1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                1,
            ],
            [
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
            ],
            [
                1,
                -1,
                1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                -1,
            ],
            [
                1,
                1,
                -1,
                1,
                -1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                1,
                -1,
                1,
                1,
            ],
        ],
        dtype=torch.bfloat16,
        device=device,
    )
