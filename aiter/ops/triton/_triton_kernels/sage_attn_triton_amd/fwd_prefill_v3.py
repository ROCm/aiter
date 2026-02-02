import torch
import triton
import triton.language as tl
from typing import Literal, Optional
from .utils import (
    compute_alibi_block,
    get_arch,
    map_dims,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d
from .sage_version import Sage_version
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
"""aiter/aiter/ops/triton/_triton_kernels/moe/quant_moe.py"""
from aiter.ops.triton._triton_kernels.moe.quant_moe import _compute_mx_quant_and_scale
from aiter.ops.triton._triton_kernels.sage_attn_triton_amd.fwd_prefill import get_sage_fwd_configs, compute_block_masking 

import triton
import triton.language as tl


@triton.jit
def _get_max_quant_val(dtype: tl.constexpr):
    if dtype == tl.uint8:
        return 6.0
    elif dtype == tl.float8e5:
        return 57344.0
    elif dtype == tl.float8e4nv:
        return 448.0
    else:
        tl.static_assert(False, f"Invalid {dtype=}")

@triton.jit
def static_compute_mx_quant_and_scale(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
    DEQUANT_SCALE_ROUNDING_MODE: tl.constexpr = 0,
):
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32
    
    dequant_scale = 1.0 / _get_max_quant_val(mx_tensor_dtype)
    dequant_scale_exponent = (
            dequant_scale.to(tl.uint32, bitcast=True) + 0x007FFFFF
        ) & 0x7F800000
    dequant_scale_rounded = dequant_scale_exponent.to(tl.float32, bitcast=True)
    # quant_scale = tl.where(dequant_scale_rounded == 0, 0, 1.0 / dequant_scale_rounded)

    f32_tensor = src_tensor.to(tl.float32)
    quant_tensor = f32_tensor / dequant_scale_rounded


    # First, we simply extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)

    dequant_scale_tensor = tl.full((BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE), dequant_scale_exponent, dtype=tl.uint8)

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

    

    return out_tensor, dequant_scale_tensor


@triton.jit
def _sage_fwd_no_mask_v3(
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
    stride_sn,
    stride_sm,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sd_mask,
    stride_sz,
    stride_sh,
    off_z,
    off_h_q,
    offs_m,
    offs_d_qk,
    offs_d_v,
    offs_d_vs,
    block_min,
    block_max,
    alibi_slope,
    q_descale,
    k_descale_ptr,
    v_descale_ptr,
    stride_ksblk,
    stride_vsn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_VS: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    ACCUMULATOR_TYPE,
):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n//2 * stride_vk

        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        # Load K
        if PADDED_HEAD_QK:
            k_mask = offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptr)
        k_descale_ptr += stride_ksblk
        v_descale_ptr += stride_vsn * BLOCK_N//2

        # Optionally preload V
        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
                v_descale_mask = offs_d_vs[None, :] < ACTUAL_BLOCK_DMODEL_VS
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                v_descale = tl.load(v_descale_ptr, mask=v_descale_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)
                v_descale = tl.load(v_descale_ptr)


        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # -- compute qk ----
        qk += tl.dot(q, k) * (q_descale * k_descale)
        # qk_scaled = qk * SM_SCALE
        qk_scaled = qk

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk_scaled - m_ij[:, None]
        )

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            # p = tl.math.exp2(q_shifted * RCP_LN2)
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        if USE_EXP2:
            # alpha = tl.math.exp2(m_diff * RCP_LN2)
            alpha = tl.math.exp2(m_diff)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v_descale_mask = offs_d_vs[None, :] < ACTUAL_BLOCK_DMODEL_VS
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                v_descale = tl.load(v_descale_ptr, mask=v_descale_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)
                v_descale = tl.load(v_descale_ptr)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        p_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)
        p, p_descale = _compute_mx_quant_and_scale(p, p_mask, tl.uint8)
        acc += tl.dot_scaled(p, p_descale, "e2m1", v, v_descale, "e2m1", fast_math=True,)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_mask_v3(
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
    stride_sn,
    stride_sm,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sd_mask,
    stride_sz,
    stride_sh,
    off_z,
    off_h_q,
    offs_m,
    offs_n,
    offs_d_qk,
    offs_d_v,
    offs_d_vs,
    block_min,
    block_max,
    n_extra_tokens,
    alibi_slope,
    q_descale,
    k_descale_ptr,
    v_descale_ptr,
    stride_ksblk,
    stride_vsn,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_VS: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    ACCUMULATOR_TYPE,
):
    # seqlen diff
    seqlen_delta_qk = seqlen_k - seqlen_q


    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n//2 * stride_vk

        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        v_offs_n = start_n//2 + tl.arange(0, BLOCK_N//2) # Important! //2 for MXFP4 since two fp4 per byte
        k_mask = kv_offs_n[None, :] < seqlen_k
        v_mask = v_offs_n[:, None] < seqlen_k//2
        if PADDED_HEAD_QK:
            k_mask = k_mask & (offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK)
            v_descale_mask = None
      
        # load k and if preload_v then v
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(k_descale_ptr)
        k_descale_ptr += stride_ksblk
        v_descale_ptr += stride_vsn * BLOCK_N//2

        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
                v_descale_mask = offs_d_vs[None, :] < ACTUAL_BLOCK_DMODEL_VS
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                v_descale = tl.load(v_descale_ptr, mask=v_descale_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)
                v_descale = tl.load(v_descale_ptr)

        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        # If this is the last block / iteration, we want to
        # mask if the sequence length is not a multiple of block size
        # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
        # last step might get wasted but that is okay. check if this masking works For
        # that case.
        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + offs_n[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        qk += tl.dot(q, k) * (q_descale * k_descale)
        # qk_scaled = qk * SM_SCALE
        qk_scaled = qk

        # compute qk mask
        qk_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)

        # compute bias
        if bias_base_ptrs is not None:
            bias_ptrs = bias_base_ptrs + start_n * stride_bn
            bias = tl.load(bias_ptrs, mask=qk_mask, other=0.0)
            qk_scaled += bias

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        q_shifted = qk_scaled - m_ij[:, None]

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            # p = tl.math.exp2(q_shifted * RCP_LN2)
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
       
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        if USE_EXP2:
            # alpha = tl.math.exp2(m_diff * RCP_LN2)
            alpha = tl.math.exp2(m_diff)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
                v_descale_mask = offs_d_vs[None, :] < ACTUAL_BLOCK_DMODEL_VS
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
                v_descale = tl.load(v_descale_ptr, mask=v_descale_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)
                v_descale = tl.load(v_descale_ptr)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        p_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)
        p, p_descale = _compute_mx_quant_and_scale(p, p_mask, tl.uint8)

        acc += tl.dot_scaled(p, p_descale, "e2m1", v, v_descale, "e2m1")

    return acc, l_i, m_i


@triton.jit
def sage_fwd_v3(
    Q,
    K,
    V,
    bias,
    Q_Descale,
    K_Descale,
    V_Descale,
    V_mean,
    stride_qsz,
    stride_qsh,
    stride_qsblk,
    stride_ksz,
    stride_ksh,
    stride_ksblk,
    stride_vsz,
    stride_vsh,
    stride_vsn,
    stride_vmz,
    stride_vmh,
    LSE,
    Out,
    SD_MASK,
    ALIBI_SLOPES,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,  # Add seqused parameters
    dropout_p,
    philox_seed,
    philox_offset_base,
    RETURN_LSE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_SEQUSED: tl.constexpr,
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    # set params
    ACCUMULATOR_TYPE = tl.float32  # for q*k product
    ACTUAL_BLOCK_DMODEL_VS: tl.constexpr = ACTUAL_BLOCK_DMODEL_V
    # ACTUAL_BLOCK_DMODEL_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V // 2 
    # compute offsets
    off_z = tl.program_id(0)
    off_h_q = tl.program_id(1)
    start_m = tl.program_id(2)
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q
    # Determine if we need to mask the heads
    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_vn = tl.arange(0, BLOCK_N//2)
    offs_vsn = tl.arange(0, BLOCK_N//SCALE_GROUP_SIZE)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_o = tl.arange(0, BLOCK_DMODEL_V)
    offs_d_vs = tl.arange(0, BLOCK_DMODEL_V)

    # handle seqlen
    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

        # If seqused is provided, use it to limit the actual sequence length
        if USE_SEQUSED:
            actual_seqlen_q = (
                tl.load(seqused_q + off_z)
                if seqused_q is not None
                else cu_seqlens_q_end - cu_seqlens_q_start
            )
            seqlen_q = tl.minimum(
                actual_seqlen_q, cu_seqlens_q_end - cu_seqlens_q_start
            )
        else:
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

        # we have a one-size-fits-all grid in id(0). Some seqlens might be too small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)

        # If seqused is provided, use it to limit the actual sequence length for keys
        if USE_SEQUSED:
            actual_seqlen_k = (
                tl.load(seqused_k + off_z)
                if seqused_k is not None
                else cu_seqlens_k_end - cu_seqlens_k_start
            )
            seqlen_k = tl.minimum(
                actual_seqlen_k, cu_seqlens_k_end - cu_seqlens_k_start
            )
        else:
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # figure out masking pattern
    (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    ) = compute_block_masking(
        seqlen_k,
        seqlen_q,
        start_m,
        IS_CAUSAL,
        USE_SLIDING_WINDOW,
        WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT,
        BLOCK_M,
        BLOCK_N,
    )

    # ============================================================
    #          PROGRAM EARLY EXIT (All K Blocks Skipped)
    # ============================================================
    total_visible_blocks = n_front_masked_blocks + n_full_blocks + n_back_masked_blocks
    if total_visible_blocks == 0:
        """
        No K blocks visible - write zeros and exit.
        """
        # Write zeros to output
        o_offset = (
            Out
            + off_z * stride_oz
            + off_h_q * stride_oh
            + cu_seqlens_q_start * stride_om
        )
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_o[None, :] * stride_on
        o_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_V:
            o_mask = o_mask & (offs_d_o[None, :] < ACTUAL_BLOCK_DMODEL_V)
        tl.store(
            o_ptrs,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty),
            mask=o_mask,
        )

        # Write zeros to LSE
        if RETURN_LSE:
            l_ptrs = (
                LSE
                + off_z * stride_lse_z
                + off_h_q * stride_lse_h
                + cu_seqlens_q_start * stride_lse_m
                + offs_m * stride_lse_m
            )
            tl.store(
                l_ptrs, tl.zeros([BLOCK_M], dtype=tl.float32), mask=offs_m < seqlen_q
            )
        return

    # ============================================================
    #         NORMAL PROCESSING (Some K Blocks Visible)
    # ============================================================
    """
    This program has visible K blocks to process.
    We'll use two calls to handle different block types efficiently.
    """

    # Initialize for processing
    # Compute pointers for all the tensors used in this kernel.
    q_offset = (
        Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    )
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    k_offset = (
        K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    )
    k_ptrs = k_offset + offs_d_qk[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = (
        V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    )
    v_ptrs = v_offset + offs_vn[:, None] * stride_vk + offs_d_v[None, :] * stride_vn
    q_descale_ptr = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + (start_m + cu_seqlens_q_start) * stride_qsblk
    )
    k_descale_offset = (
        K_Descale
        + off_z * stride_ksz
        + off_h_k * stride_ksh
        + cu_seqlens_k_start * stride_ksblk
    )
    v_descale_offset = off_z * stride_vsz + off_h_k * stride_vsh
    
    v_mean_offset = (
        V_mean
        + off_z * stride_vmz
        + off_h_k * stride_vmh
        + offs_d_v[None, :]
    )
    v_mean = tl.load(v_mean_offset)
    q_descale = tl.load(q_descale_ptr)  # MHA: use q head index


    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = (
            bias
            + bias_offset
            + offs_m[:, None] * stride_bm
            + offs_n[None, :] * stride_bn
        )
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(ALIBI_SLOPES + a_offset)
    else:
        alibi_slope = None

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACCUMULATOR_TYPE)

    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk[None, :] < ACTUAL_BLOCK_DMODEL_QK)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # ========== Process MASKED K Blocks in the front ==========
    # NOTE: we use USE_SLIDING_WINDOW as guard because the compiler will crash other wise. front masking is only for sliding window so that is fine.
    if n_front_masked_blocks > 0 and USE_SLIDING_WINDOW:
        block_min = n_front_skip_blocks * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N

        k_descale_ptr = k_descale_offset + n_front_skip_blocks * stride_ksblk

        v_descale_ptr = (
            V_Descale
            + v_descale_offset
            + (block_min//SCALE_GROUP_SIZE + offs_vsn[None, :]) * stride_vsn
            + offs_d_vs[:, None]
        )

        acc, l_i, m_i = _sage_fwd_mask_v3(
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
            stride_sn,
            stride_sm,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            philox_offset_base,
            SD_MASK,
            stride_sz,
            stride_sh,
            off_z,
            off_h_q,
            offs_m,
            offs_n,
            offs_d_qk,
            offs_d_v,
            offs_d_vs,
            block_min,  # Start of front masked blocks
            block_max,  # End of front masked blocks
            0,  # n_extra_tokens (0 for front blocks, only relevant for last block)
            alibi_slope,
            q_descale,
            k_descale_ptr,
            v_descale_ptr,
            stride_ksblk,
            stride_vsn,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_VS,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
        )

    # ========== Process FULL K Blocks (Fast Path) ==========
    if n_full_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        block_max = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N

        k_descale_ptr = (
            k_descale_offset
            + (n_front_skip_blocks + n_front_masked_blocks) * stride_ksblk
        )
        v_descale_ptr = (
            V_Descale
            + v_descale_offset
            + (block_min//SCALE_GROUP_SIZE + offs_vsn[None, :]) * stride_vsn
            + offs_d_vs[:, None]
        )

        acc, l_i, m_i = _sage_fwd_no_mask_v3(
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
            stride_sn,
            stride_sm,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            philox_offset_base,
            SD_MASK,
            stride_sz,
            stride_sh,
            off_z,
            off_h_q,
            offs_m,
            offs_d_qk,
            offs_d_v,
            offs_d_vs,
            block_min,  # Start of range: 0
            block_max,  # End of range: n_full_blocks * BLOCK_N
            alibi_slope,
            q_descale,
            k_descale_ptr,
            v_descale_ptr,
            stride_ksblk,
            stride_vsn,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_VS,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
        )

    # ========== Process MASKED K Blocks in the back ==========
    if n_back_masked_blocks > 0:
        block_min = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N
        block_max = (
            n_front_skip_blocks
            + n_front_masked_blocks
            + n_full_blocks
            + n_back_masked_blocks
        ) * BLOCK_N

        k_descale_ptr = (
            k_descale_offset
            + (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks)
            * stride_ksblk
        )
        v_descale_ptr = (
            V_Descale
            + v_descale_offset
            + (block_min//SCALE_GROUP_SIZE + offs_n[None, :]) * stride_vsn
            + offs_d_vs[:, None]
        )
        acc, l_i, m_i = _sage_fwd_mask_v3(
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
            stride_sn,
            stride_sm,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            philox_offset_base,
            SD_MASK,
            stride_sz,
            stride_sh,
            off_z,
            off_h_q,
            offs_m,
            offs_n,
            offs_d_qk,
            offs_d_v,
            offs_d_vs,
            block_min,  # Start of range: n_full_blocks * BLOCK_N
            block_max,  # End of range: n_visible_k_blocks * BLOCK_N
            n_extra_tokens,  # Padding tokens in last block
            alibi_slope,
            q_descale,
            k_descale_ptr,
            v_descale_ptr,
            stride_ksblk,
            stride_vsn,
            IS_CAUSAL,  # Use actual causal flag
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_VS,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
        )

    # ============================================================
    #                        EPILOGUE
    # ============================================================
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    # Instead of directly computing 1/l_i which can be inf,
    # we check for the invalid case first
    if USE_SLIDING_WINDOW:
        # For rows where m_i is still -inf, no keys were valid
        # Set l_i to 1.0 to avoid division by zero (acc is already 0)
        invalid_mask = m_i == float("-inf")
        l_i_safe = tl.where(invalid_mask, 1.0, l_i)
        l_recip = 1 / l_i_safe[:, None]
    else:
        invalid_mask = None
        l_recip = 1 / l_i[:, None]

    acc = acc * l_recip
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale

    # compute log-sum-exp
    if RETURN_LSE:
        if USE_EXP2:
            # RCP_LN2: tl.constexpr = 1.4426950408889634
            LN2: tl.constexpr = 0.6931471824645996
            # compute log-sum-exp in base 2 units
            # mi_base2 = m_i * RCP_LN2
            mi_base2 = m_i
            # For invalid rows, log(l_i) would be -inf, but we want LSE to be -inf
            # So we handle this case explicitly
            if USE_SLIDING_WINDOW:
                log_l_i = tl.where(invalid_mask, 0.0, tl.math.log2(l_i))
                softmax_lse = mi_base2 + log_l_i
                # Ensure invalid rows have LSE = -inf
                softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)
            else:
                softmax_lse = mi_base2 + tl.math.log2(l_i)
            # convert back to natural units
            softmax_lse *= LN2
        else:
            if USE_SLIDING_WINDOW:
                log_l_i = tl.where(invalid_mask, 0.0, tl.math.log(l_i))
                softmax_lse = m_i + log_l_i
                softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)
            else:
                softmax_lse = m_i + tl.math.log(l_i)

    # handle masking edge cases
    if USE_SLIDING_WINDOW:
        if IS_CAUSAL:
            pass
        else:
            pass
    else:
        if IS_CAUSAL:
            # When seqlen_q > seqlen_k, some rows are completely above the causal diagonal
            # These rows have all -inf attention scores, resulting in NaN after softmax
            # e.g.
            # Q length: 6, K length: 4
            # Causal mask (X = can attend, . = cannot):
            #    K0 K1 K2 K3
            # Q0   .  .  .  .  <- All masked, would give NaN
            # Q1   .  .  .  .  <- All masked, would give NaN
            # Q2   X  .  .  .  <- First valid row
            # Q3   X  X  .  .
            # Q4   X  X  X  .
            # Q5   X  X  X  X
            causal_start_idx = seqlen_q - seqlen_k
            start_m_idx = start_m * BLOCK_M

            # Create mask for rows that need zeroing
            row_indices = start_m_idx + tl.arange(0, BLOCK_M)
            causal_mask = row_indices < causal_start_idx

            # Zero out both acc and LSE for these rows
            if causal_start_idx > start_m_idx:
                end_m_idx = (start_m + 1) * BLOCK_M
                if causal_start_idx < end_m_idx:
                    # This block contains the boundary - need to mask acc
                    out_mask_boundary = tl.full(
                        (BLOCK_DMODEL_V,), causal_start_idx, dtype=tl.int32
                    )
                    out_ptrs_mask = row_indices[:, None] >= out_mask_boundary[None, :]
                    z = 0.0
                    acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

            # Zero out LSE for rows above diagonal
            if RETURN_LSE:
                softmax_lse = tl.where(causal_mask, 0.0, softmax_lse)

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    if RETURN_LSE:
        l_offset = (
            LSE
            + off_z * stride_lse_z
            + off_h_q * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
        )
        l_ptrs = l_offset + offs_m * stride_lse_m

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last Q block. For others, overflow_size will be -ve
    end_m_idx = (start_m + 1) * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    if RETURN_LSE:
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
            l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
        else:
            tl.store(l_ptrs, softmax_lse)

    # write back O
    o_offset = (
        Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    )
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_o[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL_V], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD_V:
        o_ptrs_mask = o_ptrs_mask & (offs_d_o[None, :] < ACTUAL_BLOCK_DMODEL_V)

    # add v_mean back
    acc = acc.to(Out.dtype.element_ty) + v_mean
    
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


def fav3_sage_triton_impl_v3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: Optional[torch.Tensor],
    sd_mask: Optional[torch.Tensor],
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    layout: Literal["bshd", "bhsd", "thd"],
    # varlen
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlens_q: int,
    max_seqlens_k: int,
    # dropout
    dropout_p: float,
    philox_seed: Optional[int],
    philox_offset: Optional[int],
    # misc
    return_scores: bool,
    use_exp2: bool,
    # fp8
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    FP8_MAX: float,
    # seqused for FA v3
    v_mean: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    # rotary (optional)
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
    seqlens_rotary: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    # get params, strides and shape
    IS_VARLEN = layout == "thd"

    # common assertions
    assert (
        0.0 <= dropout_p <= 1.0
    ), f"dropout_p must be between 0 and 1, got {dropout_p}"
    assert (
        q.device == k.device == v.device == o.device
    ), f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}"
    assert (
        q.dtype == k.dtype == torch.int8
    ), f"q, k must have the dtype torch.int8. Got {q.dtype} for q and {k.dtype} for k"
    current_device = torch.cuda.current_device()
    assert (
        q.is_cuda and q.device.index == current_device
    ), f"Device mismatch: Kernel will launch on cuda:{current_device}, but tensors are on {q.device}"
    
    
    # get shapes and strides
    if IS_VARLEN:
        # shape
        total_seqlen_q, nheads_q, head_size_q = q.shape
        total_seqlen_k, nheads_k, head_size_k = k.shape
        total_seqlen_v, nheads_v, head_size_v = v.shape

        # assert shapes
        assert (
            cu_seqlens_q is not None
        ), "cu_seqlens_q must be provided for varlen layout"
        assert (
            cu_seqlens_k is not None
        ), "cu_seqlens_k must be provided for varlen layout"
        assert (
            max_seqlens_q is not None and max_seqlens_q > 0
        ), "max_seqlens_q must be provided and positive for varlen layout"
        assert (
            max_seqlens_k is not None and max_seqlens_k > 0
        ), "max_seqlens_k must be provided and positive for varlen layout"

        # assert head dimensions
        assert (
            head_size_q == head_size_k
        ), f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert (
            nheads_k == nheads_v
        ), f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert (
            nheads_q % nheads_k == 0
        ), f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"

        # assert output shapes
        assert o.shape == (
            total_seqlen_q,
            nheads_q,
            head_size_v,
        ), f"o shape {o.shape} != expected {(total_seqlen_q, nheads_q, head_size_v)}"

        # assert cu_seqlens
        assert (
            cu_seqlens_q.dtype == torch.int32
        ), f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}"
        assert (
            cu_seqlens_k.dtype == torch.int32
        ), f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}"
        assert cu_seqlens_q[0] == 0, "cu_seqlens_q must start with 0"
        assert cu_seqlens_k[0] == 0, "cu_seqlens_k must start with 0"
        assert (
            cu_seqlens_q[-1] == total_seqlen_q
        ), f"cu_seqlens_q[-1] {cu_seqlens_q[-1]} != total_seqlen_q {total_seqlen_q}"
        assert (
            cu_seqlens_k[-1] == total_seqlen_k
        ), f"cu_seqlens_k[-1] {cu_seqlens_k[-1]} != total_seqlen_k {total_seqlen_k}"

        # set vars
        batch = len(cu_seqlens_q) - 1
        head_size_qk = head_size_q

        # Assert softmax_lse tensor is large enough
        if softmax_lse is not None:
            assert (
                softmax_lse.shape[0] >= nheads_q
            ), f"softmax_lse.shape[0]={softmax_lse.shape[0]} must be >= nheads_q={nheads_q}"
            assert (
                softmax_lse.shape[1] >= total_seqlen_q
            ), f"softmax_lse.shape[1]={softmax_lse.shape[1]} must be >= total_seqlen_q={total_seqlen_q}"
            assert (
                softmax_lse.dtype == torch.float32
            ), f"softmax_lse must be float32, got {softmax_lse.dtype}"
            assert (
                softmax_lse.device == q.device
            ), "softmax_lse must be on same device as q"

        # strides
        stride_qb, stride_qh, stride_qm, stride_qd = (
            0,
            q.stride(1),
            q.stride(0),
            q.stride(2),
        )
        stride_kb, stride_kh, stride_kn, stride_kd = (
            0,
            k.stride(1),
            k.stride(0),
            k.stride(2),
        )
        stride_vb, stride_vh, stride_vn, stride_vd = (
            0,
            v.stride(1),
            v.stride(0),
            v.stride(2),
        )
        stride_ob, stride_oh, stride_om, stride_od = (
            0,
            o.stride(1),
            o.stride(0),
            o.stride(2),
        )
        stride_lse_z, stride_lse_h, stride_lse_m = (
            (
                0,
                softmax_lse.stride(0),
                softmax_lse.stride(1),
            )
            if softmax_lse is not None
            else (0, 0, 0)
        )
    else:
        bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        # shapes
        batch_q, seqlen_q, nheads_q, head_size_q = map_dims(q.shape, bshd)
        batch_k, seqlen_k, nheads_k, head_size_k = map_dims(k.shape, bshd)
        batch_v, seqlen_v, nheads_v, head_size_v = map_dims(v.shape, bshd)

        # assert batch dimensions
        assert (
            batch_q == batch_k == batch_v
        ), f"batch sizes must match: q={batch_q}, k={batch_k}, v={batch_v}"

        # assert head dimensions
        assert (
            head_size_q == head_size_k
        ), f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert (
            nheads_k == nheads_v
        ), f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert (
            nheads_q % nheads_k == 0
        ), f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"

        # assert sequence lengths
        assert (
            seqlen_k == seqlen_v*2
        ), f"k and v sequence lengths must match: k={seqlen_k}, v={seqlen_v}"

        # assert output shapes
        assert o.shape == (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            v.shape[-1],
        ), f"o shape {o.shape} != expected {(batch_q, seqlen_q, nheads_q, head_size_v)}"

        # set vars
        batch = batch_q
        head_size_qk = head_size_q
        max_seqlens_q = seqlen_q
        max_seqlens_k = seqlen_k

        # Assert softmax_lse tensor is large enough
        if softmax_lse is not None:
            assert (
                softmax_lse.shape[0] >= batch
            ), f"softmax_lse.shape[0]={softmax_lse.shape[0]} must be >= batch={batch}"
            assert (
                softmax_lse.shape[1] >= nheads_q
            ), f"softmax_lse.shape[1]={softmax_lse.shape[1]} must be >= nheads_q={nheads_q}"
            assert (
                softmax_lse.shape[2] >= seqlen_q
            ), f"softmax_lse.shape[2]={softmax_lse.shape[2]} must be >= seqlen_q={seqlen_q}"
            assert (
                softmax_lse.dtype == torch.float32
            ), f"softmax_lse must be float32, got {softmax_lse.dtype}"
            assert (
                softmax_lse.device == q.device
            ), "softmax_lse must be on same device as q"

        # strides
        stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
        stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)
        stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd)
        stride_ob, stride_om, stride_oh, stride_od = map_dims(o.stride(), bshd)
        stride_lse_z, stride_lse_h, stride_lse_m = (
            softmax_lse.stride() if softmax_lse is not None else (0, 0, 0)
        )

    # apply rotary embeddings
    if rotary_cos is not None and rotary_sin is not None:
        raise NotImplementedError("Rotary embeddings prefill are not implemented yet.")
    else:
        assert o.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], f"Output tensor o must be fp16, bf16, or fp32 when using fp8, got {o.dtype}"

    stride_qsz, stride_qsh, stride_qsblk = q_descale.stride()
    stride_ksz, stride_ksh, stride_ksblk = k_descale.stride()

    stride_vsz, stride_vsn, stride_vsh, _ = map_dims(
        v_descale.stride(), bshd
    )
    # print("v_mean", v_mean.shape, v_mean.stride())
    stride_vmz, stride_vmh, _ = v_mean.stride()

    # check features
    use_sliding_window = window_size_left != -1 or window_size_right != -1
    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )
    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if bias is not None:
        assert bias.numel() < 2**31

    # Get closest power of 2 over or equal to 32 for both QK and V dimensions
    padded_d_model_qk = 1 << (head_size_qk - 1).bit_length()
    padded_d_model_v = 1 << (head_size_v - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model_qk = max(padded_d_model_qk, 16)
    padded_d_model_v = max(padded_d_model_v, 16)

    # sd_mask assertions and strides
    if sd_mask is not None:
        assert dropout_p > 0.0 or return_scores, "sd_mask provided but not used"
        assert (
            sd_mask is not None
        ), "sd_mask must be provided when return_scores=True or dropout_p > 0"
        # Assert sd_mask tensor is large enough
        assert (
            sd_mask.shape[0] >= batch
        ), f"sd_mask.shape[0]={sd_mask.shape[0]} must be >= batch={batch}"
        assert (
            sd_mask.shape[1] >= nheads_q
        ), f"sd_mask.shape[1]={sd_mask.shape[1]} must be >= nheads_q={nheads_q}"
        assert (
            sd_mask.shape[2] >= max_seqlens_q
        ), f"sd_mask.shape[2]={sd_mask.shape[2]} must be >= max_seqlens_q={max_seqlens_q}"
        assert (
            sd_mask.shape[3] >= max_seqlens_k
        ), f"sd_mask.shape[3]={sd_mask.shape[3]} must be >= max_seqlens_k={max_seqlens_k}"
        assert sd_mask.device == q.device, "sd_mask must be on same device as q"

        stride_sz, stride_sh, stride_sm, stride_sn = (
            sd_mask.stride(0),
            sd_mask.stride(1),
            sd_mask.stride(2),
            sd_mask.stride(3),
        )
    else:
        stride_sz, stride_sh, stride_sm, stride_sn = (0, 0, 0, 0)

    if bias is not None:
        stride_bz, stride_bh, stride_bm, stride_bn = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )
    else:
        stride_bz, stride_bh, stride_bm, stride_bn = (0, 0, 0, 0)

    return_lse = True if softmax_lse is not None else False

    # launch kernel
    def grid(META):
        return (batch, nheads_q, triton.cdiv(max_seqlens_q, META["BLOCK_M"]))

    if config is None:
        config = get_sage_fwd_configs(
            False, seqlen_q=max_seqlens_q, seqlen_k=max_seqlens_k, num_heads=nheads_q
        )
    sage_fwd_v3[grid](
        q,
        k,
        v,
        bias,
        q_descale,
        k_descale,
        v_descale,
        v_mean,
        stride_qsz,
        stride_qsh,
        stride_qsblk,
        stride_ksz,
        stride_ksh,
        stride_ksblk,
        stride_vsz,
        stride_vsh,
        stride_vsn,
        stride_vmz,
        stride_vmh,
        softmax_lse,
        o,
        sd_mask,
        alibi_slopes,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_bz,
        stride_bh,
        stride_bm,
        stride_bn,
        stride_az,
        stride_ah,
        stride_sz,
        stride_sh,
        stride_sm,
        stride_sn,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,  # Pass seqused tensors
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        RETURN_LSE=return_lse,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        USE_SLIDING_WINDOW=use_sliding_window,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        IS_VARLEN=IS_VARLEN,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=use_alibi,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        USE_SEQUSED=(seqused_q is not None or seqused_k is not None),
        **config,
    )


def sage_quant_v3(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    layout="bshd",
    smooth_k=True,
):
    """
    Quantize Q and K tensors to INT8 with per-block scaling.

    Args:
        q: Query tensor
        k: Key tensor
        km: Optional pre-computed K smoothing factors (if None and smooth_k=True, will be computed)
        BLKQ: Block size for Q quantization
        BLKK: Block size for K quantization
        sm_scale: Softmax scale factor (defaults to head_dim^-0.5)
        layout: Either "bshd" or "bhsd"
        smooth_k: Whether to apply SageAttention-style smoothing to K tensor (default: True)

    Returns:
        q_int8: Quantized Q tensor
        q_scale: Per-block scales for Q
        k_int8: Quantized K tensor
        k_scale: Per-block scales for K
        k_smooth: K smoothing factors applied (or None if smooth_k=False)
    """
    q_int8 = torch.empty_like(q, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty_like(k, dtype=torch.int8, device=k.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    Q_NUM_BLKS = (qo_len + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    # Apply K tensor smoothing following SageAttention approach
    if smooth_k:
        k = k - k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    q_scale = torch.empty((b, h_qo, Q_NUM_BLKS), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, K_NUM_BLKS), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q_task_count = b * h_qo * Q_NUM_BLKS
    k_task_count = b * h_kv * K_NUM_BLKS

    grid = (q_task_count + k_task_count, )

    # call sage_quant_kernel
    sage_quant_kernel[grid](
        q,
        q_int8,
        q_scale,
        k,
        k_int8,
        k_scale,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        (sm_scale * 1.4426950408889634),
        q_task_count,
        k_task_count,
        b,
        h_qo,
        h_kv,
        Q_NUM_BLKS,
        K_NUM_BLKS,
        qo_len,
        kv_len,
        triton.next_power_of_2(kv_len),
        FP8_MAX=FP8_MAX,
        INT8_MAX=torch.iinfo(q_int8.dtype).max,
        D=head_dim,
        BLK_Q=BLKQ,
        BLK_K=BLKK,
    )
    # v = v.permute(0, 1, 3, 2) if layout == "bhsd" else v.permute(0, 2, 3, 1)
    v_mean = v.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
    v = v - v_mean
    
    v_fp4, v_scale = downcast_to_mxfp(v, torch.uint8, axis=-2 if layout == "bhsd" else -3)

    # print("v_fp4.shape:", v_fp4.shape)
    v_mean = v_mean.squeeze(1 if layout == "bshd" else 2)

    return q_int8, q_scale, k_int8, k_scale, v_fp4, v_scale, v_mean

@triton.jit
def sage_quant_kernel(
    Q_Input,
    Q_Output,
    Q_Scale,
    K_Input,
    K_Output,
    K_Scale,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_qsz,
    stride_qsh,
    stride_ksz,
    stride_ksh,
    sm_scale,
    q_task_count,
    k_task_count,
    BATCH,
    Q_HEAD,
    K_HEAD,
    Q_NUM_BLKS,
    K_NUM_BLKS,
    SEQLEN_Q,
    SEQLEN_K,
    SEQLEN_K_PADDED: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INT8_MAX: tl.constexpr,
    D: tl.constexpr,
    BLK_Q: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_blk_q = tl.arange(0, BLK_Q)
    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    if pid < q_task_count:
        # here we do Q
        off_blk, off_h, off_b = pid_grid_3d(pid, Q_NUM_BLKS, Q_HEAD, BATCH)
        offs_qn = off_blk * BLK_Q + offs_blk_q

        q_offs = (
            off_b * stride_qz
            + off_h * stride_qh
            + offs_qn[:, None] * stride_qn
            + offs_d[None, :]
        )

        q_input_ptrs = Q_Input + q_offs
        q_output_ptrs = Q_Output + q_offs
        q_scale_ptrs = Q_Scale + off_b * stride_qsz + off_h * stride_qsh + off_blk

        _general_quant_kernel(
            q_input_ptrs,
            q_output_ptrs,
            q_scale_ptrs,
            INT8_MAX,
            offs_qn[:, None] < SEQLEN_Q,
            sm_scale=sm_scale,
        )
    elif pid >= q_task_count and pid < q_task_count + k_task_count:
        # here we do K
        _pid = pid - q_task_count
        off_blk, off_h, off_b = pid_grid_3d(_pid, K_NUM_BLKS, K_HEAD, BATCH)

        offs_kn = off_blk * BLK_K + offs_blk_k

        k_offs = (
            off_b * stride_kz
            + off_h * stride_kh
            + offs_kn[:, None] * stride_kn
            + offs_d[None, :]
        )

        k_input_ptrs = K_Input + k_offs
        k_output_ptrs = K_Output + k_offs
        k_scale_ptrs = K_Scale + off_b * stride_ksz + off_h * stride_ksh + off_blk

        _general_quant_kernel(
            k_input_ptrs,
            k_output_ptrs,
            k_scale_ptrs,
            INT8_MAX,
            offs_kn[:, None] < SEQLEN_K,
        )


@triton.jit
def _general_quant_kernel(
    input_ptrs, output_ptrs, scale_ptrs, DTYPE_MAX, mask, sm_scale=None
):
    if mask is not None:
        x = tl.load(input_ptrs, mask=mask, other=0.0)
    else:
        x = tl.load(input_ptrs)
    x = x.to(tl.float32)
    if sm_scale is not None:
        x *= sm_scale
    scale = tl.max(tl.abs(x)) / DTYPE_MAX
    x_quant = x / scale
    if output_ptrs.dtype.element_ty == tl.int8:
        x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)
    x_quant = x_quant.to(output_ptrs.dtype.element_ty)
    tl.store(output_ptrs, x_quant, mask=mask)
    tl.store(scale_ptrs, scale)
