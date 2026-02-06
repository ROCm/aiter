import aiter
import torch
import triton
import triton.language as tl
from typing import Literal, Optional
from .utils import (
    compute_alibi_block,
    map_dims,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton._triton_kernels.sage_attn_triton_amd.fwd_prefill import get_sage_fwd_configs, _general_quant_kernel

from .scale_mxfp import downcast_to_mxfp_rne
from .hadamard_rotation import apply_hadamard_rotation_qk

@triton.jit
def _sage_fwd_no_mask_v2(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    delta_s_ptr,
    stride_kn,
    stride_vk,
    stride_dsk,
    seqlen_k,
    seqlen_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    q_descale,
    q_descale_pre,
    k_descale_base_ptr,
    k_descale_pre_base_ptr,
    stride_ksn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_K: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ACCUMULATOR_TYPE,
):
    k_descale_ptr = k_descale_base_ptr
    k_descale_pre_ptr = k_descale_pre_base_ptr

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        # Load K
        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_K
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        #k_descale = tl.load(k_descale_ptr)
        k_descale = tl.load(
            k_descale_ptr, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )
        k_descale_ptr += stride_ksn * BLOCK_N
        k_descale_pre = tl.load(k_descale_pre_ptr)
        k_descale_pre_ptr += 1

        # SMOOTH_Q is always True - always load delta_s
        delta_s_ptrs = delta_s_ptr + tl.arange(0, BLOCK_N)
        delta_s = tl.load(delta_s_ptrs)
        delta_s_ptr += stride_dsk

        # Optionally preload V
        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        # SMOOTH_Q is always True
        qk += delta_s[None, :]

        # -- compute qk ----
        # Hardcoded for e2m1 dtype
        qk += tl.dot_scaled(q, q_descale, "e2m1", k, k_descale, "e2m1", fast_math=True) * (q_descale_pre * k_descale_pre)

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # scale and subtract max
        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk - m_ij[:, None]
        )

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

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
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc += tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_mask_v2(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    delta_s_ptr,
    stride_kn,
    stride_vk,
    stride_dsk,
    start_m,
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
    q_descale_pre,
    k_descale_base_ptr,
    k_descale_pre_base_ptr,
    stride_ksn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_K: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_EXP2: tl.constexpr,
    ACCUMULATOR_TYPE,
):
    # Initialize pointers for this masked block range
    k_descale_ptr = k_descale_base_ptr
    k_descale_pre_ptr = k_descale_pre_base_ptr
    
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = kv_offs_n[None, :] < seqlen_k
        v_mask = kv_offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask = k_mask & (offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_K)
        if PADDED_HEAD_V:
            v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

        # load k and if preload_v then v
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptr, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )
        k_descale_ptr += stride_ksn * BLOCK_N
        k_descale_pre = tl.load(k_descale_pre_ptr)
        k_descale_pre_ptr += 1

        # SMOOTH_Q is always True - always load delta_s
        delta_s_ptrs = delta_s_ptr + tl.arange(0, BLOCK_N)
        delta_s = tl.load(delta_s_ptrs)
        delta_s_ptr += stride_dsk

        if PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)
        # SMOOTH_Q is always True
        qk += delta_s[None, :]

        # -- compute qk ----
        # Hardcoded for e2m1 dtype
        qk += tl.dot_scaled(q, q_descale, "e2m1", k, k_descale, "e2m1", fast_math=True) * (q_descale_pre * k_descale_pre)

        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # If this is the last block / iteration, we want to
        # mask if the sequence length is not a multiple of block size
        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + offs_n[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # scale and subtract max
        q_shifted = qk - m_ij[:, None]

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

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
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc += tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def compute_window_bounds(
    q_start,
    q_end,
    diag,
    seqlen_k,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Calculate the window boundaries for a query block."""
    # Left boundary
    if WINDOW_SIZE_LEFT < 0:
        left_min = 0
        left_max = 0
    else:
        left_min = tl.maximum(0, q_start + diag - WINDOW_SIZE_LEFT)
        left_max = tl.maximum(0, q_end + diag - WINDOW_SIZE_LEFT)

    # Right boundary
    if IS_CAUSAL:
        # Causal cap: col ≤ row + diag
        right_min = tl.minimum(seqlen_k - 1, q_start + diag)
        right_max = tl.minimum(seqlen_k - 1, q_end + diag)
    else:
        if WINDOW_SIZE_RIGHT < 0:
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)
        else:
            # Non-causal doesn't have the diagonal constraint
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)

    return left_min, left_max, right_min, right_max


@triton.jit
def classify_window_blocks(
    left_min, left_max, right_min, right_max, BLOCK_N: tl.constexpr
):
    """Classify blocks based on window boundaries."""
    # First and last blocks that have ANY overlap with window
    first_block = left_min // BLOCK_N
    last_block = right_max // BLOCK_N

    # First block that is FULLY visible for all rows in Q block
    full_left_block = left_max // BLOCK_N + (left_max % BLOCK_N != 0)
    clipped_left = tl.minimum(full_left_block, last_block + 1)

    # Last block that is FULLY visible for all rows in Q block
    last_full_block_candidate = right_min // BLOCK_N
    if (last_full_block_candidate + 1) * BLOCK_N - 1 > right_min:
        last_full_block_candidate -= 1
    full_right_block = tl.maximum(last_full_block_candidate, clipped_left - 1)

    # Calculate counts
    n_front_skip_blocks = first_block
    n_front_masked_blocks = tl.maximum(0, clipped_left - first_block)
    n_full_blocks = tl.maximum(0, full_right_block - clipped_left + 1)
    n_back_masked_blocks = tl.maximum(0, last_block - full_right_block)

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        clipped_left,
    )  # Return clipped_left for padded block handling


@triton.jit
def handle_padded_last_block(
    n_extra_tokens,
    last_block,
    total_k_blocks,
    clipped_left,
    n_front_masked_blocks,
    n_full_blocks,
    n_back_masked_blocks,
):
    """Ensure a padded last K-block is never classified as 'full'.

    We move the padded last block (if visible) into the back-masked bucket.
    If it's already back-masked, we do nothing.  If it was counted in the
    front-masked range, we decrement front-masked; if it was counted as full,
    we decrement full.  Then we increment back-masked.
    """
    padded_last_k = (n_extra_tokens != 0) & (last_block == total_k_blocks - 1)

    if padded_last_k:
        # current 'full' range right edge
        full_right_block = clipped_left + n_full_blocks - 1

        # If last_block is already beyond full_right_block, it's already in back-masked → nothing to do
        last_already_back_masked = last_block > full_right_block
        if not last_already_back_masked:
            # If the window starts past last_block, it was counted in front-masked
            if clipped_left > last_block:
                n_front_masked_blocks = tl.maximum(0, n_front_masked_blocks - 1)
            else:
                # Otherwise it was counted 'full' → move it out of full
                n_full_blocks = tl.maximum(0, n_full_blocks - 1)
            # In both cases we need one more back-masked block
            n_back_masked_blocks = n_back_masked_blocks + 1

    return n_front_masked_blocks, n_full_blocks, n_back_masked_blocks


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simplified block masking for attention computation without causal or sliding window.
    Only handles padding in the last K block.

    Returns:
        - n_front_skip_blocks: Always 0 (no skipping)
        - n_front_masked_blocks: Always 0 (no front masking)
        - n_full_blocks: All blocks except possibly the last
        - n_back_masked_blocks: 1 if last block is padded, 0 otherwise
        - n_extra_tokens: Padding tokens in last K block
    """
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)
    
    # No causal or sliding window - all positions can attend to all positions
    # Only need to handle padding in the last block
    n_front_skip_blocks = 0
    n_front_masked_blocks = 0
    
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
def sage_fwd_v2(
    Q,
    K,
    V,
    Q_Descale,
    Q_Descale_Pre,
    K_Descale,
    K_Descale_Pre,
    Delta_S,
    V_Descale,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_qspz,
    stride_qsph,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_kspz,
    stride_ksph,
    stride_dsz,
    stride_dsh,
    stride_dsq,
    stride_dsk,
    stride_vsz,
    stride_vsh,
    Out,
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
    USE_EXP2: tl.constexpr,
):
    # Hardcoded for e2m1 dtype - 2 elements per int8
    Q_HEAD_DIM_DIVISOR: tl.constexpr = 2
    K_HEAD_DIM_DIVISOR: tl.constexpr = 2
    # SMOOTH_Q is always True
    # set params
    ACCUMULATOR_TYPE: tl.constexpr = tl.float32  # for q*k product
    SCALE_GROUP_SIZE: tl.constexpr = 32

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
    offs_d_q = tl.arange(0, BLOCK_DMODEL_QK // Q_HEAD_DIM_DIVISOR)  # we fit 2 fp4 elements per int8
    offs_d_k = tl.arange(0, BLOCK_DMODEL_QK // K_HEAD_DIM_DIVISOR)  # we fit 2 fp4 elements per int8
    offs_d_qk_s = tl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP_SIZE)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    # Simplified seqlen handling - no varlen support
    seqlen_q = MAX_SEQLENS_Q
    seqlen_k = MAX_SEQLENS_K

    # Load scale factors
    # For MQA/GQA (GROUP_SIZE != 1), q_descale uses the same indexing as k/v (off_h_k)
    # For MHA (GROUP_SIZE == 1), q_descale uses off_h_q (same as off_h_k)
    # tl.static_print("Q_Descale:", Q_Descale)

    q_descale_ptrs = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + offs_m[:, None] * stride_qsm
        + offs_d_qk_s[None, :]
    )

    q_descale = tl.load(
        q_descale_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0
    )  # MHA: use q head index

    q_descale_pre_ptr = (
        Q_Descale_Pre
        + off_z * stride_qspz
        + off_h_q * stride_qsph
        + start_m
    )

    q_descale_pre = tl.load(q_descale_pre_ptr)

    # Defer computing offsets until needed to reduce register pressure
    # These will be recomputed cheaply in each block section

    # Defer computing offsets until needed to reduce register pressure
    # These will be recomputed cheaply in each block section

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
        o_offset = Out + off_z * stride_oz + off_h_q * stride_oh
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
        o_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_V:
            o_mask = o_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
        tl.store(
            o_ptrs,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty),
            mask=o_mask,
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
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_q[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh
    k_ptrs = k_offset + offs_d_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vn

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACCUMULATOR_TYPE)

    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (offs_d_q[None, :] < ACTUAL_BLOCK_DMODEL_QK // Q_HEAD_DIM_DIVISOR)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # Compute offsets here - closer to usage to minimize live range
    k_descale_offset = off_z * stride_ksz + off_h_k * stride_ksh
    k_descale_pre_offset = K_Descale_Pre + off_z * stride_kspz + off_h_k * stride_ksph
    delta_s_offset = off_z * stride_dsz + off_h_k * stride_dsh + start_m * stride_dsq

    # ========== Process MASKED K Blocks in the front ==========
    # Front masking no longer needed without sliding window
    # This section can be removed entirely as n_front_masked_blocks will always be 0
    
    # ========== Process FULL K Blocks (Fast Path) ==========
    if n_full_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        block_max = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N

        k_descale_ptr = (
            K_Descale
            + k_descale_offset
            + (block_min + offs_n[:, None]) * stride_ksn
            + offs_d_qk_s[None, :]
        )

        k_descale_pre_ptr = k_descale_pre_offset + n_front_skip_blocks + n_front_masked_blocks

        # SMOOTH_Q is always True
        delta_s_ptr = Delta_S + delta_s_offset + (n_front_skip_blocks + n_front_masked_blocks) * stride_dsk

        acc, l_i, m_i = _sage_fwd_no_mask_v2(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            delta_s_ptr,
            stride_kn,
            stride_vk,
            stride_dsk,
            seqlen_k,
            seqlen_q,
            offs_m,
            offs_n,
            offs_d_k,
            offs_d_v,
            block_min,
            block_max,
            q_descale,
            q_descale_pre,
            k_descale_ptr,
            k_descale_pre_ptr,
            stride_ksn,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK // K_HEAD_DIM_DIVISOR,
            ACTUAL_BLOCK_DMODEL_V,
            USE_EXP2=USE_EXP2,
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
            K_Descale
            + k_descale_offset
            + (block_min + offs_n[:, None]) * stride_ksn
            + offs_d_qk_s[None, :]
        )

        k_descale_pre_ptr = k_descale_pre_offset + n_front_skip_blocks + n_front_masked_blocks + n_full_blocks

        # SMOOTH_Q is always True
        delta_s_ptr = Delta_S + delta_s_offset + (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks) * stride_dsk

        acc, l_i, m_i = _sage_fwd_mask_v2(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            delta_s_ptr,
            stride_kn,
            stride_vk,
            stride_dsk,
            start_m,
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
            q_descale_pre,
            k_descale_ptr,
            k_descale_pre_ptr,
            stride_ksn,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK // K_HEAD_DIM_DIVISOR,
            ACTUAL_BLOCK_DMODEL_V,
            USE_EXP2=USE_EXP2,
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
        )

    # ============================================================
    #                        EPILOGUE
    # ============================================================
    # Load v_descale here, right before epilogue where it's used
    v_descale = tl.load(
        V_Descale
        + off_z * stride_vsz
        + off_h_k * stride_vsh
        + offs_d_v,
        mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V,
        other=0.0,
    )
    
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip * v_descale

    # ============================================================
    #                       FINAL STORE
    # ============================================================
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    o_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_V:
        o_mask = o_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
    tl.store(o_ptrs, acc, mask=o_mask)


def fav3_sage_triton_impl_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,  # not used
    sd_mask: torch.Tensor,  # not used
    softmax_scale: float,  # not used
    alibi_slopes: torch.Tensor,  # not used
    causal: bool,  # not used
    window_size_left: int,  # not used
    window_size_right: int,  # not used
    bias: torch.Tensor,  # not used
    layout: Literal["bshd", "bhsd"],
    cu_seqlens_q: torch.Tensor,  # not used
    cu_seqlens_k: torch.Tensor,  # not used
    max_seqlens_q: int,
    max_seqlens_k: int,
    dropout_p: float,  # not used
    philox_seed: int,  # not used
    philox_offset: int,  # not used
    return_softmax: bool,  # not used
    use_exp2: bool,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    FP8_MAX: float,  # not used
    seqused_q: torch.Tensor,  # not used
    seqused_k: torch.Tensor,  # not used
    rotary_cos: torch.Tensor,  # not used
    rotary_sin: torch.Tensor,  # not used
    rotary_interleaved: bool,  # not used
    seqlens_rotary: torch.Tensor,  # not used
    config: Optional[dict],
    q_descale_pre: torch.Tensor,
    k_descale_pre: torch.Tensor,
    delta_s: torch.Tensor,
):
    # common assertions
    assert (
        q.device == k.device == v.device == o.device
    ), f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}"
    
    current_device = torch.cuda.current_device()
    assert (
        q.is_cuda and q.device.index == current_device
    ), f"Device mismatch: Kernel will launch on cuda:{current_device}, but tensors are on {q.device}"

    assert layout in ["bshd", "bhsd"], f"layout must be 'bshd' or 'bhsd', got {layout}"

    head_size_q = q.shape[-1]
    head_size_k = k.shape[-1]
    is_q_fp4 = q.dtype == torch.uint8
    is_k_fp4 = k.dtype == torch.uint8
    if is_q_fp4:
        head_size_q *= 2
    if is_k_fp4:
        head_size_k *= 2

    # get shapes and strides
    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    # shapes
    batch_q, seqlen_q, nheads_q, _ = map_dims(q.shape, bshd)
    batch_k, seqlen_k, nheads_k, _ = map_dims(k.shape, bshd)
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
        seqlen_k == seqlen_v
    ), f"k and v sequence lengths must match: k={seqlen_k}, v={seqlen_v}"

    # assert output shapes
    assert o.shape == (
        q.shape[0],
        q.shape[1],
        q.shape[2],
        v.shape[-1],
    ), f"o shape {o.shape} != expected {(batch_q, seqlen_q, nheads_q, head_size_v)}"

    batch = batch_q
    head_size_qk = head_size_q
    max_seqlens_q = seqlen_q
    max_seqlens_k = seqlen_k

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)
    stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd)
    stride_ob, stride_om, stride_oh, stride_od = map_dims(o.stride(), bshd)

    # No rotary embeddings support
    assert o.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], f"Output tensor o must be fp16, bf16, or fp32 when using fp8, got {o.dtype}"

    stride_qsz, stride_qsm, stride_qsh, _ = map_dims(
        q_descale.stride(), bshd
    )
    stride_ksz, stride_ksn, stride_ksh, _ = map_dims(
        k_descale.stride(), bshd
    )

    stride_vsz, stride_vsh, _ = v_descale.stride()

    stride_qspz, stride_qsph, _ = q_descale_pre.stride()
    stride_kspz, stride_ksph, _ = k_descale_pre.stride()

    if delta_s is not None:
        stride_dsz, stride_dsh, stride_dsq, stride_dsk, _ = delta_s.stride()
    else:
        stride_dsz, stride_dsh, stride_dsq, stride_dsk = (0, 0, 0, 0)

    # Get closest power of 2 over or equal to 32 for both QK and V dimensions
    padded_d_model_qk = 1 << (head_size_qk - 1).bit_length()
    padded_d_model_v = 1 << (head_size_v - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model_qk = max(padded_d_model_qk, 16)
    padded_d_model_v = max(padded_d_model_v, 16)

    # launch kernel
    grid = lambda META: (batch, nheads_q, triton.cdiv(max_seqlens_q, META["BLOCK_M"]))
    if config is None:
        config = get_sage_fwd_configs(False, seqlen_k=max_seqlens_k)

    sage_fwd_v2[grid](
        q,
        k,
        v,
        q_descale,
        q_descale_pre,
        k_descale,
        k_descale_pre,
        delta_s, 
        v_descale,
        stride_qsz,
        stride_qsh,
        stride_qsm,
        stride_qspz,
        stride_qsph,
        stride_ksz,
        stride_ksh,
        stride_ksn,
        stride_kspz,
        stride_ksph,
        stride_dsz,
        stride_dsh,
        stride_dsq,
        stride_dsk,
        stride_vsz,
        stride_vsh,
        o,
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
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_EXP2=use_exp2,
        **config,
    )


def sage_quant_v2(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX=None,
    FP4_MAX=None,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    layout="bshd",
    smooth_k=True,
    smooth_q=True,
    is_q_fp4=True,
    is_k_fp4=True,
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
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if FP8_MAX is None:
        FP8_MAX = torch.finfo(FP8_TYPE).max
    if FP4_MAX is None:
        FP4_MAX = 6.0

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

    # Apply K/Q tensor smoothing following SageAttention approach
    if smooth_k:
        k = k - k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)
    # call sage_quant_kernel
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
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8
    )

    ## rotate q and k
    q, k = apply_hadamard_rotation_qk(q, k)

    ## quantize q and k
    q_bf16 = torch.empty_like(q, dtype=torch.bfloat16, device=q.device)
    k_bf16 = torch.empty_like(k, dtype=torch.bfloat16, device=k.device)
    q_scale_pre = torch.empty((b, h_qo, Q_NUM_BLKS), device=q.device, dtype=torch.float32)
    k_scale_pre = torch.empty((b, h_kv, K_NUM_BLKS), device=k.device, dtype=torch.float32)

    # smooth Q + prescale Q
    delta_s = None
    q_mean = None
    stride_qmz, stride_qmh, stride_qmblk = (0, 0, 0)
    stride_dsz, stride_dsh, stride_dsm, stride_dsn = (0, 0, 0, 0)
    if smooth_q:
        q_mean = torch.empty((b, h_qo, Q_NUM_BLKS, head_dim), device=q.device, dtype=q.dtype)
        stride_qmz, stride_qmh, stride_qmblk, _ = q_mean.stride()
        delta_s = torch.empty((b, h_kv, Q_NUM_BLKS, K_NUM_BLKS, BLKK), device=q.device, dtype=torch.float32) # float32 since result of qk in the main kernel is float32
        stride_dsz, stride_dsh, stride_dsm, stride_dsn, _ = delta_s.stride()

    Q_MAX = FP4_MAX if is_q_fp4 else FP8_MAX
    grid = (b, h_qo, Q_NUM_BLKS)

    smooth_prescale_q_kernel[grid](
        q,
        q_bf16,
        q_scale_pre,
        smooth_q,
        q_mean,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        q_scale_pre.stride(0),
        q_scale_pre.stride(1),
        stride_qmz,
        stride_qmh,
        stride_qmblk,
        (sm_scale * 1.4426950408889634),
        b,
        h_qo,
        Q_NUM_BLKS,
        qo_len,
        Q_MAX=Q_MAX,
        D=head_dim,
        BLK_Q=BLKQ,
    )

    ## calculate delta_s if smooth_q and prescale K
    K_MAX = FP4_MAX if is_k_fp4 else FP8_MAX
    grid = (b, h_kv, K_NUM_BLKS)

    prescale_k_kernel[grid](
        k,
        k_bf16,
        k_scale_pre,
        smooth_q,
        q_mean,
        delta_s,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        k_scale_pre.stride(0),
        k_scale_pre.stride(1),
        stride_qmz,
        stride_qmh,
        stride_qmblk,
        stride_dsz,
        stride_dsh,
        stride_dsm,
        stride_dsn,
        b,
        h_kv, # should be equal to h_qo in this implementation
        K_NUM_BLKS,
        Q_NUM_BLKS,
        kv_len,
        qo_len,
        K_MAX=K_MAX,
        D=head_dim,
        BLK_K=BLKK,
    )

    if is_q_fp4:
        q_fp4, q_scale = downcast_to_mxfp_rne(q_bf16, torch.uint8, axis=-1)
    else:
        q_fp4, q_scale = downcast_to_mxfp_rne(q_bf16, aiter.dtypes.fp8, axis=-1)
    if is_k_fp4:
        k_fp4, k_scale = downcast_to_mxfp_rne(k_bf16, torch.uint8, axis=-1)
    else:
        k_fp4, k_scale = downcast_to_mxfp_rne(k_bf16, aiter.dtypes.fp8, axis=-1)

    return q_fp4, q_scale, q_scale_pre, k_fp4, k_scale, k_scale_pre, v_fp8, v_scale, delta_s


@triton.jit
def sage_v2_pre_quant_kernel(
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
    BATCH,
    Q_HEAD,
    K_HEAD,
    Q_NUM_BLKS,
    K_NUM_BLKS,
    SEQLEN_Q,
    SEQLEN_K,
    SEQLEN_K_PADDED: tl.constexpr,
    Q_MAX: tl.constexpr,
    K_MAX: tl.constexpr,
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
            Q_MAX,
            offs_qn[:, None] < SEQLEN_Q,
            sm_scale=sm_scale,
        )
    else:
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
            K_MAX,
            offs_kn[:, None] < SEQLEN_K,
        )

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
def smooth_prescale_q_kernel(
    Q_Input,
    Q_Output,
    Q_Scale,
    SMOOTH_Q: tl.constexpr,
    Q_Mean,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_qsz,
    stride_qsh,
    stride_qmz,
    stride_qmh,
    stride_qmblk,
    sm_scale,
    BATCH,
    Q_HEAD,
    Q_NUM_BLKS,
    SEQLEN_Q,
    Q_MAX: tl.constexpr,
    D: tl.constexpr,
    BLK_Q: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_blk = tl.program_id(2)

    offs_blk_q = tl.arange(0, BLK_Q)
    offs_d = tl.arange(0, D)

    offs_qn = off_blk * BLK_Q + offs_blk_q

    q_offs = (
        off_b * stride_qz
        + off_h * stride_qh
        + offs_qn[:, None] * stride_qn
        + offs_d[None, :]
    )

    q_input_ptrs = Q_Input + q_offs
    q_output_ptrs = Q_Output + q_offs
    mask = offs_qn[:, None] < SEQLEN_Q

    q_scale_ptrs = Q_Scale + off_b * stride_qsz + off_h * stride_qsh + off_blk

    x = tl.load(q_input_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    if sm_scale is not None:
        x *= sm_scale
    if SMOOTH_Q:
        q_mean_ptrs = Q_Mean + off_b * stride_qmz + off_h * stride_qmh + off_blk * stride_qmblk + offs_d
        actual_num = tl.minimum(BLK_Q, SEQLEN_Q - (off_blk * BLK_Q))
        x_mean = tl.sum(x, 0) / actual_num # [D]
        x = x - x_mean[None, :]
    scale = tl.max(tl.abs(x)) / Q_MAX
    x_quant = x / scale
    if q_output_ptrs.dtype.element_ty == tl.int8:
        x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)
    x_quant = x_quant.to(q_output_ptrs.dtype.element_ty)
    tl.store(q_output_ptrs, x_quant, mask=mask)
    tl.store(q_scale_ptrs, scale)
    if SMOOTH_Q:
        tl.store(q_mean_ptrs, x_mean.to(q_mean_ptrs.dtype.element_ty))

@triton.jit
def prescale_k_kernel(
    K_Input,
    K_Output,
    K_Scale,
    SMOOTH_Q: tl.constexpr,
    Q_Mean,
    Delta_S,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_ksz,
    stride_ksh,
    stride_qmz,
    stride_qmh,
    stride_qmblk,
    stride_dsz,
    stride_dsh,
    stride_dsm,
    stride_dsn,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    Q_NUM_BLKS,
    SEQLEN_K,
    SEQLEN_Q,
    K_MAX: tl.constexpr,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)
    off_blk = tl.program_id(2)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    offs_kn = off_blk * BLK_K + offs_blk_k

    k_offs = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_kn[:, None] * stride_kn
        + offs_d[None, :]
    )

    k_input_ptrs = K_Input + k_offs
    k_output_ptrs = K_Output + k_offs
    mask = offs_kn[:, None] < SEQLEN_K

    k_scale_ptrs = K_Scale + off_b * stride_ksz + off_h * stride_ksh + off_blk

    x = tl.load(k_input_ptrs, mask=mask, other=0.0) # [BLK_K, D]
    if SMOOTH_Q:
        q_mean_ptrs = Q_Mean + off_b * stride_qmz + off_h * stride_qmh + offs_d
        delta_s_ptrs = Delta_S + off_b * stride_dsz + off_h * stride_dsh + off_blk * stride_dsn + offs_d
        for m in range(Q_NUM_BLKS):
            q_mean = tl.load(q_mean_ptrs) # [D]
            ## x and q_mean should have the same dtype, e.g., bf16 here, the dot gives in float32 result
            delta_s = tl.dot(x, q_mean[:, None]) # [BLK_K, 1]
            delta_s = tl.reshape(delta_s, [BLK_K])
            tl.store(delta_s_ptrs, delta_s)

            q_mean_ptrs += stride_qmblk
            delta_s_ptrs += stride_dsm

    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / K_MAX
    x_quant = x / scale
    if k_output_ptrs.dtype.element_ty == tl.int8:
        x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)
    x_quant = x_quant.to(k_output_ptrs.dtype.element_ty)
    tl.store(k_output_ptrs, x_quant, mask=mask)
    tl.store(k_scale_ptrs, scale)
