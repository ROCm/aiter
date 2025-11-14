# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import torch
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.amd.cdna4.async_copy as acp

from ...utils._triton import arch_info
from ...utils.core import AITER_TRITON_CONFIGS_PATH
from ...utils._triton.pid_preprocessing import remap_xcd


@gluon.jit
def _full_like(tensor, value, shape=None, layout=None, dtype=None):
    """
    Creates a full tensor like the given tensor, with specified shape, layout, and dtype.

    NOTE: gluon.language has full_like, but it currently throws a "Not implemented" error.
    """

    return gl.full(
        tensor.shape if shape is None else shape,
        value,
        tensor.dtype if dtype is None else dtype,
        tensor.type.layout if layout is None else layout,
    )


@gluon.jit
def _create_mask(offset_first, offset_second, boundary_first, boundary_second):
    """
    Creates a boolean mask based on the given offsets and boundaries.
    If there's no boundary, we do not need explicit masking for that axis.
    """

    if boundary_first is not None and boundary_second is not None:
        return (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
    elif boundary_first is not None:
        return offset_first[:, None] < boundary_first
    elif boundary_second is not None:
        return offset_second[None, :] < boundary_second
    else:
        return _full_like(
            offset_first[:, None] + offset_second[None, :], True, dtype=gl.int1
        )


@gluon.jit
def _create_offsets(offset_first, offset_second, stride_first, stride_second):
    """
    Creates offsets based on the given first and second offsets, must be int32.
    """

    offsets = (
        offset_first[:, None] * stride_first + offset_second[None, :] * stride_second
    )
    return tl.cast(offsets, gl.int32)


@gluon.jit
def _load_fn(
    base_ptr,
    offset_first,
    offset_second,
    stride_first,
    stride_second,
    boundary_first,
    boundary_second,
):
    """
    Loads from the given pointer using mask determined by given offsets and offset boundaries,
    across axis 0 and axis 1 of a two-dimensional tensor.

    NOTE: Assumes the given axis offsets are already strided appropriately.
    """

    mask = _create_mask(offset_first, offset_second, boundary_first, boundary_second)
    offsets = _create_offsets(offset_first, offset_second, stride_first, stride_second)
    return gl.amd.cdna4.buffer_load(ptr=base_ptr, offsets=offsets, mask=mask, other=0.0)


@gluon.jit
def _load_fn_with_smem(
    load_idx,
    smem,
    base_ptr,
    offset_first,
    offset_second,
    stride_first,
    stride_second,
    boundary_first,
    boundary_second,
    NUM_STAGES: gl.constexpr,
):
    """
    Loads from the given pointers using smem as staging buffer.
    """

    mask = _create_mask(offset_first, offset_second, boundary_first, boundary_second)
    offsets = _create_offsets(offset_first, offset_second, stride_first, stride_second)
    acp.buffer_load_to_shared(
        dest=smem.index(load_idx % NUM_STAGES),
        ptr=base_ptr,
        offsets=offsets,
        mask=mask,
        other=0.0,
    )

    return load_idx + 1


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_base_ptr,
    k_pe_base_ptr,
    v_base_ptr,
    stride_kk,
    stride_kn,
    stride_vn,
    stride_vk,
    seqlen_k,
    block_min,
    block_max,
    n_extra_tokens,
    OFFS_K_N: gl.constexpr,
    OFFS_QK_M: gl.constexpr,
    OFFS_QK_N: gl.constexpr,
    OFFS_QK_N_CAUSAL: gl.constexpr,
    OFFS_V_N: gl.constexpr,
    OFFS_K_D: gl.constexpr,
    OFFS_KPE_D: gl.constexpr,
    OFFS_V_D: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,  # it's zero or a power of 2
    MFMA_LAYOUT: gl.constexpr,
    SHARED_K_LAYOUT: gl.constexpr,
    SHARED_V_LAYOUT: gl.constexpr,
    SM_SCALE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,  # whether to apply masking at some point while stepping (for causal or padding)
    ENABLE_PIPELINING: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """
    This function computes the inner loop of Flash Attention 2 over blocks of k/v,
    between given block_min and block_max.
    """

    RCP_LN2: gl.constexpr = 1.4426950408889634
    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0

    dot_left_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=MFMA_LAYOUT, k_width=8
    )
    dot_right_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=MFMA_LAYOUT, k_width=8
    )
    qk_zeros = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_LAYOUT)
    pv_zeros = gl.zeros(
        [BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=MFMA_LAYOUT
    )

    # TODO: Attempt to pipeline the loads of K and V with compute, using async_copy

    for start_n in tl.range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.

        # Compute k offsets across head dimension, only needed if padded head
        k = _load_fn(
            k_base_ptr,
            OFFS_K_D,
            OFFS_K_N,
            stride_kk,
            stride_kn,
            None if BLOCK_DMODEL == BLOCK_DMODEL_POW2 else BLOCK_DMODEL,
            None if not MASK_STEPS else seqlen_k - start_n,
        )
        if HAS_PE:
            k_pe = _load_fn(
                k_pe_base_ptr,
                OFFS_KPE_D,
                OFFS_K_N,
                stride_kk,
                stride_kn,
                None,
                None if not MASK_STEPS else seqlen_k - start_n,
            )

        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=MFMA_LAYOUT)
        qk_mask = gl.full([BLOCK_M, BLOCK_N], True, dtype=gl.int1, layout=MFMA_LAYOUT)
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.

            # remove the old if condition
            # if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
            # Though this will unconditionally compute mask_partial at runtime,
            # the causal for loop does not have the if-else block any more, which
            # helps instruction scheduling and register pressure.
            bound_cond = (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0)
            boundary_m = gl.full(
                [BLOCK_M],
                seqlen_k,
                dtype=gl.int32,
                layout=gl.SliceLayout(dim=1, parent=MFMA_LAYOUT),
            )
            size_n = start_n + OFFS_QK_N[None, :]
            qk_mask_partial = size_n < boundary_m[:, None]
            qk_mask = gl.where(bound_cond, qk_mask_partial, qk_mask)

        # Compute qk^T
        dot_q = gl.convert_layout(q, dot_left_layout)
        dot_k = gl.convert_layout(k, dot_right_layout)
        qk += gl.amd.cdna4.mfma(dot_q, dot_k, qk_zeros)
        if HAS_PE:
            dot_q_pe = gl.convert_layout(q_pe, dot_left_layout)
            dot_k_pe = gl.convert_layout(k_pe, dot_right_layout)
            qk += gl.amd.cdna4.mfma(dot_q_pe, dot_k_pe, qk_zeros)

        if IS_CAUSAL:
            causal_boundary = start_n + OFFS_QK_N_CAUSAL
            causal_mask = OFFS_QK_M[:, None] >= causal_boundary[None, :]
            qk_mask = qk_mask & causal_mask

        # Perform causal and padding masking
        qk = gl.where(qk_mask, qk, float("-inf"))

        # Get max scores so far
        m_ij = gl.maximum(m_i, gl.max(qk, 1))
        m_ij_scaled = m_ij * SM_SCALE * RCP_LN2

        # Scale and subtract max
        q_shifted = qk * SM_SCALE * RCP_LN2 - m_ij_scaled[:, None]

        # Compute scaled QK and softmax probabilities
        p = gl.exp2(q_shifted)
        l_ij = gl.sum(p, 1)

        # Update the output accumulator
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff_scaled = m_i * SM_SCALE * RCP_LN2 - m_ij_scaled
        alpha = gl.exp2(m_diff_scaled)
        acc = acc * alpha[:, None]
        v = _load_fn(
            v_base_ptr,
            OFFS_V_N,
            OFFS_V_D,
            stride_vn,
            stride_vk,
            None if not MASK_STEPS else seqlen_k - start_n,
            None if BLOCK_DMODEL == BLOCK_DMODEL_POW2 else BLOCK_DMODEL,
        )
        dot_p = gl.convert_layout(p, dot_left_layout).to(v.type.element_ty)
        dot_v = gl.convert_layout(v, dot_right_layout)
        acc += gl.amd.cdna4.mfma(dot_p, dot_v, pv_zeros)

        # Update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Update pointers for k and v
        k_base_ptr += BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_base_ptr += BLOCK_N * stride_kn
        v_base_ptr += BLOCK_N * stride_vn

    return acc, l_i, m_i


@gluon.jit
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    descale_q_ptr: torch.Tensor,
    descale_k_ptr: torch.Tensor,
    descale_v_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    alibi_slopes_ptr: torch.Tensor,
    s_dmask_ptr: torch.Tensor,
    dropout_mask_ptr: torch.Tensor,
    softmax_lse_ptr: torch.Tensor,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    stride_alibi_z_in,
    stride_alibi_h_in,
    stride_sd_z_in,
    stride_sd_h_in,
    stride_sd_m_in,
    stride_sd_n_in,
    stride_lse_z_in,
    stride_lse_h_in,
    stride_lse_m_in,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    SEQLEN_Q,
    SEQLEN_K,
    IS_CAUSAL: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,  # it's zero or a power of 2
    RETURN_SCORES: gl.constexpr,
    ENABLE_DROPOUT: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    NUM_XCD: gl.constexpr,
    USE_INT64_STRIDES: gl.constexpr,
):
    if IS_FP8:
        raise NotImplementedError("FP8 is not supported in Gluon MHA yet.")
    if ENABLE_DROPOUT:
        raise NotImplementedError("Dropout is not supported in Gluon MHA yet.")
    if NUM_Q_HEADS > NUM_K_HEADS:
        raise NotImplementedError(
            "Grouped query and multi-query attention not supported yet in Gluon MHA."
        )
    if RETURN_SCORES:
        raise NotImplementedError(
            "Returning attention scores is not supported yet in Gluon MHA."
        )

    NUM_BLOCKS = gl.cdiv(SEQLEN_Q, BLOCK_M)
    wid = gl.program_id(
        0
    )  # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)

    off_q_head = wid % NUM_Q_HEADS  # across num q heads
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH  # across batch size

    """
    NOTE:
    The following layouts come from the compiler-generated IR for the basic Triton kernel.
        
    This one is for loading/storing 1D tensors of 32-bit dtype (like lse and associated offsets/masks):
    #blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
    
    This one is for loading from Q tensor:
    #blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
    
    This one is for loading from K and V tensors:
    #blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
    
    This one is for storing the output tensor:
    #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], 
                           lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], 
                           warp = [[32, 0], [64, 0]], 
                           block = []}>
    
    This one is for matmul operations:
    #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
    
    This is shared memory for the Q block:
    #shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
    
    This is shared memory for the K block:
    #shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
    
    This is shared memory for the V block:
    #shared2 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
    """

    # TODO: These layouts need to eventually be generalized across input data sizes
    # Right now, ththeyese are hardcoded for dtype=float16
    blocked_lse: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[64],
        warps_per_cta=[gl.num_warps()],
        order=[0],
    )  # analogous to #blocked in the comment above
    blocked_md: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[gl.num_warps(), 1],
        order=[1, 0],
    )  # analogous to #blocked1 in the comment above
    blocked_dn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[8, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, gl.num_warps()],
        order=[0, 1],
    )  # analogous to #blocked2 in the comment above
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[gl.num_warps(), 1],
    )  # analogous to #mma in the comment above
    out_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]],
        lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]],
        warp_bases=[[32, 0], [64, 0]],
        block_bases=[],
        shape=[BLOCK_M, BLOCK_DMODEL_POW2],
    )  # analogous to #linear in the comment above
    shared_q: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[1, 0]
    )  # analogous to #shared in the comment above
    shared_k: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=2, max_phase=8, order=[0, 1]
    )  # analogous to #shared1 in the comment above
    shared_v: gl.constexpr = gl.SwizzledSharedLayout(
        vec=4, per_phase=2, max_phase=8, order=[1, 0]
    )  # analogous to #shared2 in the comment above

    # Create offsets across sequence length
    offs_q_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=blocked_md)
    )
    offs_lse_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=blocked_lse)
    offs_k_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=blocked_dn))
    offs_qk_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=mfma_layout)
    )
    offs_qk_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=0, parent=mfma_layout))
    offs_v_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(dim=1, parent=mfma_layout))
    offs_v_d = gl.arange(
        0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(dim=0, parent=mfma_layout)
    )
    offs_out_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=out_layout)
    )

    # Create offsets across dimension size
    offs_q_d = gl.arange(
        0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(dim=0, parent=blocked_md)
    )
    offs_k_d = gl.arange(
        0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(dim=1, parent=blocked_dn)
    )
    offs_out_d = gl.arange(
        0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(dim=0, parent=out_layout)
    )
    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0
    if HAS_PE:
        offs_qpe_d = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(dim=0, parent=blocked_md)
        )  # across dimension size of q for positional encoding
        offs_kpe_d = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(dim=1, parent=blocked_dn)
        )  # across dimension size of q for positional encoding
    else:
        offs_qpe_d = None
        offs_kpe_d = None

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(gl.int64), that does not work
    # because strides are gl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: gl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of gl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    if USE_INT64_STRIDES:
        stride_qz = tl.cast(stride_qz_in, gl.int64)
        stride_qh = tl.cast(stride_qh_in, gl.int64)
        stride_qm = tl.cast(stride_qm_in, gl.int64)
        stride_qk = tl.cast(stride_qk_in, gl.int64)
        stride_kz = tl.cast(stride_kz_in, gl.int64)
        stride_kh = tl.cast(stride_kh_in, gl.int64)
        stride_kn = tl.cast(stride_kn_in, gl.int64)
        stride_kk = tl.cast(stride_kk_in, gl.int64)
        stride_vz = tl.cast(stride_vz_in, gl.int64)
        stride_vh = tl.cast(stride_vh_in, gl.int64)
        stride_vn = tl.cast(stride_vn_in, gl.int64)
        stride_vk = tl.cast(stride_vk_in, gl.int64)

        stride_oz = tl.cast(stride_oz_in, gl.int64)
        stride_oh = tl.cast(stride_oh_in, gl.int64)
        stride_om = tl.cast(stride_om_in, gl.int64)
        stride_on = tl.cast(stride_on_in, gl.int64)

        stride_lse_z = tl.cast(stride_lse_z_in, gl.int64)
        stride_lse_h = tl.cast(stride_lse_h_in, gl.int64)
        stride_lse_m = tl.cast(stride_lse_m_in, gl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in

    gl.assume(stride_qz_in >= 0)
    gl.assume(stride_qh_in >= 0)
    gl.assume(stride_qm_in >= 0)
    gl.assume(stride_qk_in >= 0)
    gl.assume(stride_kz_in >= 0)
    gl.assume(stride_kh_in >= 0)
    gl.assume(stride_kn_in >= 0)
    gl.assume(stride_kk_in >= 0)
    gl.assume(stride_vz_in >= 0)
    gl.assume(stride_vh_in >= 0)
    gl.assume(stride_vn_in >= 0)
    gl.assume(stride_vk_in >= 0)

    gl.assume(stride_lse_z_in >= 0)
    gl.assume(stride_lse_h_in >= 0)
    gl.assume(stride_lse_m_in >= 0)

    # Get sequence lengths
    if VARLEN:
        cu_seqlens_q_start = gl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = gl.load(cu_seqlens_q + off_z + 1)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = gl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = gl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    # Create offsets to return log-sum-exp
    if softmax_lse_ptr is not None:
        lse_offs = (
            off_z * stride_lse_z
            + off_q_head * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + offs_lse_m * stride_lse_m
        )
        lse_offs = tl.cast(lse_offs, gl.int32)
        lse_mask = offs_lse_m < seqlen_q
    else:
        lse_offs = None
        lse_mask = None

    # Create offsets to write output
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    out_offs = (
        off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + offs_out_m[:, None] * stride_om
        + offs_out_d[None, :] * stride_on
    )
    out_offs = tl.cast(out_offs, gl.int32)
    out_mask = gl.full(
        [BLOCK_M, BLOCK_DMODEL_POW2], True, dtype=gl.int1, layout=out_layout
    )
    if overflow_size > 0:
        out_mask = out_mask & (offs_out_m[:, None] < seqlen_q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_out_d[None, :] < BLOCK_DMODEL)

    # Compute number of blocks along seqlen_k
    n_blocks = gl.cdiv(seqlen_k, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.

        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = gl.cdiv(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            acc = gl.zeros(
                [BLOCK_M, BLOCK_DMODEL_POW2],
                dtype=out_ptr.type.element_ty,
                layout=out_layout,
            )
            gl.amd.cdna4.buffer_store(
                stored_value=acc, ptr=out_ptr, offsets=out_offs, mask=out_mask
            )

            if softmax_lse_ptr is not None:
                lse = gl.full(
                    [BLOCK_M], value=0.0, dtype=gl.float32, layout=blocked_lse
                )
                gl.amd.cdna4.buffer_store(
                    stored_value=lse,
                    ptr=softmax_lse_ptr,
                    offsets=lse_offs,
                    mask=lse_mask,
                )

            return

    off_k_head = off_q_head  # for now, no grouped q/k attention

    # Create q offsets
    q_offs = (
        off_z * stride_qz
        + off_q_head * stride_qh
        + cu_seqlens_q_start * stride_qm
        + offs_q_m[:, None] * stride_qm
        + offs_q_d[None, :] * stride_qk
    )
    q_offs = tl.cast(q_offs, gl.int32)
    if HAS_PE:
        q_pe_offs = (
            off_z * stride_qz
            + off_q_head * stride_qh
            + cu_seqlens_q_start * stride_qm
            + offs_q_m[:, None] * stride_qm
            + offs_qpe_d[None, :] * stride_qk
        )
        q_pe_offs = tl.cast(q_pe_offs, gl.int32)
    else:
        q_pe_offs = None

    # Create initial k offsets
    k_offs = off_z * stride_kz + off_k_head * stride_kh + cu_seqlens_k_start * stride_kn
    k_base_ptr = k_ptr + k_offs
    if HAS_PE:
        k_pe_offs = (
            off_z * stride_kz + off_k_head * stride_kh + cu_seqlens_k_start * stride_kn
        )
        k_pe_base_ptr = k_base_ptr + k_pe_offs
    else:
        k_pe_base_ptr = None

    # Create new base pointer for v
    v_offs = off_z * stride_vz + off_k_head * stride_vh + cu_seqlens_k_start * stride_vn
    v_base_ptr = v_ptr + v_offs

    # Load q block
    m_i = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(dim=1, parent=mfma_layout),
    )
    l_i = gl.full(
        [BLOCK_M],
        1.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(dim=1, parent=mfma_layout),
    )
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=mfma_layout)
    q_mask = offs_q_m[:, None] < seqlen_q
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        q_mask = q_mask & (offs_q_d[None, :] < BLOCK_DMODEL)
    q = gl.amd.cdna4.buffer_load(ptr=q_ptr, offsets=q_offs, mask=q_mask, other=0.0)
    if HAS_PE:
        q_pe = gl.amd.cdna4.buffer_load(
            ptr=q_ptr, offsets=q_pe_offs, mask=q_mask, other=0.0
        )
    else:
        q_pe = None

    # Any extra tokens to pad on the K side?
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    # If CAUSAL, then determine masked_blocks and full blocks
    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k

    # If IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks

    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_min = 0
        block_max = n_full_blocks * BLOCK_N

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_base_ptr,
            k_pe_base_ptr,
            v_base_ptr,
            stride_kk,
            stride_kn,
            stride_vn,
            stride_vk,
            seqlen_k,
            block_min,
            block_max,
            0,
            offs_k_n,
            offs_qk_m,
            offs_qk_n,
            0,
            offs_v_n,
            offs_k_d,
            0 if not HAS_PE else offs_kpe_d,
            offs_v_d,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            mfma_layout,
            shared_k,
            shared_v,
            sm_scale,
            False,
            MASK_STEPS=False,
            ENABLE_PIPELINING=True,
            NUM_STAGES=2,
        )

    # Compute for masked blocks
    if masked_blocks > 0:
        block_min = n_full_blocks * BLOCK_N
        block_max = n_blocks * BLOCK_N

        # If a causal mask is needed, compute the offset
        if IS_CAUSAL:
            offs_qk_n_causal = offs_qk_n + (seqlen_q - seqlen_k)
        else:
            offs_qk_n_causal = 0

        # Update pointers if we computed for any full blocks
        k_base_ptr += n_full_blocks * BLOCK_N * stride_kn
        if HAS_PE:
            k_pe_base_ptr += n_full_blocks * BLOCK_N * stride_kn
        v_base_ptr += n_full_blocks * BLOCK_N * stride_vn

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_base_ptr,
            k_pe_base_ptr,
            v_base_ptr,
            stride_kk,
            stride_kn,
            stride_vn,
            stride_vk,
            seqlen_k,
            block_min,
            block_max,
            n_extra_tokens,
            offs_k_n,
            offs_qk_m,
            offs_qk_n,
            offs_qk_n_causal,
            offs_v_n,
            offs_k_d,
            offs_kpe_d,
            offs_v_d,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            mfma_layout,
            shared_k,
            shared_v,
            sm_scale,
            IS_CAUSAL,
            MASK_STEPS=True,
            ENABLE_PIPELINING=False,
            NUM_STAGES=2,
        )

    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.

    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = gl.full(
                (BLOCK_DMODEL_POW2,),
                causal_start_idx,
                dtype=gl.int32,
                layout=gl.SliceLayout(dim=0, parent=mfma_layout),
            )
            mask_m_offsets = start_m_idx + gl.arange(
                0, BLOCK_M, layout=gl.SliceLayout(dim=1, parent=mfma_layout)
            )
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = gl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # Write the log-sum-exp back
    if softmax_lse_ptr is not None:
        RCP_LN2: gl.constexpr = 1.4426950408889634
        LN2: gl.constexpr = 0.6931471824645996
        # Compute log-sum-exp in base-2 units
        mi_base2 = m_i * RCP_LN2 * sm_scale
        softmax_lse = mi_base2 + gl.log2(l_i)
        # Convert back to natural units
        softmax_lse *= LN2

        softmax_lse = gl.convert_layout(softmax_lse, blocked_lse)

        if IS_CAUSAL:
            # Zero out nans caused by -infs when doing causal
            lse_causal_mask = (start_m_idx + offs_lse_m) < causal_start_idx
            softmax_lse = gl.where(lse_causal_mask, 0.0, softmax_lse)

        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        # This is only true for the last M block. For others, overflow_size will be -ve
        if overflow_size > 0:
            boundary = BLOCK_M - overflow_size
            lse_mask = offs_lse_m < boundary
            gl.amd.cdna4.buffer_store(
                stored_value=softmax_lse,
                ptr=softmax_lse_ptr,
                offsets=lse_offs,
                mask=lse_mask,
            )
        else:
            gl.amd.cdna4.buffer_store(
                stored_value=softmax_lse, ptr=softmax_lse_ptr, offsets=lse_offs
            )

    # Write back O
    op = gl.convert_layout(acc.to(out_ptr.dtype.element_ty), out_layout)
    gl.amd.cdna4.buffer_store(
        stored_value=op, ptr=out_ptr, offsets=out_offs, mask=out_mask
    )


@functools.lru_cache(maxsize=1024)
def _get_config(
    enable_dropout: bool,
    dtype: torch.dtype,
    has_pe: bool = False,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    if has_pe and "pe" in _get_config._config_dict["default"]["fwd"]:
        return _get_config._config_dict["default"]["fwd"]["pe"]
    elif enable_dropout or dtype == torch.float32:
        return _get_config._config_dict["default"]["fwd"]["dropout_or_fp32"]
    else:
        return _get_config._config_dict["default"]["fwd"]["default"]
