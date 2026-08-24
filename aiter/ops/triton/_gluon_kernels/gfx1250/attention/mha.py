# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.mha_kernel_utils import (
    _compute_fp8_scaling_factors,
)


@gluon.jit
def _cdiv_fn(x, y):
    # Mirrors _triton_kernels/attention/mha.py:_cdiv_fn. Both operands are
    # non-negative here, so the truncating `//` is a true ceiling divide.
    return (x + y - 1) // y


@gluon.jit
def _attn_fwd_inner(
    # Mirrors the parameter order of _triton_kernels/attention/mha.py::_attn_fwd_inner.
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_desc,
    k_pe_desc,
    v_desc,
    stride_sd_m,
    stride_sd_n,
    seqlen_k,
    seqlen_q,
    dropout_p,
    sd_ptr,
    dm_ptr,
    philox_seed,
    philox_base,
    block_min,
    block_max,
    alibi_slope,
    descale_q,
    descale_k,
    descale_v,
    offs_m,
    offs_n,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    sm_scale,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    ENABLE_DROPOUT: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    # Gluon-only: the hand-rolled LDS pipeline and the layouts Triton infers.
    k_smem,
    k_pe_smem,
    v_smem,
    buf,
    LAST: gl.constexpr,
    QK_LAYOUT: gl.constexpr,
    PV_LAYOUT: gl.constexpr,
    q_layout: gl.constexpr,
    k_layout: gl.constexpr,
    p_layout: gl.constexpr,
    v_layout: gl.constexpr,
):
    RCP_LN2 = gl.constexpr(1.4426950408889634)
    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0
    # Async ops issued per block. All wait counts derive from this so the
    # bookkeeping stays in one place (getting it wrong reads a half-written
    # LDS buffer silently, with no fault).
    ASYNC_OPS: gl.constexpr = 3 if HAS_PE else 2

    for start_n in range(block_min, block_max, BLOCK_N):
        if LAST:
            gl.amd.gfx1250.tdm.async_wait(0)
        else:
            gl.amd.gfx1250.tdm.async_wait(ASYNC_OPS - 1)
        k = k_smem.index(buf).permute([1, 0]).load(layout=k_layout)

        if HAS_PE:
            if not LAST:
                gl.amd.gfx1250.tdm.async_wait(ASYNC_OPS - 2)
            k_pe = k_pe_smem.index(buf).permute([1, 0]).load(layout=k_layout)

        if not LAST:
            gl.amd.gfx1250.tdm.async_load(
                k_desc, [start_n + BLOCK_N, 0], k_smem.index(1 - buf)
            )
            if HAS_PE:
                gl.amd.gfx1250.tdm.async_load(
                    k_pe_desc, [start_n + BLOCK_N, 0], k_pe_smem.index(1 - buf)
                )
            gl.amd.gfx1250.tdm.async_load(
                v_desc, [start_n + BLOCK_N, 0], v_smem.index(1 - buf)
            )

        qk_scale = sm_scale * RCP_LN2
        # -- compute qk ----
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=QK_LAYOUT)
        if HAS_PE:
            # PE contribution accumulates first, matching the Triton kernel's
            # dot order so the two agree to fp32 rounding.
            qk = gl.amd.gfx1250.wmma(q_pe, k_pe, qk)
        qk = gl.amd.gfx1250.wmma(q, k, qk)
        # Relabel into the PV layout so the softmax, the mask and `acc` all
        # share one layout. Only the instruction K dim differs between the two,
        # which does not affect the [M, N] accumulator distribution -- hence
        # trivial. The assert is the canary if that ever stops being true.
        qk = gl.convert_layout(qk, PV_LAYOUT, assert_trivial=True)
        if IS_FP8:
            qk = qk * (qk_scale * descale_q * descale_k)
        else:
            qk = qk * qk_scale

        # ---- masking ----
        need_mask: gl.constexpr = MASK_STEPS or IS_CAUSAL or (SLIDING_WINDOW > 0)
        if need_mask:
            mask = gl.full([BLOCK_M, BLOCK_N], 1, gl.int1, layout=PV_LAYOUT)
            if MASK_STEPS:
                # Partial trailing tile. Applied unconditionally in this phase:
                # it is a no-op for full tiles and avoids a branch.
                mask = mask & ((start_n + offs_n)[None, :] < seqlen_k)
            if IS_CAUSAL:
                # Bottom-right aligned diagonal when seqlen_q != seqlen_k.
                causal_boundary = start_n + offs_n + (seqlen_q - seqlen_k)
                mask = mask & (offs_m[:, None] >= causal_boundary[None, :])
            if SLIDING_WINDOW > 0:
                q_adj = offs_m + seqlen_k - seqlen_q
                mask = mask & (
                    (start_n + offs_n)[None, :] >= (q_adj[:, None] - SLIDING_WINDOW)
                )
            qk = gl.where(mask, qk, float("-inf"))

        if alibi_slope is not None:
            # Distance from the diagonal, which stays bottom-right aligned when
            # seqlen_q != seqlen_k (same alignment as the causal mask). The
            # diagonal itself gets no penalty; it grows with distance.
            # ``offs_m`` is already global; ``offs_n`` is tile-local.
            relative_pos = (
                offs_m[:, None] + seqlen_k - seqlen_q - (start_n + offs_n)[None, :]
            )
            alibi_block = -1 * alibi_slope * gl.abs(relative_pos)
            # Added after the -inf masking: a finite bias leaves -inf intact.
            qk += alibi_block * RCP_LN2

        # get max scores so far
        m_ij = gl.maximum(m_i, gl.max(qk, axis=1))

        # Compute scaled QK and softmax probabilities
        p = gl.exp2(qk - m_ij[:, None])

        if SLIDING_WINDOW > 0:
            # A fully out-of-window row leaves qk all -inf and m_ij == -inf, so
            # exp2(-inf - -inf) = NaN. Zero those elements.
            p = gl.where(mask, p, 0.0)

        # CAVEAT: l_ij must be summed before dropout is applied -- the LSE is
        # the pre-dropout normalizer.
        l_ij = gl.sum(p, axis=1)

        if ENABLE_DROPOUT or RETURN_SCORES:
            # Element (m, n) of the score matrix for this (batch, q head).
            sd_offs = (
                offs_m[:, None] * stride_sd_m
                + (start_n + offs_n)[None, :] * stride_sd_n
            )
            p_mask = (offs_m < seqlen_q)[:, None] & (
                (start_n + offs_n) < seqlen_k
            )[None, :]

        if ENABLE_DROPOUT:
            # Philox is counter-based: the draw for element (m, n) depends only
            # on the seed and that element's offset, so the mask is
            # reproducible without storing it. `tl.rand` is a @triton.jit
            # device helper -- it inlines into gluon and needs no gl.* port.
            #
            # NOTE: this diverges from the Triton kernel. Triton advances its
            # philox pointer for the skipped blocks but not across the
            # full/masked phase boundary, so its RNG stream repeats over the
            # trailing blocks. Deriving the offset from `start_n` avoids that,
            # at the cost of the two kernels drawing different masks.
            keep = (
                tl.rand(philox_seed, philox_base + sd_offs) > dropout_p
            ) # TODO: use tl.randint for better performance
            gl.store(dm_ptr + sd_offs, keep.to(gl.float32), mask=p_mask)
            # Dropped scores are returned negated so the caller can recover the
            # mask from the sign.
            gl.store(sd_ptr + sd_offs, gl.where(keep, p, -p), mask=p_mask)
            p = gl.where(keep, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            gl.store(sd_ptr + sd_offs, p, mask=p_mask)

        alpha = gl.exp2(m_i - m_ij)
        if SLIDING_WINDOW > 0:
            # When m_i == m_ij == -inf, exp2(-inf - (-inf)) = NaN. alpha should be 1.0
            # (no rescaling needed since max didn't change).
            alpha = gl.where(m_i == m_ij, 1.0, alpha)
        acc = acc * alpha[:, None]
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        if not LAST:
            gl.amd.gfx1250.tdm.async_wait(ASYNC_OPS)
        v = v_smem.index(buf).load(layout=v_layout)

        if IS_FP8:
            # p is rescaled into the fp8 range per block, so the dot cannot
            # accumulate into `acc` directly -- it runs into a fresh zero
            # accumulator and the result is descaled before being added. Same
            # shape as the Triton kernel's `acc += dot(...) * dp * dv`.
            scale_p, descale_p = _compute_fp8_scaling_factors(p, FP8_MAX)
            p_fp8 = gl.convert_layout((p * scale_p).to(v.dtype), p_layout)
            acc += (
                gl.amd.gfx1250.wmma(
                    p_fp8,
                    v,
                    gl.zeros(
                        [BLOCK_M, BLOCK_DMODEL_POW2],
                        dtype=gl.float32,
                        layout=PV_LAYOUT,
                    ),
                )
                * descale_p
                * descale_v
            )
        else:
            p = gl.convert_layout(p.to(v.dtype, fp_downcast_rounding="rtz"), p_layout)
            acc = gl.amd.gfx1250.wmma(p, v, acc)

        buf = 1 - buf

    return acc, l_i, m_i, buf


_attn_fwd_repr = make_kernel_repr(
    # Deliberately distinct from the Triton kernel's "_attn_fwd" so the two are
    # distinguishable in profiles, cache dumps and logs -- the dispatch between
    # them is otherwise silent.
    "_attn_fwd_gluon",
    [
        "IS_CAUSAL",
        "NUM_Q_HEADS",
        "NUM_K_HEADS",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_DMODEL",
        "VARLEN",
        "ENABLE_SINK",
        "SLIDING_WINDOW",
        "num_warps",
    ],
)


@gluon.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    descale_q_ptr,
    descale_k_ptr,
    descale_v_ptr,
    out_ptr,
    alibi_slopes_ptr,
    s_dmask_ptr,
    dropout_mask_ptr,
    softmax_lse_ptr,
    sink_ptr,
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
    PRELOAD_V: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    ENABLE_DROPOUT: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    NUM_XCD: gl.constexpr,
    SWIZZLE: gl.constexpr,
    USE_INT64_STRIDES: gl.constexpr,
    ENABLE_SINK: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    HEAD_STRIDE_ALIGNED_8: gl.constexpr = False,
):
    # ---- unsupported-feature gates ----
    gl.static_assert(SWIZZLE == "default", "gluon MHA: only the default swizzle")

    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    # calculate offsets
    wid = gl.program_id(0)
    # NUM_XCD=1 on gfx1250
    off_q_head = wid % NUM_Q_HEADS
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    GRP_SZ: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // GRP_SZ

    HAS_PE: gl.constexpr = BLOCK_DMODEL_PE > 0

    # Head dim not a power of two: the tiles are BLOCK_DMODEL_POW2 wide and the
    # trailing columns carry no data. Q is zero-masked at the load, K/V are
    # zero-filled by the TDM descriptor bound (CDNA5 descriptors pad with zero),
    # and the O store drops them again.
    PADDED_HEAD: gl.constexpr = BLOCK_DMODEL != BLOCK_DMODEL_POW2

    NUM_WARPS: gl.constexpr = gl.num_warps()
    WARP_SIZE: gl.constexpr = 32  # gfx1250 is wave32
    gl.static_assert(
        BLOCK_M // 16 >= NUM_WARPS, "BLOCK_M must give each warp >=1 WMMA M-tile"
    )

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(tl.int64), that does not work
    # because strides are tl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: tl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of tl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    if USE_INT64_STRIDES:
        stride_qz = gl.cast(stride_qz_in, gl.int64)
        stride_qh = gl.cast(stride_qh_in, gl.int64)
        stride_qm = gl.cast(stride_qm_in, gl.int64)
        stride_qk = gl.cast(stride_qk_in, gl.int64)
        stride_kz = gl.cast(stride_kz_in, gl.int64)
        stride_kh = gl.cast(stride_kh_in, gl.int64)
        stride_kn = gl.cast(stride_kn_in, gl.int64)
        stride_vz = gl.cast(stride_vz_in, gl.int64)
        stride_vh = gl.cast(stride_vh_in, gl.int64)
        stride_vn = gl.cast(stride_vn_in, gl.int64)
        if IS_FP8:
            stride_descale_q_z = gl.cast(stride_descale_q_z_in, gl.int64)
            stride_descale_k_z = gl.cast(stride_descale_k_z_in, gl.int64)
            stride_descale_v_z = gl.cast(stride_descale_v_z_in, gl.int64)
        stride_oz = gl.cast(stride_oz_in, gl.int64)
        stride_oh = gl.cast(stride_oh_in, gl.int64)
        stride_om = gl.cast(stride_om_in, gl.int64)
        stride_on = gl.cast(stride_on_in, gl.int64)
        stride_alibi_z = gl.cast(stride_alibi_z_in, gl.int64)
        stride_alibi_h = gl.cast(stride_alibi_h_in, gl.int64)
        stride_sd_z = gl.cast(stride_sd_z_in, gl.int64)
        stride_sd_h = gl.cast(stride_sd_h_in, gl.int64)
        stride_sd_m = gl.cast(stride_sd_m_in, gl.int64)
        stride_sd_n = gl.cast(stride_sd_n_in, gl.int64)
        stride_lse_z = gl.cast(stride_lse_z_in, gl.int64)
        stride_lse_h = gl.cast(stride_lse_h_in, gl.int64)
        stride_lse_m = gl.cast(stride_lse_m_in, gl.int64)
        # NOTE: philox offset is needed in dropout pointer calculations
        philox_offset_base = gl.cast(philox_offset_base_in, gl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in
        stride_alibi_z = stride_alibi_z_in
        stride_alibi_h = stride_alibi_h_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        philox_offset_base = philox_offset_base_in

    # ---- layouts: ONE mma layout, M-split, shared by QK and PV ----
    if NUM_WARPS == 1:
        warp_bases: gl.constexpr = []
    elif NUM_WARPS == 2:
        warp_bases: gl.constexpr = [[1, 0]]
    elif NUM_WARPS == 4:
        warp_bases: gl.constexpr = [[1, 0], [2, 0]]
    else:
        warp_bases: gl.constexpr = [[1, 0], [2, 0], [4, 0]]

    # Two MMA layouts, one per dot. FP8 WMMA reduces a deeper K per instruction,
    # and the two dots reduce over different extents -- QK over the head dim, PV
    # over BLOCK_N -- so their instruction shapes can differ. In bf16 both are
    # [16, 16, 32] and the pair collapses to a single layout, which is why the
    # QK->PV relabel in the inner loop is a no-op there (and asserted as such).
    K_WIDTH: gl.constexpr = 16 if IS_FP8 else 8
    QK_INSTR_K: gl.constexpr = (128 if BLOCK_DMODEL_POW2 > 64 else 64) if IS_FP8 else 32
    PV_INSTR_K: gl.constexpr = (128 if BLOCK_N > 64 else 64) if IS_FP8 else 32
    QK_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, QK_INSTR_K],
        warp_bases=warp_bases,
    )
    PV_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, PV_INSTR_K],
        warp_bases=warp_bases,
    )
    q_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=QK_LAYOUT, k_width=K_WIDTH
    )
    k_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=QK_LAYOUT, k_width=K_WIDTH
    )
    p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=PV_LAYOUT, k_width=K_WIDTH
    )
    v_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=PV_LAYOUT, k_width=K_WIDTH
    )

    k_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_POW2, 8]], [BLOCK_N, BLOCK_DMODEL_POW2], [1, 0]
    )
    v_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_POW2, 16]], [BLOCK_N, BLOCK_DMODEL_POW2], [1, 0]
    )
    if HAS_PE:
        # Same transposed-read padding as K, sized to the PE width.
        k_pe_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_DMODEL_PE, 8]], [BLOCK_N, BLOCK_DMODEL_PE], [1, 0]
        )

    ELEMENT_SIZE: gl.constexpr = 8 if IS_FP8 else 16  # bits per element
    MAX_LOAD: gl.constexpr = 128
    SIZE_PER_THREAD: gl.constexpr = MAX_LOAD // ELEMENT_SIZE
    HEAD_SIZE_DIV: gl.constexpr = BLOCK_DMODEL_POW2 // SIZE_PER_THREAD
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, SIZE_PER_THREAD],
        threads_per_warp=[WARP_SIZE // HEAD_SIZE_DIV, HEAD_SIZE_DIV],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

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

    offs_m_mma = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, PV_LAYOUT))
    offs_n_mma = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, PV_LAYOUT))
    offs_d_mma = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, PV_LAYOUT))
    offs_m = start_m * BLOCK_M + offs_m_mma

    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

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
        n_blocks_seqlen = _cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = gl.minimum(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            offs_out = (
                off_z * stride_oz
                + off_q_head * stride_oh
                + cu_seqlens_q_start * stride_om
                + offs_m[:, None] * stride_om
                + offs_d_mma[None, :] * stride_on
            )
            acc = gl.zeros(
                [BLOCK_M, BLOCK_DMODEL_POW2],
                dtype=out_ptr.type.element_ty,
                layout=PV_LAYOUT,
            )
            out_mask = (offs_m < seqlen_q)[:, None]
            if PADDED_HEAD:
                out_mask = out_mask & (offs_d_mma < BLOCK_DMODEL)[None, :]
            gl.store(out_ptr + offs_out, acc, mask=out_mask)

            if softmax_lse_ptr is not None:
                offs_lse = (
                    off_z * stride_lse_z
                    + off_q_head * stride_lse_h
                    + cu_seqlens_q_start * stride_lse_m
                    + offs_m * stride_lse_m
                )
                lse_mask = offs_m < SEQLEN_Q
                lse = gl.full(
                    [BLOCK_M], 0.0, gl.float32, gl.SliceLayout(1, PV_LAYOUT)
                )
                gl.store(softmax_lse_ptr + offs_lse, lse, mask=lse_mask)

            return

    # ---- Q: loaded once, straight into registers ----
    offs_m_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_q))
    offs_d_q = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, blocked_q))
    q_base = (
        off_z * stride_qz
        + off_q_head * stride_qh
        + (cu_seqlens_q_start + start_m * BLOCK_M) * stride_qm
    )
    q_mask = ((start_m * BLOCK_M + offs_m_q) < seqlen_q)[:, None]
    if PADDED_HEAD:
        q_mask = q_mask & (offs_d_q < BLOCK_DMODEL)[None, :]
    q = gl.load(
        q_ptr + q_base + offs_m_q[:, None] * stride_qm + offs_d_q[None, :] * stride_qk,
        mask=q_mask,
        other=0.0,
    )
    q = gl.convert_layout(q, q_layout)

    if HAS_PE:
        # The PE columns sit immediately after the BLOCK_DMODEL data columns of
        # the same Q tensor, so this is one more strided load, not a new tensor.
        PE_SIZE_DIV: gl.constexpr = BLOCK_DMODEL_PE // SIZE_PER_THREAD
        blocked_q_pe: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, SIZE_PER_THREAD],
            threads_per_warp=[WARP_SIZE // PE_SIZE_DIV, PE_SIZE_DIV],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        )
        offs_m_q_pe = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked_q_pe))
        offs_pe = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(0, blocked_q_pe)
        )
        q_pe = gl.load(
            q_ptr
            + q_base
            + offs_m_q_pe[:, None] * stride_qm
            + offs_pe[None, :] * stride_qk,
            mask=((start_m * BLOCK_M + offs_m_q_pe) < seqlen_q)[:, None],
            other=0.0,
        )
        q_pe = gl.convert_layout(q_pe, q_layout)
    else:
        q_pe = None

    # ---- K/V: TDM descriptors bounded by this sequence's length ----
    # The per-sequence `shape` bound removes the OOB row mask from the loads;
    # trailing rows of a partial tile are zero-filled and masked out of the
    # scores anyway. Requires the head-dim stride to be 1 (checked by caller).
    k_base = off_z * stride_kz + off_k_head * stride_kh + cu_seqlens_k_start * stride_kn
    v_base = off_z * stride_vz + off_k_head * stride_vh + cu_seqlens_k_start * stride_vn
    # `shape` is the real extent in both axes: rows past seqlen_k and columns
    # past BLOCK_DMODEL are zero-filled by the descriptor, which is what makes
    # the OOB row mask and the padded-head column mask unnecessary here.
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + k_base,
        shape=[seqlen_k, BLOCK_DMODEL],
        strides=[stride_kn, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL_POW2],
        layout=k_shared,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + v_base,
        shape=[seqlen_k, BLOCK_DMODEL],
        strides=[stride_vn, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL_POW2],
        layout=v_shared,
    )
    k_smem = gl.allocate_shared_memory(
        k_ptr.type.element_ty, [2, BLOCK_N, BLOCK_DMODEL_POW2], k_shared
    )
    v_smem = gl.allocate_shared_memory(
        v_ptr.type.element_ty, [2, BLOCK_N, BLOCK_DMODEL_POW2], v_shared
    )
    if HAS_PE:
        k_pe_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=k_ptr + k_base + BLOCK_DMODEL,
            shape=[seqlen_k, BLOCK_DMODEL_PE],
            strides=[stride_kn, 1],
            block_shape=[BLOCK_N, BLOCK_DMODEL_PE],
            layout=k_pe_shared,
        )
        k_pe_smem = gl.allocate_shared_memory(
            k_ptr.type.element_ty, [2, BLOCK_N, BLOCK_DMODEL_PE], k_pe_shared
        )
    else:
        k_pe_desc = None
        k_pe_smem = None

    # ---- accumulators ----
    if ENABLE_SINK:
        RCP_LN2 = gl.constexpr(1.4426950408889634)
        m_i_value = gl.load(sink_ptr + off_q_head).to(gl.float32) * RCP_LN2
    else:
        m_i_value = float("-inf")
    m_i = gl.full(
        [BLOCK_M], m_i_value, dtype=gl.float32, layout=gl.SliceLayout(1, PV_LAYOUT)
    )
    l_i = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, PV_LAYOUT)
    )
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=PV_LAYOUT)

    # ---- fp8 per-tensor descale factors ----
    if IS_FP8:
        descale_q = gl.load(descale_q_ptr + off_z * stride_descale_q_z + off_q_head)
        descale_k = gl.load(descale_k_ptr + off_z * stride_descale_k_z + off_k_head)
        descale_v = gl.load(descale_v_ptr + off_z * stride_descale_v_z + off_k_head)
    else:
        descale_q = 1.0
        descale_k = 1.0
        descale_v = 1.0

    # ---- s_dmask / dropout: base offset for this (batch, q head) ----
    # s_dmask is allocated whenever scores are returned *or* dropout is on, so
    # the gate is the pointer itself rather than RETURN_SCORES (mirrors the
    # Triton kernel).
    sd_base = off_z * stride_sd_z + off_q_head * stride_sd_h
    if s_dmask_ptr is not None:
        sd_ptr = s_dmask_ptr + sd_base
    else:
        sd_ptr = None
    if ENABLE_DROPOUT:
        dm_ptr = dropout_mask_ptr + sd_base
        philox_base = philox_offset_base + sd_base
    else:
        dm_ptr = None
        philox_base = 0

    # ---- alibi: one scalar slope per (batch, q head) ----
    if alibi_slopes_ptr is not None:
        alibi_slope = gl.load(
            alibi_slopes_ptr + off_z * stride_alibi_z + off_q_head * stride_alibi_h
        )
    else:
        alibi_slope = None

    # ---- full / masked block split (mirrors _attn_fwd) ----
    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = (not padded_block_k) and (seqlen_q % BLOCK_M == 0)

    skipped_blocks = 0
    if SLIDING_WINDOW > 0:
        # Skip K blocks that are fully left of the earliest key position
        # reachable by this Q block. The first retained block can still be
        # partially outside the window, so we keep the per-element mask below.
        window_start_n = start_m * BLOCK_M + seqlen_k - seqlen_q - SLIDING_WINDOW
        skipped_blocks = gl.maximum(window_start_n, 0) // BLOCK_N
        skipped_blocks = gl.minimum(skipped_blocks, gl.maximum(n_blocks - 1, 0))
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k.to(gl.int32)
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    visible_blocks = n_blocks - skipped_blocks
    masked_blocks = gl.minimum(masked_blocks, visible_blocks)
    n_full_blocks = visible_blocks - masked_blocks

    block_min = skipped_blocks * BLOCK_N
    block_last = (n_blocks - 1) * BLOCK_N
    block_full_end = gl.minimum(block_min + n_full_blocks * BLOCK_N, block_last)
    block_max = n_blocks * BLOCK_N

    # Prime the pipeline. Issue order [K, K_PE, V] must match the loop body's.
    gl.amd.gfx1250.tdm.async_load(k_desc, [block_min, 0], k_smem.index(0))
    if HAS_PE:
        gl.amd.gfx1250.tdm.async_load(k_pe_desc, [block_min, 0], k_pe_smem.index(0))
    gl.amd.gfx1250.tdm.async_load(v_desc, [block_min, 0], v_smem.index(0))
    buf: gl.int32 = 0

    # n_blocks == 0 only for a zero-length KV sequence (varlen). Fall through
    # with acc=0, l_i=1, m_i=-inf so the epilogue emits out=0 / lse=-inf, which
    # is what the Triton kernel produces for that case.
    if n_blocks > 0:
        if block_full_end > block_min:
            acc, l_i, m_i, buf = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                q_pe,
                k_desc,
                k_pe_desc,
                v_desc,
                stride_sd_m,
                stride_sd_n,
                seqlen_k,
                seqlen_q,
                dropout_p,
                sd_ptr,
                dm_ptr,
                philox_seed,
                philox_base,
                block_min,
                block_full_end,
                alibi_slope,
                descale_q,
                descale_k,
                descale_v,
                offs_m,
                offs_n_mma,
                BLOCK_M,
                BLOCK_N,
                BLOCK_DMODEL_POW2,
                BLOCK_DMODEL_PE,
                sm_scale,
                IS_CAUSAL=False,
                MASK_STEPS=False,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                RETURN_SCORES=RETURN_SCORES,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                SLIDING_WINDOW=SLIDING_WINDOW,
                k_smem=k_smem,
                k_pe_smem=k_pe_smem,
                v_smem=v_smem,
                buf=buf,
                LAST=False,
                QK_LAYOUT=QK_LAYOUT,
                PV_LAYOUT=PV_LAYOUT,
                q_layout=q_layout,
                k_layout=k_layout,
                p_layout=p_layout,
                v_layout=v_layout,
            )

        if block_last > block_full_end:
            acc, l_i, m_i, buf = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                q_pe,
                k_desc,
                k_pe_desc,
                v_desc,
                stride_sd_m,
                stride_sd_n,
                seqlen_k,
                seqlen_q,
                dropout_p,
                sd_ptr,
                dm_ptr,
                philox_seed,
                philox_base,
                block_full_end,
                block_last,
                alibi_slope,
                descale_q,
                descale_k,
                descale_v,
                offs_m,
                offs_n_mma,
                BLOCK_M,
                BLOCK_N,
                BLOCK_DMODEL_POW2,
                BLOCK_DMODEL_PE,
                sm_scale,
                IS_CAUSAL=IS_CAUSAL,
                MASK_STEPS=True,
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                RETURN_SCORES=RETURN_SCORES,
                IS_FP8=IS_FP8,
                FP8_MAX=FP8_MAX,
                SLIDING_WINDOW=SLIDING_WINDOW,
                k_smem=k_smem,
                k_pe_smem=k_pe_smem,
                v_smem=v_smem,
                buf=buf,
                LAST=False,
                QK_LAYOUT=QK_LAYOUT,
                PV_LAYOUT=PV_LAYOUT,
                q_layout=q_layout,
                k_layout=k_layout,
                p_layout=p_layout,
                v_layout=v_layout,
            )

        acc, l_i, m_i, buf = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_pe,
            k_desc,
            k_pe_desc,
            v_desc,
            stride_sd_m,
            stride_sd_n,
            seqlen_k,
            seqlen_q,
            dropout_p,
            sd_ptr,
            dm_ptr,
            philox_seed,
            philox_base,
            block_last,
            block_max,
            alibi_slope,
            descale_q,
            descale_k,
            descale_v,
            offs_m,
            offs_n_mma,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_POW2,
            BLOCK_DMODEL_PE,
            sm_scale,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
            ENABLE_DROPOUT=ENABLE_DROPOUT,
            RETURN_SCORES=RETURN_SCORES,
            IS_FP8=IS_FP8,
            FP8_MAX=FP8_MAX,
            SLIDING_WINDOW=SLIDING_WINDOW,
            k_smem=k_smem,
            k_pe_smem=k_pe_smem,
            v_smem=v_smem,
            buf=buf,
            LAST=True,
            QK_LAYOUT=QK_LAYOUT,
            PV_LAYOUT=PV_LAYOUT,
            q_layout=q_layout,
            k_layout=k_layout,
            p_layout=p_layout,
            v_layout=v_layout,
        )

    gl.amd.gfx1250.tdm.async_wait(0)

    # ---- epilogue ----
    # Reciprocal on l_i (BLOCK_M) rather than acc (BLOCK_M x D) so the compiler's
    # Newton-Raphson runs on the small tensor.
    acc = acc * (1.0 / l_i[:, None])
    if ENABLE_DROPOUT:
        # The surviving scores carry the full row mass, so scale back up. LSE is
        # deliberately left alone -- it is the pre-dropout normalizer.
        acc = acc * (1.0 / (1.0 - dropout_p))

    start_m_idx = start_m * BLOCK_M
    end_m_idx = (start_m + 1) * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    if IS_CAUSAL:
        # Rows entirely above the diagonal softmax'd a row of -inf -> NaN.
        # They should be zero.
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            acc = gl.where(offs_m[:, None] >= causal_start_idx, acc, 0.0)

    if softmax_lse_ptr is not None:
        _LN2 = gl.constexpr(0.6931471824645996)
        # compute log-sum-exp in base 2 units and convert back to natural units
        softmax_lse = (m_i + gl.log2(l_i)) * _LN2
        if IS_CAUSAL:
            softmax_lse = gl.where(offs_m < causal_start_idx, 0.0, softmax_lse)
        offs_lse = (
            off_z * stride_lse_z
            + off_q_head * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
            + offs_m * stride_lse_m
        )
        gl.store(softmax_lse_ptr + offs_lse, softmax_lse, mask=offs_m < seqlen_q)

    # write back O
    offs_out = (
        off_z * stride_oz
        + off_q_head * stride_oh
        + cu_seqlens_q_start * stride_om
        + offs_m[:, None] * stride_om
        + offs_d_mma[None, :] * stride_on
    )
    out_mask = (offs_m < seqlen_q)[:, None]
    if PADDED_HEAD:
        out_mask = out_mask & (offs_d_mma < BLOCK_DMODEL)[None, :]
    gl.store(
        out_ptr + offs_out,
        acc.to(out_ptr.type.element_ty),
        mask=out_mask,
    )
