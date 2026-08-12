# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _cdiv_fn(x, y):
    # Mirrors _triton_kernels/attention/mha.py:_cdiv_fn. Both operands are
    # non-negative here, so the truncating `//` is a true ceiling divide.
    return (x + y - 1) // y


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc,
    v_desc,
    k_smem,
    v_smem,
    buf,
    blk_lo,
    blk_hi,
    n_blocks,
    seqlen_q,
    seqlen_k,
    offs_m,
    offs_n,
    sm_scale,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    a_layout: gl.constexpr,
    b_layout: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    LAST: gl.constexpr = False,
):
    """One phase of the KV loop over block indices ``[blk_lo, blk_hi)``.

    Steady state (``LAST=False``): on entry to every iteration exactly two TDM
    ops are in flight, in issue order ``[K[j], V[j]]``:
        async_wait(1) -> K[j] landed (V[j] still in flight)
        issue K[j+1], V[j+1]
        async_wait(2) -> V[j] landed (K[j+1], V[j+1] still in flight)
    The caller issues K[blk_lo]/V[blk_lo]; the invariant carries across both
    steady-state phase calls with no drain/refill at the boundary. These phases
    never cover the final block, so ``j + 1`` is always a real block index.

    Final block (``LAST=True``, one iteration): drained with ``async_wait(0)``
    and issues no prefetch. Peeling it is not just an optimization -- relying on
    the steady-state counts for the last tile races (observed: V read before its
    load landed, giving a zero ``acc`` for a few workgroups) because the ops
    issued in that iteration are the ones being counted. Same reason
    ``unified_attention_2d.attention_loop_standard`` peels its last tile.
    """
    RCP_LN2 = gl.constexpr(1.4426950408889634)

    for j in range(blk_lo, blk_hi):
        start_n = j * BLOCK_N

        if LAST:
            gl.amd.gfx1250.tdm.async_wait(0)
        else:
            gl.amd.gfx1250.tdm.async_wait(1)
        k = k_smem.index(buf).permute([1, 0]).load(layout=b_layout)

        if not LAST:
            gl.amd.gfx1250.tdm.async_load(
                k_desc, [(j + 1) * BLOCK_N, 0], k_smem.index(1 - buf)
            )
            gl.amd.gfx1250.tdm.async_load(
                v_desc, [(j + 1) * BLOCK_N, 0], v_smem.index(1 - buf)
            )

        qk_scale = sm_scale * RCP_LN2
        # -- compute qk ----
        qk = gl.amd.gfx1250.wmma(
            q, k, gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=WMMA_LAYOUT)
        )
        qk = qk * qk_scale

        # ---- masking ----
        need_mask: gl.constexpr = MASK_STEPS or IS_CAUSAL or (SLIDING_WINDOW > 0)
        if need_mask:
            mask = gl.full([BLOCK_M, BLOCK_N], 1, gl.int1, layout=WMMA_LAYOUT)
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
        # get max scores so far
        m_ij = gl.maximum(m_i, gl.max(qk, axis=1))

        # Compute scaled QK and softmax probabilities
        p = gl.exp2(qk - m_ij[:, None])

        if SLIDING_WINDOW > 0:
            # A fully out-of-window row leaves qk all -inf and m_ij == -inf, so
            # exp2(-inf - -inf) = NaN. Zero those elements.
            p = gl.where(mask, p, 0.0)

        l_ij = gl.sum(p, axis=1)
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
            gl.amd.gfx1250.tdm.async_wait(2)
        v = v_smem.index(buf).load(layout=b_layout)

        p = gl.convert_layout(p.to(v.dtype, fp_downcast_rounding="rtz"), a_layout)
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
    # ---- unsupported-feature gates (the caller must route these to Triton) ----
    gl.static_assert(not ENABLE_DROPOUT, "gluon MHA: dropout not supported")
    gl.static_assert(not RETURN_SCORES, "gluon MHA: return_scores not supported")
    gl.static_assert(not IS_FP8, "gluon MHA: fp8 not supported")
    gl.static_assert(BLOCK_DMODEL_PE == 0, "gluon MHA: PE head dim not supported")
    gl.static_assert(
        BLOCK_DMODEL == BLOCK_DMODEL_POW2, "gluon MHA: padded head not supported"
    )
    gl.static_assert(SWIZZLE == "default", "gluon MHA: only the default swizzle")

    NUM_WARPS: gl.constexpr = gl.num_warps()
    WARP_SIZE: gl.constexpr = 32  # gfx1250 is wave32
    gl.static_assert(
        BLOCK_M // 16 >= NUM_WARPS, "BLOCK_M must give each warp >=1 WMMA M-tile"
    )

    # NOTE: mirrors the Triton kernel's USE_INT64_STRIDES block
    # (_triton_kernels/attention/mha.py:412). Strides arrive as i32 -- Triton
    # specializes a Python int arg to i32 whenever the value fits -- so a bare
    # `index * stride` overflows on tensors with more than 2**31 elements.
    # Widening the stride once here promotes every offset expression it feeds,
    # so the indices themselves stay untouched, exactly as in the Triton kernel.
    #
    # Declaring the parameters int64 in the signature instead would lose the arg
    # specialization AxisInfo needs to prove contiguity, and with it the
    # vectorized loads -- hence the cast inside the kernel.
    #
    # Use gl.cast, not `.to()`: Triton specializes a stride whose value is 0 or 1
    # (contiguous head dim, varlen batch stride) into a plain Python int, which
    # has no `.to()` method.
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
        stride_oz = gl.cast(stride_oz_in, gl.int64)
        stride_oh = gl.cast(stride_oh_in, gl.int64)
        stride_om = gl.cast(stride_om_in, gl.int64)
        stride_on = gl.cast(stride_on_in, gl.int64)
        stride_lse_z = gl.cast(stride_lse_z_in, gl.int64)
        stride_lse_h = gl.cast(stride_lse_h_in, gl.int64)
        stride_lse_m = gl.cast(stride_lse_m_in, gl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qh = stride_qh_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
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

    # ---- layouts: ONE mma layout, M-split, shared by QK and PV ----
    if NUM_WARPS == 1:
        warp_bases: gl.constexpr = []
    elif NUM_WARPS == 2:
        warp_bases: gl.constexpr = [[1, 0]]
    elif NUM_WARPS == 4:
        warp_bases: gl.constexpr = [[1, 0], [2, 0]]
    else:
        warp_bases: gl.constexpr = [[1, 0], [2, 0], [4, 0]]

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        instr_shape=[16, 16, 32],
        warp_bases=warp_bases,
    )
    K_WIDTH: gl.constexpr = 8  # bf16
    a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=K_WIDTH
    )
    b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=K_WIDTH
    )

    k_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_POW2, 8]], [BLOCK_N, BLOCK_DMODEL_POW2], [1, 0]
    )
    v_shared: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_POW2, 16]], [BLOCK_N, BLOCK_DMODEL_POW2], [1, 0]
    )

    ELEMENT_SIZE: gl.constexpr = 16  # bf16; fp8 would be 8
    MAX_LOAD: gl.constexpr = 128
    SIZE_PER_THREAD: gl.constexpr = MAX_LOAD // ELEMENT_SIZE
    HEAD_SIZE_DIV: gl.constexpr = BLOCK_DMODEL_POW2 // SIZE_PER_THREAD
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, SIZE_PER_THREAD],
        threads_per_warp=[WARP_SIZE // HEAD_SIZE_DIV, HEAD_SIZE_DIV],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    # ---- program decomposition (SWIZZLE == "default"; NUM_XCD is 1) ----
    wid = gl.program_id(0)
    NUM_BLOCKS = _cdiv_fn(SEQLEN_Q, BLOCK_M)
    off_q_head = wid % NUM_Q_HEADS
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH

    GRP_SZ: gl.constexpr = NUM_Q_HEADS // NUM_K_HEADS
    off_k_head = off_q_head // GRP_SZ

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

    offs_m_mma = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_n_mma = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
    offs_d_mma = gl.arange(0, BLOCK_DMODEL_POW2, layout=gl.SliceLayout(0, WMMA_LAYOUT))
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
                layout=WMMA_LAYOUT,
            )
            out_mask = (offs_m < seqlen_q)[:, None]
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
                    [BLOCK_M], 0.0, gl.float32, gl.SliceLayout(1, WMMA_LAYOUT)
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
    q = gl.load(
        q_ptr + q_base + offs_m_q[:, None] * stride_qm + offs_d_q[None, :] * stride_qk,
        mask=((start_m * BLOCK_M + offs_m_q) < seqlen_q)[:, None],
        other=0.0,
    )
    q = gl.convert_layout(q, a_layout)

    # ---- K/V: TDM descriptors bounded by this sequence's length ----
    # The per-sequence `shape` bound removes the OOB row mask from the loads;
    # trailing rows of a partial tile are zero-filled and masked out of the
    # scores anyway. Requires the head-dim stride to be 1 (checked by caller).
    k_base = off_z * stride_kz + off_k_head * stride_kh + cu_seqlens_k_start * stride_kn
    v_base = off_z * stride_vz + off_k_head * stride_vh + cu_seqlens_k_start * stride_vn
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr + k_base,
        shape=[seqlen_k, BLOCK_DMODEL_POW2],
        strides=[stride_kn, 1],
        block_shape=[BLOCK_N, BLOCK_DMODEL_POW2],
        layout=k_shared,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr + v_base,
        shape=[seqlen_k, BLOCK_DMODEL_POW2],
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

    # ---- accumulators ----
    if ENABLE_SINK:
        RCP_LN2 = gl.constexpr(1.4426950408889634)
        m_i_value = gl.load(sink_ptr + off_q_head).to(gl.float32) * RCP_LN2
    else:
        m_i_value = float("-inf")
    m_i = gl.full(
        [BLOCK_M], m_i_value, dtype=gl.float32, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    l_i = gl.full(
        [BLOCK_M], 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=gl.float32, layout=WMMA_LAYOUT)

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

    # Prime the pipeline
    gl.amd.gfx1250.tdm.async_load(
        k_desc, [skipped_blocks * BLOCK_N, 0], k_smem.index(0)
    )
    gl.amd.gfx1250.tdm.async_load(
        v_desc, [skipped_blocks * BLOCK_N, 0], v_smem.index(0)
    )
    buf: gl.int32 = 0

    # The final block is always peeled (and always masked -- a conservative mask
    # on a block that needs none is a no-op). The two steady-state phases are
    # clamped to stop before it so their prefetch of j+1 is always in range.
    last_blk = n_blocks - 1
    end_full = gl.minimum(skipped_blocks + n_full_blocks, last_blk)

    # n_blocks == 0 only for a zero-length KV sequence (varlen). Fall through
    # with acc=0, l_i=1, m_i=-inf so the epilogue emits out=0 / lse=-inf, which
    # is what the Triton kernel produces for that case.
    if n_blocks > 0:
        if end_full > skipped_blocks:
            acc, l_i, m_i, buf = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                k_desc,
                v_desc,
                k_smem,
                v_smem,
                buf,
                skipped_blocks,
                end_full,
                n_blocks,
                seqlen_q,
                seqlen_k,
                offs_m,
                offs_n_mma,
                sm_scale,
                BLOCK_M,
                BLOCK_N,
                WMMA_LAYOUT,
                a_layout,
                b_layout,
                IS_CAUSAL=False,
                MASK_STEPS=False,
                SLIDING_WINDOW=SLIDING_WINDOW,
            )

        if last_blk > end_full:
            acc, l_i, m_i, buf = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                k_desc,
                v_desc,
                k_smem,
                v_smem,
                buf,
                end_full,
                last_blk,
                n_blocks,
                seqlen_q,
                seqlen_k,
                offs_m,
                offs_n_mma,
                sm_scale,
                BLOCK_M,
                BLOCK_N,
                WMMA_LAYOUT,
                a_layout,
                b_layout,
                IS_CAUSAL=IS_CAUSAL,
                MASK_STEPS=True,
                SLIDING_WINDOW=SLIDING_WINDOW,
            )

        acc, l_i, m_i, buf = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            v_desc,
            k_smem,
            v_smem,
            buf,
            last_blk,
            n_blocks,
            n_blocks,
            seqlen_q,
            seqlen_k,
            offs_m,
            offs_n_mma,
            sm_scale,
            BLOCK_M,
            BLOCK_N,
            WMMA_LAYOUT,
            a_layout,
            b_layout,
            IS_CAUSAL=IS_CAUSAL,
            MASK_STEPS=True,
            SLIDING_WINDOW=SLIDING_WINDOW,
            LAST=True,
        )

    gl.amd.gfx1250.tdm.async_wait(0)

    # ---- epilogue ----
    # Reciprocal on l_i (BLOCK_M) rather than acc (BLOCK_M x D) so the compiler's
    # Newton-Raphson runs on the small tensor.
    acc = acc * (1.0 / l_i[:, None])

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
    gl.store(
        out_ptr + offs_out,
        acc.to(out_ptr.type.element_ty),
        mask=(offs_m < seqlen_q)[:, None],
    )
