# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd

KW = 8
KW_BIG = 8


@functools.cache
def build_layouts(nw, kw=KW, kw_big=KW_BIG):
    """The layout set, derived from one decision: the state stays in registers.

    ``instr_shape[0:2] = [16, 16]`` with ``transposed=False`` is what makes an
    MFMA accumulator a legal B operand, so the state can be the accumulator of
    ``dot(kr^T, U)`` and the B operand of ``dot(kd, h)`` without a round trip.
    ``kw`` names the dots that contract over C and ``kw_big`` the one that
    contracts over K; the two accumulator distributions are the same, since an
    accumulator's layout is set by M and N and not by K.
    """
    mma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4 * kw],
        transposed=False,
        warps_per_cta=[1, nw],
    )
    mma_b = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4 * kw_big],
        transposed=False,
        warps_per_cta=[1, nw],
    )
    return {
        "MMA": mma,
        "A_OP": gl.DotOperandLayout(0, mma, kw),
        "B_OP": gl.DotOperandLayout(1, mma, kw),
        "MMA_B": mma_b,
        "A_OP_B": gl.DotOperandLayout(0, mma_b, kw_big),
        "B_OP_B": gl.DotOperandLayout(1, mma_b, kw_big),
        # Sources of the global -> LDS copies: 128 (or 32) bits per thread and
        # no lane replication, as buffer_load_to_shared requires.
        "BLK": gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        "BLK_CC": gl.BlockedLayout([1, 2], [4, 16], [nw, 1], [1, 0]),
        # gt and beta: one element per thread.
        "BLK_1D": gl.BlockedLayout([1], [64], [nw], [0]),
        "SH_WS": gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
        "SH_PLAIN": gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        "SH_1D": gl.SwizzledSharedLayout(1, 1, 1, [0]),
    }


@gluon.jit
def _recur(
    h,
    kd_a,
    inv_a,
    kr_a,
    gt,
    beta,
    v,
    m_c,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    INV_TY: gl.constexpr,
    HAS_V: gl.constexpr,
):
    h_op = gl.convert_layout(h.to(gl.bfloat16), B_OP_B)
    tmp = gl.convert_layout(
        gl.amd.cdna4.mfma(kd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
    )
    if HAS_V:
        u = (v - tmp) * beta[:, None]
    else:
        u = (-tmp) * beta[:, None]
    # Tail rows must stay zero: U feeds the state update, which sums over all C
    # rows regardless of how many are real.
    u = gl.where(m_c[:, None], u, 0.0)
    big_u = gl.amd.cdna4.mfma(
        inv_a, gl.convert_layout(u.to(INV_TY), B_OP), gl.zeros([C, BW], gl.float32, MMA)
    )
    h_next = gl.amd.cdna4.mfma(
        kr_a, gl.convert_layout(big_u.to(gl.bfloat16), B_OP), h * gt[:, None]
    )
    return h_next, big_u, h_op


@gluon.jit
def _issue_chunk(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_beta,
    ws_inv_mqk,
    v_input,
    ws_qd,
    s_kd,
    s_kr,
    s_inv,
    s_v,
    s_gt,
    s_beta,
    s_qd,
    s_mqk,
    ws_idx,
    t0,
    tok_end,
    ws_off,
    cc_off,
    v_off,
    o_c_v,
    o_t,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    WITH_OUTPUT: gl.constexpr,
):
    """Start fetching one chunk into LDS by async copy, as one commit group.

    ``gt`` and ``beta`` (K1 applied its sigmoid) have fewer elements than
    threads, and the async copy cannot lower lane replication, so they are
    copied one element per thread (``o_t``) with the excess threads masked.
    Tail rows of ``v`` are masked here; ``beta`` of a tail row is harmless
    because the recurrence zeroes those rows of U anyway. ``WITH_OUTPUT``
    (pass C) also fetches the output projection's ``qd`` and ``Mqk`` tiles.
    """
    tb = t0 * H
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s_gt, ws_gt + ws_idx * K, o_t, mask=o_t < K
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s_beta, ws_beta + ws_idx * C, o_t, mask=o_t < C
    )
    ck = ws_idx * (C * K)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s_kd, ws_kd + ck, ws_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s_kr, ws_kr + ck, ws_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s_inv, ws_inv_mqk + ws_idx * (2 * C * C), cc_off
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s_v,
        v_input + tb * V,
        v_off,
        mask=((t0 + o_c_v) < tok_end)[:, None],
        other=0.0,
    )
    if WITH_OUTPUT:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(s_qd, ws_qd + ck, ws_off)
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            s_mqk, ws_inv_mqk + ws_idx * (2 * C * C) + C * C, cc_off
        )
    gl.amd.cdna4.async_copy.commit_group()


_k2_ab_fused_repr = make_kernel_repr(
    "k2_ab_fused_gluon",
    ["C", "K", "V", "BW", "NUM_WARPS"],
)


@gluon.jit(repr=_k2_ab_fused_repr)
def k2_ab_fused_gluon(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_beta,
    ws_inv_mqk,
    v_input,
    h_out_b,
    h_out_a,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    A_OP_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    BLK: gl.constexpr,
    BLK_CC: gl.constexpr,
    BLK_1D: gl.constexpr,
    SH_WS: gl.constexpr,
    SH_PLAIN: gl.constexpr,
    SH_1D: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Both pass-A recurrences in one launch, sharing every operand load.

    The two differ only in seeding -- zero with the real v against the identity
    with v = 0 -- so they read the same chunk and can share it. Register
    residency is what makes the sharing worth having: the two chains are
    independent, so the scheduler interleaves them, and that is what covers the
    serial dependence each one has on its own. Requires K == V, since the two
    states are then the same width and one program covers a column block of
    both.

    Chunk j+1's operands are fetched while chunk j computes: its tiles by async
    global -> LDS copy into the other half of a double buffer, so the prefetch
    costs no registers (a register prefetch halves occupancy). Without it the
    kernel spent a third or more of its wave-cycles waiting on loads issued and
    consumed in the same iteration.
    """
    # Keeps a (segment, head)'s V blocks on one XCD, so their re-reads of the
    # chunk workspace share an L2. See the Triton K2 for the full reasoning.
    n_w = gl.num_programs(0)
    pid = remap_xcd(
        gl.program_id(1) * n_w + gl.program_id(0), n_w * gl.num_programs(1), NUM_XCDS
    )
    i_w = (pid % n_w).to(gl.int64)
    i_sh = (pid // n_w).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)

    # The launch's warp count, as a constexpr so it can name the kernel.
    gl.static_assert(NUM_WARPS == gl.num_warps())
    BLK_V: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // (BW // 8), BW // 8], [NUM_WARPS, 1], [1, 0]
    )
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_r_cc = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_CC))
    o_c_cc = gl.arange(0, C, layout=gl.SliceLayout(0, BLK_CC))
    o_c_v = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_V))
    o_w_v = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, BLK_V))
    NT: gl.constexpr = 64 * NUM_WARPS
    gl.static_assert(NT >= K and NT >= C)
    o_t = gl.arange(0, NT, layout=BLK_1D)
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    ws_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    cc_off = (o_r_cc[:, None] * C + o_c_cc[None, :]).to(gl.int32)
    v_off = (i_h * V + o_c_v[:, None] * (H * V) + o_w_v[None, :]).to(gl.int32)

    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    s_kd = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_kr = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_inv = gl.allocate_shared_memory(inv_ty, [2, C, C], SH_PLAIN)
    s_v = gl.allocate_shared_memory(v_input.dtype.element_ty, [2, C, BW], SH_PLAIN)
    # gt / beta live in LDS until their use, so the per-thread copies of the MMA
    # slice layout are only materialized there.
    s_gt = gl.allocate_shared_memory(gl.float32, [2, NT], SH_1D)
    s_beta = gl.allocate_shared_memory(gl.float32, [2, NT], SH_1D)

    h_b = gl.zeros([K, BW], gl.float32, MMA)
    h_a = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)

    ws0 = i_h * TOTAL_TILES + chunk_base
    if n_chunks > 0:
        _issue_chunk(ws_kd, ws_kr, ws_gt, ws_beta, ws_inv_mqk, v_input, None,
                     s_kd.index(0), s_kr.index(0), s_inv.index(0), s_v.index(0),
                     s_gt.index(0), s_beta.index(0), None, None, ws0, tok_base,
                     tok_end, ws_off, cc_off, v_off, o_c_v, o_t, H, K, V, C,
                     False)  # fmt: skip

    for j in range(n_chunks):
        cur = j % 2
        # Chunk j has landed in half `cur` (every thread's copies, once past the
        # barrier the wait carries), and every thread is done reading the other
        # half, which held chunk j-1, before it is refilled.
        gl.amd.cdna4.async_copy.wait_group(0)
        if j + 1 < n_chunks:
            _issue_chunk(ws_kd, ws_kr, ws_gt, ws_beta, ws_inv_mqk, v_input, None,
                         s_kd.index(1 - cur), s_kr.index(1 - cur),
                         s_inv.index(1 - cur), s_v.index(1 - cur),
                         s_gt.index(1 - cur), s_beta.index(1 - cur), None, None,
                         ws0 + j + 1, tok_base + (j + 1) * C, tok_end, ws_off,
                         cc_off, v_off, o_c_v, o_t, H, K, V, C, False)  # fmt: skip

        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kd.index(cur), A_OP_B)
        # ws_kr is [C, K] in memory; the permuted view is the kr^T A operand.
        kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_kr.index(cur).permute((1, 0)), A_OP
        )
        inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_inv.index(cur), A_OP)
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(
            gl.float32
        )
        gt = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_gt.index(cur).slice(0, K), gl.SliceLayout(1, MMA)
        )
        beta = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_beta.index(cur).slice(0, C), gl.SliceLayout(1, MMA)
        )
        m_c = (tok_base + j * C + o_c_m) < tok_end
        # Written one after the other so the scheduler has two independent MFMA
        # chains to interleave, which is what covers the serial dependence.
        h_b, _u, _h = _recur(h_b, kd_a, inv_a, kr_a, gt, beta, b_v, m_c, C, BW,
                             MMA, B_OP, MMA_B, B_OP_B, inv_ty, True)  # fmt: skip
        h_a, _u, _h = _recur(h_a, kd_a, inv_a, kr_a, gt, beta, b_v, m_c, C, BW,
                             MMA, B_OP, MMA_B, B_OP_B, inv_ty, False)  # fmt: skip

    s_base = (i_seg * H + i_h) * (K * V)
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)
    gl.amd.cdna4.buffer_store(h_b.to(h_out_b.dtype.element_ty), h_out_b + s_base, s_off)
    gl.amd.cdna4.buffer_store(h_a.to(h_out_a.dtype.element_ty), h_out_a + s_base, s_off)


_k2_c_repr = make_kernel_repr(
    "k2_c_gluon",
    [
        "C",
        "K",
        "V",
        "BW",
        "HAS_H_IN",
        "STORE_FINAL",
        "STATE_V_FIRST",
        "PAGED_CACHE",
        "PAGED_H_IN",
        "NUM_WARPS",
    ],
)


@gluon.jit(repr=_k2_c_repr)
def k2_c_gluon(
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_beta,
    ws_inv_mqk,
    v_input,
    out,
    h_in,
    final_state,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    seg_seq,
    seg_is_last,
    state_cache,
    state_indices,
    has_initial_state,
    cache_stride,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    B_OP: gl.constexpr,
    MMA_B: gl.constexpr,
    A_OP_B: gl.constexpr,
    B_OP_B: gl.constexpr,
    BLK: gl.constexpr,
    BLK_CC: gl.constexpr,
    BLK_1D: gl.constexpr,
    SH_WS: gl.constexpr,
    SH_PLAIN: gl.constexpr,
    SH_1D: gl.constexpr,
    HAS_H_IN: gl.constexpr,
    STORE_FINAL: gl.constexpr,
    STATE_V_FIRST: gl.constexpr,
    PAGED_CACHE: gl.constexpr,
    PAGED_H_IN: gl.constexpr,
    CM_OUT: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Pass C (the output pass): the recurrence of one segment from its incoming state.

    Same contract as ``_flash_kda_segment_kernel`` with ``COMPUTE_OUTPUT`` and
    ``HAS_V`` and without ``STORE_H_OUT``, paged ``state_cache`` included. Chunk
    j+1's tiles are prefetched into LDS while chunk j computes, as in the fused
    pass A. The output is stored straight from the MFMA layout: converting it
    for a wider store costs an LDS round trip and enough scratch to cost
    occupancy at BW=128.
    """
    n_w = gl.num_programs(0)
    pid = remap_xcd(
        gl.program_id(1) * n_w + gl.program_id(0), n_w * gl.num_programs(1), NUM_XCDS
    )
    i_w = (pid % n_w).to(gl.int64)
    i_sh = (pid // n_w).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)
    # Read up front, not after the loop: past the output stores the compiler
    # must assume these words may have changed and loads them per lane, which
    # leaves the final-state store a waterfall loop over a non-uniform pointer.
    # When STORE_FINAL is false the final-state pointers are null and these
    # must not be read.
    if STORE_FINAL or PAGED_CACHE:
        i_n = gl.load(seg_seq + i_seg).to(gl.int64)
    if STORE_FINAL:
        is_last = gl.load(seg_is_last + i_seg)
    if PAGED_CACHE:
        slot = gl.load(state_indices + i_n).to(gl.int64)

    # The launch's warp count, as a constexpr so it can name the kernel.
    gl.static_assert(NUM_WARPS == gl.num_warps())
    BLK_V: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // (BW // 8), BW // 8], [NUM_WARPS, 1], [1, 0]
    )
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_r_cc = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_CC))
    o_c_cc = gl.arange(0, C, layout=gl.SliceLayout(0, BLK_CC))
    o_c_v = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_V))
    o_w_v = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, BLK_V))
    NT: gl.constexpr = 64 * NUM_WARPS
    gl.static_assert(NT >= K and NT >= C)
    o_t = gl.arange(0, NT, layout=BLK_1D)
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    ws_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    cc_off = (o_r_cc[:, None] * C + o_c_cc[None, :]).to(gl.int32)
    v_off = (i_h * V + o_c_v[:, None] * (H * V) + o_w_v[None, :]).to(gl.int32)
    o_off = (i_h * V + o_c_m[:, None] * (H * V) + o_w_m[None, :]).to(gl.int32)
    # K-first [K, V] and V-first [V, K] offsets of this program's state block.
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)
    s_off_vk = (o_w_m[None, :] * K + o_k_m[:, None]).to(gl.int32)

    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    out_ty: gl.constexpr = out.dtype.element_ty
    s_kd = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_qd = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_kr = gl.allocate_shared_memory(gl.bfloat16, [2, C, K], SH_WS)
    s_inv = gl.allocate_shared_memory(inv_ty, [2, C, C], SH_PLAIN)
    s_mqk = gl.allocate_shared_memory(inv_ty, [2, C, C], SH_PLAIN)
    s_v = gl.allocate_shared_memory(v_input.dtype.element_ty, [2, C, BW], SH_PLAIN)
    s_gt = gl.allocate_shared_memory(gl.float32, [2, NT], SH_1D)
    s_beta = gl.allocate_shared_memory(gl.float32, [2, NT], SH_1D)

    if PAGED_H_IN:
        # A zero-start sequence's slot may hold a previous owner's state: mask
        # the load so it is never read, not read and then discarded.
        take = gl.load(has_initial_state + i_n).to(gl.int1)
        m_h = (o_k_m[:, None] < K) & (o_w_m[None, :] < V) & take
        h = gl.amd.cdna4.buffer_load(
            ptr=state_cache + slot * cache_stride + i_h * (V * K),
            offsets=s_off_vk,
            mask=m_h,
            other=0.0,
        ).to(gl.float32)
    elif HAS_H_IN:
        h = gl.amd.cdna4.buffer_load(
            ptr=h_in + (i_seg * H + i_h) * (K * V), offsets=s_off
        ).to(gl.float32)
    else:
        h = gl.zeros([K, BW], gl.float32, MMA)

    ws0 = i_h * TOTAL_TILES + chunk_base
    if n_chunks > 0:
        _issue_chunk(ws_kd, ws_kr, ws_gt, ws_beta, ws_inv_mqk, v_input, ws_qd,
                     s_kd.index(0), s_kr.index(0), s_inv.index(0), s_v.index(0),
                     s_gt.index(0), s_beta.index(0), s_qd.index(0),
                     s_mqk.index(0), ws0, tok_base, tok_end, ws_off, cc_off,
                     v_off, o_c_v, o_t, H, K, V, C, True)  # fmt: skip

    for j in range(n_chunks):
        cur = j % 2
        # As in pass A: chunk j is in half `cur`, and half 1 - cur is free.
        gl.amd.cdna4.async_copy.wait_group(0)
        if j + 1 < n_chunks:
            _issue_chunk(ws_kd, ws_kr, ws_gt, ws_beta, ws_inv_mqk, v_input, ws_qd,
                         s_kd.index(1 - cur), s_kr.index(1 - cur),
                         s_inv.index(1 - cur), s_v.index(1 - cur),
                         s_gt.index(1 - cur), s_beta.index(1 - cur),
                         s_qd.index(1 - cur), s_mqk.index(1 - cur), ws0 + j + 1,
                         tok_base + (j + 1) * C, tok_end, ws_off, cc_off, v_off,
                         o_c_v, o_t, H, K, V, C, True)  # fmt: skip

        t0 = tok_base + j * C
        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kd.index(cur), A_OP_B)
        inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_inv.index(cur), A_OP)
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(
            gl.float32
        )
        gt = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_gt.index(cur).slice(0, K), gl.SliceLayout(1, MMA)
        )
        beta = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_beta.index(cur).slice(0, C), gl.SliceLayout(1, MMA)
        )
        m_c = (t0 + o_c_m) < tok_end

        # _recur's steps, split around the output projection: updating the state
        # after it keeps the next state and kr out of registers meanwhile
        # (1.1-1.2x on this kernel).
        h_op = gl.convert_layout(h.to(gl.bfloat16), B_OP_B)
        tmp = gl.convert_layout(
            gl.amd.cdna4.mfma(kd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
        )
        u = gl.where(m_c[:, None], (b_v - tmp) * beta[:, None], 0.0)
        big_u = gl.amd.cdna4.mfma(
            inv_a,
            gl.convert_layout(u.to(inv_ty), B_OP),
            gl.zeros([C, BW], gl.float32, MMA),
        )

        # o = qd @ h + Mqk @ U, h being the state entering the chunk.
        qd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_qd.index(cur), A_OP_B)
        mqk_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_mqk.index(cur), A_OP)
        o = gl.convert_layout(
            gl.amd.cdna4.mfma(qd_a, h_op, gl.zeros([C, BW], gl.float32, MMA_B)), MMA
        )
        o = gl.amd.cdna4.mfma(mqk_a, gl.convert_layout(big_u.to(inv_ty), B_OP), o)
        gl.amd.cdna4.buffer_store(o.to(out_ty), out + t0 * (H * V), o_off,
                                  mask=m_c[:, None], cache=CM_OUT)  # fmt: skip

        kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_kr.index(cur).permute((1, 0)), A_OP
        )
        h = gl.amd.cdna4.mfma(
            kr_a, gl.convert_layout(big_u.to(gl.bfloat16), B_OP), h * gt[:, None]
        )

    if STORE_FINAL:  # noqa: SIM102
        if is_last == 1:
            if PAGED_CACHE:
                gl.amd.cdna4.buffer_store(
                    h,
                    state_cache + slot * cache_stride + i_h * (V * K),
                    s_off_vk,
                )
            else:
                f_ptr = final_state + (i_n * H + i_h) * (K * V)
                f_val = h.to(final_state.dtype.element_ty)
                if STATE_V_FIRST:
                    gl.amd.cdna4.buffer_store(f_val, f_ptr, s_off_vk)
                else:
                    gl.amd.cdna4.buffer_store(f_val, f_ptr, s_off)


k2_ab_fused_fast = fast_launch(k2_ab_fused_gluon)
k2_c_fast = fast_launch(k2_c_gluon)
