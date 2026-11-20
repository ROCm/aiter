# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.device_info import get_num_xcds

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
        # Global -> LDS copy sources: no lane replication, as the copy requires.
        "BLK": gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        "BLK_CC": gl.BlockedLayout([1, 2], [4, 16], [nw, 1], [1, 0]),
        # gt and beta: one element per thread.
        "BLK_1D": gl.BlockedLayout([1], [64], [nw], [0]),
        "SH_WS": gl.SwizzledSharedLayout(8, 1, 16, [1, 0]),
        "SH_PLAIN": gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        "SH_1D": gl.SwizzledSharedLayout(1, 1, 1, [0]),
    }


@gluon.jit
def _solve_u(
    h,
    kd_a,
    inv_a,
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
    """U = inv @ (beta * (v - kd @ h)), and h as a dot operand."""
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
    return big_u, h_op


@gluon.jit
def _update(h, kr_a, big_u, gt, B_OP: gl.constexpr):
    """The next state: diag(gt) h + kr^T U."""
    return gl.amd.cdna4.mfma(
        kr_a, gl.convert_layout(big_u.to(gl.bfloat16), B_OP), h * gt[:, None]
    )


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
    big_u, h_op = _solve_u(h, kd_a, inv_a, beta, v, m_c, C, BW, MMA, B_OP,
                           MMA_B, B_OP_B, INV_TY, HAS_V)  # fmt: skip
    return _update(h, kr_a, big_u, gt, B_OP), big_u, h_op


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
    """Start the async copy of one chunk into LDS, as one commit group.

    ``gt`` and ``beta`` are copied one element per thread (``o_t``), excess
    threads masked. ``WITH_OUTPUT`` (pass C) also fetches ``qd`` and ``Mqk``.
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
    both. Chunk j+1 is async-copied into the other half of an LDS double buffer
    while chunk j computes.
    """
    # Keeps a (segment, head)'s V blocks on one XCD, sharing an L2.
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
        # Chunk j is in half `cur`, and every thread is done with half 1 - cur.
        gl.amd.cdna4.async_copy.wait_group(0)
        if j + 1 < n_chunks:
            _issue_chunk(ws_kd, ws_kr, ws_gt, ws_beta, ws_inv_mqk, v_input, None,
                         s_kd.index(1 - cur), s_kr.index(1 - cur),
                         s_inv.index(1 - cur), s_v.index(1 - cur),
                         s_gt.index(1 - cur), s_beta.index(1 - cur), None, None,
                         ws0 + j + 1, tok_base + (j + 1) * C, tok_end, ws_off,
                         cc_off, v_off, o_c_v, o_t, H, K, V, C, False)  # fmt: skip

        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(s_kd.index(cur), A_OP_B)
        # The permuted [C, K] view is the kr^T A operand.
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


# The pipelined pass A: one program per (segment, head) and four warp-pipeline
# stages per chunk over a 4-slot LDS ring, with chunks j+1..j+3 in flight. It
# needs gl.amd.hint.sched_barrier and the buffer descriptor ops; without them
# (Triton 3.8, whose LLVM also spills this kernel), pass A is k2_ab_fused_gluon.
PASS_A_PIPELINED = (
    hasattr(gl.amd, "hint")
    and hasattr(gl.amd.hint, "sched_barrier")
    and hasattr(gl.amd.cdna4, "buffer")
)
PASS_A_C = 32
PASS_A_K = 128
PASS_A_WARPS = 8
# Without it, LLVM merges the kd and inv reads of rows r and r + 16 into
# ds_read2st64_b64, and v_movs then rebuild the MFMA operand register pairs.
PASS_A_FN_ATTRS = "target-features=-load-store-opt"
NS: gl.constexpr = gl.constexpr(4)  # LDS ring slots
NP: gl.constexpr = gl.constexpr(4)  # 32-row pieces of the state
# The bound of the pointer-based buffer ops, i.e. none: the masks keep the
# accesses in bounds.
PA_NUM_RECORDS: gl.constexpr = gl.constexpr(2**31 - 2)


@gluon.jit
def _desc(base):
    return gl.amd.cdna4.buffer.make_buffer_descriptor(base, PA_NUM_RECORDS)


@gluon.jit
def _to_lds(dest, base, offsets, mask=None):
    gl.amd.cdna4.buffer.load_to_shared(dest, _desc(base), offsets, mask=mask)


@gluon.jit
def _s_barrier(x):
    # gl.barrier() also drains vmcnt, i.e. every copy in flight.
    return gl.inline_asm_elementwise(
        "s_barrier", "=v,v", [x], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def _rows(x, i: gl.constexpr, R: gl.constexpr):
    return gl.amd.slice(x, [R, x.shape[1]], [i * R, 0])


@gluon.jit
def _cols(x, i: gl.constexpr, R: gl.constexpr):
    return gl.amd.slice(x, [x.shape[0], R], [0, i * R])


@gluon.jit
def _pieces(x, R: gl.constexpr):
    return (_rows(x, 0, R), _rows(x, 1, R), _rows(x, 2, R), _rows(x, 3, R))


@gluon.jit
def _bf16_op(x, B_OP: gl.constexpr):
    return gl.convert_layout(x.to(gl.bfloat16), B_OP)


@gluon.jit
def _bases(
    ws_kd,
    ws_inv_mqk,
    v_input,
    ws_beta,
    ws_kr,
    ws_gt,
    ws_idx,
    tok,
    C: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    H: gl.constexpr,
):
    """The global bases of the six copies of chunk ws_idx (first token tok)."""
    return (ws_kd + ws_idx * (C * K), ws_inv_mqk + ws_idx * (2 * C * C),
            v_input + tok * (H * V), ws_beta + ws_idx * C,
            ws_kr + ws_idx * (C * K), ws_gt + ws_idx * K)  # fmt: skip


@gluon.jit
def _advance(
    pk,
    n: gl.constexpr,
    C: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    H: gl.constexpr,
):
    """The _bases n chunks after pk."""
    return (pk[0] + n * (C * K), pk[1] + n * (2 * C * C), pk[2] + n * (C * H * V),
            pk[3] + n * C, pk[4] + n * (C * K), pk[5] + n * K)  # fmt: skip


@gluon.jit
def _copy_d(
    kd_w,
    inv_w,
    v_w,
    beta_w,
    pd,
    slot,
    kd_off,
    inv_off,
    v_off,
    beta_off,
    o_c,
    lim,
    C: gl.constexpr,
    MASK: gl.constexpr,
):
    """kd, inv, v and beta of one chunk into ring slot ``slot``, one commit group.

    beta is copied 2C wide, one element per lane of a wave; the upper half is
    the next chunk's. MASK: only the first ``lim`` rows of v and beta are real
    (the chunk may be the last), the rest and the upper half load as 0.
    """
    _to_lds(kd_w.index(slot), pd[0], kd_off)
    _to_lds(inv_w.index(slot), pd[1], inv_off)
    if MASK:
        _to_lds(v_w.index(slot), pd[2], v_off, (o_c < lim)[:, None])
        _to_lds(beta_w.index(slot), pd[3], beta_off, beta_off < gl.minimum(lim, C))
    else:
        _to_lds(v_w.index(slot), pd[2], v_off)
        _to_lds(beta_w.index(slot), pd[3], beta_off)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _copy_r(kr_w, gt_w, pd, slot, kr_off, gt_off):
    """kr and gt of one chunk into ring slot ``slot``, one commit group."""
    _to_lds(kr_w.index(slot), pd[4], kr_off)
    _to_lds(gt_w.index(slot), pd[5], gt_off)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _read_d(
    kd_r,
    inv_r,
    v_r,
    beta_r,
    slot,
    C: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    CVT: gl.constexpr,
):
    """dot1's operands of the chunk in ``slot``; v stays bf16 unless CVT."""
    kd = kd_r.index(slot).load(A_OP)
    inv = inv_r.index(slot).load(A_OP)
    v = v_r.index(slot).load(MMA)
    beta = beta_r.index(slot).slice(0, C).load(gl.SliceLayout(1, MMA))
    if CVT:
        v = v.to(gl.float32)
    return kd, inv, v, beta


@gluon.jit
def _gt_piece(gt_s, slot, i: gl.constexpr, R: gl.constexpr, MMA: gl.constexpr):
    return gt_s.index(slot).slice(i * R, R).load(gl.SliceLayout(1, MMA))


@gluon.jit
def _dot1(
    kd,
    inv,
    vf,
    beta,
    g_a,
    gb_op,
    C: gl.constexpr,
    BW: gl.constexpr,
    R: gl.constexpr,
    MMA: gl.constexpr,
    B_OP: gl.constexpr,
    INV_TY: gl.constexpr,
):
    """big = inv @ (-beta * (v + kd @ g)) for both chains, with g = -h.

    The A chain has v = 0. Seeding the B chain's accumulator with v saves the
    subtraction; the sign of big is undone by dot2.
    """
    zero = gl.zeros([C, BW], gl.float32, MMA)
    nbeta = -beta
    tmp_b = vf
    for i in gl.static_range(NP):
        tmp_b = gl.amd.cdna4.mfma(_cols(kd, i, R), gb_op[i], tmp_b)
    ga = (_bf16_op(g_a[0], B_OP), _bf16_op(g_a[1], B_OP),
          _bf16_op(g_a[2], B_OP), _bf16_op(g_a[3], B_OP))  # fmt: skip
    tmp_a = zero
    for i in gl.static_range(NP):
        tmp_a = gl.amd.cdna4.mfma(_cols(kd, i, R), ga[i], tmp_a)
    u_b = gl.convert_layout((tmp_b * nbeta[:, None]).to(INV_TY), B_OP)
    big_b = gl.amd.cdna4.mfma(inv, u_b, zero)
    u_a = gl.convert_layout((tmp_a * nbeta[:, None]).to(INV_TY), B_OP)
    big_a = gl.amd.cdna4.mfma(inv, u_a, zero)
    return big_b, big_a


@gluon.jit
def _dot2(krp, big_b_op, big_a_op, g_b, g_a, gt, B_OP: gl.constexpr):
    """g = kr^T @ big + g * gt per piece, and g_b in bf16 as dot1's next B.

    Every g * gt is issued first, then the four B MFMAs, then each A MFMA
    with the cvt of one B piece. Left to the scheduler, the packed multiplies
    sit between the MFMAs, where they cannot issue.
    """
    cb0 = g_b[0] * gt[0][:, None]
    cb1 = g_b[1] * gt[1][:, None]
    cb2 = g_b[2] * gt[2][:, None]
    cb3 = g_b[3] * gt[3][:, None]
    ca0 = g_a[0] * gt[0][:, None]
    ca1 = g_a[1] * gt[1][:, None]
    ca2 = g_a[2] * gt[2][:, None]
    ca3 = g_a[3] * gt[3][:, None]
    gl.amd.hint.sched_barrier()
    nb0 = gl.amd.cdna4.mfma(krp[0], big_b_op, cb0)
    nb1 = gl.amd.cdna4.mfma(krp[1], big_b_op, cb1)
    nb2 = gl.amd.cdna4.mfma(krp[2], big_b_op, cb2)
    nb3 = gl.amd.cdna4.mfma(krp[3], big_b_op, cb3)
    gl.amd.hint.sched_barrier()
    na0 = gl.amd.cdna4.mfma(krp[0], big_a_op, ca0)
    ob0 = _bf16_op(nb0, B_OP)
    gl.amd.hint.sched_barrier()
    na1 = gl.amd.cdna4.mfma(krp[1], big_a_op, ca1)
    ob1 = _bf16_op(nb1, B_OP)
    gl.amd.hint.sched_barrier()
    na2 = gl.amd.cdna4.mfma(krp[2], big_a_op, ca2)
    ob2 = _bf16_op(nb2, B_OP)
    gl.amd.hint.sched_barrier()
    na3 = gl.amd.cdna4.mfma(krp[3], big_a_op, ca3)
    ob3 = _bf16_op(nb3, B_OP)
    return (nb0, nb1, nb2, nb3), (na0, na1, na2, na3), (ob0, ob1, ob2, ob3)


@gluon.jit
def _step(
    kd,
    inv,
    vf,
    beta,
    g_b,
    g_a,
    gb_op,
    pk,
    kd_w,
    kr_w,
    v_w,
    inv_w,
    gt_s,
    beta_s,
    kd_r,
    kr_r,
    v_r,
    inv_r,
    kd_off,
    kr_off,
    inv_off,
    v_off,
    gt_off,
    beta_off,
    o_c,
    lim,
    S,
    COPY: gl.constexpr,
    LR: gl.constexpr,
    W1: gl.constexpr,
    W2: gl.constexpr,
    MASK: gl.constexpr,
    PO: gl.constexpr,
    ADV: gl.constexpr,
    CVT_D1: gl.constexpr,
    CVT_M2: gl.constexpr,
    C: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    H: gl.constexpr,
    MMA: gl.constexpr,
    A_OP: gl.constexpr,
    B_OP: gl.constexpr,
):
    """One chunk, in ring slot S, as four warp-pipeline stages.

    COPY: copy the chunk 3 ahead, whose bases are ``_advance(pk, PO)``, into
    slot S + 3 (the slot of the chunk before). LR: read the next chunk's dot1
    operands from slot S + 1. W1, W2: the wait_group counts before mem1 and
    mem2 (-1: none). mem2 moves pk on by ADV chunks. v crosses a loop back
    edge in bf16: a converted v there is folded into the loop header, which
    blocks loop rotation and spills, so CVT_D1 / CVT_M2 convert it in dot1 or
    in mem2.
    """
    S1 = (S + 1) % NS
    S3 = (S + 3) % NS
    R: gl.constexpr = K // NP
    inv_ty: gl.constexpr = inv_r.dtype
    with gl.amd.warp_pipeline_stage("dot1"):
        if CVT_D1:
            vf = vf.to(gl.float32)
        big_b, big_a = _dot1(kd, inv, vf, beta, g_a, gb_op, C, V, R, MMA, B_OP,
                             inv_ty)  # fmt: skip
    if W1 >= 0:
        gl.amd.cdna4.async_copy.wait_group(W1)
    with gl.amd.warp_pipeline_stage("mem1"):
        pd = _advance(pk, PO, C, K, V, H)
        # Reads before copies: the copies' VALU would wait behind the other
        # wave group's dot2, and so would the reads issued after them.
        kr = kr_r.index(S).permute([1, 0]).load(A_OP)
        krp = _pieces(kr, R)
        gl.amd.hint.sched_barrier()
        gt = (_gt_piece(gt_s, S, 0, R, MMA), _gt_piece(gt_s, S, 1, R, MMA),
              _gt_piece(gt_s, S, 2, R, MMA), _gt_piece(gt_s, S, 3, R, MMA))  # fmt: skip
        gl.amd.hint.sched_barrier()
        big_b_op = gl.convert_layout(big_b.to(kr.dtype), B_OP)
        big_a_op = gl.convert_layout(big_a.to(kr.dtype), B_OP)
        if COPY:
            _copy_d(kd_w, inv_w, v_w, beta_s, pd, S3, kd_off, inv_off, v_off,
                    beta_off, o_c, lim, C, MASK)  # fmt: skip
    with gl.amd.warp_pipeline_stage("dot2"):
        g_b, g_a, gb_op = _dot2(krp, big_b_op, big_a_op, g_b, g_a, gt, B_OP)
    if W2 >= 0:
        gl.amd.cdna4.async_copy.wait_group(W2)
    with gl.amd.warp_pipeline_stage("mem2"):
        if LR:
            kd, inv, vf, beta = _read_d(kd_r, inv_r, v_r, beta_s, S1, C, MMA,
                                        A_OP, CVT_M2)  # fmt: skip
        if COPY:
            _copy_r(kr_w, gt_s, _advance(pk, PO, C, K, V, H), S3, kr_off, gt_off)
        if ADV > 0:
            pk = _advance(pk, ADV, C, K, V, H)
    return kd, inv, vf, beta, g_b, g_a, gb_op, pk


@gluon.jit
def _store(g_b, g_a, h_out_b, h_out_a, s_base, s_off, R: gl.constexpr):
    desc_b = _desc(h_out_b + s_base)
    desc_a = _desc(h_out_a + s_base)
    for i in gl.static_range(NP):
        o = _rows(s_off, i, R)
        gl.amd.cdna4.buffer.store((-g_b[i]).to(h_out_b.dtype.element_ty), desc_b, o)
        gl.amd.cdna4.buffer.store((-g_a[i]).to(h_out_a.dtype.element_ty), desc_a, o)


# The layouts of the pipelined pass A are module constants: as kernel arguments,
# fast_launch hashes them on each launch, which doubles its host time.
PA_MMA: gl.constexpr = gl.constexpr(
    gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[1, PASS_A_WARPS],
    )
)
# k_width 4: an MFMA accumulator is the B operand of the next MFMA as is.
PA_A_OP: gl.constexpr = gl.constexpr(gl.DotOperandLayout(0, PA_MMA.value, 4))
PA_B_OP: gl.constexpr = gl.constexpr(gl.DotOperandLayout(1, PA_MMA.value, 4))
# Copy sources: 8 contiguous elements (16 bytes) per lane, no lane replicated.
PA_ASYNC_T: gl.constexpr = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]],
        warp_bases=[[4, 0], [8, 0], [16, 0]],
        block_bases=[],
        shape=[PASS_A_C, PASS_A_K],
    )
)
PA_ASYNC_INV: gl.constexpr = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4]],
        lane_bases=[[0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]],
        warp_bases=[[16, 0], [0, 0], [0, 0]],
        block_bases=[],
        shape=[PASS_A_C, PASS_A_C],
    )
)
PA_ASYNC_GT: gl.constexpr = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[64], [0], [0]],
        block_bases=[],
        shape=[PASS_A_K],
    )
)
PA_ASYNC_BETA: gl.constexpr = gl.constexpr(
    gl.DistributedLinearLayout(
        reg_bases=[],
        lane_bases=[[1], [2], [4], [8], [16], [32]],
        warp_bases=[[0], [0], [0]],
        block_bases=[],
        shape=[2 * PASS_A_C],
    )
)
# Copies write unswizzled; the reads see the swizzles of PA_SH_KD, PA_SH_TR and
# PA_SH_INV.
PA_SH_LIN: gl.constexpr = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 1, [1, 0]))
PA_SH_KD: gl.constexpr = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 16, [1, 0]))
# kr and v: their transposed reads (ds_read_b64_tr_b16) are then free of bank
# conflicts.
PA_SH_TR: gl.constexpr = gl.constexpr(gl.SwizzledSharedLayout(16, 1, 8, [1, 0]))
PA_SH_INV: gl.constexpr = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 4, [1, 0]))
PA_SH_1D: gl.constexpr = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))

_k2_ab_pipe_repr = make_kernel_repr("k2_ab_fused_pipe_gluon", ["C", "K", "V"])


@gluon.jit(repr=_k2_ab_pipe_repr)
def k2_ab_fused_pipe_gluon(
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
):
    """Both pass-A recurrences of one (segment, head): h_b from zero with the
    real v, h_a from the identity with v = 0. Their two MFMA chains cover each
    other's latency.

    Chunk j uses ring slot (j - n % 4) % 4. The first n % 4 chunks run in a
    loop with run-time slots, the rest in groups of 4 with compile-time slots;
    the last group is straight-line code. Only the copy of chunk n - 1 masks
    the rows past the segment's end. With n < 4, the copies past the end
    repeat chunk n - 1, so the wait counts hold.
    """
    gl.static_assert(K == V)
    gl.static_assert(K == NP * 32)
    i_sh = gl.program_id(1).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H
    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)
    ws0 = i_h * TOTAL_TILES + chunk_base
    R: gl.constexpr = K // NP

    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, PA_MMA))
    o_w_m = gl.arange(0, V, layout=gl.SliceLayout(0, PA_MMA))
    s_base = (i_seg * H + i_h) * (K * V)
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)
    # The state is kept negated, g = -h.
    g_b = _pieces(gl.zeros([K, V], gl.float32, PA_MMA), R)
    g_a = _pieces(gl.where(o_k_m[:, None] == o_w_m[None, :], -1.0, 0.0), R)
    if n == 0:
        _store(g_b, g_a, h_out_b, h_out_a, s_base, s_off, R)
        return
    gb_op = (_bf16_op(g_b[0], PA_B_OP), _bf16_op(g_b[1], PA_B_OP),
             _bf16_op(g_b[2], PA_B_OP), _bf16_op(g_b[3], PA_B_OP))  # fmt: skip

    # A copy writes LDS in lane order, so the read layouts' swizzles are
    # applied to the global offsets instead.
    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, PA_ASYNC_T))
    o_k = gl.arange(0, K, layout=gl.SliceLayout(0, PA_ASYNC_T))
    kd_off = (o_c[:, None] * K + (o_k[None, :] ^ ((o_c[:, None] & 15) * 8))).to(
        gl.int32
    )
    kd_off = gl.max_contiguous(gl.multiple_of(kd_off, [1, 8]), [1, 8])
    o_k_tr = o_k[None, :] ^ ((o_c[:, None] & 7) * 16)
    kr_off = (o_c[:, None] * K + o_k_tr).to(gl.int32)
    kr_off = gl.max_contiguous(gl.multiple_of(kr_off, [1, 8]), [1, 8])
    v_off = (o_c[:, None] * (H * V) + i_h * V + o_k_tr).to(gl.int32)
    v_off = gl.max_contiguous(gl.multiple_of(v_off, [1, 8]), [1, 8])
    o_r_i = gl.arange(0, C, layout=gl.SliceLayout(1, PA_ASYNC_INV))
    o_c_i = gl.arange(0, C, layout=gl.SliceLayout(0, PA_ASYNC_INV))
    inv_off = (o_r_i[:, None] * C + (o_c_i[None, :] ^ ((o_r_i[:, None] & 3) * 8))).to(
        gl.int32
    )
    inv_off = gl.max_contiguous(gl.multiple_of(inv_off, [1, 8]), [1, 8])
    gt_off = gl.arange(0, K, layout=PA_ASYNC_GT)
    beta_off = gl.arange(0, 2 * C, layout=PA_ASYNC_BETA)

    kd_w = gl.allocate_shared_memory(ws_kd.dtype.element_ty, [NS, C, K], PA_SH_LIN)
    kr_w = gl.allocate_shared_memory(ws_kr.dtype.element_ty, [NS, C, K], PA_SH_LIN)
    v_w = gl.allocate_shared_memory(v_input.dtype.element_ty, [NS, C, V], PA_SH_LIN)
    inv_w = gl.allocate_shared_memory(
        ws_inv_mqk.dtype.element_ty, [NS, C, C], PA_SH_LIN
    )
    gt_s = gl.allocate_shared_memory(gl.float32, [NS, K], PA_SH_1D)
    beta_s = gl.allocate_shared_memory(gl.float32, [NS, 2 * C], PA_SH_1D)
    kd_r = kd_w.reinterpret(kd_w.dtype, kd_w.shape, PA_SH_KD)
    kr_r = kr_w.reinterpret(kr_w.dtype, kr_w.shape, PA_SH_TR)
    v_r = v_w.reinterpret(v_w.dtype, v_w.shape, PA_SH_TR)
    inv_r = inv_w.reinterpret(inv_w.dtype, inv_w.shape, PA_SH_INV)

    # Chunks 0..2 (past the end, chunk n - 1 again).
    n_lead = n % NS
    for c in gl.static_range(NS - 1):
        cc = gl.minimum(n - 1, c)
        pd = _bases(ws_kd, ws_inv_mqk, v_input, ws_beta, ws_kr, ws_gt, ws0 + cc,
                    tok_base + cc * C, C, K, V, H)  # fmt: skip
        slot = (c + NS - n_lead) % NS
        lim = (tok_end - tok_base - cc * C).to(gl.int32)
        _copy_d(kd_w, inv_w, v_w, beta_s, pd, slot, kd_off, inv_off, v_off,
                beta_off, o_c, lim, C, True)  # fmt: skip
        _copy_r(kr_w, gt_s, pd, slot, kr_off, gt_off)
    gl.amd.cdna4.async_copy.wait_group(2 * (NS - 2))
    _s_barrier(i_h.to(gl.int32))
    kd, inv, vf, beta = _read_d(kd_r, inv_r, v_r, beta_s, (NS - n_lead) % NS, C,
                                PA_MMA, PA_A_OP, False)  # fmt: skip

    for t in range(n_lead):
        c3 = gl.minimum(t + 3, n - 1)
        pk = _bases(ws_kd, ws_inv_mqk, v_input, ws_beta, ws_kr, ws_gt, ws0 + c3,
                    tok_base + c3 * C, C, K, V, H)  # fmt: skip
        lim = (tok_end - tok_base - c3 * C).to(gl.int32)
        kd, inv, vf, beta, g_b, g_a, gb_op, pk = _step(
            kd, inv, vf, beta, g_b, g_a, gb_op, pk,
            kd_w, kr_w, v_w, inv_w, gt_s, beta_s, kd_r, kr_r, v_r, inv_r,
            kd_off, kr_off, inv_off, v_off, gt_off, beta_off, o_c, lim,
            (t + NS - n_lead) % NS, True, True, 3, 3, True, 0, 0, True, False,
            C, K, V, H, PA_MMA, PA_A_OP, PA_B_OP,
        )  # fmt: skip
    if n < NS:
        gl.amd.cdna4.async_copy.wait_group(0)
        _store(g_b, g_a, h_out_b, h_out_a, s_base, s_off, R)
        return

    pk = _bases(ws_kd, ws_inv_mqk, v_input, ws_beta, ws_kr, ws_gt,
                ws0 + n_lead + 3, tok_base + (n_lead + 3) * C, C, K, V, H)  # fmt: skip
    for b in range(n // NS - 1):
        for s in gl.static_range(NS):
            kd, inv, vf, beta, g_b, g_a, gb_op, pk = _step(
                kd, inv, vf, beta, g_b, g_a, gb_op, pk,
                kd_w, kr_w, v_w, inv_w, gt_s, beta_s, kd_r, kr_r, v_r, inv_r,
                kd_off, kr_off, inv_off, v_off, gt_off, beta_off, o_c, 0,
                s, True, True, 3, 3, False, s, NS if s == NS - 1 else 0,
                s == 0, s != NS - 1, C, K, V, H, PA_MMA, PA_A_OP, PA_B_OP,
            )  # fmt: skip

    lim = (tok_end - tok_base - (n - 1) * C).to(gl.int32)
    for t in gl.static_range(NS):
        kd, inv, vf, beta, g_b, g_a, gb_op, pk = _step(
            kd, inv, vf, beta, g_b, g_a, gb_op, pk,
            kd_w, kr_w, v_w, inv_w, gt_s, beta_s, kd_r, kr_r, v_r, inv_r,
            kd_off, kr_off, inv_off, v_off, gt_off, beta_off, o_c, lim,
            t, t == 0, t < NS - 1, 3 if t < 2 else (1 if t == 2 else -1),
            3 - t if t < 2 else (0 if t == 2 else -1), True, t, 0, t == 0, True,
            C, K, V, H, PA_MMA, PA_A_OP, PA_B_OP,
        )  # fmt: skip
    gl.amd.cdna4.async_copy.wait_group(0)
    _store(g_b, g_a, h_out_b, h_out_a, s_base, s_off, R)


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
    """Pass C: one segment's recurrence from its incoming state, writing outputs.

    Same contract as ``_flash_kda_segment_kernel`` with ``COMPUTE_OUTPUT`` and
    ``HAS_V`` and without ``STORE_H_OUT``, paged ``state_cache`` included.
    Prefetches like pass A. The output is stored from the MFMA layout.
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
    # Read before the output stores, after which the compiler reloads them per
    # lane (a waterfall loop on the final-state store). Null unless needed.
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
        # A zero-start sequence's slot may hold a stale state: never read it.
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

        # _recur's steps, with the state update moved after the output
        # projection to keep the next state and kr out of registers meanwhile.
        big_u, h_op = _solve_u(h, kd_a, inv_a, beta, b_v, m_c, C, BW, MMA, B_OP,
                               MMA_B, B_OP_B, inv_ty, True)  # fmt: skip

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
        h = _update(h, kr_a, big_u, gt, B_OP)

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


k2_ab_fused_fast = fast_launch(
    k2_ab_fused_pipe_gluon if PASS_A_PIPELINED else k2_ab_fused_gluon
)
k2_c_fast = fast_launch(k2_c_gluon)


def gluon_k2_ab_fused(
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
    H,
    K,
    V,
    C,
    num_segs,
    BW,
    num_warps,
    waves_per_eu=0,
):
    """Pass A. The pipelined kernel has a fixed schedule -- one program per
    (segment, head), 8 warps, 2 waves per SIMD -- and does not use BW,
    num_warps or waves_per_eu."""
    args = {
        "ws_kd": ws_kd,
        "ws_kr": ws_kr,
        "ws_gt": ws_gt,
        "ws_beta": ws_beta,
        "ws_inv_mqk": ws_inv_mqk,
        "v_input": v_input,
        "h_out_b": h_out_b,
        "h_out_a": h_out_a,
        "seg_chunk_base": seg_chunk_base,
        "seg_nchunks": seg_nchunks,
        "seg_tok_base": seg_tok_base,
        "seg_tok_end": seg_tok_end,
        "TOTAL_TILES": TOTAL_TILES,
        "H": H,
        "K": K,
        "V": V,
        "C": C,
    }
    if not PASS_A_PIPELINED:
        return k2_ab_fused_fast[(triton.cdiv(V, BW), num_segs * H)](
            **args,
            BW=BW,
            **build_layouts(num_warps),
            NUM_XCDS=get_num_xcds(),
            NUM_WARPS=num_warps,
            num_warps=num_warps,
            waves_per_eu=waves_per_eu,
        )
    if (C, K, V) != (PASS_A_C, PASS_A_K, PASS_A_K):
        raise ValueError(
            f"pass A needs C={PASS_A_C} and K=V={PASS_A_K}, got C={C} K={K} V={V}"
        )
    return k2_ab_fused_fast[(1, num_segs * H)](
        **args,
        num_warps=PASS_A_WARPS,
        waves_per_eu=2,
        llvm_fn_attrs=PASS_A_FN_ATTRS,
    )


def gluon_k2_c(
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
    H,
    K,
    V,
    C,
    num_segs,
    BW,
    num_warps,
    STORE_FINAL,
    STATE_V_FIRST,
    CM_OUT="",
):
    paged = state_cache is not None
    return k2_c_fast[(triton.cdiv(V, BW), num_segs * H)](
        ws_kd=ws_kd,
        ws_qd=ws_qd,
        ws_kr=ws_kr,
        ws_gt=ws_gt,
        ws_beta=ws_beta,
        ws_inv_mqk=ws_inv_mqk,
        v_input=v_input,
        out=out,
        h_in=h_in,
        final_state=final_state,
        seg_chunk_base=seg_chunk_base,
        seg_nchunks=seg_nchunks,
        seg_tok_base=seg_tok_base,
        seg_tok_end=seg_tok_end,
        seg_seq=seg_seq,
        seg_is_last=seg_is_last,
        state_cache=state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        cache_stride=cache_stride,
        TOTAL_TILES=TOTAL_TILES,
        H=H,
        K=K,
        V=V,
        C=C,
        BW=BW,
        HAS_H_IN=h_in is not None,
        STORE_FINAL=STORE_FINAL,
        STATE_V_FIRST=STATE_V_FIRST,
        PAGED_CACHE=paged,
        PAGED_H_IN=paged and h_in is None,
        **build_layouts(num_warps),
        NUM_XCDS=get_num_xcds(),
        NUM_WARPS=num_warps,
        CM_OUT=CM_OUT,
        num_warps=num_warps,
    )
