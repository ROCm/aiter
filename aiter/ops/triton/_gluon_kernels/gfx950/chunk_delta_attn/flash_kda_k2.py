# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.chunk_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

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
        "BLK": gl.BlockedLayout([1, 8], [4, 16], [nw, 1], [1, 0]),
        "SH_KR": gl.SwizzledSharedLayout(8, 1, 16, [0, 1]),
    }


@gluon.jit
def _sigmoid(x):
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp(-x.to(gl.float32)))


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
def _kr_operand(
    kr_raw,
    K: gl.constexpr,
    C: gl.constexpr,
    SH_KR: gl.constexpr,
    A_OP: gl.constexpr,
):
    return gl.allocate_shared_memory(
        gl.bfloat16, [K, C], SH_KR, gl.permute(kr_raw, 1, 0)
    ).load(A_OP)


_k2_ab_fused_repr = make_kernel_repr(
    "k2_ab_fused_gluon",
    ["C", "K", "V", "BW"],
)


@gluon.jit(repr=_k2_ab_fused_repr)
def k2_ab_fused_gluon(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
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
    SH_KR: gl.constexpr,
):
    """Both pass-A recurrences in one launch, sharing every operand load.

    The two differ only in seeding -- zero with the real v against the identity
    with v = 0 -- so they read the same chunk and can share it. Register
    residency is what makes the sharing worth having: the two chains are
    independent, so the scheduler interleaves them, and that is what covers the
    serial dependence each one has on its own. Requires K == V, since the two
    states are then the same width and one program covers a column block of
    both.
    """
    i_w = gl.program_id(0).to(gl.int64)
    i_sh = gl.program_id(1).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)

    o_c_ab = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP_B))
    o_k_ab = gl.arange(0, K, layout=gl.SliceLayout(0, A_OP_B))
    o_c_a = gl.arange(0, C, layout=gl.SliceLayout(1, A_OP))
    o_cc_a = gl.arange(0, C, layout=gl.SliceLayout(0, A_OP))
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    o_k_s = gl.arange(0, K, layout=gl.SliceLayout(0, BLK))
    o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))

    kd_off = (o_c_ab[:, None] * K + o_k_ab[None, :]).to(gl.int32)
    inv_off = (o_c_a[:, None] * C + o_cc_a[None, :]).to(gl.int32)
    kr_off = (o_c_s[:, None] * K + o_k_s[None, :]).to(gl.int32)
    gt_off = o_k_m.to(gl.int32)
    beta_off = (i_h + o_c_m * H).to(gl.int32)
    v_off = (i_h * V + o_c_m[:, None] * (H * V) + o_w_m[None, :]).to(gl.int32)

    h_b = gl.zeros([K, BW], gl.float32, MMA)
    h_a = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)

    ws0 = i_h * TOTAL_TILES + chunk_base
    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty

    for j in range(n_chunks):
        ws_idx = ws0 + j
        ck = ws_idx * (C * K)
        t0 = tok_base + j * C
        tb = t0 * H
        m_c = (t0 + o_c_m) < tok_end

        kd_a = gl.amd.cdna4.buffer_load(ptr=ws_kd + ck, offsets=kd_off)
        inv_a = gl.amd.cdna4.buffer_load(
            ptr=ws_inv_mqk + ws_idx * (2 * C * C), offsets=inv_off
        )
        gt = gl.amd.cdna4.buffer_load(ptr=ws_gt + ws_idx * K, offsets=gt_off)
        kr_raw = gl.amd.cdna4.buffer_load(ptr=ws_kr + ck, offsets=kr_off)
        beta = _sigmoid(
            gl.amd.cdna4.buffer_load(
                ptr=beta_raw + tb, offsets=beta_off, mask=m_c, other=0.0
            )
        )
        b_v = gl.amd.cdna4.buffer_load(
            ptr=v_input + tb * V, offsets=v_off, mask=m_c[:, None], other=0.0
        ).to(gl.float32)

        kr_a = _kr_operand(kr_raw, K, C, SH_KR, A_OP)
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


k2_ab_fused_fast = fast_launch(k2_ab_fused_gluon)
