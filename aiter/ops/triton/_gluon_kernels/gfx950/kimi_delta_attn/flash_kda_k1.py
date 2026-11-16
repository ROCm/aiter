# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import (
    fast_sigmoid,
    sigmoid_log2,
    sum_combine,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_BLK1: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
# One whole C = 32 column per thread, for the gate's cumulative sum.
_BLK_COL: gl.constexpr = gl.BlockedLayout([32, 1], [1, 64], [1, 2], [0, 1])

_MMA_B16: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 1]
)
_A8_16: gl.constexpr = gl.DotOperandLayout(0, _MMA_B16, 8)
_B8_16: gl.constexpr = gl.DotOperandLayout(1, _MMA_B16, 8)

_SH_A: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
_SH_B: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
_SH_ROWS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
_SH_VEC: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
_SH_G8: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
_BLK_H: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [2, 1], [1, 0])
_M3: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[2, 1, 1]
)
_A3: gl.constexpr = gl.DotOperandLayout(0, _M3, 4)
_B3: gl.constexpr = gl.DotOperandLayout(1, _M3, 4)
# [cb, rb, i, j] LDS view of the off-diagonal X; the sum over rb lands in _B3 / _A3.
_L4B: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 1, 0, 0]],
    lane_bases=[
        [0, 0, 0, 1],
        [0, 0, 0, 2],
        [0, 0, 0, 4],
        [0, 0, 0, 8],
        [0, 0, 4, 0],
        [0, 0, 8, 0],
    ],
    warp_bases=[[1, 0, 0, 0]],
    block_bases=[],
    shape=[2, 2, 16, 16],
)
_L4A: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 1, 0, 0]],
    lane_bases=[
        [0, 0, 1, 0],
        [0, 0, 2, 0],
        [0, 0, 4, 0],
        [0, 0, 8, 0],
        [0, 0, 0, 4],
        [0, 0, 0, 8],
    ],
    warp_bases=[[1, 0, 0, 0]],
    block_bases=[],
    shape=[2, 2, 16, 16],
)

_LOG2E = gl.constexpr(1.4426950408889634)


@gluon.jit
def _join_halves(a, b):
    C: gl.constexpr = a.shape[0]
    KH: gl.constexpr = a.shape[1]
    return gl.permute(gl.join(a, b), (0, 2, 1)).reshape([C, 2 * KH])


@gluon.jit
def _k1_half(
    q_raw,
    k_raw,
    rq,
    rk,
    s_gc,
    s_gl,
    s_gpe,
    s_gp,
    s_stg,
    ws_kd,
    ws_qd,
    ws_kr,
    tb,
    k0: gl.constexpr,
    CM: gl.constexpr,
    K: gl.constexpr,
    scale,
):
    C: gl.constexpr = q_raw.shape[0]
    N: gl.constexpr = q_raw.shape[1]
    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_H))
    o_k = gl.arange(0, N, layout=gl.SliceLayout(0, _BLK_H)) + k0
    kn = k_raw.to(gl.float32) * rk[:, None]
    qn = q_raw.to(gl.float32) * (rq * scale)[:, None]
    gm = s_gc.load(_BLK_H) - s_gp.load(gl.SliceLayout(0, _BLK_H))[None, :]
    dec = gl.exp2(gm)
    kp = kn * dec
    qp = qn * dec
    ki = kn * gl.exp2(-gm)
    gpe = s_gpe.load(gl.SliceLayout(0, _BLK_H))[None, :]
    s_stg.store(
        gl.permute(
            gl.join(
                (kp * gpe).to(ws_kd.dtype.element_ty),
                (qp * gpe).to(ws_qd.dtype.element_ty),
            ),
            (2, 0, 1),
        ).reshape([2 * C, N])
    )
    v = (
        s_stg.reshape([2 * C, N // 32, 2, 4, 4])
        .permute((0, 1, 3, 2, 4))
        .reshape([2 * C, N])
        .load(_BLK_H)
    )
    a, b = gl.split(gl.permute(v.reshape([2, C, N]), (1, 2, 0)))
    L: gl.constexpr = a.type.layout
    o_r = gl.arange(0, C, layout=gl.SliceLayout(1, L))
    o_n = gl.arange(0, N, layout=gl.SliceLayout(0, L))
    off = (tb + o_r[:, None].to(gl.int64) * K + k0 + o_n[None, :]).to(gl.int32)
    gl.amd.cdna4.buffer_store(a, ws_kd, off, cache=CM)
    gl.amd.cdna4.buffer_store(b, ws_qd, off, cache=CM)
    # perm32 row order: bits 2-3 of the index move to bits 3-4, bit 4 to bit 2.
    pr = (o_c & ~31) | (((o_c >> 2) & 3) << 3) | (((o_c >> 4) & 1) << 2) | (o_c & 3)
    off_r = (tb + pr[:, None].to(gl.int64) * K + o_k[None, :]).to(gl.int32)
    gl_ = s_gl.load(gl.SliceLayout(0, _BLK_H))[None, :]
    gl.amd.cdna4.buffer_store(
        (ki * gl_).to(ws_kr.dtype.element_ty), ws_kr, off_r, cache=CM
    )
    return kp.to(gl.bfloat16), qp.to(gl.bfloat16), ki.to(gl.bfloat16)


@gluon.jit
def _mm(a, bT, acc):
    return gl.amd.cdna4.mfma(
        gl.convert_layout(a.to(gl.float16), _A3),
        gl.convert_layout(gl.permute(bT, (0, 2, 1)).to(gl.float16), _B3),
        acc,
    )


@gluon.jit
def _blockdiag_inv(b_L, same, BC: gl.constexpr):
    C: gl.constexpr = b_L.shape[0]
    pd = gl.convert_layout(
        gl.sum(gl.where(same, b_L, 0.0).reshape([2, BC, 2, BC]), axis=2), _M3
    )
    pdT = gl.convert_layout(gl.permute(pd, (0, 2, 1)), _M3)
    i = gl.arange(0, BC, layout=gl.SliceLayout(0, gl.SliceLayout(2, _M3)))[
        None, :, None
    ]
    j = gl.arange(0, BC, layout=gl.SliceLayout(0, gl.SliceLayout(1, _M3)))[
        None, None, :
    ]
    eye = gl.where(i == j, 1.0, 0.0)
    x = eye + gl.where(i // 2 == j // 2, pd, 0.0)
    xT = eye + gl.where(i // 2 == j // 2, pdT, 0.0)
    z = gl.zeros(x.shape, gl.float32, _M3)
    for s in gl.static_range(1, 4):
        ms = (i >> (s + 1) == j >> (s + 1)) & (i >> s != j >> s)
        t = _mm(x, gl.where(ms, pdT, 0.0), z)
        tT = _mm(xT, gl.where(ms, pd, 0.0), z)
        xn = _mm(t, xT, x)
        xT = _mm(tT, x, xT)
        x = xn
    # Merge: the off-diagonal block is -X1 L10 X0, with X0 and X1 the diagonal inverses.
    po = gl.convert_layout(
        gl.sum(gl.where(same, 0.0, b_L).reshape([2, BC, 2, BC]), axis=2), _M3
    )
    y = gl.convert_layout(
        gl.permute(gl.join(x, x), (0, 1, 3, 2)).reshape([C, C]), b_L.type.layout
    )
    s_x = gl.allocate_shared_memory(
        gl.float32, [C, C], _SH_ROWS, gl.where(same, 0.0, y)
    )
    v4 = s_x.reshape([2, BC, 2, BC]).permute((2, 0, 1, 3))  # [cb, rb, i, j]
    xsT = gl.convert_layout(gl.sum(v4.permute((0, 1, 3, 2)).load(_L4A), axis=1), _M3)
    off = _mm(x, _mm(xsT, po, z), z)
    bb = gl.arange(0, 2, layout=gl.SliceLayout(1, gl.SliceLayout(2, _M3)))[
        :, None, None
    ]
    a0 = gl.where(bb == 0, x, off)
    a1 = gl.where(bb == 1, x, 0.0)
    return gl.convert_layout(
        gl.permute(gl.join(a0, a1), (0, 1, 3, 2)).reshape([C, C]), b_L.type.layout
    )


@gluon.jit
def _store_cc_perm(val, ws, off_base, CM: gl.constexpr):
    C: gl.constexpr = val.shape[0]
    v = val.reshape([C, C // 32, 2, 4, 4]).permute((0, 1, 3, 2, 4)).reshape([C, C])
    L: gl.constexpr = v.type.layout
    o_r = gl.arange(0, C, layout=gl.SliceLayout(1, L))
    o_c = gl.arange(0, C, layout=gl.SliceLayout(0, L))
    gl.amd.cdna4.buffer_store(
        v, ws, off_base + o_r[:, None] * C + o_c[None, :], cache=CM
    )


_k1_prepare_repr = make_kernel_repr(
    "k1_prepare_gluon",
    ["C", "K", "BC", "IS_VARLEN", "HAS_BIAS"],
)


@gluon.jit(repr=_k1_prepare_repr)
def k1_prepare_gluon(
    q,
    k,
    g_raw,
    beta_raw,
    A_log,
    dt_bias,
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    cu_seqlens,
    chunk_indices,
    scale,
    lower_bound,
    T,
    NT,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    C: gl.constexpr,
    BC: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    CM_WS: gl.constexpr = "",
    CM_LOAD: gl.constexpr = ".cg",
    epoch=None,
):
    """FlashKDA K1 on gfx950, for the Gluon K2 only: the inverse carries diag(beta),
    and the contraction indices of kd, qd, kr, inv and Mqk are in perm32 order.
    With ``IS_VARLEN``, ``chunk_indices`` is a (first token, length) row per tile."""
    gl.static_assert(C == 32 and K == 128 and BC == 16)
    if epoch is not None:
        if (gl.program_id(0) == 0) & (gl.program_id(1) == 0):
            gl.store(epoch, gl.load(epoch) + 1)
    KH: gl.constexpr = K // 2
    i_t = gl.program_id(0).to(gl.int64)
    i_bh = gl.program_id(1).to(gl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        # chunk_indices is the (first token, length) table of the tiles.
        bos = gl.load(chunk_indices + i_t * 2).to(gl.int64)
        actual_len = gl.load(chunk_indices + i_t * 2 + 1)
        g_tile = i_t
    else:
        bos = i_b * T + i_t * C
        actual_len = gl.minimum(C, T - i_t * C).to(gl.int32)
        g_tile = i_b * NT + i_t
    base = bos * H + i_h
    ws_idx = i_h * TOTAL_TILES + g_tile
    tb = ws_idx * (C * K)
    o_k_v = gl.arange(0, K, layout=_BLK1)
    o_i_r = gl.arange(0, C, layout=gl.SliceLayout(1, _MMA_B16))
    o_i_c = gl.arange(0, C, layout=gl.SliceLayout(0, _MMA_B16))

    # q and k in two K / 2 column halves (fewer live registers); rows past the
    # tile get out-of-range offsets, which load as 0.
    o_bt = gl.arange(0, C, layout=_BLK1)
    beta_raw_v = gl.amd.cdna4.buffer_load(
        ptr=beta_raw,
        offsets=gl.where(o_bt < actual_len, base.to(gl.int32) + o_bt * H, 1 << 30),
    )
    o_ch = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_H))
    o_kh = gl.arange(0, KH, layout=gl.SliceLayout(0, _BLK_H))
    off_h = (base * K + o_ch[:, None].to(gl.int64) * (H * K) + o_kh[None, :]).to(
        gl.int32
    )
    off_h = gl.where((o_ch < actual_len)[:, None], off_h, 1 << 30)
    off_h = gl.max_contiguous(gl.multiple_of(off_h, [1, 8]), [1, 8])
    q0 = gl.amd.cdna4.buffer_load(ptr=q, offsets=off_h, cache=CM_LOAD)
    k0 = gl.amd.cdna4.buffer_load(ptr=k, offsets=off_h, cache=CM_LOAD)
    q1 = gl.amd.cdna4.buffer_load(ptr=q, offsets=off_h + KH, cache=CM_LOAD)
    k1 = gl.amd.cdna4.buffer_load(ptr=k, offsets=off_h + KH, cache=CM_LOAD)

    # One column per thread, so the gate's scan and row picks are in-thread.
    o_c_col = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_COL))
    o_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_COL))
    o_c_cl = gl.minimum(o_c_col, actual_len - 1)
    g_off = (base * K).to(gl.int32) + o_c_cl[:, None] * (H * K) + o_k_col[None, :]
    b_g = gl.amd.cdna4.buffer_load(ptr=g_raw, offsets=g_off, cache=CM_LOAD).to(
        gl.float32
    )
    # sigmoid(e^A * (g + bias)) with -log2(e) * e^A folded into one scalar.
    rate = gl.exp(gl.load(A_log + i_h).to(gl.float32)) * -_LOG2E
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(
            ptr=dt_bias, offsets=(i_h * K).to(gl.int32) + o_k_col
        )
        b_z = b_g * rate + (bias * rate)[None, :]
    else:
        b_z = b_g * rate
    # Tail rows get a zero gate, so the cumulative gate does not change after
    # the last valid row. Rows C - 1 and C / 2 give the last and pivot values.
    b_gate = gl.where(
        o_c_col[:, None] < actual_len, (lower_bound * _LOG2E) * sigmoid_log2(b_z), 0.0
    )
    b_gcum_col = gl.associative_scan(b_gate, 0, sum_combine)
    b_g_last_col = gl.sum(gl.where(o_c_col[:, None] == C - 1, b_gcum_col, 0.0), axis=0)
    b_gp_col = gl.sum(gl.where(o_c_col[:, None] == C // 2, b_gcum_col, 0.0), axis=0)
    # The decays factor around the pivot: exp2(g) = exp2(g - gp) exp2(gp).
    s_gcum = gl.allocate_shared_memory(gl.float32, [C, K], _SH_G8, b_gcum_col)
    s_bt = gl.allocate_shared_memory(
        gl.float32, [C], _SH_VEC, fast_sigmoid(beta_raw_v.to(gl.float32))
    )
    s_g_last = gl.allocate_shared_memory(
        gl.float32, [K], _SH_VEC, gl.exp2(b_g_last_col - b_gp_col)
    )
    s_gpe = gl.allocate_shared_memory(gl.float32, [K], _SH_VEC, gl.exp2(b_gp_col))
    s_gp = gl.allocate_shared_memory(gl.float32, [K], _SH_VEC, b_gp_col)
    gl.amd.cdna4.buffer_store(
        gl.convert_layout(gl.exp2(b_g_last_col), _BLK1),
        ws_gt,
        (ws_idx * K).to(gl.int32) + o_k_v,
        cache=CM_WS,
    )
    q0f = q0.to(gl.float32)
    q1f = q1.to(gl.float32)
    k0f = k0.to(gl.float32)
    k1f = k1.to(gl.float32)
    rq = gl.rsqrt(gl.sum(q0f * q0f, axis=1) + gl.sum(q1f * q1f, axis=1) + 1e-6)
    rk = gl.rsqrt(gl.sum(k0f * k0f, axis=1) + gl.sum(k1f * k1f, axis=1) + 1e-6)
    s_stg = gl.allocate_shared_memory(gl.bfloat16, [2 * C, KH], _SH_ROWS)
    kp0, qp0, ki0 = _k1_half(
        q0,
        k0,
        rq,
        rk,
        s_gcum.slice(0, KH, dim=1),
        s_g_last.slice(0, KH),
        s_gpe.slice(0, KH),
        s_gp.slice(0, KH),
        s_stg,
        ws_kd,
        ws_qd,
        ws_kr,
        tb,
        0,
        CM_WS,
        K,
        scale,
    )
    kp1, qp1, ki1 = _k1_half(
        q1,
        k1,
        rq,
        rk,
        s_gcum.slice(KH, KH, dim=1),
        s_g_last.slice(KH, KH),
        s_gpe.slice(KH, KH),
        s_gp.slice(KH, KH),
        s_stg,
        ws_kd,
        ws_qd,
        ws_kr,
        tb,
        KH,
        CM_WS,
        K,
        scale,
    )
    s_kinv = gl.allocate_shared_memory(
        gl.bfloat16, [K, C], _SH_B, gl.permute(_join_halves(ki0, ki1), 1, 0)
    )
    s_kpiv = gl.allocate_shared_memory(
        gl.bfloat16, [C, K], _SH_A, _join_halves(kp0, kp1)
    )
    s_qpiv = gl.allocate_shared_memory(
        gl.bfloat16, [C, K], _SH_A, _join_halves(qp0, qp1)
    )
    b_beta = s_bt.load(gl.SliceLayout(1, _MMA_B16))
    b_kinv_b = s_kinv.load(_B8_16)

    b_L = gl.amd.cdna4.mfma(
        s_kpiv.load(_A8_16), b_kinv_b, gl.zeros([C, C], gl.float32, _MMA_B16)
    )
    b_L = gl.where(o_i_r[:, None] > o_i_c[None, :], -b_L * b_beta[:, None], 0.0)

    b_Mqk = gl.amd.cdna4.mfma(
        s_qpiv.load(_A8_16), b_kinv_b, gl.zeros([C, C], gl.float32, _MMA_B16)
    )
    b_Mqk = gl.where(o_i_r[:, None] >= o_i_c[None, :], b_Mqk, 0.0)
    cc_base = (ws_idx * (2 * C * C)).to(gl.int32)
    _store_cc_perm(
        b_Mqk.to(ws_inv_mqk.dtype.element_ty), ws_inv_mqk, cc_base + C * C, CM_WS
    )

    # (I - D)^-1 of D = -beta L with diag(beta) folded in, so K2 needs no beta.
    same = (o_i_r[:, None] // BC) == (o_i_c[None, :] // BC)
    b_x = _blockdiag_inv(b_L, same, BC)
    b_x = b_x * s_bt.load(gl.SliceLayout(0, _MMA_B16))[None, :]
    _store_cc_perm(b_x.to(ws_inv_mqk.dtype.element_ty), ws_inv_mqk, cc_base, CM_WS)


# The warp count is fixed by the layouts above.
_NUM_WARPS = math.prod(_MMA_B16.warps_per_cta)


_k1_fast = fast_launch(k1_prepare_gluon)


def gluon_k1_prepare(
    q,
    k,
    g_raw,
    beta_raw,
    A_log,
    dt_bias,
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    cu_seqlens,
    chunk_indices,
    scale,
    lower_bound,
    T,
    NT,
    TOTAL_TILES,
    H,
    K,
    C,
    BC,
    B,
    CM_WS="",
    CM_LOAD=".cg",
    num_warps=_NUM_WARPS,
    waves_per_eu=0,
    epoch=None,
):
    if num_warps != _NUM_WARPS:
        raise ValueError(
            f"k1_prepare_gluon is built for num_warps={_NUM_WARPS}, got {num_warps}"
        )
    return _k1_fast[(TOTAL_TILES if cu_seqlens is not None else NT, B * H)](
        q=q,
        k=k,
        g_raw=g_raw,
        beta_raw=beta_raw,
        A_log=A_log,
        dt_bias=dt_bias,
        ws_kd=ws_kd,
        ws_qd=ws_qd,
        ws_kr=ws_kr,
        ws_gt=ws_gt,
        ws_inv_mqk=ws_inv_mqk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        lower_bound=lower_bound,
        T=T,
        NT=NT,
        TOTAL_TILES=TOTAL_TILES,
        H=H,
        K=K,
        C=C,
        BC=BC,
        IS_VARLEN=cu_seqlens is not None,
        HAS_BIAS=dt_bias is not None,
        CM_WS=CM_WS,
        CM_LOAD=CM_LOAD,
        epoch=epoch,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
