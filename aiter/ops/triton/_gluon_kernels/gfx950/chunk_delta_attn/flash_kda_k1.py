# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.chunk_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_BLK_WARP_K: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 2], [1, 0])
_BLK1: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
_BLK_CC: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [2, 1], [1, 0])

_MMA_F16: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[2, 1]
)
_AF16: gl.constexpr = gl.DotOperandLayout(0, _MMA_F16, 1)
_BF16: gl.constexpr = gl.DotOperandLayout(1, _MMA_F16, 1)

_MMA_B16: gl.constexpr = gl.amd.AMDMFMALayout(
    version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 1]
)
_A8_16: gl.constexpr = gl.DotOperandLayout(0, _MMA_B16, 8)
_B8_16: gl.constexpr = gl.DotOperandLayout(1, _MMA_B16, 8)

_SH_A: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
_SH_B: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
_SH_CC_F: gl.constexpr = gl.SwizzledSharedLayout(1, 2, 8, [0, 1])


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _exp(x):
    return gl.exp(x.to(gl.float32))


@gluon.jit
def _exp2(x):
    return gl.exp2(x.to(gl.float32))


@gluon.jit
def _sigmoid(x):
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + _exp(-x.to(gl.float32)))


@gluon.jit
def _l2norm(x):
    f = x.to(gl.float32)
    return f * gl.rsqrt(gl.sum(f * f, axis=1) + 1e-6)[:, None]


@gluon.jit
def _via_lds(x, shared: gl.constexpr, dot: gl.constexpr):
    return gl.allocate_shared_memory(x.dtype, x.shape, shared, x).load(dot)


@gluon.jit
def _dot_f32(a, b_op, a_op: gl.constexpr, acc_layout: gl.constexpr, N: gl.constexpr):
    return gl.amd.cdna4.mfma(
        gl.convert_layout(a, a_op),
        b_op,
        gl.zeros([a.shape[0], N], gl.float32, acc_layout),
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
):
    gl.static_assert(C == 32 and K == 128)
    NUM_DOUBLING: gl.constexpr = BC.bit_length() - 2
    NUM_MERGE: gl.constexpr = (C // BC).bit_length() - 1

    i_t = gl.program_id(0).to(gl.int64)
    i_bh = gl.program_id(1).to(gl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = gl.load(chunk_indices + i_t * 2).to(gl.int64)
        i_tl = gl.load(chunk_indices + i_t * 2 + 1).to(gl.int64)
        bos = gl.load(cu_seqlens + i_n).to(gl.int64)
        eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
        T_seq = eos - bos
        g_tile = i_t
    else:
        i_tl = i_t
        bos = i_b * T
        T_seq = T
        g_tile = i_b * NT + i_t

    t_off = i_tl * C
    if t_off >= T_seq:
        return
    actual_len = gl.minimum(C, T_seq - t_off)

    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_WARP_K))
    o_k = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_WARP_K))
    o_k_v = gl.arange(0, K, layout=_BLK1)
    o_i_r = gl.arange(0, C, layout=gl.SliceLayout(1, _MMA_B16))
    o_i_c = gl.arange(0, C, layout=gl.SliceLayout(0, _MMA_B16))

    m_ck = (o_c < actual_len)[:, None]
    m_beta = o_i_r < actual_len

    base = (bos + t_off) * H + i_h
    qk_off = (base * K + o_c[:, None].to(gl.int64) * (H * K) + o_k[None, :]).to(
        gl.int32
    )

    b_q_raw = gl.amd.cdna4.buffer_load(
        ptr=q, offsets=qk_off, mask=m_ck, other=0.0, cache=CM_LOAD
    )
    b_k_raw = gl.amd.cdna4.buffer_load(
        ptr=k, offsets=qk_off, mask=m_ck, other=0.0, cache=CM_LOAD
    )
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(
            ptr=dt_bias, offsets=(i_h * K).to(gl.int32) + o_k
        )[None, :]

    b_g = gl.amd.cdna4.buffer_load(
        ptr=g_raw, offsets=qk_off, mask=m_ck, other=0.0, cache=CM_LOAD
    ).to(gl.float32)
    if HAS_BIAS:
        b_g = b_g + bias
    b_A = gl.load(A_log + i_h)
    b_gate = lower_bound * _sigmoid(_exp(b_A) * b_g)
    log2_e: gl.constexpr = 1.4426950408889634
    b_gcum = gl.associative_scan(b_gate, 0, _add) * log2_e
    b_gcum = gl.where(m_ck, b_gcum, 0.0)

    b_g_last = gl.sum(gl.where(o_c[:, None] == actual_len - 1, b_gcum, 0.0), axis=0)
    b_g_total = _exp2(b_g_last)
    b_exp_g = _exp2(b_gcum)

    b_q = _l2norm(b_q_raw)
    b_k = _l2norm(b_k_raw)

    ws_idx = i_h * TOTAL_TILES + g_tile
    ck_off = (ws_idx * (C * K) + o_c[:, None].to(gl.int64) * K + o_k[None, :]).to(
        gl.int32
    )
    gl.amd.cdna4.buffer_store(
        gl.where(m_ck, b_k * b_exp_g, 0.0).to(ws_kd.dtype.element_ty),
        ws_kd,
        ck_off,
        cache=CM_WS,
    )
    gl.amd.cdna4.buffer_store(
        gl.where(m_ck, b_q * b_exp_g * scale, 0.0).to(ws_qd.dtype.element_ty),
        ws_qd,
        ck_off,
        cache=CM_WS,
    )
    b_kr_val = gl.where(m_ck, b_k * _exp2(b_g_last[None, :] - b_gcum), 0.0).to(
        gl.bfloat16
    )
    gl.amd.cdna4.buffer_store(
        b_kr_val.to(ws_kr.dtype.element_ty), ws_kr, ck_off, cache=CM_WS
    )
    gl.amd.cdna4.buffer_store(
        gl.convert_layout(b_g_total, _BLK1),
        ws_gt,
        (ws_idx * K).to(gl.int32) + o_k_v,
        cache=CM_WS,
    )

    b_beta = _sigmoid(
        gl.amd.cdna4.buffer_load(
            ptr=beta_raw,
            offsets=(base.to(gl.int32) + o_i_r * H),
            mask=m_beta,
            other=0.0,
        ).to(gl.float32)
    )

    o_mid = gl.minimum(C // 2, actual_len - 1)
    b_gp = gl.sum(gl.where(o_c[:, None] == o_mid, b_gcum, 0.0), axis=0)
    b_gm = b_gcum - b_gp[None, :]
    b_dec = _exp2(b_gm)
    b_inc = _exp2(-b_gm)
    b_k_piv = gl.where(m_ck, b_k * b_dec, 0.0).to(gl.bfloat16)
    b_q_piv = gl.where(m_ck, b_q * b_dec * scale, 0.0).to(gl.bfloat16)
    b_k_inv = gl.where(m_ck, b_k * b_inc, 0.0).to(gl.bfloat16)

    b_kinv_b = _via_lds(gl.permute(b_k_inv, 1, 0), _SH_B, _B8_16)

    b_L = gl.amd.cdna4.mfma(
        _via_lds(b_k_piv, _SH_A, _A8_16),
        b_kinv_b,
        gl.zeros([C, C], gl.float32, _MMA_B16),
    )
    b_L = gl.where(o_i_r[:, None] > o_i_c[None, :], -b_L * b_beta[:, None], 0.0)

    b_Mqk = gl.amd.cdna4.mfma(
        _via_lds(b_q_piv, _SH_A, _A8_16),
        b_kinv_b,
        gl.zeros([C, C], gl.float32, _MMA_B16),
    )
    b_Mqk = gl.where(o_i_r[:, None] >= o_i_c[None, :], b_Mqk, 0.0)

    o_r_cc = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_CC))
    o_c_cc = gl.arange(0, C, layout=gl.SliceLayout(0, _BLK_CC))
    cc_off_raw = (
        ws_idx * (2 * C * C) + o_r_cc[:, None].to(gl.int64) * C + o_c_cc[None, :]
    ).to(gl.int32)
    gl.amd.cdna4.buffer_store(
        gl.convert_layout(b_Mqk.to(ws_inv_mqk.dtype.element_ty), _BLK_CC),
        ws_inv_mqk,
        cc_off_raw + C * C,
        cache=CM_WS,
    )

    if BC == C:
        b_D = b_L
    else:
        b_D = gl.where(o_i_r[:, None] // BC == o_i_c[None, :] // BC, b_L, 0.0)
    b_INV = gl.where(o_i_r[:, None] == o_i_c[None, :], 1.0, 0.0) + b_D
    b_INV = gl.convert_layout(b_INV, _MMA_F16)
    b_Dp = _dot_f32(b_D, _via_lds(b_D, _SH_CC_F, _BF16), _AF16, _MMA_F16, C)
    for _ in gl.static_range(NUM_DOUBLING):
        dp_b = _via_lds(b_Dp, _SH_CC_F, _BF16)
        b_INV = b_INV + _dot_f32(b_INV, dp_b, _AF16, _MMA_F16, C)
        b_Dp = _dot_f32(b_Dp, dp_b, _AF16, _MMA_F16, C)

    w = BC
    for _ in gl.static_range(NUM_MERGE):
        ne_w = o_i_r[:, None] // w != o_i_c[None, :] // w
        if 2 * w < C:
            m_off = (o_i_r[:, None] // (2 * w) == o_i_c[None, :] // (2 * w)) & ne_w
        else:
            m_off = ne_w
        b_off = gl.where(m_off, b_L, 0.0)
        inner = _dot_f32(b_off, _via_lds(b_INV, _SH_CC_F, _BF16), _AF16, _MMA_F16, C)
        b_INV = b_INV + _dot_f32(
            b_INV, _via_lds(inner, _SH_CC_F, _BF16), _AF16, _MMA_F16, C
        )
        w = 2 * w

    gl.amd.cdna4.buffer_store(
        gl.convert_layout(b_INV.to(ws_inv_mqk.dtype.element_ty), _BLK_CC),
        ws_inv_mqk,
        cc_off_raw,
        cache=CM_WS,
    )


_NUM_WARPS = math.prod(_MMA_F16.warps_per_cta)


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
):
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
        num_warps=_NUM_WARPS,
    )
