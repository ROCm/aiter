# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# [C, K] tiles: warps split the rows, so the l2norm row sums stay within a warp
# (a cross-warp sum costs three barriers).
_BLK_CK: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [2, 1], [1, 0])
_BLK1: gl.constexpr = gl.BlockedLayout([1], [64], [2], [0])
# One whole C = 32 column per thread, for the gate's cumulative sum.
_BLK_COL: gl.constexpr = gl.BlockedLayout([32, 1], [1, 64], [1, 2], [0, 1])

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
_SH_ROWS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
_SH_VEC: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

_LOG2E = gl.constexpr(1.4426950408889634)


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
def _sigmoid_log2(z):
    """sigmoid(x) given ``z = -x * log2(e)``.

    exp2 lowers to a bare v_exp_f32, where exp adds a denormal-range rescale
    that the sigmoid does not need.
    """
    return gl.extra.libdevice.fast_dividef(1.0, 1.0 + gl.exp2(z))


@gluon.jit
def _sigmoid(x):
    return _sigmoid_log2(x.to(gl.float32) * -_LOG2E)


@gluon.jit
def _l2norm(x):
    f = x.to(gl.float32)
    return f * gl.rsqrt(gl.sum(f * f, axis=1) + 1e-6)[:, None]


_k1_prepare_repr = make_kernel_repr(
    "k1_prepare_gluon",
    ["C", "K", "BC", "IS_VARLEN", "HAS_BIAS", "STORE_BETA"],
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
    ws_beta,
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
    STORE_BETA: gl.constexpr = False,
):
    """FlashKDA K1 on gfx950; same workspace contract as the Triton prepare kernel.

    With ``STORE_BETA`` also writes ``sigmoid(beta)`` to ``ws_beta``, which the
    Gluon K2 copies into LDS with the chunk's other tiles.
    """
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

    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_CK))
    o_k = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_CK))
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

    # The gate is cumulated over rows, so it is computed with each thread owning
    # a whole column: the scan and the row picks below are then in-thread.
    o_c_col = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_COL))
    o_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_COL))
    g_off = (base * K + o_c_col[:, None].to(gl.int64) * (H * K) + o_k_col[None, :]).to(
        gl.int32
    )
    # Masked: the buffer resource has no size, so rows past the tensor's end are
    # not clamped by hardware and would fault.
    b_g = gl.amd.cdna4.buffer_load(
        ptr=g_raw,
        offsets=g_off,
        mask=(o_c_col < actual_len)[:, None],
        other=0.0,
        cache=CM_LOAD,
    ).to(gl.float32)
    # sigmoid(e^A * (g + bias)) with -log2(e) * e^A folded into one scalar.
    rate = _exp(gl.load(A_log + i_h)) * -_LOG2E
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(
            ptr=dt_bias, offsets=(i_h * K).to(gl.int32) + o_k_col
        )
        b_z = b_g * rate + (bias * rate)[None, :]
    else:
        b_z = b_g * rate
    # Tail rows get a zero gate (g = 0 alone would give lower_bound / 2, plus the
    # bias), so the cumulative gate is constant from row actual_len - 1 on: the
    # last row is row C - 1, and the pivot row min(C / 2, actual_len - 1) is
    # row C / 2.
    b_gate = gl.where(
        o_c_col[:, None] < actual_len,
        (lower_bound * _LOG2E) * _sigmoid_log2(b_z),
        0.0,
    )
    b_gcum_col = gl.associative_scan(b_gate, 0, _add)
    b_g_last_col = gl.sum(gl.where(o_c_col[:, None] == C - 1, b_gcum_col, 0.0), axis=0)
    b_gp_col = gl.sum(gl.where(o_c_col[:, None] == C // 2, b_gcum_col, 0.0), axis=0)
    # One LDS round trip (one barrier) takes the cumulative gate and its last and
    # pivot rows to the row layout; separate conversions cost one or two
    # barriers each.
    s_gcum = gl.allocate_shared_memory(gl.float32, [C, K], _SH_ROWS, b_gcum_col)
    s_g_last = gl.allocate_shared_memory(gl.float32, [K], _SH_VEC, b_g_last_col)
    s_gp = gl.allocate_shared_memory(gl.float32, [K], _SH_VEC, b_gp_col)
    b_gcum = s_gcum.load(_BLK_CK)
    b_g_last = s_g_last.load(gl.SliceLayout(0, _BLK_CK))
    b_gp = s_gp.load(gl.SliceLayout(0, _BLK_CK))
    b_g_total = _exp2(b_g_last_col)
    b_exp_g = _exp2(b_gcum)

    # Tail rows of q and k load as 0 and stay 0 through the norm. Every factor
    # they meet below is finite (the cumulative gate is constant over the tail),
    # so the workspace tiles' tail rows come out zero without a mask.
    b_q = _l2norm(b_q_raw)
    b_k = _l2norm(b_k_raw)

    ws_idx = i_h * TOTAL_TILES + g_tile
    ck_off = (ws_idx * (C * K) + o_c[:, None].to(gl.int64) * K + o_k[None, :]).to(
        gl.int32
    )
    gl.amd.cdna4.buffer_store(
        (b_k * b_exp_g).to(ws_kd.dtype.element_ty),
        ws_kd,
        ck_off,
        cache=CM_WS,
    )
    gl.amd.cdna4.buffer_store(
        (b_q * b_exp_g * scale).to(ws_qd.dtype.element_ty),
        ws_qd,
        ck_off,
        cache=CM_WS,
    )
    b_kr_val = (b_k * _exp2(b_g_last[None, :] - b_gcum)).to(gl.bfloat16)
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
    if STORE_BETA:
        gl.amd.cdna4.buffer_store(
            b_beta, ws_beta, (ws_idx * C).to(gl.int32) + o_i_r, cache=CM_WS
        )

    b_gm = b_gcum - b_gp[None, :]
    b_dec = _exp2(b_gm)
    b_inc = _exp2(-b_gm)
    b_k_piv = (b_k * b_dec).to(gl.bfloat16)
    b_q_piv = (b_q * b_dec * scale).to(gl.bfloat16)
    b_k_inv = (b_k * b_inc).to(gl.bfloat16)

    # All three operands go to LDS before any is read back: one barrier for the
    # lot instead of two per operand.
    s_kinv = gl.allocate_shared_memory(
        gl.bfloat16, [K, C], _SH_B, gl.permute(b_k_inv, 1, 0)
    )
    s_kpiv = gl.allocate_shared_memory(gl.bfloat16, [C, K], _SH_A, b_k_piv)
    s_qpiv = gl.allocate_shared_memory(gl.bfloat16, [C, K], _SH_A, b_q_piv)
    b_kinv_b = s_kinv.load(_B8_16)

    b_L = gl.amd.cdna4.mfma(
        s_kpiv.load(_A8_16), b_kinv_b, gl.zeros([C, C], gl.float32, _MMA_B16)
    )
    b_L = gl.where(o_i_r[:, None] > o_i_c[None, :], -b_L * b_beta[:, None], 0.0)

    b_Mqk = gl.amd.cdna4.mfma(
        s_qpiv.load(_A8_16), b_kinv_b, gl.zeros([C, C], gl.float32, _MMA_B16)
    )
    b_Mqk = gl.where(o_i_r[:, None] >= o_i_c[None, :], b_Mqk, 0.0)

    # Mqk and INV are stored from the MFMA layout (four consecutive columns per
    # thread): converting them for a wider store costs two barriers each.
    cc_base = (ws_idx * (2 * C * C)).to(gl.int32)
    gl.amd.cdna4.buffer_store(
        b_Mqk.to(ws_inv_mqk.dtype.element_ty),
        ws_inv_mqk,
        cc_base + C * C + o_i_r[:, None] * C + o_i_c[None, :],
        cache=CM_WS,
    )

    if BC == C:
        b_D = b_L
    else:
        b_D = gl.where(o_i_r[:, None] // BC == o_i_c[None, :] // BC, b_L, 0.0)
    # I + D; D is strictly lower triangular, so its diagonal is free for the 1s.
    b_INV = gl.convert_layout(
        gl.where(o_i_r[:, None] == o_i_c[None, :], 1.0, b_D), _MMA_F16
    )
    zero_cc = gl.zeros([C, C], gl.float32, _MMA_F16)
    # Each product's operands go through LDS once: stored together, one barrier,
    # then loaded in each operand layout they are used in. Steps alternate
    # between two buffer pairs, so a step's stores never land on what the
    # previous step is still reading. Barriers are placed per allocation, so
    # every operand has its own: two stores into one buffer cost a barrier
    # between them.
    s_a0 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC_F)
    s_b0 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC_F)
    s_a1 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC_F)
    s_b1 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC_F)
    s_a1.store(b_D)
    b_Dp = gl.amd.cdna4.mfma(s_a1.load(_AF16), s_a1.load(_BF16), zero_cc)
    for i in gl.static_range(NUM_DOUBLING):
        if i % 2 == 0:
            s_dp = s_a0
            s_inv = s_b0
        else:
            s_dp = s_a1
            s_inv = s_b1
        s_dp.store(b_Dp)
        s_inv.store(b_INV)
        dp_b = s_dp.load(_BF16)
        b_INV = gl.amd.cdna4.mfma(s_inv.load(_AF16), dp_b, b_INV)
        b_Dp = gl.amd.cdna4.mfma(s_dp.load(_AF16), dp_b, zero_cc)

    # The merge steps keep to fixed buffers: off and INV in the pair the doubling
    # would use next, inner in the other pair. The first pair was last read
    # before the previous step's barrier, the second before this step's first.
    if NUM_DOUBLING % 2 == 0:
        s_off = s_a0
        s_inv = s_b0
        s_inner = s_a1
    else:
        s_off = s_a1
        s_inv = s_b1
        s_inner = s_a0
    w = BC
    for _ in gl.static_range(NUM_MERGE):
        ne_w = o_i_r[:, None] // w != o_i_c[None, :] // w
        if 2 * w < C:
            m_off = (o_i_r[:, None] // (2 * w) == o_i_c[None, :] // (2 * w)) & ne_w
        else:
            m_off = ne_w
        s_off.store(gl.where(m_off, b_L, 0.0))
        s_inv.store(b_INV)
        inner = gl.amd.cdna4.mfma(s_off.load(_AF16), s_inv.load(_BF16), zero_cc)
        s_inner.store(inner)
        b_INV = gl.amd.cdna4.mfma(s_inv.load(_AF16), s_inner.load(_BF16), b_INV)
        w = 2 * w

    o_f_r = gl.arange(0, C, layout=gl.SliceLayout(1, _MMA_F16))
    o_f_c = gl.arange(0, C, layout=gl.SliceLayout(0, _MMA_F16))
    gl.amd.cdna4.buffer_store(
        b_INV.to(ws_inv_mqk.dtype.element_ty),
        ws_inv_mqk,
        cc_base + o_f_r[:, None] * C + o_f_c[None, :],
        cache=CM_WS,
    )


_NUM_WARPS = math.prod(_MMA_F16.warps_per_cta)
# Holds K1 to 168 VGPRs; left alone it lands at 170 and drops to 2 waves per
# SIMD (0.85x).
_WAVES_PER_EU = 3


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
    ws_beta,
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
        ws_beta=ws_beta,
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
        STORE_BETA=ws_beta is not None,
        CM_WS=CM_WS,
        CM_LOAD=CM_LOAD,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )
