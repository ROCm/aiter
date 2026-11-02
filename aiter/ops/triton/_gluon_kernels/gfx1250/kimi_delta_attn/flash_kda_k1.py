# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import (
    fast_sigmoid,
    l2norm,
    sigmoid_log2,
    sum_combine,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# [C, K] tiles: warps split the rows, so the l2norm row sums stay within a warp.
_BLK_CK: gl.constexpr = gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0])
_BLK1: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
# One whole C = 32 column per thread, for the gate's cumulative sum.
_BLK_COL: gl.constexpr = gl.BlockedLayout([32, 1], [1, 32], [1, 4], [0, 1])

_MMA_B16: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3, transposed=True, warp_bases=[[0, 1], [1, 0]], instr_shape=[16, 16, 32]
)
_A_B16: gl.constexpr = gl.DotOperandLayout(0, _MMA_B16, 8)
_B_B16: gl.constexpr = gl.DotOperandLayout(1, _MMA_B16, 8)
_MMA_F32: gl.constexpr = gl.amd.AMDWMMALayout(
    version=3, transposed=True, warp_bases=[[0, 1], [1, 0]], instr_shape=[16, 16, 4]
)
_A_F32: gl.constexpr = gl.DotOperandLayout(0, _MMA_F32, 2)
_B_F32: gl.constexpr = gl.DotOperandLayout(1, _MMA_F32, 2)

# bf16 [C, K] tiles as TDM writes them and the WMMA operands read them: a 16 B
# pad per 256 B row spreads an operand read's 16 rows over the banks.
_SH_CK: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[128, 8]], [32, 128], [1, 0]
)
# fp32 [C, C] operands: an 8 B pad per 128 B row, for the same reason.
_SH_CC: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
    [[32, 2]], [32, 32], [1, 0]
)
_SH_ROWS: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
_SH_VEC: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

_LOG2E = gl.constexpr(1.4426950408889634)


@gluon.jit
def _tile(ptr, tok, i_h, stride, n_rows, K: gl.constexpr, C: gl.constexpr):
    """The [C, K] tile of head ``i_h`` from token ``tok`` of a [T, H, K] tensor
    whose tokens are ``stride`` apart; rows past ``n_rows`` load as zeros."""
    return gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=ptr + tok * stride + i_h * K,
        shape=(n_rows, K),
        strides=(stride, 1),
        block_shape=(C, K),
        layout=_SH_CK,
    )


_k1_prepare_repr = make_kernel_repr(
    "k1_prepare_gluon",
    ["C", "K", "BC", "IS_VARLEN", "HAS_BIAS"],
)


@gluon.jit(repr=_k1_prepare_repr, do_not_specialize=["T", "NT", "TOTAL_TILES"])
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
    stride_q,
    stride_k,
    stride_g,
    H: gl.constexpr,
    K: gl.constexpr,
    C: gl.constexpr,
    BC: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_BIAS: gl.constexpr,
    CM_WS: gl.constexpr = "",
    CM_LOAD: gl.constexpr = "",
):
    gl.static_assert(C == 32 and K == 128)
    gl.static_assert(gl.num_warps() == 4)
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
    tok = bos + t_off

    o_c = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_CK))
    o_k = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_CK))
    o_c_col = gl.arange(0, C, layout=gl.SliceLayout(1, _BLK_COL))
    o_k_col = gl.arange(0, K, layout=gl.SliceLayout(0, _BLK_COL))

    n_rows = actual_len.to(gl.int32)
    q_s = gl.allocate_shared_memory(q.dtype.element_ty, [C, K], _SH_CK)
    k_s = gl.allocate_shared_memory(k.dtype.element_ty, [C, K], _SH_CK)
    g_s = gl.allocate_shared_memory(g_raw.dtype.element_ty, [C, K], _SH_CK)
    # One load per warp: q, k, and g on the other two.
    gl.amd.gfx1250.tdm.async_load_fused(
        [
            (_tile(q, tok, i_h, stride_q, n_rows, K, C), q_s, 0b0001),
            (_tile(k, tok, i_h, stride_k, n_rows, K, C), k_s, 0b0010),
            (_tile(g_raw, tok, i_h, stride_g, n_rows, K, C), g_s, 0b1100),
        ],
        cache_modifier=CM_LOAD,
    )

    # sigmoid(e^A * (g + bias)) with -log2(e) * e^A folded into one scalar.
    rate = gl.exp(gl.load(A_log + i_h).to(gl.float32)) * -_LOG2E
    if HAS_BIAS:
        bias = gl.amd.gfx1250.buffer_load(
            ptr=dt_bias, offsets=(i_h * K).to(gl.int32) + o_k_col
        )
    o_i_r = gl.arange(0, C, layout=gl.SliceLayout(1, _MMA_B16))
    o_i_c = gl.arange(0, C, layout=gl.SliceLayout(0, _MMA_B16))
    b_beta = fast_sigmoid(
        gl.amd.gfx1250.buffer_load(
            ptr=beta_raw + tok * H + i_h,
            offsets=o_i_r * H,
            mask=o_i_r < actual_len,
            other=0.0,
        )
    )

    gl.amd.gfx1250.tdm.async_wait(0)
    b_g = g_s.load(_BLK_COL).to(gl.float32)
    b_q_raw = q_s.load(_BLK_CK)
    b_k_raw = k_s.load(_BLK_CK)

    if HAS_BIAS:
        b_z = b_g * rate + (bias * rate)[None, :]
    else:
        b_z = b_g * rate
    # Tail rows get a zero gate, so the cumulative gate does not change after
    # the last valid row. Rows C - 1 and C / 2 give the last and pivot values.
    b_gate = gl.where(
        o_c_col[:, None] < actual_len,
        (lower_bound * _LOG2E) * sigmoid_log2(b_z),
        0.0,
    )
    b_gcum_col = gl.associative_scan(b_gate, 0, sum_combine)
    b_g_last_col = gl.sum(gl.where(o_c_col[:, None] == C - 1, b_gcum_col, 0.0), axis=0)
    b_gp_col = gl.sum(gl.where(o_c_col[:, None] == C // 2, b_gcum_col, 0.0), axis=0)
    s_gcum = gl.allocate_shared_memory(gl.float32, [C, K], _SH_ROWS, b_gcum_col)
    s_g_last = gl.allocate_shared_memory(gl.float32, [K], _SH_VEC, b_g_last_col)
    s_gp = gl.allocate_shared_memory(gl.float32, [K], _SH_VEC, b_gp_col)
    b_gcum = s_gcum.load(_BLK_CK)
    b_g_last = s_g_last.load(gl.SliceLayout(0, _BLK_CK))
    b_gp = s_gp.load(gl.SliceLayout(0, _BLK_CK))
    b_g_total = gl.exp2(b_g_last_col)
    b_exp_g = gl.exp2(b_gcum)

    # Tail rows of q and k load as 0 and all factors below are finite, so the
    # tail rows store as 0 without a mask.
    b_q = l2norm(b_q_raw)
    b_k = l2norm(b_k_raw)

    ws_idx = i_h * TOTAL_TILES + g_tile
    # The tile start goes in the 64-bit base: buffer offsets are 32-bit.
    ws_ck = ws_idx * (C * K)
    ck_off = o_c[:, None] * K + o_k[None, :]
    gl.amd.gfx1250.buffer_store(
        (b_k * b_exp_g).to(ws_kd.dtype.element_ty), ws_kd + ws_ck, ck_off, cache=CM_WS
    )
    gl.amd.gfx1250.buffer_store(
        (b_q * b_exp_g * scale).to(ws_qd.dtype.element_ty),
        ws_qd + ws_ck,
        ck_off,
        cache=CM_WS,
    )
    gl.amd.gfx1250.buffer_store(
        (b_k * gl.exp2(b_g_last[None, :] - b_gcum)).to(ws_kr.dtype.element_ty),
        ws_kr + ws_ck,
        ck_off,
        cache=CM_WS,
    )
    gl.amd.gfx1250.buffer_store(
        gl.convert_layout(b_g_total, _BLK1),
        ws_gt + ws_idx * K,
        gl.arange(0, K, layout=_BLK1),
        cache=CM_WS,
    )

    b_gm = b_gcum - b_gp[None, :]
    b_dec = gl.exp2(b_gm)
    b_inc = gl.exp2(-b_gm)
    # All three operands go to LDS before any is read back: one barrier for all.
    s_kinv = gl.allocate_shared_memory(
        gl.bfloat16, [C, K], _SH_CK, (b_k * b_inc).to(gl.bfloat16)
    )
    s_kpiv = gl.allocate_shared_memory(
        gl.bfloat16, [C, K], _SH_CK, (b_k * b_dec).to(gl.bfloat16)
    )
    s_qpiv = gl.allocate_shared_memory(
        gl.bfloat16, [C, K], _SH_CK, (b_q * b_dec * scale).to(gl.bfloat16)
    )
    b_kinv_b = s_kinv.permute([1, 0]).load(_B_B16)

    b_L = gl.amd.gfx1250.wmma(
        s_kpiv.load(_A_B16), b_kinv_b, gl.zeros([C, C], gl.float32, _MMA_B16)
    )
    b_L = gl.where(o_i_r[:, None] > o_i_c[None, :], -b_L * b_beta[:, None], 0.0)

    b_Mqk = gl.amd.gfx1250.wmma(
        s_qpiv.load(_A_B16), b_kinv_b, gl.zeros([C, C], gl.float32, _MMA_B16)
    )
    b_Mqk = gl.where(o_i_r[:, None] >= o_i_c[None, :], b_Mqk, 0.0)

    ws_cc = ws_inv_mqk + ws_idx * (2 * C * C)
    gl.amd.gfx1250.buffer_store(
        b_Mqk.to(ws_inv_mqk.dtype.element_ty),
        ws_cc,
        C * C + o_i_r[:, None] * C + o_i_c[None, :],
        cache=CM_WS,
    )

    if BC == C:
        b_D = b_L
    else:
        b_D = gl.where(o_i_r[:, None] // BC == o_i_c[None, :] // BC, b_L, 0.0)
    # I + D; D is strictly lower triangular, so its diagonal is free for the 1s.
    b_INV = gl.convert_layout(
        gl.where(o_i_r[:, None] == o_i_c[None, :], 1.0, b_D), _MMA_F32
    )
    zero_cc = gl.zeros([C, C], gl.float32, _MMA_F32)
    # Steps alternate between two buffer pairs, so the stores of a step do not
    # overwrite what the previous step still reads.
    s_a0 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC)
    s_b0 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC)
    s_a1 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC)
    s_b1 = gl.allocate_shared_memory(gl.float32, [C, C], _SH_CC)
    s_a1.store(b_D)
    b_Dp = gl.amd.gfx1250.wmma(s_a1.load(_A_F32), s_a1.load(_B_F32), zero_cc)
    for i in gl.static_range(NUM_DOUBLING):
        if i % 2 == 0:
            s_dp = s_a0
            s_inv = s_b0
        else:
            s_dp = s_a1
            s_inv = s_b1
        s_dp.store(b_Dp)
        s_inv.store(b_INV)
        dp_b = s_dp.load(_B_F32)
        b_INV = gl.amd.gfx1250.wmma(s_inv.load(_A_F32), dp_b, b_INV)
        b_Dp = gl.amd.gfx1250.wmma(s_dp.load(_A_F32), dp_b, zero_cc)

    # The merge steps keep to fixed buffers: off and INV in the pair the doubling
    # would use next, inner in the other pair.
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
        inner = gl.amd.gfx1250.wmma(s_off.load(_A_F32), s_inv.load(_B_F32), zero_cc)
        s_inner.store(inner)
        b_INV = gl.amd.gfx1250.wmma(s_inv.load(_A_F32), s_inner.load(_B_F32), b_INV)
        w = 2 * w

    o_f_r = gl.arange(0, C, layout=gl.SliceLayout(1, _MMA_F32))
    o_f_c = gl.arange(0, C, layout=gl.SliceLayout(0, _MMA_F32))
    gl.amd.gfx1250.buffer_store(
        b_INV.to(ws_inv_mqk.dtype.element_ty),
        ws_cc,
        o_f_r[:, None] * C + o_f_c[None, :],
        cache=CM_WS,
    )


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
    stride_q,
    stride_k,
    stride_g,
    H,
    K,
    C,
    BC,
    B,
    num_warps,
    waves_per_eu,
    CM_WS="",
    CM_LOAD="",
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
        stride_q=stride_q,
        stride_k=stride_k,
        stride_g=stride_g,
        H=H,
        K=K,
        C=C,
        BC=BC,
        IS_VARLEN=cu_seqlens is not None,
        HAS_BIAS=dt_bias is not None,
        CM_WS=CM_WS,
        CM_LOAD=CM_LOAD,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
