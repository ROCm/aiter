# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import softplus
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_replayssm_repr = make_kernel_repr(
    "_replayssm_kda_fwd_cta_kernel", ["BH", "BT", "TPW_K", "WARPS", "FUSED"]
)


@gluon.jit
def _kda_cm_in(x):
    """[R, 2, C, 4] -> [R, C, 8] (or [2, C, 4] -> [C, 8]): slot 4h + e of lane j is column 64h + 4j + e."""
    if len(x.shape) == 4:
        return gl.reshape(gl.permute(x, [0, 2, 1, 3]), [x.shape[0], x.shape[2], 8])
    else:
        return gl.reshape(gl.permute(x, [1, 0, 2]), [x.shape[1], 8])


@gluon.jit
def _kda_cm_out(x, L: gl.constexpr):
    """Inverse of `_kda_cm_in` for [R, C, 8] -> [R, 2, C, 4] in layout L."""
    y = gl.reshape(x, [x.shape[0], x.shape[1], 2, 4])
    return gl.convert_layout(gl.permute(y, [0, 2, 1, 3]), L)


@gluon.jit
def _kda_split8(x, L: gl.constexpr):
    """[R, C, 8] or [C, 8] (last axis in-thread) -> its 8 slices, in layout L."""
    if x.shape[len(x.shape) - 1] == 4:
        x = _kda_cm_in(x)
    if len(x.shape) == 3:
        a, b = gl.split(gl.reshape(x, [x.shape[0], x.shape[1], 4, 2]))
        a0, a1 = gl.split(gl.reshape(a, [x.shape[0], x.shape[1], 2, 2]))
        b0, b1 = gl.split(gl.reshape(b, [x.shape[0], x.shape[1], 2, 2]))
    else:
        a, b = gl.split(gl.reshape(x, [x.shape[0], 4, 2]))
        a0, a1 = gl.split(gl.reshape(a, [x.shape[0], 2, 2]))
        b0, b1 = gl.split(gl.reshape(b, [x.shape[0], 2, 2]))
    x0, x4 = gl.split(a0)
    x2, x6 = gl.split(a1)
    x1, x5 = gl.split(b0)
    x3, x7 = gl.split(b1)
    return (
        gl.convert_layout(x0, L),
        gl.convert_layout(x1, L),
        gl.convert_layout(x2, L),
        gl.convert_layout(x3, L),
        gl.convert_layout(x4, L),
        gl.convert_layout(x5, L),
        gl.convert_layout(x6, L),
        gl.convert_layout(x7, L),
    )


@gluon.jit
def _kda_join8(x0, x1, x2, x3, x4, x5, x6, x7, L: gl.constexpr):
    """Inverse of `_kda_split8` for [R, C] slices -> [R, C, 8] in layout L."""
    a = gl.join(gl.join(x0, x4), gl.join(x2, x6))
    b = gl.join(gl.join(x1, x5), gl.join(x3, x7))
    x = gl.join(a, b)
    return gl.convert_layout(gl.reshape(x, [x0.shape[0], x0.shape[1], 8]), L)


@gluon.jit
def _kda_lanesum16(p):
    """Sum over the 16 lanes of a DPP row (the K lanes), left in every lane."""
    p = gl.inline_asm_elementwise(
        "v_add_f32_dpp $0, $1, $1 row_xmask:1 row_mask:0xf bank_mask:0xf",
        "=v,v",
        [p],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    p = gl.inline_asm_elementwise(
        "v_add_f32_dpp $0, $1, $1 row_xmask:2 row_mask:0xf bank_mask:0xf",
        "=v,v",
        [p],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    p = gl.inline_asm_elementwise(
        "v_add_f32_dpp $0, $1, $1 row_xmask:4 row_mask:0xf bank_mask:0xf",
        "=v,v",
        [p],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    p = gl.inline_asm_elementwise(
        "v_add_f32_dpp $0, $1, $1 row_xmask:8 row_mask:0xf bank_mask:0xf",
        "=v,v",
        [p],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    return p


# Transpose-reduce of 8 per-lane row partials over a 16-lane DPP row:
# bank-masked DPP adds halve the live rows per step (rows 0-3 | 4-7 by
# lane bit 3, then by bit 2), v_bfi picks by lane bit 1 (m = -1 where set),
# and lane j ends with the full sum of row j >> 1 (14 DPP adds + 2 bfi
# instead of 32 butterfly adds).


@gluon.jit
def _kda_rowsum8(p, m):
    """Row sums over the 16 K lanes, left in every lane (drop-in for `_kda_lanesum16`)."""
    return gl.inline_asm_elementwise(
        (
            "v_add_f32_dpp $0, $8, $8 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $0, $12, $12 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $1, $9, $9 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $1, $13, $13 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $2, $10, $10 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $2, $14, $14 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $3, $11, $11 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $3, $15, $15 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $4, $0, $0 row_xmask:4 row_mask:0xf bank_mask:0x5\n"
            "v_add_f32_dpp $4, $2, $2 row_xmask:4 row_mask:0xf bank_mask:0xa\n"
            "v_add_f32_dpp $5, $1, $1 row_xmask:4 row_mask:0xf bank_mask:0x5\n"
            "v_add_f32_dpp $5, $3, $3 row_xmask:4 row_mask:0xf bank_mask:0xa\n"
            "v_bfi_b32 $7, $16, $4, $5\n"
            "v_bfi_b32 $6, $16, $5, $4\n"
            "v_add_f32_dpp $7, $7, $6 row_xmask:2 row_mask:0xf bank_mask:0xf\n"
            "v_add_f32_dpp $7, $7, $7 row_xmask:1 row_mask:0xf bank_mask:0xf\n"
        )
        + "v_mov_b32_dpp $0, $7 row_share:0 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $1, $7 row_share:2 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $2, $7 row_share:4 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $3, $7 row_share:6 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $4, $7 row_share:8 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $5, $7 row_share:10 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $6, $7 row_share:12 row_mask:0xf bank_mask:0xf\n"
        "v_mov_b32_dpp $7, $7 row_share:14 row_mask:0xf bank_mask:0xf\n",
        "=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v",
        [p, m],
        dtype=gl.float32,
        is_pure=True,
        pack=8,
    )


@gluon.jit
def _kda_rowsum8_t(p, m):
    """Row sums over the 16 K lanes: lane j's last row slot holds row j >> 1."""
    return gl.inline_asm_elementwise(
        (
            "v_add_f32_dpp $0, $8, $8 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $0, $12, $12 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $1, $9, $9 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $1, $13, $13 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $2, $10, $10 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $2, $14, $14 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $3, $11, $11 row_xmask:8 row_mask:0xf bank_mask:0x3\n"
            "v_add_f32_dpp $3, $15, $15 row_xmask:8 row_mask:0xf bank_mask:0xc\n"
            "v_add_f32_dpp $4, $0, $0 row_xmask:4 row_mask:0xf bank_mask:0x5\n"
            "v_add_f32_dpp $4, $2, $2 row_xmask:4 row_mask:0xf bank_mask:0xa\n"
            "v_add_f32_dpp $5, $1, $1 row_xmask:4 row_mask:0xf bank_mask:0x5\n"
            "v_add_f32_dpp $5, $3, $3 row_xmask:4 row_mask:0xf bank_mask:0xa\n"
            "v_bfi_b32 $7, $16, $4, $5\n"
            "v_bfi_b32 $6, $16, $5, $4\n"
            "v_add_f32_dpp $7, $7, $6 row_xmask:2 row_mask:0xf bank_mask:0xf\n"
            "v_add_f32_dpp $7, $7, $7 row_xmask:1 row_mask:0xf bank_mask:0xf\n"
        ),
        "=&v,=&v,=&v,=&v,=&v,=&v,=&v,=&v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v,v",
        [p, m],
        dtype=gl.float32,
        is_pure=True,
        pack=8,
    )


@gluon.jit
def _kda_dot8m(
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    m,
    MODE: gl.constexpr,
):
    p = S0 * x0[None, :]
    p = gl.fma(S1, x1[None, :], p)
    p = gl.fma(S2, x2[None, :], p)
    p = gl.fma(S3, x3[None, :], p)
    p = gl.fma(S4, x4[None, :], p)
    p = gl.fma(S5, x5[None, :], p)
    p = gl.fma(S6, x6[None, :], p)
    p = gl.fma(S7, x7[None, :], p)
    if MODE == 0:
        return _kda_lanesum16(p)
    elif MODE == 1:
        return _kda_rowsum8(p, m)
    else:
        return _kda_rowsum8_t(p, m)


@gluon.jit
def _kda_dot8(S0, S1, S2, S3, S4, S5, S6, S7, x0, x1, x2, x3, x4, x5, x6, x7):
    """Row sums of S * x: an fma chain over the blocks, then across lanes."""
    p = S0 * x0[None, :]
    p = gl.fma(S1, x1[None, :], p)
    p = gl.fma(S2, x2[None, :], p)
    p = gl.fma(S3, x3[None, :], p)
    p = gl.fma(S4, x4[None, :], p)
    p = gl.fma(S5, x5[None, :], p)
    p = gl.fma(S6, x6[None, :], p)
    p = gl.fma(S7, x7[None, :], p)
    return gl.sum(p, axis=1)


@gluon.jit
def _kda_upd8(S0, S1, S2, S3, S4, S5, S6, S7, u, x0, x1, x2, x3, x4, x5, x6, x7):
    """S += u x^T, block by block."""
    if len(u.shape) == 1:
        u = u[:, None]
    return (
        gl.fma(u, x0[None, :], S0),
        gl.fma(u, x1[None, :], S1),
        gl.fma(u, x2[None, :], S2),
        gl.fma(u, x3[None, :], S3),
        gl.fma(u, x4[None, :], S4),
        gl.fma(u, x5[None, :], S5),
        gl.fma(u, x6[None, :], S6),
        gl.fma(u, x7[None, :], S7),
    )


@gluon.jit
def _kda_tok_load(
    q,
    k,
    v,
    a,
    b,
    bos,
    t0,
    T,
    i_h,
    i_hv,
    o_t,
    o_kt,
    o_vt,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    stride_a_token: gl.constexpr,
    stride_b_token: gl.constexpr,
    stride_b_head: gl.constexpr,
):
    """Raw token tiles of chunk t0 (q, k, v, a, b), masked past T."""
    m_t = (t0 + o_t) < T
    tok = bos + t0 + o_t
    m_tk = m_t[:, None] & (o_kt < K)[None, :]
    m_tv = m_t[:, None] & (o_vt < V)[None, :]
    r_q = gl.load(
        q + (tok * H + i_h)[:, None] * K + o_kt[None, :], mask=m_tk, other=0.0
    )
    r_k = gl.load(
        k + (tok * H + i_h)[:, None] * K + o_kt[None, :], mask=m_tk, other=0.0
    )
    r_v = gl.load(
        v + (tok * HV + i_hv)[:, None] * V + o_vt[None, :], mask=m_tv, other=0.0
    )
    r_x = gl.load(
        a + tok[:, None] * stride_a_token + i_hv * K + o_kt[None, :],
        mask=m_tk,
        other=0.0,
    )
    if b.dtype.element_ty == gl.float16:
        # A predicated half load is miscompiled on this toolchain.
        # Clamp unused rows to a valid token and use an unmasked load.
        r_b = gl.load(
            b + gl.where(m_t, tok, bos) * stride_b_token + i_hv * stride_b_head
        )
    else:
        r_b = gl.load(
            b + tok * stride_b_token + i_hv * stride_b_head, mask=m_t, other=0.0
        )
    return r_q, r_k, r_v, r_x, r_b


@gluon.jit
def _kda_tok_math(
    r_q,
    r_k,
    r_v,
    r_x,
    r_b,
    b_dtb,
    b_Alog,
    scale,
    LOWER_BOUND,
    s_eg,
    s_kn,
    s_qn,
    s_v,
    s_beta,
    BT: gl.constexpr,
    V: gl.constexpr,
    LT: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
):
    """Gates, beta and normalised q/k of one chunk into the token LDS; returns (k, g) for the appends."""
    b_q = r_q.to(gl.float32)
    b_k = r_k.to(gl.float32)
    b_v = r_v.to(gl.float32)
    x = r_x.to(gl.float32) + b_dtb[None, :]
    b_b = r_b.to(gl.float32)
    b_beta = 1.0 / (1.0 + gl.exp(-b_b))
    if USE_LOWER_BOUND:
        b_g = LOWER_BOUND * (1.0 / (1.0 + gl.exp(-(gl.exp(b_Alog) * x))))
    else:
        b_g = -gl.exp(b_Alog) * softplus(x)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q * gl.rsqrt(gl.sum(b_q * b_q, axis=1) + 1e-6)[:, None]
        b_k = b_k * gl.rsqrt(gl.sum(b_k * b_k, axis=1) + 1e-6)[:, None]
    b_q = b_q * scale
    zv = gl.zeros([BT, V], gl.float32, LT)
    s_eg.store(gl.exp(b_g))
    s_kn.store(b_k)
    s_qn.store(b_q)
    s_v.store(b_v)
    s_beta.store(b_beta[:, None] + zv)
    return b_k, b_g


@gluon.jit
def _kda_tok_append(
    buf_k,
    buf_g,
    b_k,
    b_g,
    slot,
    base,
    t0,
    T,
    i_hv,
    o_t,
    o_kt,
    stride_bufk_slot: gl.constexpr,
    stride_bufk_pos: gl.constexpr,
    stride_bufk_hv: gl.constexpr,
    stride_bufg_slot: gl.constexpr,
    stride_bufg_pos: gl.constexpr,
    stride_bufg_hv: gl.constexpr,
    K: gl.constexpr,
):
    m_tk = ((t0 + o_t) < T)[:, None] & (o_kt < K)[None, :]
    pos = base + t0 + o_t
    gl.store(
        buf_k
        + slot * stride_bufk_slot
        + pos[:, None] * stride_bufk_pos
        + i_hv * stride_bufk_hv
        + o_kt[None, :],
        b_k.to(buf_k.dtype.element_ty),
        mask=m_tk,
    )
    gl.store(
        buf_g
        + slot * stride_bufg_slot
        + pos[:, None] * stride_bufg_pos
        + i_hv * stride_bufg_hv
        + o_kt[None, :],
        b_g.to(buf_g.dtype.element_ty),
        mask=m_tk,
    )


@gluon.jit
def _kda_tok_loop(
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    s_eg,
    s_kn,
    s_qn,
    s_v,
    s_beta,
    o,
    buf_u,
    bos,
    t0,
    n_t,
    slot,
    base,
    i_hv,
    o_v,
    o_k0,
    m_k0,
    m_b1,
    o_vo,
    m_o,
    stride_bufu_slot: gl.constexpr,
    stride_bufu_pos: gl.constexpr,
    stride_bufu_hv: gl.constexpr,
    HV: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    KB: gl.constexpr,
    NB: gl.constexpr,
    LK: gl.constexpr,
    LV: gl.constexpr,
    LK2: gl.constexpr,
    SH1: gl.constexpr,
    SH2: gl.constexpr,
):
    """The serial token recurrence over n_t tokens of one chunk (token LDS already published)."""
    for i_t in range(n_t):
        g0, g1, g2, g3, g4, g5, g6, g7 = _kda_split8(
            s_eg.reinterpret(gl.float32, [BT, 2, KB, 4], SH2).index(i_t).load(LK2), LK
        )
        k0, k1, k2, k3, k4, k5, k6, k7 = _kda_split8(
            s_kn.reinterpret(gl.float32, [BT, 2, KB, 4], SH2).index(i_t).load(LK2), LK
        )
        S0 = S0 * g0[None, :]
        S1 = S1 * g1[None, :]
        S2 = S2 * g2[None, :]
        S3 = S3 * g3[None, :]
        S4 = S4 * g4[None, :]
        S5 = S5 * g5[None, :]
        S6 = S6 * g6[None, :]
        S7 = S7 * g7[None, :]
        b_hk = _kda_dot8m(
            S0, S1, S2, S3, S4, S5, S6, S7, k0, k1, k2, k3, k4, k5, k6, k7, m_b1, 1
        )
        b_vn = (
            s_v.reinterpret(gl.float32, [BT, V], SH1).index(i_t).load(LV)[:, None]
            - b_hk
        ) * s_beta.reinterpret(gl.float32, [BT, V], SH1).index(i_t).load(LV)[:, None]
        S0, S1, S2, S3, S4, S5, S6, S7 = _kda_upd8(
            S0, S1, S2, S3, S4, S5, S6, S7, b_vn, k0, k1, k2, k3, k4, k5, k6, k7
        )
        # o = S_t q on the updated state: one contraction live at a time.
        q0, q1, q2, q3, q4, q5, q6, q7 = _kda_split8(
            s_qn.reinterpret(gl.float32, [BT, 2, KB, 4], SH2).index(i_t).load(LK2), LK
        )
        tt = bos + t0 + i_t
        if o.dtype.element_ty == gl.float32:
            # Use a compiler reduction for full-width output stores.
            b_o = _kda_dot8(
                S0, S1, S2, S3, S4, S5, S6, S7, q0, q1, q2, q3, q4, q5, q6, q7
            )
            gl.store(o + (tt * HV + i_hv) * V + o_v, b_o)
        else:
            b_o = _kda_dot8m(
                S0, S1, S2, S3, S4, S5, S6, S7, q0, q1, q2, q3, q4, q5, q6, q7, m_b1, 2
            )
            gl.store(
                o + (tt * HV + i_hv) * V + o_vo, b_o.to(o.dtype.element_ty), mask=m_o
            )
        gl.store(
            buf_u
            + slot * stride_bufu_slot
            + (base + t0 + i_t) * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v[:, None]
            + o_k0[None, :],
            b_vn.to(buf_u.dtype.element_ty),
            mask=m_k0,
        )
    return S0, S1, S2, S3, S4, S5, S6, S7


@gluon.jit(repr=_replayssm_repr)
def _replayssm_kda_fwd_cta_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    LOWER_BOUND,
    T_MAX,
    CAP,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BH: gl.constexpr,
    BT: gl.constexpr,
    stride_ckpt_slot: gl.constexpr,
    stride_bufk_slot: gl.constexpr,
    stride_bufk_pos: gl.constexpr,
    stride_bufk_hv: gl.constexpr,
    stride_bufu_slot: gl.constexpr,
    stride_bufu_pos: gl.constexpr,
    stride_bufu_hv: gl.constexpr,
    stride_bufg_slot: gl.constexpr,
    stride_bufg_pos: gl.constexpr,
    stride_bufg_hv: gl.constexpr,
    stride_a_token: gl.constexpr,
    stride_b_token: gl.constexpr,
    stride_b_head: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
    TPW_K: gl.constexpr,
    WARPS: gl.constexpr,
    FUSED: gl.constexpr,
):
    TPW_V: gl.constexpr = 32 // TPW_K
    RPT: gl.constexpr = V // (TPW_V * WARPS)
    # State [V, K]: lane owns RPT rows x K/TPW_K contiguous columns.
    # Held as 8 column blocks S_0..S_7 (see L3 below); a lane owns RPT rows
    # of one column of each block, so every contraction is an fma chain over
    # the blocks (row pairs pack) followed by a 16-lane DPP-row reduction.
    LS: gl.constexpr = gl.BlockedLayout([RPT, 1], [TPW_V, TPW_K], [WARPS, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(0, LS)
    LV: gl.constexpr = gl.SliceLayout(1, LS)
    KB: gl.constexpr = TPW_K
    NB: gl.constexpr = K // TPW_K
    # `_kda_lanesum16` needs the K lanes to be one 16-lane DPP row, and the
    # block helpers are written for 8 blocks.
    gl.static_assert(TPW_K == 16)
    gl.static_assert(NB == 8)
    # Block c is the interleaved column set {j * NB + c}: lane j's NB columns
    # are contiguous in memory and LDS, so they load as vectors and split.
    L3: gl.constexpr = gl.BlockedLayout(
        [RPT, 1, NB], [TPW_V, TPW_K, 1], [WARPS, 1, 1], [2, 1, 0]
    )
    # State column map: the 4-D view [V, 2, KB, 4] puts K column 64h + 4j + e in lane j,
    # so each b128 of the ckpt load / flush store is contiguous across the 16 K lanes.
    L4: gl.constexpr = gl.BlockedLayout(
        [RPT, 2, 1, 4], [TPW_V, 1, TPW_K, 1], [WARPS, 1, 1, 1], [3, 1, 2, 0]
    )
    LK2: gl.constexpr = gl.SliceLayout(0, L4)
    SH2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    # Record tile [BH, K]: 2 rows x 4 contiguous columns per thread, 8 lanes down
    # a column, no replication -- vector loads, and the row cumsum is in-warp.
    # (One column per thread made 32 scalar loads per thread, each element twice.)
    LR: gl.constexpr = gl.BlockedLayout([2, 4], [8, 4], [1, WARPS], [1, 0])
    # Token tile [BT, K]: rows split over warps, row reductions in-warp.
    LT: gl.constexpr = gl.BlockedLayout([1, K // 32], [1, 32], [WARPS, 1], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SH1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    i_nh = gl.program_id(0)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    # All metadata loads are issued before the exits: slot_idx[i_n] is in
    # bounds for every program and write_pos[max(slot, 0)] is too.
    bos = gl.load(cu_seqlens + i_n).to(gl.int64)
    eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
    slot = gl.load(slot_idx + i_n).to(gl.int64)
    wp = gl.load(write_pos + gl.maximum(slot, 0)).to(gl.int32)
    T = (eos - bos).to(gl.int32)
    if T == 0:
        return
    if slot < 0:
        return
    o_sp = gl.arange(0, 32 * WARPS, layout=gl.BlockedLayout([1], [32], [WARPS], [0]))
    gl.inline_asm_elementwise(
        "s_setprio 2\nv_mov_b32 $0, $1",
        "=v,v",
        [o_sp],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    # Clamp the prefill sentinel (see `_replayssm_kda_fwd_kernel`).
    h = gl.maximum(wp, 0)
    do_flush = h + 2 * T_MAX > CAP
    base = gl.where(do_flush, 0, h).to(gl.int64)

    # ---- 0. every load that does not need the state, then the ckpt --------
    # Records and chunk-0 token tiles are issued first and the 64 KB ckpt
    # last, so the record cumsum and the token-side math run while the
    # ckpt is in flight (loads retire in order).  One barrier then
    # publishes both LDS sets.
    o_hr = gl.arange(0, BH, layout=gl.SliceLayout(1, LR))
    o_kr = gl.arange(0, K, layout=gl.SliceLayout(0, LR))
    h_ld = h
    m_r = (o_hr < h_ld)[:, None] & (o_kr < K)[None, :]
    t_g = gl.load(
        buf_g
        + slot * stride_bufg_slot
        + o_hr[:, None] * stride_bufg_pos
        + i_hv * stride_bufg_hv
        + o_kr[None, :],
        mask=m_r,
        other=0.0,
    ).to(gl.float32)
    t_k = gl.load(
        buf_k
        + slot * stride_bufk_slot
        + o_hr[:, None] * stride_bufk_pos
        + i_hv * stride_bufk_hv
        + o_kr[None, :],
        mask=m_r,
        other=0.0,
    ).to(gl.float32)
    # The u records need no cumsum: 8 contiguous values per thread, no replication.
    LU: gl.constexpr = gl.BlockedLayout(
        [1, 8], [32 * 8 // V, V // 8], [WARPS, 1], [1, 0]
    )
    o_hu = gl.arange(0, BH, layout=gl.SliceLayout(1, LU))
    o_vr = gl.arange(0, V, layout=gl.SliceLayout(0, LU))
    t_u = gl.load(
        buf_u
        + slot * stride_bufu_slot
        + o_hu[:, None] * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_vr[None, :],
        mask=(o_hu < h_ld)[:, None] & (o_vr < V)[None, :],
        other=0.0,
    )
    o_t = gl.arange(0, BT, layout=gl.SliceLayout(1, LT))
    o_kt = gl.arange(0, K, layout=gl.SliceLayout(0, LT))
    o_vt = gl.arange(0, V, layout=gl.SliceLayout(0, LT))
    b_Alog = gl.load(A_log + i_hv).to(gl.float32)
    b_dtb = gl.load(dt_bias + i_hv * K + o_kt).to(gl.float32)
    if FUSED:
        r_q, r_k, r_v, r_x, r_b = _kda_tok_load(
            q,
            k,
            v,
            a,
            b,
            bos,
            0,
            T,
            i_h,
            i_hv,
            o_t,
            o_kt,
            o_vt,
            H,
            HV,
            K,
            V,
            stride_a_token,
            stride_b_token,
            stride_b_head,
        )
    o_v4 = gl.arange(
        0, V, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L4)))
    )
    o_h4 = gl.arange(
        0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L4)))
    )
    o_j4 = gl.arange(
        0, KB, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L4)))
    )
    o_e4 = gl.arange(
        0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L4)))
    )
    # Rows j >= h load as zeros: g = 0 leaves the sums alone, u = 0 drops them.
    # Gates are log-decays (<= 0): every weight exp(C - C_j) is in (0, 1].
    b_ctot = gl.sum(t_g, axis=0)
    t_kw = t_k * gl.exp(b_ctot[None, :] - gl.cumsum(t_g, axis=0))
    s_kw = gl.allocate_shared_memory(gl.float32, [BH, K], SH, t_kw)
    s_u = gl.allocate_shared_memory(t_u.dtype, [BH, V], SH, t_u)
    s_dec = gl.allocate_shared_memory(gl.float32, [K], SH1, gl.exp(b_ctot))
    p_ckpt_hv = (
        ckpt
        + slot * stride_ckpt_slot
        + i_hv * V * K
        + o_v4[:, None, None, None] * K
        + o_h4[None, :, None, None] * (K // 2)
        + o_j4[None, None, :, None] * 4
        + o_e4[None, None, None, :]
    )
    b_ck = gl.load(p_ckpt_hv)
    o_v = gl.arange(0, V, layout=LV)
    o_k0 = gl.arange(0, KB, layout=LK) * 0
    m_k0 = (gl.arange(0, KB, layout=LK) == 0)[None, :] & (o_v >= 0)[:, None]
    o_j = gl.arange(0, KB, layout=LK)
    m_b1 = gl.where((o_j // 2) % 2 == 1, -1, 0)[None, :] + gl.zeros(
        [V, KB], gl.int32, LS
    )
    # After a transpose-reduce, row slot RPT-1 of lane j holds row (j >> 1) of its 8-row group.
    o_vo = (o_v - o_v % RPT)[:, None] + (o_j // 2)[None, :]
    m_o = ((o_v % RPT) == RPT - 1)[:, None] & ((o_j % 2) == 0)[None, :]

    s_eg = gl.allocate_shared_memory(gl.float32, [BT, K], SH)
    s_kn = gl.allocate_shared_memory(gl.float32, [BT, K], SH)
    s_qn = gl.allocate_shared_memory(gl.float32, [BT, K], SH)
    s_v = gl.allocate_shared_memory(gl.float32, [BT, V], SH)
    s_beta = gl.allocate_shared_memory(gl.float32, [BT, V], SH)
    if FUSED:
        a_k, a_g = _kda_tok_math(
            r_q,
            r_k,
            r_v,
            r_x,
            r_b,
            b_dtb,
            b_Alog,
            scale,
            LOWER_BOUND,
            s_eg,
            s_kn,
            s_qn,
            s_v,
            s_beta,
            BT,
            V,
            LT,
            USE_QK_L2NORM_IN_KERNEL,
            USE_LOWER_BOUND,
        )
    gl.barrier()
    gl.inline_asm_elementwise(
        "s_setprio 0\nv_mov_b32 $0, $1",
        "=v,v",
        [o_sp],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )
    if FUSED:
        # Appends.  Every thread's replay record loads were consumed into LDS
        # before the barrier, so overwriting rows a flush replayed is safe.
        _kda_tok_append(
            buf_k,
            buf_g,
            a_k,
            a_g,
            slot,
            base,
            0,
            T,
            i_hv,
            o_t,
            o_kt,
            stride_bufk_slot,
            stride_bufk_pos,
            stride_bufk_hv,
            stride_bufg_slot,
            stride_bufg_pos,
            stride_bufg_hv,
            K,
        )
    # ---- 1. rebuild S_h -------------------------------------------------
    S0, S1, S2, S3, S4, S5, S6, S7 = _kda_split8(b_ck.to(gl.float32), LS)
    d0, d1, d2, d3, d4, d5, d6, d7 = _kda_split8(
        s_dec.reinterpret(gl.float32, [2, KB, 4], SH2).load(LK2), LK
    )
    S0 = S0 * d0[None, :]
    S1 = S1 * d1[None, :]
    S2 = S2 * d2[None, :]
    S3 = S3 * d3[None, :]
    S4 = S4 * d4[None, :]
    S5 = S5 * d5[None, :]
    S6 = S6 * d6[None, :]
    S7 = S7 * d7[None, :]
    n_rep = gl.minimum(h, BH)
    for j in range(n_rep):
        w0, w1, w2, w3, w4, w5, w6, w7 = _kda_split8(
            s_kw.reinterpret(gl.float32, [BH, 2, KB, 4], SH2).index(j).load(LK2), LK
        )
        b_uj = s_u.reinterpret(t_u.dtype, [BH, V], SH1).index(j).load(LV).to(gl.float32)
        S0, S1, S2, S3, S4, S5, S6, S7 = _kda_upd8(
            S0, S1, S2, S3, S4, S5, S6, S7, b_uj, w0, w1, w2, w3, w4, w5, w6, w7
        )
    if do_flush:
        gl.store(
            p_ckpt_hv,
            _kda_cm_out(_kda_join8(S0, S1, S2, S3, S4, S5, S6, S7, L3), L4).to(
                ckpt.dtype.element_ty
            ),
        )

    # ---- 2. this step's tokens, BT at a time --------------------------------
    t_lo = 0
    if FUSED:
        S0, S1, S2, S3, S4, S5, S6, S7 = _kda_tok_loop(
            S0,
            S1,
            S2,
            S3,
            S4,
            S5,
            S6,
            S7,
            s_eg,
            s_kn,
            s_qn,
            s_v,
            s_beta,
            o,
            buf_u,
            bos,
            0,
            gl.minimum(T, BT),
            slot,
            base,
            i_hv,
            o_v,
            o_k0,
            m_k0,
            m_b1,
            o_vo,
            m_o,
            stride_bufu_slot,
            stride_bufu_pos,
            stride_bufu_hv,
            HV,
            V,
            BT,
            KB,
            NB,
            LK,
            LV,
            LK2,
            SH1,
            SH2,
        )
        t_lo = BT
    for t0 in range(t_lo, T, BT):
        # Token-side math needs no state: once per CTA, then shared via LDS.
        r_q, r_k, r_v, r_x, r_b = _kda_tok_load(
            q,
            k,
            v,
            a,
            b,
            bos,
            t0,
            T,
            i_h,
            i_hv,
            o_t,
            o_kt,
            o_vt,
            H,
            HV,
            K,
            V,
            stride_a_token,
            stride_b_token,
            stride_b_head,
        )
        gl.barrier()  # the previous chunk is done reading the token LDS
        a_k, a_g = _kda_tok_math(
            r_q,
            r_k,
            r_v,
            r_x,
            r_b,
            b_dtb,
            b_Alog,
            scale,
            LOWER_BOUND,
            s_eg,
            s_kn,
            s_qn,
            s_v,
            s_beta,
            BT,
            V,
            LT,
            USE_QK_L2NORM_IN_KERNEL,
            USE_LOWER_BOUND,
        )
        _kda_tok_append(
            buf_k,
            buf_g,
            a_k,
            a_g,
            slot,
            base,
            t0,
            T,
            i_hv,
            o_t,
            o_kt,
            stride_bufk_slot,
            stride_bufk_pos,
            stride_bufk_hv,
            stride_bufg_slot,
            stride_bufg_pos,
            stride_bufg_hv,
            K,
        )
        gl.barrier()
        S0, S1, S2, S3, S4, S5, S6, S7 = _kda_tok_loop(
            S0,
            S1,
            S2,
            S3,
            S4,
            S5,
            S6,
            S7,
            s_eg,
            s_kn,
            s_qn,
            s_v,
            s_beta,
            o,
            buf_u,
            bos,
            t0,
            gl.minimum(T - t0, BT),
            slot,
            base,
            i_hv,
            o_v,
            o_k0,
            m_k0,
            m_b1,
            o_vo,
            m_o,
            stride_bufu_slot,
            stride_bufu_pos,
            stride_bufu_hv,
            HV,
            V,
            BT,
            KB,
            NB,
            LK,
            LV,
            LK2,
            SH1,
            SH2,
        )
