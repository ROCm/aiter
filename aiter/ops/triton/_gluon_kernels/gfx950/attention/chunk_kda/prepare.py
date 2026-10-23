# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

LOG2E = gl.constexpr(1.4426950408889634)


def _get_config(overrides=None):
    cfg_dir = resolve_config_dir("attention", "CHUNK_KDA", backend="gluon")
    config = load_config_json(f"{cfg_dir}/DEFAULT.json")["prepare"]
    return {**config, **(overrides or {})}


_repr = make_kernel_repr(
    "chunk_kda_prepare_kernel", ["BT", "K", "V", "NUM_WARPS", "NC"]
)


@gluon.jit(repr=_repr)
def chunk_kda_prepare_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    A_log_ptr,
    dt_bias_ptr,
    qg_ptr,
    w_ptr,
    u_ptr,
    kg_t_ptr,
    aqk_ptr,
    decay_ptr,
    cu_seqlens_ptr,
    chunk_indices_ptr,
    lower_bound,
    stride_q_token: gl.constexpr,
    stride_k_token: gl.constexpr,
    stride_v_token: gl.constexpr,
    stride_g_token: gl.constexpr,
    stride_beta_token: gl.constexpr,
    scale: gl.constexpr,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NC: gl.constexpr,
):
    """One (chunk, head) per program: l2norm, gate cumsum, the intra-chunk products and
    the (I + L)^-1 solve, written as the walk's workspace (qg, w, u, aqk, kg_t, decay).
    """
    gl.static_assert(
        BT == 64 and K == 128 and V == 128, "specialised to BT=64, K=V=128"
    )
    gl.static_assert(NUM_WARPS == 2 or NUM_WARPS == 4)
    BC: gl.constexpr = 16  # band: 2^-lc stays finite over 16 tokens
    NB: gl.constexpr = BT // BC

    MMA: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1],
    )
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MMA, 8)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MMA, 8)
    MB: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1, 1],
    )
    A3: gl.constexpr = gl.DotOperandLayout(0, MB, 8)
    B3: gl.constexpr = gl.DotOperandLayout(1, MB, 8)
    MF: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[NUM_WARPS, 1, 1],
    )
    AF: gl.constexpr = gl.DotOperandLayout(0, MF, 1)
    BF: gl.constexpr = gl.DotOperandLayout(1, MF, 1)
    # [band, row, channel], a band per warp: cumsum and l2norm stay in-warp
    BL: gl.constexpr = gl.BlockedLayout(
        [1, 4, 8], [1, 4, 16], [NUM_WARPS, 1, 1], [2, 1, 0]
    )
    # every band in each lane: the inter-band scan is a register reduction
    TOT: gl.constexpr = gl.BlockedLayout([NB, 2], [1, 64], [NUM_WARPS, 1], [1, 0])
    # [K, BT], 8 tokens per lane: 16 B kg_t stores
    KT: gl.constexpr = gl.BlockedLayout([1, 8], [16, 4], [NUM_WARPS, 1], [1, 0])
    FLAT: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    FLAT3: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    # mha.py's swizzles for 128- and 64-wide bf16 rows
    SL_K: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    SL_A: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, [1, 0])

    i_t = gl.program_id(0)
    i_h = gl.program_id(1)
    i_n = gl.load(chunk_indices_ptr + 2 * i_t).to(gl.int32)
    bos = gl.load(cu_seqlens_ptr + i_n).to(gl.int32)
    t0 = bos + gl.load(chunk_indices_ptr + 2 * i_t + 1).to(gl.int32) * BT
    n = gl.minimum(gl.load(cu_seqlens_ptr + i_n + 1).to(gl.int32) - t0, BT)
    tok = t0.to(gl.int64)

    band = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, BL)))[:, None, None]
    row = (
        band * BC
        + gl.arange(0, BC, gl.SliceLayout(0, gl.SliceLayout(2, BL)))[None, :, None]
    )
    chan = gl.arange(0, K, gl.SliceLayout(0, gl.SliceLayout(1, BL)))
    # gk = lb log2(e) sigmoid(a (z + dt_bias)) = lb log2(e) / (1 + 2^(z c1 + c0))
    c1 = gl.exp(gl.load(A_log_ptr + i_h).to(gl.float32)) * -LOG2E
    c0 = c1 * gl.amd.cdna4.buffer_load(dt_bias_ptr + i_h * K, chan).to(gl.float32)
    gate_c = lower_bound * LOG2E
    b_p = beta_ptr + tok * stride_beta_token + i_h
    ch = (i_t * H + i_h).to(gl.int64)
    ws_q = tok * (H * K) + i_h * K
    m_row = row < n

    # every input in flight before any math
    zr = gl.amd.cdna4.buffer_load(
        g_ptr + tok * stride_g_token + i_h * K,
        row * stride_g_token + chan[None, None, :],
        mask=m_row,
        other=0.0,
    )
    kr = gl.amd.cdna4.buffer_load(
        k_ptr + tok * stride_k_token + i_h * K,
        row * stride_k_token + chan[None, None, :],
        mask=m_row,
        other=0.0,
    )
    qr = gl.amd.cdna4.buffer_load(
        q_ptr + tok * stride_q_token + i_h * K,
        row * stride_q_token + chan[None, None, :],
        mask=m_row,
        other=0.0,
    )
    vv = gl.amd.cdna4.buffer_load(
        v_ptr + tok * stride_v_token + i_h * V,
        row * stride_v_token + chan[None, None, :],
        mask=m_row,
        other=0.0,
    )

    z = zr.to(gl.float32)
    # v_rcp_f32: fdiv lowers to the full IEEE division sequence even when relaxed
    gk = gate_c * gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        [1.0 + gl.exp2(z * c1 + c0[None, None, :])],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    gk = gl.where(m_row, gk, 0.0)
    lc = tl.cumsum(gk, 1)
    tot = gl.sum(gk, axis=1)
    g_last = gl.sum(tot, axis=0)
    gl.amd.cdna4.buffer_store(gl.exp2(g_last), decay_ptr + ch * K, chan)

    # G = e + lc: the band's pivot e plus the in-band cumsum lc
    tot_s = gl.allocate_shared_memory(gl.float32, [NB, K], FLAT, tot)
    t2 = tot_s.load(TOT)
    e_s = gl.allocate_shared_memory(gl.float32, [NB, K], FLAT, tl.cumsum(t2, 0) - t2)
    e_b = e_s.load(TOT)
    e = e_s.load(gl.SliceLayout(1, BL))

    # bridge[J, b] = 2^min(e_b - e_J, 0) moves block row b to band J's pivot
    br_s = gl.allocate_shared_memory(gl.float32, [NB, NB, K], FLAT)
    for J in gl.static_range(NB):
        e_j = e_s.reshape([NB * K]).slice(J * K, K).load(gl.SliceLayout(0, TOT))
        br_s.index(J).store(gl.exp2(gl.minimum(e_b - e_j[None, :], 0.0)))

    kf = kr.to(gl.float32)
    kf = kf * gl.rsqrt(gl.sum(kf * kf, axis=2) + 1e-6)[:, :, None]

    # kg^T through a [K, BT] tile, read back as 16 B token runs
    t_s = gl.allocate_shared_memory(kg_t_ptr.dtype.element_ty, [K, BT], SL_A)
    kg = (kf * gl.exp2((g_last[None, :] - e)[:, None, :] - lc)).to(
        kg_t_ptr.dtype.element_ty
    )
    t_s.permute([1, 0]).store(gl.reshape(kg, [BT, K]))
    kt = t_s.load(KT)
    kt_r = gl.arange(0, K, gl.SliceLayout(1, KT))[:, None]
    kt_c = gl.arange(0, BT, gl.SliceLayout(0, KT))[None, :]
    gl.amd.cdna4.buffer_store(kt, kg_t_ptr + ch * (K * BT), kt_r * BT + kt_c)

    ki_s = gl.allocate_shared_memory(gl.bfloat16, [BT, K], SL_K)
    ki_s.reshape([NB, BC, K]).store((kf * gl.exp2(-lc)).to(gl.bfloat16))
    # 2^G = 2^e 2^lc; 2^e flushing to 0 only drops 2^G < 2^-126
    el = gl.exp2(lc)
    ee = gl.exp2(e)[:, None, :]
    kn_s = gl.allocate_shared_memory(gl.bfloat16, [BT, K], SL_K)
    kn_s.reshape([NB, BC, K]).store((kf * el).to(gl.bfloat16))
    # read kn back now so qn_s reuses kn_s: LDS < 80 KB, two workgroups per CU
    kn = kn_s.reshape([NB, BC, K]).load(A3)
    kbv = (kf * el * ee).to(gl.bfloat16)

    qf = qr.to(gl.float32)
    qf = qf * gl.rsqrt(gl.sum(qf * qf, axis=2) + 1e-6)[:, :, None]
    qn_s = gl.allocate_shared_memory(gl.bfloat16, [BT, K], SL_K)
    qn_s.reshape([NB, BC, K]).store((qf * (el * scale)).to(gl.bfloat16))
    qn = qn_s.reshape([NB, BC, K]).load(A3)
    gl.amd.cdna4.buffer_store(
        (qf * el * ee).to(qg_ptr.dtype.element_ty),
        qg_ptr + ws_q,
        row * (H * K) + chan[None, None, :],
        mask=m_row,
    )

    # L and Aqk by column band J, block rows batched over warps
    b_m = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, MB)))[:, None, None]
    r_m = (
        b_m * BC
        + gl.arange(0, BC, gl.SliceLayout(0, gl.SliceLayout(2, MB)))[None, :, None]
    )
    c_m = gl.arange(0, BC, gl.SliceLayout(0, gl.SliceLayout(1, MB)))[None, None, :]
    # sigmoid(beta) once per chunk, shared by L's rows and A_inv's columns
    b1 = gl.arange(0, BT, gl.BlockedLayout([1], [64], [NUM_WARPS], [0]))
    beta = gl.amd.cdna4.buffer_load(
        b_p, b1 * stride_beta_token, mask=b1 < n, other=0.0
    ).to(gl.float32)
    beta = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1",
        "=v,v",
        [1.0 + gl.exp2(beta * -LOG2E)],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    beta_s = gl.allocate_shared_memory(
        gl.float32, [BT], gl.SwizzledSharedLayout(1, 1, 1, [0]), beta
    )
    beta = beta_s.reshape([NB, BC]).load(gl.SliceLayout(2, MB))[:, :, None]
    n0 = gl.zeros([NB, BC, BC], gl.float32, MB)
    n1 = gl.zeros([NB, BC, BC], gl.float32, MB)
    n2 = gl.zeros([NB, BC, BC], gl.float32, MB)
    n3 = gl.zeros([NB, BC, BC], gl.float32, MB)
    ws_a = tok * (H * BT) + i_h * BT
    for J in gl.static_range(NB):
        k_j = (
            ki_s.slice(J * BC, BC)
            .permute([1, 0])
            .load(gl.SliceLayout(0, B3))
            .to(gl.float32)
        )
        b_j = (
            k_j[None, :, :] * br_s.index(J).load(gl.SliceLayout(2, B3))[:, :, None]
        ).to(gl.bfloat16)
        akk = gl.amd.cdna4.mfma(kn, b_j, gl.zeros([NB, BC, BC], gl.float32, MB))
        aqk = gl.amd.cdna4.mfma(qn, b_j, gl.zeros([NB, BC, BC], gl.float32, MB))
        col = J * BC + c_m
        gl.amd.cdna4.buffer_store(
            gl.where(r_m >= col, aqk, 0.0).to(aqk_ptr.dtype.element_ty),
            aqk_ptr + ws_a + J * BC,
            r_m * (H * BT) + c_m,
            mask=r_m < n,
        )
        # N = -beta L by block diagonal: n_d[b] = N_{b, b - d}
        akk = gl.where(r_m > col, -beta * akk, 0.0)
        n0 = gl.where(b_m == J, akk, n0)
        n1 = gl.where(b_m == J + 1, akk, n1)
        n2 = gl.where(b_m == J + 2, akk, n2)
        n3 = gl.where(b_m == J + 3, akk, n3)
    ns = (n0, n1, n2, n3)

    # X = (I - N)^-1; diagonal blocks by doubling: (I + N)(I + N^2)(I + N^4)(I + N^8)
    b3 = gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, MF)))[:, None, None]
    r3 = gl.arange(0, BC, gl.SliceLayout(0, gl.SliceLayout(2, MF)))[None, :, None]
    c3 = gl.arange(0, BC, gl.SliceLayout(0, gl.SliceLayout(1, MF)))[None, None, :]
    x = gl.convert_layout(n0, MF, assert_trivial=True)
    p = gl.amd.cdna4.mfma(
        gl.convert_layout(x, AF),
        gl.convert_layout(x, BF),
        gl.zeros([NB, BC, BC], gl.float32, MF),
    )
    x = x + gl.where(r3 == c3, 1.0, 0.0)
    for i in gl.static_range(3):
        x = x + gl.amd.cdna4.mfma(
            gl.convert_layout(x, AF),
            gl.convert_layout(p, BF),
            gl.zeros([NB, BC, BC], gl.float32, MF),
        )
        if i < 2:
            p = gl.amd.cdna4.mfma(
                gl.convert_layout(p, AF),
                gl.convert_layout(p, BF),
                gl.zeros([NB, BC, BC], gl.float32, MF),
            )

    # off-diagonals: X_{b, b-d} = X_bb sum_s N_{b, b-s} X_{b-s, b-d}; zero batches shift
    x_s = gl.allocate_shared_memory(gl.float32, [NB, 2 * NB, BC, BC], FLAT3)
    bb = gl.zeros([NB, BC, BC], gl.int32, BF)
    bb = bb + gl.arange(0, NB, gl.SliceLayout(1, gl.SliceLayout(2, BF)))[:, None, None]
    for d in gl.static_range(NB - 1):
        x_s.index(d).slice(0, NB).store(gl.zeros([NB, BC, BC], gl.float32, MF))
    x_s.index(0).slice(NB, NB).store(x)
    for d in gl.static_range(1, NB):
        acc = gl.zeros([NB, BC, BC], gl.float32, MF)
        for s in gl.static_range(1, d + 1):
            acc = gl.amd.cdna4.mfma(
                gl.convert_layout(ns[s], AF),
                x_s.index(d - s).gather(bb + (NB - s), 0),
                acc,
            )
        x_s.index(d).slice(NB, NB).store(
            gl.amd.cdna4.mfma(
                gl.convert_layout(x, AF),
                gl.convert_layout(acc, BF),
                gl.zeros([NB, BC, BC], gl.float32, MF),
            )
        )

    # A_inv diag(beta) as one bf16 [BT, BT] operand
    inv_s = gl.allocate_shared_memory(gl.bfloat16, [BT, BT], SL_A)
    for J in gl.static_range(NB):
        a_j = gl.zeros([NB, BC, BC], gl.float32, MF)
        for d in gl.static_range(NB - J):
            a_j = gl.where(b3 == J + d, x_s.index(d).slice(NB, NB).load(MF), a_j)
        beta_j = beta_s.slice(J * BC, BC).load(gl.SliceLayout(0, gl.SliceLayout(1, MF)))
        a_j = (a_j * beta_j[None, None, :]).to(gl.bfloat16)
        inv_s.slice(J * BC, BC, dim=1).store(gl.reshape(a_j, [BT, BC]))

    # w, u by NC-column chunk, from transposed [K, BT] tiles (16 B token runs)
    inv = inv_s.load(A_OP)
    ro = gl.arange(0, BT, gl.SliceLayout(1, MMA))[:, None]
    wu_off = ro * (H * K) + gl.arange(0, NC, gl.SliceLayout(0, MMA))[None, :]
    ws_u = tok * (H * V) + i_h * V
    m_ro = ro < n
    kb_s = gl.allocate_shared_memory(gl.bfloat16, [K, BT], SL_A)
    kb_s.permute([1, 0]).store(gl.reshape(kbv, [BT, K]))
    for c in gl.static_range(K // NC):
        w = gl.amd.cdna4.mfma(
            inv,
            kb_s.slice(c * NC, NC, dim=0).permute([1, 0]).load(B_OP),
            gl.zeros([BT, NC], gl.float32, MMA),
        )
        gl.amd.cdna4.buffer_store(
            w.to(w_ptr.dtype.element_ty), w_ptr + ws_q + c * NC, wu_off, mask=m_ro
        )
    v_s = gl.allocate_shared_memory(v_ptr.dtype.element_ty, [V, BT], SL_A)
    v_s.permute([1, 0]).store(gl.reshape(vv, [BT, V]))
    for c in gl.static_range(V // NC):
        u = gl.amd.cdna4.mfma(
            inv,
            v_s.slice(c * NC, NC, dim=0).permute([1, 0]).load(B_OP),
            gl.zeros([BT, NC], gl.float32, MMA),
        )
        gl.amd.cdna4.buffer_store(
            u.to(u_ptr.dtype.element_ty), u_ptr + ws_u + c * NC, wu_off, mask=m_ro
        )
