# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import remap_xcd
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.device_info import get_num_xcds

SH_WS = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 16, [1, 0]))
SH_FLAT = gl.constexpr(gl.SwizzledSharedLayout(8, 1, 1, [1, 0]))
SH_PLAIN = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
SH_1D = gl.constexpr(gl.SwizzledSharedLayout(1, 1, 1, [0]))

_LN6 = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]]
CW_LK = gl.constexpr(
    gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4], [16, 0]],
        [[0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0]],
        [[4, 0], [8, 0], [0, 0]],
        [],
        [32, 128],
    )
)
CW_LCC = gl.constexpr(
    gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4]],
        [[0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [8, 0]],
        [[16, 0], [32, 0], [0, 0]],
        [],
        [64, 32],
    )
)
CW_LV = gl.constexpr(
    gl.DistributedLinearLayout(
        [[0, 1], [0, 2], [0, 4]], _LN6, [[8, 0], [16, 0], [0, 0]], [], [32, 64]
    )
)
CW_LG = gl.constexpr(
    gl.DistributedLinearLayout(
        [], [[1], [2], [4], [8], [16], [32]], [[64], [128], [0]], [], [256]
    )
)
CW_LH8 = gl.constexpr(
    gl.DistributedLinearLayout(
        [[1, 0], [2, 0], [0, 32], [32, 0]],
        [[4, 0], [8, 0], [16, 0], [0, 1], [0, 2], [0, 4]],
        [[0, 8], [0, 16], [64, 0]],
        [],
        [128, 64],
    )
)
CW_D = gl.constexpr(5)  # chunk c is normalized at copier step c + CW_D + 1
CW_NT = gl.constexpr(9)  # raw-output ring slots
# Row partials, warp-replicated so a wait loop on them stays inside each wave.
CW_LP = gl.constexpr(
    gl.DistributedLinearLayout(
        [], [[1], [2], [4], [8], [16], [0]], [[0], [0], [0]], [], [32]
    )
)


@gluon.constexpr_function
def _sl(layout, dim, nd):
    for d in reversed(range(nd)):
        if d != dim:
            layout = gl.SliceLayout(d, layout)
    return layout


@gluon.jit
def _offs(L: gl.constexpr, SHAPE: gl.constexpr, STRIDES: gl.constexpr):
    """int32 offsets sum_d i_d * STRIDES[d] of an L-layout tile of SHAPE."""
    ND: gl.constexpr = len(SHAPE)
    off = gl.zeros(SHAPE, gl.int32, L)
    for d in gl.static_range(ND):
        x = gl.arange(0, SHAPE[d], layout=_sl(L, d, ND)) * STRIDES[d]
        for e in gl.static_range(ND):
            if e != d:
                x = gl.expand_dims(x, e)
        off = off + x
    return off


@gluon.jit
def _cp(s, ptr, off, mask=None, cm: gl.constexpr = ""):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s, ptr, off, mask=mask, cache_modifier=cm
    )


@gluon.jit
def _prow(x):
    R: gl.constexpr = x.shape[0]
    N: gl.constexpr = x.shape[1]
    return x.reshape([R // 32, 2, 4, 4, N]).permute((0, 2, 1, 3, 4)).reshape([R, N])


@gluon.jit
def _solve_u(h, kd_a, inv_a, v, MMA: gl.constexpr, HAS_V: gl.constexpr):
    z = gl.zeros(v.shape, gl.float32, MMA)
    h_op = gl.convert_layout(_prow(h.to(gl.bfloat16)), gl.DotOperandLayout(1, MMA, 8))
    tmp = gl.amd.cdna4.mfma(kd_a, h_op, z)
    u = v - tmp if HAS_V else -tmp
    u_op = gl.convert_layout(_prow(u.to(inv_a.dtype)), gl.DotOperandLayout(1, MMA, 8))
    return gl.amd.cdna4.mfma(inv_a, u_op, z), h_op


@gluon.jit
def _update(h, kr_a, big_u, gt, MMA: gl.constexpr):
    u_op = gl.convert_layout(
        _prow(big_u.to(gl.bfloat16)), gl.DotOperandLayout(1, MMA, 8)
    )
    return gl.amd.cdna4.mfma(kr_a, u_op, h * gt[:, None])


@gluon.jit
def _output(h_op, big_u, qd_a, mqk_a, MMA: gl.constexpr):
    MT: gl.constexpr = gl.amd.AMDMFMALayout(
        4, [16, 16, 32], True, [1, MMA.warps_per_cta[1]]
    )
    z = gl.zeros(big_u.shape, gl.float32, MT)
    qd_t = gl.convert_layout(qd_a, gl.DotOperandLayout(0, MT, 8))
    o = gl.amd.cdna4.mfma(
        qd_t, gl.convert_layout(h_op, gl.DotOperandLayout(1, MT, 8)), z
    )
    u_t = gl.convert_layout(_prow(big_u.to(mqk_a.dtype)), gl.DotOperandLayout(1, MT, 8))
    return gl.amd.cdna4.mfma(
        gl.convert_layout(mqk_a, gl.DotOperandLayout(0, MT, 8)), u_t, o
    )


@gluon.jit
def _psw_off(o_c, o_k, K: gl.constexpr):
    vj = o_k // 8
    off = (
        o_c[:, None] * K + (vj[None, :] ^ (o_c[:, None] % 16)) * 8 + (o_k % 8)[None, :]
    )
    return off.to(gl.int32)


@gluon.jit
def _sw_b_off(k, v, K: gl.constexpr):
    return ((v // 16) * (K * 16) + k * 16 + (v % 16)).to(gl.int32)


@gluon.jit
def _issue_chunk(
    ws,
    ring,
    slot,
    wi,
    t0,
    offs,
    tok_end,
    gate_stride,
    H: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    OUT: gl.constexpr,
    PARTS: gl.constexpr,
    GATE: gl.constexpr = False,
):
    ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, ws_qd, gate = ws
    s_kd, s_kr, s_inv, s_v, s_gt, s_qd, s_mqk, s_g = ring
    ws_off, cc_off, v_off, o_c_v, o_t, g_off = offs
    K: gl.constexpr = s_kd.shape[2]
    wi = wi.to(gl.int32)
    ck = wi * (C * K)
    cc = wi * (2 * C * C)
    if PARTS & 1:
        _cp(s_gt.index(slot), ws_gt, o_t + wi * K, o_t < K)
        _cp(s_kd.index(slot)._reinterpret(layout=SH_FLAT), ws_kd, ws_off + ck)
    if PARTS & 2:
        _cp(s_kr.index(slot)._reinterpret(layout=SH_FLAT), ws_kr, ws_off + ck)
        _cp(s_inv.index(slot), ws_inv_mqk, cc_off + cc)
    if PARTS & 4:
        m_v = (o_c_v < (tok_end - t0).to(gl.int32))[:, None]
        _cp(s_v.index(slot), v_input, v_off + (t0 * (H * V)).to(gl.int32), m_v)
        if OUT:
            _cp(s_qd.index(slot)._reinterpret(layout=SH_FLAT), ws_qd, ws_off + ck)
            if s_inv.shape[1] == C:  # else inv | Mqk is one [2C, C] tile
                _cp(s_mqk.index(slot), ws_inv_mqk, cc_off + (cc + C * C))
        if GATE:
            # Output norm gate rows [C, BW] (token stride gate_stride), tail rows unread.
            g_t = g_off + (t0 * gate_stride).to(gl.int32)
            _cp(s_g.index(slot), gate, g_t, m_v, ".cs")
    if PARTS & 8:
        gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _ring_setup(
    i_h,
    i_w,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    OUT: gl.constexpr,
    NB: gl.constexpr,
    v_input,
    inv_ty: gl.constexpr,
    gate_stride=0,
    g_ty: gl.constexpr = None,
):
    NW: gl.constexpr = gl.num_warps()
    BLK: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
    BLK_V: gl.constexpr = gl.BlockedLayout(
        [1, 8], [64 // (BW // 8), BW // 8], [NW, 1], [1, 0]
    )
    # [C, C] copies: whole tile rows per warp row; 4 warps copy inv | Mqk as one tile.
    CC2: gl.constexpr = OUT and NW == 4
    R2: gl.constexpr = 2 * C if CC2 else C
    VC: gl.constexpr = 8 if CC2 else 2
    L_CC: gl.constexpr = gl.BlockedLayout(
        [1, VC], [64 // (C // VC), C // VC], [NW, 1], [1, 0]
    )
    o_c_s = gl.arange(0, C, layout=gl.SliceLayout(1, BLK))
    ws_off = _psw_off(o_c_s, gl.arange(0, K, layout=gl.SliceLayout(0, BLK)), K)
    cc_off = _offs(L_CC, [R2, C], [C, 1])
    o_c_v = gl.arange(0, C, layout=gl.SliceLayout(1, BLK_V))
    o_w_v = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, BLK_V))
    v_off = (i_h * V + o_c_v[:, None] * (H * V) + o_w_v[None, :]).to(gl.int32)
    g_off = (i_h * V + o_c_v[:, None] * gate_stride + o_w_v[None, :]).to(gl.int32)
    o_t = gl.arange(0, 64 * NW, layout=gl.BlockedLayout([1], [64], [NW], [0]))
    s_kd = gl.allocate_shared_memory(gl.bfloat16, [NB, C, K], SH_WS)
    s_kr = gl.allocate_shared_memory(gl.bfloat16, [NB, C, K], SH_WS)
    s_inv = gl.allocate_shared_memory(inv_ty, [NB, R2, C], SH_PLAIN)
    s_v = gl.allocate_shared_memory(v_input.dtype.element_ty, [NB, C, BW], SH_PLAIN)
    s_gt = gl.allocate_shared_memory(gl.float32, [NB, 64 * NW], SH_1D)
    s_qd = s_kd
    s_mqk = s_kd
    s_g = s_kd
    if OUT:
        s_qd = gl.allocate_shared_memory(gl.bfloat16, [NB, C, K], SH_WS)
        s_mqk = s_inv
        if not CC2:
            s_mqk = gl.allocate_shared_memory(inv_ty, [NB, C, C], SH_PLAIN)
    if g_ty is not None:
        s_g = gl.allocate_shared_memory(g_ty, [NB, C, BW], SH_WS)
    ring = (s_kd, s_kr, s_inv, s_v, s_gt, s_qd, s_mqk, s_g)
    return (ws_off, cc_off, v_off, o_c_v, o_t, g_off), ring


@gluon.jit
def _gnorm(o, ss, g, w, eps, V: gl.constexpr):
    r = gl.rsqrt(ss * (1.0 / V) + eps)
    e = gl.exp2(g.to(gl.float32) * -1.4426950408889634)
    sg = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [1.0 + e], dtype=gl.float32, is_pure=True, pack=1
    )
    return o * r[:, None] * (w[None, :] * sg)


@gluon.jit
def _store_o(
    o, out, t0, o_off, rows, H: gl.constexpr, V: gl.constexpr, CM: gl.constexpr
):
    o = o.to(out.dtype.element_ty)
    gl.amd.cdna4.buffer_store(
        o, out + t0 * (H * V), o_off, mask=rows[:, None], cache=CM
    )


@gluon.jit
def _rowss_put(o, s_ss):
    NW: gl.constexpr = s_ss.shape[0]
    C: gl.constexpr = o.shape[0]
    x = (o * o).reshape([C, o.shape[1] // (16 * NW), NW, 16])
    s_ss.store(gl.permute(gl.sum(gl.sum(x, axis=3), axis=1), (1, 0)))


@gluon.jit
def _rowss_get(s_ss, LO: gl.constexpr):
    NW: gl.constexpr = s_ss.shape[0]
    s_1 = s_ss._reinterpret(layout=SH_1D)
    ss = s_1.index(0).load(gl.SliceLayout(1, LO))
    for w in gl.static_range(1, NW):
        ss = ss + s_1.index(w).load(gl.SliceLayout(1, LO))
    return ss


@gluon.jit
def _wait_flags(flags, idx, epoch, NWP: gl.constexpr, LD: gl.constexpr = ".cv"):
    NW: gl.constexpr = gl.num_warps()
    gl.static_assert(NWP <= 64)
    # One flag per lane; every warp polls all of them itself (no cross-warp reduction).
    LF: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    o = gl.arange(0, 64, layout=LF)
    m = o < NWP
    VOL: gl.constexpr = LD == ".cv"
    f = gl.load(
        flags + idx * NWP + o, mask=m, other=epoch, cache_modifier=LD, volatile=VOL
    )
    while gl.min(f - epoch, axis=0) < 0:  # wrap-safe: epochs are int32 counters
        z = gl.inline_asm_elementwise(
            "s_sleep 1\n v_mov_b32 $0, 0",
            "=v,v",
            [o],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        f = gl.load(
            flags + idx * NWP + o + z,
            mask=m,
            other=epoch,
            cache_modifier=LD,
            volatile=VOL,
        )
    return gl.min(
        gl.inline_asm_elementwise(
            "v_mov_b32 $0, 0", "=v,v", [f], dtype=gl.int32, is_pure=False, pack=1
        ),
        axis=0,
    )


@gluon.jit
def _cw_wave_id(NW: gl.constexpr):
    L2: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW, 1], [1, 0])
    LB: gl.constexpr = gl.BlockedLayout([NW, 1], [1, 64], [1, NW], [1, 0])
    HW: gl.constexpr = "s_getreg_b32 $0, hwreg(HW_REG_HW_ID)"
    tab = gl.allocate_shared_memory(gl.int32, [NW, 64], SH_PLAIN)
    r = gl.arange(0, NW, layout=gl.SliceLayout(1, L2))
    z = gl.zeros([NW, 64], gl.int32, L2) + r[:, None]
    tab.store(gl.inline_asm_elementwise(HW, "=s,v", [z], gl.int32, False, 1))
    gl.barrier()
    T = tab.slice(0, 1, dim=1).load(LB)
    me = gl.inline_asm_elementwise(HW, "=s,v", [T], gl.int32, False, 1)
    i = gl.arange(0, NW, layout=gl.SliceLayout(1, LB))[:, None] + T * 0
    return gl.sum(gl.sum(gl.where(T == me, i, 0), 1), 0)


@gluon.jit
def _cw_fin(cw, c, pp, pv, g, H: gl.constexpr, V: gl.constexpr, C: gl.constexpr):
    s_tail, out, xp, xepoch, prt0, w, eps, tok_base, tok_end, v_off, rr = cw
    cv = gl.maximum(c, 0)
    t0 = tok_base + cv * C
    m = (rr < (tok_end - t0).to(gl.int32)) & (c >= 0)
    x = s_tail.index(cv % CW_NT).load(CW_LV).to(gl.float32)
    rp = gl.arange(0, C, layout=CW_LP)
    mp = (rp < (tok_end - t0).to(gl.int32)) & (c >= 0)
    stale = mp & ((pp >> 32) != xepoch.to(gl.int64))
    if gl.max(stale.to(gl.int32), axis=0) != 0:
        while gl.max(stale.to(gl.int32), axis=0) != 0:
            nv = gl.load(
                xp + (prt0 + 2 * cv) * C + rp,
                mask=stale,
                cache_modifier=".cv",
                volatile=True,
            )
            pp = gl.where(stale, nv, pp)
            stale = mp & ((pp >> 32) != xepoch.to(gl.int64))
        pv = gl.load(
            xp + (prt0 + 2 * cv) * C + rr, mask=m, cache_modifier=".cv", volatile=True
        )
    ss = gl.sum(x * x, axis=1) + pv.to(gl.int32).to(gl.float32, bitcast=True)
    y = _gnorm(x, ss, g, w, eps, V).to(out.dtype.element_ty)
    gl.amd.cdna4.buffer_store(
        y, out + t0 * (H * V), v_off, mask=m[:, None], cache=".cs"
    )


@gluon.jit
def _cw_pub(cw, own0, c, C: gl.constexpr):
    s_tail, _out, xp, xepoch, _prt0, _w, _eps, _tok_base, _tok_end, _v_off, rr = cw
    x = s_tail.index(c % CW_NT).load(CW_LV).to(gl.float32)
    ss = gl.sum(x * x, axis=1).to(gl.int32, bitcast=True).to(gl.int64)
    v64 = (ss & 0xFFFFFFFF) | (xepoch.to(gl.int64) << 32)
    gl.amd.cdna4.buffer_store(v64, xp + (own0 + 2 * c) * C, rr, cache=".wt")


@gluon.jit
def _cw_ldp(cw, c, C: gl.constexpr):
    _s_tail, _out, xp, _xepoch, prt0, _w, _eps, tok_base, tok_end, _v_off, rr = cw
    cc = gl.maximum(c, 0)
    rp = gl.arange(0, C, layout=CW_LP)
    mp = (rp < (tok_end - tok_base - cc * C).to(gl.int32)) & (c >= 0)
    m = (rr < (tok_end - tok_base - cc * C).to(gl.int32)) & (c >= 0)
    base = xp + (prt0 + 2 * cc) * C
    a = gl.amd.cdna4.buffer_load(base, rp, mask=mp, cache=".cv")
    return a, gl.amd.cdna4.buffer_load(base, rr, mask=m, cache=".cv")


@gluon.jit
def _cw_ldg(cw, gate, gate_stride, g_off, c, C: gl.constexpr):
    _s_tail, _out, _xp, _xepoch, _prt0, _w, _eps, tok_base, tok_end, _v_off, rr = cw
    t0 = tok_base + gl.maximum(c, 0) * C
    m = (rr < (tok_end - t0).to(gl.int32)) & (c >= 0)
    return gl.amd.cdna4.buffer_load(gate + t0 * gate_stride, g_off, mask=m[:, None])


@gluon.jit
def _cw_walk(
    ws,
    out,
    seg,
    st,
    TOTAL_TILES,
    nrm,
    i_sh,
    i_w,
    H: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
):
    _ws_kd, _ws_kr, _ws_gt, ws_inv_mqk, v_input, _ws_qd, gate = ws
    seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end, seg_seq = seg
    state_cache, state_indices, has_initial_state, cache_stride = st
    norm_w, gate_stride, norm_eps, xp, xepoch = nrm
    K: gl.constexpr = 128
    BW: gl.constexpr = V // 2
    gl.static_assert(C == 32 and BW == 64 and gl.num_warps() == 8)
    MMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [1, 8])
    wid = _cw_wave_id(8)
    gl.barrier()  # all waves: the wave-id table's LDS is reused by the rings
    i_seg = i_sh // H
    i_h = i_sh % H
    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)
    i_n = gl.load(seg_seq + i_seg).to(gl.int64)
    slot = gl.load(state_indices + i_n).to(gl.int64)
    f_ptr = state_cache + slot * cache_stride + i_h * (V * K)
    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    out_ty: gl.constexpr = out.dtype.element_ty
    # kd ring: one spare slot, so the fp32 final state (32 KB) fits it for the hand-off.
    s_kd = gl.allocate_shared_memory(gl.bfloat16, [4, C, K], SH_WS)
    s_kr = gl.allocate_shared_memory(gl.bfloat16, [3, C, K], SH_WS)
    s_qd = gl.allocate_shared_memory(gl.bfloat16, [3, C, K], SH_WS)
    s_inv = gl.allocate_shared_memory(inv_ty, [3, 2 * C, C], SH_PLAIN)
    s_v = gl.allocate_shared_memory(v_input.dtype.element_ty, [3, C, BW], SH_PLAIN)
    s_gt = gl.allocate_shared_memory(gl.float32, [3, 256], SH_1D)
    SH_T: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    s_tail = gl.allocate_shared_memory(out_ty, [CW_NT, C, BW], SH_T)
    s_g = gl.allocate_shared_memory(gate.dtype.element_ty, [3, C, BW], SH_PLAIN)
    s_pp = gl.allocate_shared_memory(gl.int32, [3, 256], SH_1D)
    s_h = s_kd._reinterpret(gl.float32, [K, BW], SH_PLAIN)
    ws0 = i_h * TOTAL_TILES + chunk_base
    if wid < 4:
        # ---------------- copier waves ----------------
        o_c = gl.arange(0, C, layout=gl.SliceLayout(1, CW_LK))
        ws_off = _psw_off(o_c, gl.arange(0, K, layout=gl.SliceLayout(0, CW_LK)), K)
        cc_off = _offs(CW_LCC, [2 * C, C], [C, 1])
        rr = gl.arange(0, C, layout=gl.SliceLayout(1, CW_LV))
        o_w_v = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, CW_LV))
        v_off = (i_h * V + rr[:, None] * (H * V) + o_w_v[None, :]).to(gl.int32)
        g_off = (i_h * V + rr[:, None] * gate_stride + o_w_v[None, :]).to(gl.int32)
        o_t = gl.arange(0, 256, layout=CW_LG)
        offs = (ws_off, cc_off, v_off, rr, o_t, g_off)
        ring = (s_kd, s_kr, s_inv, s_v, s_gt, s_qd, s_inv, s_g)
        wi0 = ws0.to(gl.int32)
        if n_chunks > 0:
            _issue_chunk(
                ws, ring, 0, wi0, tok_base, offs, tok_end, 0, H, V, C, True, 15
            )
            # Pairs with the compute waves' pre-loop barrier.
            gl.barrier()
            j1 = gl.minimum(n_chunks - 1, 1)
            t1 = tok_base + j1 * C
            _issue_chunk(ws, ring, 1, wi0 + j1, t1, offs, tok_end, 0, H, V, C, True, 15)
        w_n = gl.load(norm_w + o_w_v).to(gl.float32)
        own0 = ws0 * 2 + i_w
        prt0 = ws0 * 2 + (1 - i_w)
        xp32 = xp.to(gl.pointer_type(gl.int32), bitcast=True)
        s_pp64 = s_pp._reinterpret(gl.int64, [3, 128], SH_1D)
        cw = (
            s_tail,
            out,
            xp,
            xepoch,
            prt0,
            w_n,
            norm_eps,
            tok_base,
            tok_end,
            v_off,
            rr,
        )
        for j in range(n_chunks):
            gl.amd.cdna4.async_copy.wait_group(1)  # chunk j landed + B_j
            cg = gl.maximum(j + 1 - CW_D, 0)
            tg = tok_base + cg * C
            lg = j + 1 >= CW_D
            # partner partial + gate of chunk j + 1 - D (read at step j + 2)
            pp_off = ((prt0 + 2 * cg) * (2 * C)).to(gl.int32) + o_t
            _cp(s_pp.index(cg % 3), xp32, pp_off, (o_t < 2 * C) & lg, ".cv")
            m_g = ((rr < (tok_end - tg).to(gl.int32)) & lg)[:, None]
            _cp(s_g.index(cg % 3), gate, g_off + (tg * gate_stride).to(gl.int32), m_g)
            if j + 2 < n_chunks:
                t2 = tok_base + (j + 2) * C
                nb = (j + 2) % 3
                _issue_chunk(
                    ws, ring, nb, wi0 + j + 2, t2, offs, tok_end, 0, H, V, C, True, 15
                )
            else:
                gl.amd.cdna4.async_copy.commit_group()
            if j >= 1:
                _cw_pub(cw, own0, j - 1, C)
            if j >= CW_D + 1:
                cf = j - 1 - CW_D
                g1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    s_g.index(cf % 3), CW_LV
                )
                pp1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    s_pp64.index(cf % 3).slice(0, C), CW_LP
                )
                pv1 = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    s_pp64.index(cf % 3).slice(0, C), gl.SliceLayout(1, CW_LV)
                )
                _cw_fin(cw, cf, pp1, pv1, g1, H, V, C)
        # Drain: normalize the last D + 1 chunks.
        n = n_chunks
        pq1 = _cw_ldp(cw, n - 5, C)
        pq2 = _cw_ldp(cw, n - 4, C)
        pq3 = _cw_ldp(cw, n - 3, C)
        pq4 = _cw_ldp(cw, n - 2, C)
        gq2 = _cw_ldg(cw, gate, gate_stride, g_off, n - 4, C)
        gq3 = _cw_ldg(cw, gate, gate_stride, g_off, n - 3, C)
        gq4 = _cw_ldg(cw, gate, gate_stride, g_off, n - 2, C)
        gq5 = _cw_ldg(cw, gate, gate_stride, g_off, n - 1, C)
        gl.inline_asm_elementwise(
            "s_waitcnt vmcnt(0)\n v_mov_b32 $0, 0",
            "=v,v",
            [o_t],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        c6 = gl.maximum(n - 6, 0)
        g6 = gl.amd.cdna4.async_copy.load_shared_relaxed(s_g.index(c6 % 3), CW_LV)
        g5 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_g.index(gl.maximum(n - 5, 0) % 3), CW_LV
        )
        pp6 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_pp64.index(c6 % 3).slice(0, C), CW_LP
        )
        pv6 = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_pp64.index(c6 % 3).slice(0, C), gl.SliceLayout(1, CW_LV)
        )
        _cw_fin(cw, n - 6, pp6, pv6, g6, H, V, C)
        _cw_fin(cw, n - 5, pq1[0], pq1[1], g5, H, V, C)
        _cw_fin(cw, n - 4, pq2[0], pq2[1], gq2, H, V, C)
        _cw_fin(cw, n - 3, pq3[0], pq3[1], gq3, H, V, C)
        _cw_fin(cw, n - 2, pq4[0], pq4[1], gq4, H, V, C)
        gl.barrier()  # B_final: every chunk's raw o is in s_tail
        if n > 0:
            _cw_pub(cw, own0, n - 1, C)
        gl.barrier()  # B_h
        pq5 = _cw_ldp(cw, n - 1, C)
        _cw_fin(cw, n - 1, pq5[0], pq5[1], gq5, H, V, C)
    else:
        # ---------------- compute waves ----------------
        o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
        o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))
        take = gl.load(has_initial_state + i_n).to(gl.int1)
        h_off = (o_w_m[None, :] * K + o_k_m[:, None]).to(gl.int32)
        h = gl.amd.cdna4.buffer_load(f_ptr, h_off, mask=(o_k_m[:, None] < K) & take)
        h = h.to(gl.float32)
        o_c_m = gl.arange(0, C, layout=gl.SliceLayout(1, MMA))
        if n_chunks > 0:
            gl.barrier()  # pairs with the copiers' prologue barrier
        for j in range(n_chunks):
            # B_j (chunk j - 1's s_tail writes land first: the copiers read them after B_j)
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                "=v,v",
                [o_c_m],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.barrier()
            cur = j % 3
            kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                s_kd.index(cur), gl.DotOperandLayout(0, MMA, 8)
            )
            inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                s_inv.index(cur).slice(0, C), gl.DotOperandLayout(0, MMA, 8)
            )
            b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(
                gl.float32
            )
            gt = gl.amd.cdna4.async_copy.load_shared_relaxed(
                s_gt.index(cur).slice(0, K), gl.SliceLayout(1, MMA)
            )
            big_u, h_op = _solve_u(h, kd_a, inv_a, b_v, MMA, True)
            qd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                s_qd.index(cur), gl.DotOperandLayout(0, MMA, 8)
            )
            mqk_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                s_inv.index(cur).slice(C, C), gl.DotOperandLayout(0, MMA, 8)
            )
            s_tail.index(j % CW_NT).store(
                _output(h_op, big_u, qd_a, mqk_a, MMA).to(out_ty)
            )
            kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                s_kr.index(cur).permute((1, 0)), gl.DotOperandLayout(0, MMA, 8)
            )
            h = _update(h, kr_a, big_u, gt, MMA)
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
            "=v,v",
            [o_c_m],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.barrier()  # B_final
        # State hand-off to the copier waves: fp32 [K, BW] in the (free) kd ring.
        s_h.store(h)
        gl.inline_asm_elementwise(
            "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
            "=v,v",
            [o_c_m],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.barrier()  # B_h
    # Final state, V-first: rows 0-63 by the copier waves, 64-127 by the compute waves.
    hk = gl.arange(0, K, layout=gl.SliceLayout(1, CW_LH8))
    hv = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, CW_LH8))
    h_off = (hv[None, :] * K + hk[:, None]).to(gl.int32)
    gl.amd.cdna4.buffer_store(s_h.load(CW_LH8), f_ptr, h_off)


@gluon.constexpr_function
def _vmwait(n):
    return f"s_waitcnt vmcnt({n})\n v_mov_b32 $0, $1"


@gluon.jit
def _scf_rel(pfl, pst, seg, val, N: gl.constexpr, CM: gl.constexpr = ""):
    NW: gl.constexpr = gl.num_warps()
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    lane = gl.arange(0, 64 * NW, layout=L1)
    v = gl.full([64 * NW], 0, gl.int32, L1) + val
    v = gl.inline_asm_elementwise(
        _vmwait(N), "=v,v", [v], dtype=gl.int32, is_pure=False, pack=1
    )
    gl.store(pfl + seg * pst + lane // 64, v, mask=(lane % 64) == 0, cache_modifier=CM)


@gluon.jit
def _sv2_step(h, j, n_upd, sc, ra, rb, s_hh, MASKED: gl.constexpr):
    s0, s_last, A_base, b_base, h_base, a_off, b_off, HKK, HKV, pfl, pst, pep = sc
    MMA: gl.constexpr = h.type.layout
    # Publish h_in of segment s0 + j - 1 (its store is older than the last loads).
    _scf_rel(
        pfl,
        pst,
        s0 + gl.maximum(j - 1, 0),
        gl.where(j >= 1, pep, 0),
        5 if s_hh.shape[1] == 16 else 7,
    )
    # [8 t, 2 h, 4 g, 16 r, 16 b] memory order -> the [128, 128] A operand (register no-op)
    x = ra.reshape([8, 2, 4, 16, 2, 8]).permute((0, 3, 1, 4, 2, 5)).reshape([128, 128])
    a_op = gl.convert_layout(x, gl.DotOperandLayout(0, MMA, 8)).to(gl.bfloat16)
    b_cur = rb.to(gl.float32)
    sl = gl.minimum(s0 + j + 2, s_last)
    s_hh.store(h.to(gl.bfloat16))
    gl.barrier()
    hn = gl.amd.cdna4.mfma(a_op, s_hh.load(gl.DotOperandLayout(1, MMA, 8)), b_cur)
    ra = gl.amd.cdna4.buffer_load(A_base + sl * HKK, a_off)
    rb = gl.amd.cdna4.buffer_load(b_base + sl * HKV, b_off)
    h_ptr = h_base + (s0 + j + 1) * HKV
    if MASKED:
        live = j < n_upd
        h = gl.where(live, hn, h)
        gl.amd.cdna4.buffer_store(
            h.to(gl.bfloat16), h_ptr, b_off, mask=(b_off >= 0) & live
        )
    else:
        h = hn
        gl.amd.cdna4.buffer_store(h.to(gl.bfloat16), h_ptr, b_off)
    return h, ra, rb


@gluon.jit
def _scan_item(
    e,
    n_cap,
    h_in,
    sflags,
    scan,
    state_cache,
    state_indices,
    has_initial_state,
    cache_stride,
    xepoch,
    HSLAB,
    FSLAB,
    H: gl.constexpr,
    V: gl.constexpr,
    BV: gl.constexpr,
    RMAX: gl.constexpr,
    HAS_H0: gl.constexpr,
    PAGED: gl.constexpr,
):
    A_seg, b_seg, h0, seq_seg_off, NH = scan
    r = (e >> 26) & 7
    nh = (e >> 4) & 0x3FFFFF
    i_v = e & 15
    NW: gl.constexpr = gl.num_warps()
    K: gl.constexpr = 128
    gl.static_assert(NW == 4 or NW == 8)
    MMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [NW, 1])
    i_n = nh // H
    i_h = nh % H
    s0 = gl.load(seq_seg_off + i_n)
    s_end = gl.load(seq_seg_off + i_n + 1) - 1
    n_upd = gl.minimum(s_end - s0, n_cap)
    s_last = gl.maximum(s_end - 1, s0)
    o_k = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_v = i_v * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, MMA))
    b_off = _sw_b_off(o_k[:, None], o_v[None, :], K)
    LA: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, 1, 16], [1, 1, 4, 16, 1], [NW, 1, 1, 1, 1], [4, 3, 2, 1, 0]
    )
    a_off = _offs(LA, [8, 2, 4, 16, 16], [2048, 1024, 256, 16, 1])
    HKK = H * K * K
    HKV = H * K * V
    A_base = A_seg + i_h * (K * K)
    b_base = b_seg + i_h * (K * V)
    h_base = h_in + r.to(gl.int64) * HSLAB + i_h * (K * V)
    pfl = sflags + r * FSLAB + ((nh % H) * (V // BV) + i_v) * NW
    s1 = gl.minimum(s0 + 1, s_last)
    ra0 = gl.amd.cdna4.buffer_load(A_base + s0 * HKK, a_off)
    rb0 = gl.amd.cdna4.buffer_load(b_base + s0 * HKV, b_off)
    ra1 = gl.amd.cdna4.buffer_load(A_base + s1 * HKK, a_off)
    rb1 = gl.amd.cdna4.buffer_load(b_base + s1 * HKV, b_off)
    if PAGED:
        slot = gl.load(state_indices + i_n).to(gl.int64)
        take = gl.load(has_initial_state + i_n).to(gl.int1)
        h = gl.amd.cdna4.buffer_load(
            ptr=state_cache + slot * cache_stride + i_h * (V * K),
            offsets=(o_v[None, :] * K + o_k[:, None]).to(gl.int32),
            mask=(o_k[:, None] < K) & take,
            other=0.0,
        ).to(gl.float32)
    elif HAS_H0:
        h = gl.amd.cdna4.buffer_load(
            ptr=h0 + (i_n * H + i_h).to(gl.int64) * (K * V),
            offsets=(o_k[:, None] * V + o_v[None, :]).to(gl.int32),
        ).to(gl.float32)
    else:
        h = gl.zeros([K, BV], gl.float32, MMA)
    gl.amd.cdna4.buffer_store(h.to(h_in.dtype.element_ty), h_base + s0 * HKV, b_off)
    SH_H: gl.constexpr = gl.SwizzledSharedLayout(8, 1, BV // 8, [1, 0])
    s_ha = gl.allocate_shared_memory(gl.bfloat16, [K, BV], SH_H)
    s_hb = gl.allocate_shared_memory(gl.bfloat16, [K, BV], SH_H)
    sc = (
        s0,
        s_last,
        A_base,
        b_base,
        h_base,
        a_off,
        b_off,
        HKK,
        HKV,
        pfl,
        H * V // BV * NW,
        xepoch,
    )
    n_full = n_upd - n_upd % 2
    for i in range(0, n_full, 2):
        h, ra0, rb0 = _sv2_step(h, i, n_upd, sc, ra0, rb0, s_ha, False)
        h, ra1, rb1 = _sv2_step(h, i + 1, n_upd, sc, ra1, rb1, s_hb, False)
    if n_full < n_upd:
        h, ra0, rb0 = _sv2_step(h, n_full, n_upd, sc, ra0, rb0, s_ha, True)
    _scf_rel(pfl, H * V // BV * NW, s0 + gl.maximum(n_upd - 1, 0), xepoch, 0)
    _scf_rel(pfl, H * V // BV * NW, s0 + n_upd, xepoch, 0)
    if ((e >> 29) & 1) != 0:
        # Done flag for a split stream's last segment on another XCD (write-through).
        dn = sflags + RMAX * FSLAB + ((r * NH + nh) * (V // BV) + i_v) * NW
        _scf_rel(dn, 0, 0, xepoch, 0, ".wt")


_k2_ab_fused_repr = make_kernel_repr(
    "k2_ab_fused_gluon",
    ["C", "K", "V", "BW", "NUM_WARPS"],
)


@gluon.jit(repr=_k2_ab_fused_repr)
def k2_ab_fused_gluon(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    h_out_b,
    h_out_a,
    seg_rows,
    TOTAL_TILES,
    H: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    C: gl.constexpr,
    BW: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    # Keeps a (segment, head)'s V blocks on one XCD, sharing an L2.
    n_w = gl.num_programs(0)
    pid = remap_xcd(
        gl.program_id(1) * n_w + gl.program_id(0), n_w * gl.num_programs(1), NUM_XCDS
    )
    i_w = (pid % n_w).to(gl.int64)
    row = seg_rows + (pid // n_w).to(gl.int64) * 8
    i_seg = gl.load(row).to(gl.int64)
    chunk_base = gl.load(row + 1).to(gl.int64)
    n_chunks = gl.load(row + 2)
    tok_base = gl.load(row + 3).to(gl.int64)
    tok_end = gl.load(row + 4).to(gl.int64)
    i_h = gl.load(row + 5).to(gl.int64)

    # The launch's warp count, as a constexpr so it can name the kernel.
    gl.static_assert(NUM_WARPS == gl.num_warps())
    MMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [1, NUM_WARPS])
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))
    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    offs, ring = _ring_setup(i_h, i_w, H, K, V, C, BW, False, 3, v_input, inv_ty)
    s_kd, s_kr, s_inv, s_v, s_gt, _, _, _ = ring
    ws = (ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, ws_kd, ws_kd)

    h_b = gl.zeros([K, BW], gl.float32, MMA)
    h_a = gl.where(o_k_m[:, None] == o_w_m[None, :], 1.0, 0.0)
    ws0 = i_h * TOTAL_TILES + chunk_base
    for q in gl.static_range(2):
        jq = gl.minimum(n_chunks - 1, q)
        t_q = tok_base + jq * C
        _issue_chunk(ws, ring, q, ws0 + jq, t_q, offs, tok_end, 0, H, V, C, False, 15)
    for j in range(n_chunks):
        cur = j % 3
        # Chunk j is in slot `cur`; the copy of min(j + 2, n - 1) goes to the free
        # slot in three parts spread over the step, so the group count stays static.
        gl.amd.cdna4.async_copy.wait_group(1)
        jn = gl.minimum(j + 2, n_chunks - 1)
        nb = (j + 2) % 3
        tn = tok_base + jn * C
        nxt = (ws, ring, nb, ws0 + jn, tn, offs, tok_end, 0)
        _issue_chunk(*nxt, H, V, C, False, 1)
        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_kd.index(cur), gl.DotOperandLayout(0, MMA, 8)
        )
        # The permuted [C, K] view is the kr^T A operand.
        kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_kr.index(cur).permute((1, 0)), gl.DotOperandLayout(0, MMA, 8)
        )
        inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_inv.index(cur), gl.DotOperandLayout(0, MMA, 8)
        )
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(
            gl.float32
        )
        gt = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_gt.index(cur).slice(0, K), gl.SliceLayout(1, MMA)
        )
        # Two independent chains, so the scheduler interleaves their MFMAs.
        u_b, _h = _solve_u(h_b, kd_a, inv_a, b_v, MMA, True)
        h_b = _update(h_b, kr_a, u_b, gt, MMA)
        _issue_chunk(*nxt, H, V, C, False, 2)
        u_a, _h = _solve_u(h_a, kd_a, inv_a, b_v, MMA, False)
        h_a = _update(h_a, kr_a, u_a, gt, MMA)
        _issue_chunk(*nxt, H, V, C, False, 12)

    s_base = (i_seg * H + i_h) * (K * V)
    gl.amd.cdna4.async_copy.wait_group(0)
    gl.barrier()
    # Store A_seg and b_seg in the scan's load order, 16 B per thread.
    NH: gl.constexpr = BW // 64
    LA: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, 16], [1, 1, 64, 1], [NUM_WARPS, 1, 1, 1], [3, 2, 1, 0]
    )
    xa = h_a.to(h_out_a.dtype.element_ty).reshape([8, 16, NH, 2, 4, 8])
    xa = xa.permute((0, 2, 4, 1, 3, 5)).reshape([8, NH, 64, 16])
    oa = _offs(LA, [8, NH, 64, 16], [2048, 1024, 16, 1])
    pa = h_out_a + s_base + i_w * (NH * 1024)
    gl.amd.cdna4.buffer_store(gl.convert_layout(xa, LA), pa, oa, cache=".wt")
    LB: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8], [1, 32, 2], [1, NUM_WARPS, 1], [2, 1, 0]
    )
    xb = h_b.to(h_out_b.dtype.element_ty).reshape([K, BW // 16, 16])
    ob = _offs(LB, [BW // 16, K, 16], [K * 16, 16, 1])
    pb = h_out_b + s_base + i_w * (BW // 16) * (K * 16)
    xb = gl.convert_layout(xb.permute((1, 0, 2)), LB)
    gl.amd.cdna4.buffer_store(xb, pb, ob, cache=".wt")


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
        "NORM",
        "CW",
    ],
)


@gluon.jit(repr=_k2_c_repr)
def k2_c_gluon(
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
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
    HAS_H_IN: gl.constexpr,
    STORE_FINAL: gl.constexpr,
    STATE_V_FIRST: gl.constexpr,
    PAGED_CACHE: gl.constexpr,
    PAGED_H_IN: gl.constexpr,
    CM_OUT: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    gate=None,
    norm_w=None,
    gate_stride=0,
    norm_eps=1e-5,
    NORM: gl.constexpr = False,
    xepoch=None,
    xbuf=None,
    n_str=0,
    A_seg=None,
    b_seg=None,
    h0=None,
    seq_seg_off=None,
    NH=0,
    tab=None,
    sflags=None,
    HSLAB=0,
    FSLAB=0,
    BV: gl.constexpr = 16,
    RMAX: gl.constexpr = 1,
    HAS_H0: gl.constexpr = False,
    ckpt_state=None,
    ckpt_offsets=None,
    ckpt_rows=None,
    ckpt_bos=None,
    CKPT_STRIDE: gl.constexpr = 0,
):
    """Pass C: one segment's recurrence from its incoming state, writing outputs.

    Same contract as ``_flash_kda_segment_kernel`` with ``COMPUTE_OUTPUT`` and
    ``HAS_V`` and without ``STORE_H_OUT``, paged ``state_cache`` included. The
    operands stream through a 3-slot LDS ring (2 with ``NORM``, BW == V: the norm of
    each chunk is one step late, once the row partials crossed the next barrier).
    ``xbuf``: the copier-wave walk instead. ``tab``: the scan in the same launch (BW
    = V); a pass C program reads h_in once its scan replica's flags rise.
    ``ckpt_state``: mid-prefill checkpoints. Sequence ``n`` stores its state after
    ``ckpt_offsets[n]`` tokens to row ``ckpt_rows[n]`` (default ``n``); ``ckpt_bos``
    holds each sequence's first token.
    """
    CW: gl.constexpr = xbuf is not None
    SCW: gl.constexpr = tab is not None
    NWS: gl.constexpr = (V // BV) * NUM_WARPS
    # This call's epoch is read from the device, so a replayed CUDA graph gets a new one.
    ep = 0
    if xepoch is not None:
        ep = gl.load(xepoch)
    if CW:
        # Stream s runs on XCD s % NUM_XCDS with its two V halves (pid and pid +
        # NUM_XCDS): the partner exchange of the norm stays in one L2.
        p = gl.program_id(1)
        i_w = (p % (2 * NUM_XCDS)) // NUM_XCDS
        i_sh = (p // (2 * NUM_XCDS)) * NUM_XCDS + p % NUM_XCDS
        if i_sh < n_str:
            ws = (ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, ws_qd, gate)
            st = (state_cache, state_indices, has_initial_state, cache_stride)
            seg = (seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end, seg_seq)
            nrm = (norm_w, gate_stride, norm_eps, xbuf, ep)
            _cw_walk(ws, out, seg, st, TOTAL_TILES, nrm, i_sh, i_w, H, V, C)
        return
    if SCW:
        scan = (A_seg, b_seg, h0 if HAS_H0 else A_seg, seq_seg_off, NH)
        pid = gl.program_id(0)
        e = gl.load(tab + 2 * pid)
        a = gl.load(tab + 2 * pid + 1)
        if (e >> 30) == 1:
            _scan_item(
                e,
                a,
                h_in,
                sflags,
                scan,
                state_cache,
                state_indices,
                has_initial_state,
                cache_stride,
                ep,
                HSLAB,
                FSLAB,
                H,
                V,
                BV,
                RMAX,
                HAS_H0,
                PAGED_CACHE,
            )
        if (e < 0) or ((e >> 30) == 1):
            return
        r = (e >> 26) & 7
        msk = a & 255
        if msk != 0:
            for rr in gl.static_range(RMAX):
                if (msk >> rr) & 1:
                    fl = sflags + RMAX * FSLAB + rr * NH * NWS
                    _wait_flags(fl, a >> 8, ep, NWS)
        h_in = h_in + r.to(gl.int64) * HSLAB
        sflags = sflags + r * FSLAB
        i_w = (e * 0).to(gl.int64)
        i_sh = (e & 0x3FFFFFF).to(gl.int64)
    else:
        n_w = gl.num_programs(0)
        pid = remap_xcd(
            gl.program_id(1) * n_w + gl.program_id(0),
            n_w * gl.num_programs(1),
            NUM_XCDS,
        )
        i_w = (pid % n_w).to(gl.int64)
        i_sh = (pid // n_w).to(gl.int64)
    i_seg = i_sh // H
    i_h = i_sh % H

    chunk_base = gl.load(seg_chunk_base + i_seg).to(gl.int64)
    n_chunks = gl.load(seg_nchunks + i_seg)
    tok_base = gl.load(seg_tok_base + i_seg).to(gl.int64)
    tok_end = gl.load(seg_tok_end + i_seg).to(gl.int64)
    # Load before the output stores to avoid a waterfall loop on the final-state store.
    if STORE_FINAL or PAGED_CACHE or ckpt_state is not None:
        i_n = gl.load(seg_seq + i_seg).to(gl.int64)
    if STORE_FINAL:
        is_last = gl.load(seg_is_last + i_seg)
    if PAGED_CACHE:
        slot = gl.load(state_indices + i_n).to(gl.int64)
    if ckpt_state is not None:
        # The snapshot point, in tokens into this segment; stored after chunk
        # ckpt_j if it falls inside the segment and before the sequence end.
        c_pos = gl.load(ckpt_offsets + i_n).to(gl.int64)
        rel = c_pos + gl.load(ckpt_bos + i_n).to(gl.int64) - tok_base
        n_tok = tok_end - tok_base
        c_last = gl.load(seg_is_last + i_seg)
        ckpt_row = i_n
        if ckpt_rows is not None:
            ckpt_row = gl.load(ckpt_rows + i_n).to(gl.int64)
        ok = (c_pos > 0) & (c_pos % C == 0) & (rel > 0) & (ckpt_row >= 0)
        ok = ok & ((rel < n_tok) | ((rel == n_tok) & (c_last == 0)))
        ckpt_j = gl.where(ok, rel // C, -1)
        ckpt_off = ckpt_row * CKPT_STRIDE

    # The launch's warp count, as a constexpr so it can name the kernel.
    gl.static_assert(NUM_WARPS == gl.num_warps())
    MMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], False, [1, NUM_WARPS])
    MT: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, NUM_WARPS])
    o_k_m = gl.arange(0, K, layout=gl.SliceLayout(1, MMA))
    o_w_m = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MMA))
    o_c_t = gl.arange(0, C, layout=gl.SliceLayout(1, MT))
    o_w_t = i_w * BW + gl.arange(0, BW, layout=gl.SliceLayout(0, MT))
    o_off = (i_h * V + o_c_t[:, None] * (H * V) + o_w_t[None, :]).to(gl.int32)
    # K-first [K, V] and V-first [V, K] offsets of this program's state block.
    s_off = (o_k_m[:, None] * V + o_w_m[None, :]).to(gl.int32)
    s_off_vk = (o_w_m[None, :] * K + o_k_m[:, None]).to(gl.int32)

    inv_ty: gl.constexpr = ws_inv_mqk.dtype.element_ty
    NBC: gl.constexpr = 2 if NORM else 3
    g_ty: gl.constexpr = gate.dtype.element_ty if NORM else None
    offs, ring = _ring_setup(
        i_h, i_w, H, K, V, C, BW, True, NBC, v_input, inv_ty, gate_stride, g_ty
    )
    s_kd, s_kr, s_inv, s_v, s_gt, s_qd, s_mqk, s_g = ring
    ws = (ws_kd, ws_kr, ws_gt, ws_inv_mqk, v_input, ws_qd, gate if NORM else v_input)
    if NORM:
        gl.static_assert(BW == V)
        s_ss = gl.allocate_shared_memory(gl.float32, [NUM_WARPS, C], SH_PLAIN)
        w_n = gl.amd.cdna4.buffer_load(ptr=norm_w, offsets=o_w_t.to(gl.int32)).to(
            gl.float32
        )
        o_p = gl.zeros([C, BW], gl.float32, MT)
        g_p = gl.zeros([C, BW], gate.dtype.element_ty, MT)
        t_p = tok_base

    if PAGED_H_IN and not SCW:
        # A zero-start sequence's slot may hold a stale state: never read it.
        take = gl.load(has_initial_state + i_n).to(gl.int1)
        m_h = (o_k_m[:, None] < K) & (o_w_m[None, :] < V) & take
        h = gl.amd.cdna4.buffer_load(
            ptr=state_cache + slot * cache_stride + i_h * (V * K),
            offsets=s_off_vk,
            mask=m_h,
            other=0.0,
        ).to(gl.float32)
    elif HAS_H_IN and not SCW:
        h = gl.amd.cdna4.buffer_load(
            ptr=h_in + (i_seg * H + i_h) * (K * V), offsets=s_off
        ).to(gl.float32)
    else:
        h = gl.zeros([K, BW], gl.float32, MMA)

    ws0 = i_h * TOTAL_TILES + chunk_base
    for q in gl.static_range(NBC - 1):
        jq = gl.minimum(n_chunks - 1, q)
        t_q = tok_base + jq * C
        _issue_chunk(
            ws,
            ring,
            q,
            ws0 + jq,
            t_q,
            offs,
            tok_end,
            gate_stride,
            H,
            V,
            C,
            True,
            15,
            NORM,
        )
    if SCW:
        # h_in in the scan's [V // 16, K, 16] layout, behind its flags (same XCD).
        zf = _wait_flags(sflags, i_sh, ep, NWS, ".cg")
        hin_off = _sw_b_off(o_k_m[:, None], o_w_m[None, :], K) + zf
        h = gl.amd.cdna4.buffer_load(ptr=h_in + i_sh * (K * V), offsets=hin_off)
        h = h.to(gl.float32)

    for j in range(n_chunks):
        if NORM:
            # One chunk in flight, copied (with its gate rows) as one group.
            cur = j % 2
            gl.amd.cdna4.async_copy.wait_group(0)
            if j + 1 < n_chunks:
                t_n = tok_base + (j + 1) * C
                _issue_chunk(
                    ws,
                    ring,
                    1 - cur,
                    ws0 + j + 1,
                    t_n,
                    offs,
                    tok_end,
                    gate_stride,
                    H,
                    V,
                    C,
                    True,
                    15,
                    True,
                )
            g_c = gl.amd.cdna4.async_copy.load_shared_relaxed(s_g.index(cur), MT)
        else:
            # Chunks j, j + 1 in flight; chunk j + 2's copy is spread over the step.
            cur = j % 3
            gl.amd.cdna4.async_copy.wait_group(1)
            jn = gl.minimum(j + 2, n_chunks - 1)
            nxt = (ws, ring, (j + 2) % 3, ws0 + jn, tok_base + jn * C, offs, tok_end)
            _issue_chunk(*nxt, 0, H, V, C, True, 1)

        t0 = tok_base + j * C
        kd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_kd.index(cur), gl.DotOperandLayout(0, MMA, 8)
        )
        inv_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_inv.index(cur).slice(0, C), gl.DotOperandLayout(0, MMA, 8)
        )
        b_v = gl.amd.cdna4.async_copy.load_shared_relaxed(s_v.index(cur), MMA).to(
            gl.float32
        )
        gt = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_gt.index(cur).slice(0, K), gl.SliceLayout(1, MMA)
        )
        # _recur's steps, with the state update moved after the output
        # projection to keep the next state and kr out of registers meanwhile.
        big_u, h_op = _solve_u(h, kd_a, inv_a, b_v, MMA, True)
        if not NORM:
            _issue_chunk(*nxt, 0, H, V, C, True, 2)
        qd_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_qd.index(cur), gl.DotOperandLayout(0, MMA, 8)
        )
        mqk_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_mqk.index(cur).slice(C if NUM_WARPS == 4 else 0, C),
            gl.DotOperandLayout(0, MMA, 8),
        )
        o = _output(h_op, big_u, qd_a, mqk_a, MMA)
        if NORM:
            # Finish chunk j - 1 (masked off at j == 0): its partials crossed the barrier.
            o_n = _gnorm(o_p, _rowss_get(s_ss, MT), g_p, w_n, norm_eps, V)
            m_p = (o_c_t < (tok_end - t_p).to(gl.int32)) & (j > 0)
            _store_o(o_n, out, t_p, o_off, m_p, H, V, CM_OUT)
            _rowss_put(o, s_ss)
            o_p = o
            g_p = g_c
            t_p = t0
        else:
            _store_o(
                o, out, t0, o_off, o_c_t < (tok_end - t0).to(gl.int32), H, V, CM_OUT
            )

        kr_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            s_kr.index(cur).permute((1, 0)), gl.DotOperandLayout(0, MMA, 8)
        )
        h = _update(h, kr_a, big_u, gt, MMA)
        if ckpt_state is not None:  # noqa: SIM102
            if j + 1 == ckpt_j:
                gl.amd.cdna4.buffer_store(
                    h, ckpt_state + ckpt_off + i_h * (V * K), s_off_vk
                )
        if not NORM:
            _issue_chunk(*nxt, 0, H, V, C, True, 12)
    if NORM:  # noqa: SIM102
        if n_chunks > 0:
            o_n = _gnorm(o_p, _rowss_get(s_ss, MT), g_p, w_n, norm_eps, V)
            m_p = o_c_t < (tok_end - t_p).to(gl.int32)
            _store_o(o_n, out, t_p, o_off, m_p, H, V, CM_OUT)

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


def gluon_k2_ab_fused(
    ws_kd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    h_out_b,
    h_out_a,
    seg_rows,
    TOTAL_TILES,
    H,
    K,
    V,
    C,
    BW,
    num_warps,
    waves_per_eu=0,
):
    return k2_ab_fused_fast[(triton.cdiv(V, BW), seg_rows.shape[0])](
        ws_kd=ws_kd,
        ws_kr=ws_kr,
        ws_gt=ws_gt,
        ws_inv_mqk=ws_inv_mqk,
        v_input=v_input,
        h_out_b=h_out_b,
        h_out_a=h_out_a,
        seg_rows=seg_rows,
        TOTAL_TILES=TOTAL_TILES,
        H=H,
        K=K,
        V=V,
        C=C,
        BW=BW,
        NUM_XCDS=get_num_xcds(),
        NUM_WARPS=num_warps,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )


def gluon_k2_c(
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
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
    out_gate=None,
    norm_weight=None,
    norm_eps=1e-5,
    xbuf=None,
    xepoch=None,
    scf=None,
    ckpt_state=None,
    ckpt_offsets=None,
    ckpt_rows=None,
    ckpt_bos=None,
):
    paged = state_cache is not None
    nx = get_num_xcds()
    extra = {}
    if xbuf is not None:
        num_warps = 8
        grid = (1, triton.cdiv(num_segs * H, nx) * nx * (V // BW))
    elif scf is not None:
        grid = (scf["tab"].numel() // 2,)
        extra = scf
    else:
        grid = (triton.cdiv(V, BW), num_segs * H)
    return k2_c_fast[grid](
        ws_kd=ws_kd,
        ws_qd=ws_qd,
        ws_kr=ws_kr,
        ws_gt=ws_gt,
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
        NUM_XCDS=nx,
        NUM_WARPS=num_warps,
        CM_OUT=CM_OUT,
        gate=out_gate,
        norm_w=norm_weight,
        gate_stride=out_gate.stride(1) if out_gate is not None else 0,
        norm_eps=float(norm_eps),
        NORM=out_gate is not None and xbuf is None,
        xepoch=xepoch if scf is None else scf.pop("xepoch"),
        xbuf=xbuf,
        n_str=num_segs * H,
        ckpt_state=ckpt_state,
        ckpt_offsets=ckpt_offsets,
        ckpt_rows=ckpt_rows,
        ckpt_bos=ckpt_bos,
        CKPT_STRIDE=0 if ckpt_state is None else ckpt_state.stride(0),
        **extra,
        num_warps=num_warps,
        waves_per_eu=2 if scf is not None and out_gate is None else 0,
        llvm_fn_attrs=(("amdgpu-sched-strategy", "max-ilp"),),
    )
