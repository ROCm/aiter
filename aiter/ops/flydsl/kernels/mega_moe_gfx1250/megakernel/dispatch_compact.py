# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""compact's dispatch (phases 1a. to 1c. of the package docstring):
tally, X1 count exchange, layout and send."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import ballot, ds_bpermute, readlane
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _cvt_scalef32_pk8_fp4_bf16 as emit_cvt_scalef32_pk8_fp4_bf16,
)
from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE as _MX_ROUND_UP
from aiter.utility.mx_types import MxDtypeInt as _MxDtype

from ..compact_plan import _wave32_inclusive_scan_i32
from ..config import _WAVE_SIZE as WAVE
from ..s1f_stage1 import (
    _OOB,
    S1F_SLOTS,
    SCOPE_DEV,
    SCOPE_SYS,
    _buffer_atomic_add_ret,
    _PCtx,
    _rawbar,
    _uniform,
    _vld,
    _vst4,
    s1f_produce,
)
from .protocol import (
    AMAX_FLOOR,
    CTL_PHEAD,
    CTL_QDONE,
    CTL_TALLY,
    GATE_STRIDE,
    LEAD_SLEEP,
    N_SREP,
    SEG_BITS,
    SEG_STRIDE,
    TR_PROD0,
    TR_PROD1,
    _load_i32_dev,
    device_fn,
    vdst_fence,
)


@device_fn
def produce(kc, n_mine, n_routes):
    """1a.-1c.; returns the routes this wave sent."""
    DEST_STRIDE, E, EPR, EPR_CH, G = kc.DEST_STRIDE, kc.E, kc.EPR, kc.EPR_CH, kc.G
    GATE_S, H, K, MAXR, NG, PAYLOAD = kc.GATE_S, kc.H, kc.K, kc.MAXR, kc.NG, kc.PAYLOAD
    ROW, SEGS_PAD, SEG_CNT_W, SREP_W = kc.ROW, kc.SEGS_PAD, kc.SEG_CNT_W, kc.SREP_W
    TM, T_CHUNK, T_MAX, addr_ctl = kc.TM, kc.T_CHUNK, kc.T_MAX, kc.addr_ctl
    addr_route, alive, ar, bc, bid = kc.addr_route, kc.alive, kc.ar, kc.bc, kc.bid
    i32_rows, is_t0, l_m, l_rc, l_sb = kc.i32_rows, kc.is_t0, kc.l_m, kc.l_rc, kc.l_sb
    lane, lead_plan_lds = kc.lane, kc.lead_plan_lds
    lead_plan_replicas, npes, off_ctl = kc.lead_plan_replicas, kc.npes, kc.off_ctl
    off_disp, off_rowmap, open_gates = kc.off_disp, kc.off_rowmap, kc.open_gates
    par, peer_va, rank, rsrc_actl = kc.par, kc.peer_va, kc.rank, kc.rsrc_actl
    rsrc_ctl, rsrc_ids, rsrc_wts = kc.rsrc_ctl, kc.rsrc_ids, kc.rsrc_wts
    rsrc_x, spin_end, spin_fail, stamp = kc.rsrc_x, kc.spin_end, kc.spin_fail, kc.stamp
    tag16, tid, tok, wait_gate, wave = kc.tag16, kc.tid, kc.tok, kc.wait_gate, kc.wave
    STREAM = kc.STREAM
    if const_expr(STREAM):
        GATE_R, P, SEGCAP, gen = kc.GATE_R, kc.P, kc.SEGCAP, kc.gen
        is_prod, l_st = kc.is_prod, kc.l_st
        rsrc_send = create_buffer_resource_from_addr(
            kc.addr_send, num_records_bytes=4 * kc.SEND_N
        )
        rsrc_wire = create_buffer_resource_from_addr(
            kc.addr_wire, num_records_bytes=kc.WIRE_B
        )
    # ======== 1. DISPATCH (compact layout) ========
    # LDS scratch in the GEMM arena (no GEMM item runs before the
    # plan gate): a tally CTA's histogram, bases and route table;
    # CTA 0's X1 matrix, send table and receive counts.
    rsrc_route = create_buffer_resource_from_addr(
        addr_route, num_records_bytes=4 * MAXR
    )
    T_ = (n_routes + fx.Int32(T_CHUNK - 1)) // fx.Int32(T_CHUNK)
    T_ = fx.Int32((T_ < fx.Int32(1)).select(fx.Int32(1), T_))
    T_ = fx.Int32((T_ > fx.Int32(T_MAX)).select(fx.Int32(T_MAX), T_))
    # ---- tally: CTAs 1..T_ count a chunk of routes each. LDS
    # first, then one device atomic per (CTA, segment): the
    # segment's global base for this chunk, so a hot segment (the
    # shared expert) takes T_ atomics, not one per route.
    if (bid >= fx.Int32(1)) & (bid <= T_):
        l_hist = ar
        l_hb = ar + fx.Int32(SEGS_PAD * 4)
        l_rt = ar + fx.Int32(2 * SEGS_PAD * 4)
        C_ = (n_routes + T_ - fx.Int32(1)) // T_
        r0 = (bid - fx.Int32(1)) * C_
        nloc = n_routes - r0
        nloc = fx.Int32((nloc > C_).select(C_, nloc))
        nloc = fx.Int32((nloc < fx.Int32(0)).select(fx.Int32(0), nloc))
        for i in range(tid, SEGS_PAD, 4 * WAVE):
            comm_ops.store_i32_lds(fx.Int64(l_hist + i * fx.Int32(4)), 0)
        _rawbar()
        for i in range(tid, nloc, 4 * WAVE):
            gid = fx.Int32(buffer_load(rsrc_ids, r0 + i, vec_width=1, dtype=T.i32))
            ok = (gid >= fx.Int32(0)) & (gid < fx.Int32(E))
            g_s = fx.Int32(ok.select(gid, fx.Int32(0)))
            loc = fx.Int32(
                comm_ops.atomic_add_lds(
                    fx.Int64(l_hist + g_s * fx.Int32(4)),
                    fx.Int32(ok.select(fx.Int32(1), fx.Int32(0))),
                )
            )
            comm_ops.store_i32_lds(
                fx.Int64(l_rt + i * fx.Int32(4)),
                fx.Int32(ok.select(g_s | (loc << fx.Int32(SEG_BITS)), fx.Int32(-1))),
            )
        _rawbar()
        # Every segment's atomic in flight before any return is
        # used, so the round trips overlap.
        bases = []
        for c in range_constexpr(SEGS_PAD // (4 * WAVE)):
            i = tid + fx.Int32(c * 4 * WAVE)
            n_ = fx.Int32(comm_ops.load_i32_lds(fx.Int64(l_hist + i * fx.Int32(4))))
            bases.append(
                _buffer_atomic_add_ret(
                    rsrc_ctl,
                    fx.Int32(
                        (n_ > fx.Int32(0)).select(
                            (fx.Int32(SEG_CNT_W) + i * fx.Int32(SEG_STRIDE))
                            * fx.Int32(4),
                            fx.Int32(_OOB),
                        )
                    ),
                    n_,
                    SCOPE_DEV,
                )
            )
        for c in range_constexpr(SEGS_PAD // (4 * WAVE)):
            i = tid + fx.Int32(c * 4 * WAVE)
            comm_ops.store_i32_lds(fx.Int64(l_hb + i * fx.Int32(4)), bases[c])
        _rawbar()
        for i in range(tid, nloc, 4 * WAVE):
            v = fx.Int32(comm_ops.load_i32_lds(fx.Int64(l_rt + i * fx.Int32(4))))
            sg = v & fx.Int32((1 << SEG_BITS) - 1)
            hb = fx.Int32(comm_ops.load_i32_lds(fx.Int64(l_hb + sg * fx.Int32(4))))
            if const_expr(STREAM):
                # The send list: segment sg's routes at sg * SEGCAP + index.
                ok_v = v >= fx.Int32(0)
                buffer_store(
                    r0 + i,
                    rsrc_send,
                    fx.Int32(
                        ok_v.select(
                            sg * fx.Int32(SEGCAP) + hb + (v >> fx.Int32(SEG_BITS)),
                            fx.Int32(0),
                        )
                    ),
                    mask=ok_v,
                    cache_modifier=SCOPE_DEV,
                )
            else:
                slot = fx.Int32(
                    (v >= fx.Int32(0)).select(
                        sg | ((hb + (v >> fx.Int32(SEG_BITS))) << fx.Int32(SEG_BITS)),
                        fx.Int32(-1),
                    )
                )
                buffer_store(slot, rsrc_route, r0 + i, cache_modifier=SCOPE_DEV)
        rocdl.s_wait_storecnt(0)
        _rawbar()
        if is_t0:
            old_t = fx.Int32(
                comm_ops.atomic_add_agent(
                    addr_ctl + fx.Int64(CTL_TALLY * 4) + fx.Int64(par) * 128,
                    fx.Int32(1),
                )
            )
            comm_ops.store_i32_lds(fx.Int64(bc + fx.Int32(20)), old_t)
        _rawbar()
        old_t = _uniform(comm_ops.load_i32_lds(fx.Int64(bc + fx.Int32(20))))
        # The last tally CTA reads (and clears) the segment totals and
        # sends them, tagged per dword, to every peer: X1.
        if (old_t == T_ - fx.Int32(1)) & (wave == fx.Int32(0)):
            # Every count load in flight at once (SCOPE_SYS, not
            # volatile: a volatile load waits before the next).
            # They sit in this branch, after the tally atomic that
            # chose the completer, so none can move above it.
            cws = []
            cvs = []
            for j in range_constexpr(SEGS_PAD // 128):
                for q in range_constexpr(4):
                    w_ = fx.Int32(SEG_CNT_W) + (
                        fx.Int32(128 * j + q) + lane * fx.Int32(4)
                    ) * fx.Int32(SEG_STRIDE)
                    cws.append(w_)
                    cvs.append(
                        fx.Int32(
                            buffer_load(
                                rsrc_ctl,
                                w_,
                                vec_width=1,
                                dtype=T.i32,
                                cache_modifier=SCOPE_SYS,
                            )
                        )
                    )
            for j in range_constexpr(SEGS_PAD // 128):
                cs = []
                cw = cws[4 * j : 4 * j + 4]
                for q in range_constexpr(4):
                    c_ = cvs[4 * j + q]
                    c_ = fx.Int32((c_ > fx.Int32(0xFFFF)).select(fx.Int32(0xFFFF), c_))
                    cs.append((tag16 << fx.Int32(16)) | c_)
                vec = fx.Vector.from_elements(cs, dtype=fx.Int32)
                for pe in range_constexpr(npes):
                    rsrc_x1 = create_buffer_resource_from_addr(
                        peer_va(pe, off_ctl),
                        num_records_bytes=npes * SEGS_PAD * 4,
                    )
                    buffer_store(
                        vec,
                        rsrc_x1,
                        fx.Int32(rank * SEGS_PAD + 128 * j) + lane * fx.Int32(4),
                        cache_modifier=SCOPE_SYS,
                    )
                for q in range_constexpr(4):
                    buffer_store(
                        fx.Int32(0),
                        rsrc_ctl,
                        cw[q],
                        cache_modifier=SCOPE_DEV,
                    )
            stamp(23)
    # ---- CTA 0: X1 from every source -> every destination's
    # layout (the staged compact one: expert-major, each expert
    # padded to tile_m, sources in rank order inside it) -> the
    # send table (replicated) -> send gates. Every rank lays out
    # from the same matrix, so senders and receivers agree.
    if bid == fx.Int32(0):
        tag_hi = tag16 << fx.Int32(16)

        def x1_load(w0):
            return fx.Vector(
                buffer_load(
                    rsrc_actl,
                    w0,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=SCOPE_SYS | (1 << 31),
                )
            )

        def x1_bad(v4):
            """Ballot of the lanes holding another launch's tag."""
            badl = fx.Int32(0)
            for q in range_constexpr(4):
                badl = badl | fx.Int32(
                    ((fx.Int32(v4[q]) & fx.Int32(-65536)) != tag_hi).select(
                        fx.Int32(1), fx.Int32(0)
                    )
                )
            return fx.Int32(ballot(T.i32, badl != fx.Int32(0)))

        for sj in range_constexpr((npes + 3) // 4):
            s_ = wave + fx.Int32(4 * sj)
            if s_ < fx.Int32(npes):
                for j in range_constexpr(SEGS_PAD // 128):
                    w0 = (
                        s_ * fx.Int32(SEGS_PAD) + fx.Int32(128 * j) + lane * fx.Int32(4)
                    )
                    v4 = x1_load(w0)
                    bad = x1_bad(v4)
                    t_end = spin_end()
                    live = fx.Int32(1)
                    while (bad != fx.Int32(0)) & (live != fx.Int32(0)):
                        fx.rocdl.s_sleep(LEAD_SLEEP)
                        v4 = x1_load(w0)
                        bad = x1_bad(v4)
                        live = alive(t_end)
                    if bad != fx.Int32(0):
                        spin_fail(0)
                    for q in range_constexpr(4):
                        comm_ops.store_i32_lds(
                            fx.Int64(l_m + (w0 + fx.Int32(q)) * fx.Int32(4)),
                            fx.Int32(v4[q]) & fx.Int32(0xFFFF),
                        )
        _rawbar()
        stamp(20)
        for dj in range_constexpr((npes + 3) // 4):
            d_ = wave + fx.Int32(4 * dj)
            if d_ < fx.Int32(npes):
                carry = fx.Int32(0)
                for ch in range_constexpr(EPR_CH):
                    le = lane + fx.Int32(ch * WAVE)
                    live = le < fx.Int32(EPR)
                    sg = fx.Int32(live.select(d_ * fx.Int32(EPR) + le, fx.Int32(0)))
                    tot = fx.Int32(0)
                    pre = fx.Int32(0)
                    for s2 in range_constexpr(npes):
                        m_ = fx.Int32(
                            comm_ops.load_i32_lds(
                                fx.Int64(
                                    l_m + (fx.Int32(s2 * SEGS_PAD) + sg) * fx.Int32(4)
                                )
                            )
                        )
                        tot = tot + m_
                        if const_expr(s2 < rank):
                            pre = pre + m_
                    tot = fx.Int32(live.select(tot, fx.Int32(0)))
                    al = (tot + fx.Int32(TM - 1)) // fx.Int32(TM) * fx.Int32(TM)
                    incl = _wave32_inclusive_scan_i32(al, lane) + carry
                    start = incl - al
                    carry = _uniform(readlane(T.i32, arith.unwrap(incl), WAVE - 1))
                    if live:
                        comm_ops.store_i32_lds(
                            fx.Int64(l_sb + sg * fx.Int32(4)), start + pre
                        )
                    # This rank's own receive side: rows at or past
                    # i32_rows are dropped by senders and receiver
                    # alike (an expert starts on a tile, so whole
                    # tiles go).
                    if (d_ == fx.Int32(rank)) & live:
                        if const_expr(STREAM):
                            comm_ops.store_i32_lds(
                                fx.Int64(l_st + le * fx.Int32(4)), start
                            )
                        room = i32_rows - start
                        room = fx.Int32((room < fx.Int32(0)).select(fx.Int32(0), room))
                        comm_ops.store_i32_lds(
                            fx.Int64(l_rc + le * fx.Int32(4)),
                            fx.Int32((tot > room).select(room, tot)),
                        )
        _rawbar()
        if const_expr(STREAM):
            # The producers need only the segment table and the wire rows:
            # their gate (GATE_R) opens before the plan and the masks.
            stream_segtable(kc)
            rocdl.s_wait_storecnt(0)
            _rawbar()
            if wave == fx.Int32(0):
                # The producers gather the wire rows every other CTA quantized.
                qd_w = fx.Int32(CTL_QDONE) + par * fx.Int32(32)
                qd = _uniform(_load_i32_dev(rsrc_ctl, qd_w))
                t_end = spin_end()
                live = fx.Int32(1)
                while (qd < fx.Int32(G - 1)) & (live != fx.Int32(0)):
                    fx.rocdl.s_sleep(LEAD_SLEEP)
                    qd = _uniform(_load_i32_dev(rsrc_ctl, qd_w))
                    live = alive(t_end)
                if qd < fx.Int32(G - 1):
                    spin_fail(1)
                for c in range_constexpr((P + WAVE - 1) // WAVE):
                    b_ = lane + fx.Int32(c * WAVE + 1)
                    buffer_store(
                        gen,
                        rsrc_ctl,
                        fx.Int32(GATE_R) + b_ * fx.Int32(GATE_STRIDE),
                        mask=b_ <= fx.Int32(P),
                        cache_modifier=SCOPE_DEV,
                    )
            stamp(15)
        # The plan is known now (this rank's receive counts, which
        # the wave laying out this rank wrote): wave 0 lays it
        # out, every wave publishes it with the send table below.
        if wave == fx.Int32(0):
            cnts = []
            for j in range_constexpr(EPR_CH):
                e_ = lane + fx.Int32(j * WAVE)
                c_ = fx.Int32(comm_ops.load_i32_lds(fx.Int64(l_rc + e_ * fx.Int32(4))))
                cnts.append(fx.Int32((e_ < fx.Int32(EPR)).select(c_, fx.Int32(0))))
            lead_plan_lds(cnts)
        _rawbar()
        lead_plan_replicas()
        if const_expr(STREAM):
            stream_masks(kc)
            stamp(16)
        else:
            sbv = []
            for j in range_constexpr(SEGS_PAD // 128):
                for q in range_constexpr(4):
                    sbv.append(
                        fx.Int32(
                            comm_ops.load_i32_lds(
                                fx.Int64(
                                    l_sb
                                    + (fx.Int32(128 * j + q) + lane * fx.Int32(4))
                                    * fx.Int32(4)
                                )
                            )
                        )
                    )
            for c in range_constexpr(N_SREP // 4):
                rep = wave * fx.Int32(N_SREP // 4) + fx.Int32(c)
                for j in range_constexpr(SEGS_PAD // 128):
                    buffer_store(
                        fx.Vector.from_elements(sbv[4 * j : 4 * j + 4], dtype=fx.Int32),
                        rsrc_ctl,
                        fx.Int32(SREP_W + 128 * j)
                        + rep * fx.Int32(SEGS_PAD)
                        + lane * fx.Int32(4),
                        cache_modifier=SCOPE_DEV,
                    )
        rocdl.s_wait_storecnt(0)
        _rawbar()
        stamp(21)
        if wave == fx.Int32(0):
            open_gates(GATE_S, GATE_STRIDE)
        stamp(22)
    # ---- every CTA but 0: one wave per token. The row is
    # quantized into registers once, then sent to each of its
    # routes' rows once the send gate is open.
    if bid > fx.Int32(0):
        rep_w = fx.Int32(SREP_W) + (bid & fx.Int32(N_SREP - 1)) * fx.Int32(SEGS_PAD)
        for t in range((bid - fx.Int32(1)) * fx.Int32(4) + wave, tok, (G - 1) * 4):
            n_mine = n_mine + fx.Int32(K)
            raws = []
            for jj in range_constexpr(NG):
                in_dw = t * fx.Int32(H // 2) + (lane + fx.Int32(jj * WAVE)) * fx.Int32(
                    16
                )
                for c in range_constexpr(4):
                    raws.append(
                        buffer_load(
                            rsrc_x,
                            in_dw + fx.Int32(4 * c),
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
            packed = []
            scales = []
            for jj in range_constexpr(NG):
                amax = fx.Float32(AMAX_FLOOR)
                chunks = []
                for c in range_constexpr(4):
                    bf = fx.Vector(raws[4 * jj + c]).bitcast(fx.BFloat16)
                    chunks.append(bf)
                    f32x8 = bf.to(fx.Float32)
                    for e_ in range_constexpr(8):
                        amax = fx.max(amax, abs(f32x8[e_]))
                e8m0 = emit_mx_e8m0_scale(
                    amax, mode=_MX_ROUND_UP, dtype=_MxDtype.FP4_E2M1
                )
                scale_f32 = (fx.Int32(e8m0) << fx.Int32(23)).bitcast(fx.Float32)
                for c in range_constexpr(4):
                    packed.append(
                        fx.Int32(
                            emit_cvt_scalef32_pk8_fp4_bf16(
                                chunks[c], scale_f32, i32_ty=T.i32
                            )
                        )
                    )
                scales.append(fx.Int32(e8m0) & fx.Int32(0xFF))
            packed_v = []
            for jj in range_constexpr(NG):
                packed_v.append(
                    vdst_fence(
                        fx.Vector.from_elements(
                            packed[4 * jj : 4 * jj + 4], dtype=fx.Int32
                        )
                    )
                )
            sc_dw = []
            for jj in range_constexpr(NG):
                w_ = fx.Int32(0)
                for b in range_constexpr(4):
                    src = ((lane & fx.Int32(7)) * fx.Int32(4) + fx.Int32(b)) * fx.Int32(
                        4
                    )
                    w_ = w_ | (
                        fx.Int32(ds_bpermute(T.i32, src, scales[jj])) << fx.Int32(8 * b)
                    )
                sc_dw.append(w_)
            if const_expr(STREAM):
                for jj in range_constexpr(NG):
                    buffer_store(
                        packed_v[jj],
                        rsrc_wire,
                        t * fx.Int32(ROW // 4)
                        + (lane + fx.Int32(jj * WAVE)) * fx.Int32(4),
                        cache_modifier=SCOPE_DEV,
                    )
                for jj in range_constexpr(NG):
                    buffer_store(
                        sc_dw[jj],
                        rsrc_wire,
                        t * fx.Int32(ROW // 4) + fx.Int32(PAYLOAD // 4 + jj * 8) + lane,
                        mask=lane < fx.Int32(8),
                        cache_modifier=SCOPE_DEV,
                    )
            else:
                wait_gate(GATE_S, GATE_STRIDE, 1)
                rk = fx.Int32((lane < fx.Int32(K)).select(lane, fx.Int32(0)))
                rs = _load_i32_dev(rsrc_route, t * fx.Int32(K) + rk)
                w_l = fx.Int32(
                    buffer_load(
                        rsrc_wts, t * fx.Int32(K) + rk, vec_width=1, dtype=T.i32
                    )
                )
                okr = (lane < fx.Int32(K)) & (rs >= fx.Int32(0))
                sg = fx.Int32(
                    okr.select(rs & fx.Int32((1 << SEG_BITS) - 1), fx.Int32(0))
                )
                sb = _load_i32_dev(rsrc_ctl, rep_w + sg)
                row_l = sb + (rs >> fx.Int32(SEG_BITS))
                okr = okr & (row_l < i32_rows)
                d_l = sg // fx.Int32(EPR)
                okv = fx.Int32(okr.select(fx.Int32(1), fx.Int32(0)))
                for k in range_constexpr(K):
                    okk = _uniform(readlane(T.i32, arith.unwrap(okv), k)) != fx.Int32(0)
                    row = _uniform(readlane(T.i32, arith.unwrap(row_l), k))
                    dst = _uniform(readlane(T.i32, arith.unwrap(d_l), k))
                    w_bits = _uniform(readlane(T.i32, arith.unwrap(w_l), k))
                    if okk:
                        rsrc_row = create_buffer_resource_from_addr(
                            peer_va(dst, off_disp) + fx.Int64(row) * fx.Int64(ROW),
                            num_records_bytes=ROW,
                        )
                        for jj in range_constexpr(NG):
                            buffer_store(
                                packed_v[jj],
                                rsrc_row,
                                (lane + fx.Int32(jj * WAVE)) * fx.Int32(4),
                                cache_modifier=SCOPE_SYS,
                            )
                        for jj in range_constexpr(NG):
                            buffer_store(
                                sc_dw[jj],
                                rsrc_row,
                                fx.Int32(PAYLOAD // 4 + jj * 8) + lane,
                                mask=lane < fx.Int32(8),
                                cache_modifier=SCOPE_SYS,
                            )
                        rsrc_rm = create_buffer_resource_from_addr(
                            peer_va(dst, off_rowmap) + fx.Int64(row) * fx.Int64(8),
                            num_records_bytes=8,
                        )
                        meta = (
                            fx.Int32(rank * DEST_STRIDE) + t * fx.Int32(K) + fx.Int32(k)
                        )
                        buffer_store(
                            fx.Vector.from_elements([meta, w_bits], dtype=fx.Int32),
                            rsrc_rm,
                            fx.Int32(0),
                            mask=lane == fx.Int32(0),
                            cache_modifier=SCOPE_SYS,
                        )
    if const_expr(STREAM):
        if bid > fx.Int32(0):
            # Every wave's wire rows retired (SCOPE_DEV) before the count.
            rocdl.s_wait_storecnt(0)
            _rawbar()
            if is_t0:
                comm_ops.atomic_add_agent(
                    addr_ctl + fx.Int64(CTL_QDONE * 4) + fx.Int64(par) * 128,
                    fx.Int32(1),
                )
        if is_prod:
            stream_produce(kc)
    return n_mine


@device_fn
def stream_segtable(kc):
    """STREAM, CTA 0 (every wave), after the layout: the segment table the
    producers decode their runs from.

    Run order (S1F's): local expert outer, the last SF first; destination
    rotated by rank inside; then the segment's rows, cut at flag tiles into
    runs of at most R_RUN rows."""
    EPR, FM, FSH, NPOS, NPOS2 = kc.EPR, kc.FM, kc.FSH, kc.NPOS, kc.NPOS2
    R, SEGS_PAD, SF = kc.R_RUN, kc.SEGS_PAD, kc.SF
    i32_rows, l_m = kc.i32_rows, kc.l_m
    l_scan, l_sb, lane = kc.l_scan, kc.l_sb, kc.lane
    npes, rank, tid, wave = kc.npes, kc.rank, kc.tid, kc.wave
    rsrc_segt = create_buffer_resource_from_addr(
        kc.addr_segt, num_records_bytes=(4 * NPOS2 + 32) * 4
    )

    def lds(a):
        return fx.Int32(comm_ops.load_i32_lds(fx.Int64(a)))

    def imin(a, b):
        return fx.Int32((a < b).select(a, b))

    # ---- segment table: thread t owns positions [Q t, Q t + Q) ----
    Q = NPOS2 // (4 * WAVE)
    ents = []
    tot = fx.Int32(0)
    for q in range_constexpr(Q):
        p = tid * fx.Int32(Q) + fx.Int32(q)
        valid = p < fx.Int32(NPOS)
        lo = p // fx.Int32(npes)
        j = p - lo * fx.Int32(npes)
        le = lo
        if const_expr(SF):
            le = lo + fx.Int32(EPR - SF)
            le = fx.Int32((le >= fx.Int32(EPR)).select(le - fx.Int32(EPR), le))
        d = j + fx.Int32(rank)
        d = fx.Int32((d >= fx.Int32(npes)).select(d - fx.Int32(npes), d))
        seg = fx.Int32(valid.select(d * fx.Int32(EPR) + le, fx.Int32(0)))
        c_raw = lds(l_m + (fx.Int32(rank * SEGS_PAD) + seg) * fx.Int32(4))
        sb = lds(l_sb + seg * fx.Int32(4))
        end = imin(sb + c_raw, i32_rows)
        c = end - sb
        c = fx.Int32(((c > fx.Int32(0)) & valid).select(c, fx.Int32(0)))
        fl = imin(c, fx.Int32(FM) - (sb & fx.Int32(FM - 1)))
        rest = c - fl
        nr = (
            (fl + fx.Int32(R - 1)) // fx.Int32(R)
            + (rest >> fx.Int32(FSH)) * fx.Int32(FM // R)
            + ((rest & fx.Int32(FM - 1)) + fx.Int32(R - 1)) // fx.Int32(R)
        )
        ents.append((p, valid, seg, sb, c, nr))
        tot = tot + nr
    incl = _wave32_inclusive_scan_i32(tot, lane)
    if lane == fx.Int32(WAVE - 1):
        comm_ops.store_i32_lds(fx.Int64(l_scan + wave * fx.Int32(4)), incl)
    _rawbar()
    pre = fx.Int32(0)
    all_runs = fx.Int32(0)
    for w in range_constexpr(4):
        wt = lds(l_scan + fx.Int32(4 * w))
        pre = pre + fx.Int32((wave > fx.Int32(w)).select(wt, fx.Int32(0)))
        all_runs = all_runs + wt
    run0 = pre + incl - tot
    for q in range_constexpr(Q):
        p, valid, seg, sb, c, nr = ents[q]
        vals = (fx.Int32(valid.select(run0, fx.Int32(0x7FFFFFFF))), sb, c, seg)
        for k_ in range_constexpr(4):
            buffer_store(
                vals[k_],
                rsrc_segt,
                fx.Int32(k_ * NPOS2) + p,
                cache_modifier=SCOPE_DEV,
            )
        run0 = run0 + nr
    if tid == fx.Int32(0):
        buffer_store(all_runs, rsrc_segt, fx.Int32(4 * NPOS2), cache_modifier=SCOPE_DEV)


@device_fn
def stream_masks(kc):
    """STREAM, CTA 0 (every wave): each of this rank's flag tiles' source
    mask, a thread per local expert."""
    EPR, FM, FSH, SEGS_PAD, SUB = kc.EPR, kc.FM, kc.FSH, kc.SEGS_PAD, kc.SUB
    TILE_CAP, TM, i32_rows, l_m = kc.TILE_CAP, kc.TM, kc.i32_rows, kc.l_m
    l_rc, l_st, npes, rank, tid = kc.l_rc, kc.l_st, kc.npes, kc.rank, kc.tid
    rsrc_tsrc = create_buffer_resource_from_addr(
        kc.addr_tsrc, num_records_bytes=TILE_CAP * 4
    )

    def lds(a):
        return fx.Int32(comm_ops.load_i32_lds(fx.Int64(a)))

    for le in range(tid, EPR, 4 * WAVE):
        base = lds(l_st + le * fx.Int32(4))
        rows = lds(l_rc + le * fx.Int32(4))
        starts = []
        ms = []
        acc = base
        for s_ in range_constexpr(npes):
            m_s = lds(l_m + (fx.Int32(s_ * SEGS_PAD + rank * EPR) + le) * fx.Int32(4))
            starts.append(acc)
            ms.append(m_s)
            acc = acc + m_s
        t0 = base >> fx.Int32(FSH)
        t1 = t0 + (rows + fx.Int32(TM - 1)) // fx.Int32(TM) * fx.Int32(SUB)
        t1 = fx.Int32((t1 < fx.Int32(TILE_CAP)).select(t1, fx.Int32(TILE_CAP)))
        for tau in range(t0, t1):
            lo_r = tau << fx.Int32(FSH)
            hi_r = lo_r + fx.Int32(FM)
            msk = fx.Int32(0)
            for s_ in range_constexpr(npes):
                hit = (
                    (ms[s_] > fx.Int32(0))
                    & (starts[s_] < hi_r)
                    & (starts[s_] + ms[s_] > lo_r)
                    & (starts[s_] < i32_rows)
                )
                msk = msk | fx.Int32(hit.select(fx.Int32(1 << s_), fx.Int32(0)))
            buffer_store(msk, rsrc_tsrc, tau, cache_modifier=SCOPE_DEV)


@device_fn
def stream_produce(kc):
    """STREAM, producer CTAs (1..P), every wave: once the send gate is open,
    S1F's producer pipeline over this rank's runs, decoded from the segment
    table (copied into LDS past the producer's ring)."""
    EPR, FM, FSH, GATE_R, NPOS2 = kc.EPR, kc.FM, kc.FSH, kc.GATE_R, kc.NPOS2
    R, RBK, ROW, SEGCAP, TBL_OFF = kc.R_RUN, kc.RBK, kc.ROW, kc.SEGCAP, kc.TBL_OFF
    CAP, H, K, MAXR, npes, rank = kc.CAP, kc.H, kc.K, kc.MAXR, kc.npes, kc.rank
    ar, gen, lane, par, tid, wave = kc.ar, kc.gen, kc.lane, kc.par, kc.tid, kc.wave
    stamp, tok, wait_gate = kc.stamp, kc.tok, kc.wait_gate
    wait_gate(GATE_R, GATE_STRIDE, 1)
    rsrc_segt = create_buffer_resource_from_addr(
        kc.addr_segt, num_records_bytes=(4 * NPOS2 + 32) * 4
    )
    tb = ar + fx.Int32(TBL_OFF)
    # Every load in flight before the first LDS store.
    NV = 4 * NPOS2 // (4 * 4 * WAVE)
    vs = []
    for c in range_constexpr(NV):
        vs.append(
            fx.Vector(
                buffer_load(
                    rsrc_segt,
                    (tid + fx.Int32(c * 4 * WAVE)) * fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=SCOPE_DEV,
                )
            )
        )
    n_runs = _uniform(_load_i32_dev(rsrc_segt, fx.Int32(4 * NPOS2)))
    for c in range_constexpr(NV):
        _vst4(
            tb + (tid + fx.Int32(c * 4 * WAVE)) * fx.Int32(16),
            [fx.Int32(vs[c][q]) for q in range_constexpr(4)],
        )
    _rawbar()
    stamp(TR_PROD0)
    peer_bits = max(1, (int(npes) - 1).bit_length())

    def run_entry(q):
        """Lane's run q: the last position whose first run is <= q, then
        the run's rows inside that segment."""
        qq = fx.Int32((q > n_runs - fx.Int32(1)).select(n_runs - fx.Int32(1), q))
        qq = fx.Int32((qq < fx.Int32(0)).select(fx.Int32(0), qq))
        pos = fx.Int32(0)
        for b_ in range_constexpr(NPOS2.bit_length() - 1):
            st = NPOS2 >> (b_ + 1)
            v = _vld(tb + (pos + fx.Int32(st)) * fx.Int32(4))
            pos = fx.Int32((v <= qq).select(pos + fx.Int32(st), pos))
        k = qq - _vld(tb + pos * fx.Int32(4))
        sb = _vld(tb + (pos + fx.Int32(NPOS2)) * fx.Int32(4))
        c = _vld(tb + (pos + fx.Int32(2 * NPOS2)) * fx.Int32(4))
        seg = _vld(tb + (pos + fx.Int32(3 * NPOS2)) * fx.Int32(4))
        dest = seg // fx.Int32(EPR)
        fl = fx.Int32(FM) - (sb & fx.Int32(FM - 1))
        fl = fx.Int32((c < fl).select(c, fl))
        f = (fl + fx.Int32(R - 1)) // fx.Int32(R)
        row0 = fx.Int32(
            (k < f).select(sb + k * fx.Int32(R), sb + fl + (k - f) * fx.Int32(R))
        )
        lim = ((row0 >> fx.Int32(FSH)) + fx.Int32(1)) << fx.Int32(FSH)
        end = sb + c
        lim = fx.Int32((end < lim).select(end, lim))
        n = lim - row0
        n = fx.Int32((n > fx.Int32(R)).select(fx.Int32(R), n))
        w0 = (row0 << fx.Int32(peer_bits)) | dest
        w1 = ((seg * fx.Int32(SEGCAP) + row0 - sb) << fx.Int32(4)) | n
        return w0, w1

    PAYLOAD_B = H // 2
    SCALE_B = H // 32
    s1f_produce(
        _PCtx(
            APRE=False,
            APRE_SCALE_OFF=0,
            FSH=FSH,
            OP_STORES=1,
            PAY_ITERS=0,
            R=R,
            RBK=RBK,
            S=S1F_SLOTS,
            SC_CHUNKS=0,
            SC_ITERS=0,
            ST_CAP=(min(3, (S1F_SLOTS + 1) // 2), min(3, S1F_SLOTS // 2)),
            WAIT_STEPS=[0, 1, 2, 3],
            addr_ph=kc.addr_ctl
            + fx.Int64(CTL_PHEAD * 4)
            + fx.Int64(par) * fx.Int64(128),
            addr_runs=kc.addr_segt,
            addr_send=kc.addr_send,
            block=4 * WAVE,
            compact_cap=CAP,
            gather_policy=SCOPE_DEV,
            gen=gen,
            inp_cur_tok=tok,
            inp_tok=kc.addr_wire,
            inp_wts=kc.addr_wts,
            lane=lane,
            meta_stride=kc.DEST_STRIDE,
            n_runs_in=n_runs,
            npes=npes,
            off_ep_rowmap=kc.off_rowmap,
            off_out_tok=kc.off_disp,
            off_tile_flag=kc.FLAG_OFF,
            payload_bytes=PAYLOAD_B,
            peer_bits=peer_bits,
            peer_mask=(1 << peer_bits) - 1,
            rank=rank,
            ring=ar,
            row_stride=ROW,
            run_cap=MAXR,
            run_entry=run_entry,
            run_hdr=0,
            scale_bytes=SCALE_B,
            send_cache=SCOPE_DEV,
            send_cap=kc.SEND_N,
            tid=tid,
            topk=K,
            wave=wave,
            win_base=kc.win_base,
            win_stride=kc.win_stride,
            wts_cap=MAXR,
        )
    )
    stamp(TR_PROD1)
