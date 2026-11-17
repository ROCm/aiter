# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The slots front end (phase 1. of the package docstring): one wave per
route reserves its destination slot and writes the row there; the
reserved signal (EARLY) and the W1 prefetch."""

from __future__ import annotations

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import ds_bpermute, readlane
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

from ..config import _WAVE_SIZE as WAVE
from ..s1f_stage1 import (
    _OOB,
    SCOPE_SYS,
    _buffer_atomic_add_ret,
    _now,
    _store_i64_sys,
    _uniform,
)
from .core import (
    AMAX_FLOOR,
    CTL_ERR,
    CTL_TALLY,
    FLAG_STRIDE,
    GATE_STRIDE,
    PREF_DEPTH,
    PREF_MIN_TOK,
    RT_N,
    RUN_STRIDE,
    device_fn,
)


@device_fn
def start_handshake(kc):
    """AITER_DMK_NO_HS=0: wait for every peer's ``started`` flag."""
    GATE_S, ST_OFF, bid, gen, lane = kc.GATE_S, kc.ST_OFF, kc.bid, kc.gen, kc.lane
    npes, off_ctl, open_gates, peer_va = kc.npes, kc.off_ctl, kc.open_gates, kc.peer_va
    poll_peer_flags, rank, wave = kc.poll_peer_flags, kc.rank, kc.wave
    # ---- start handshake (slots, AITER_DMK_NO_HS=0) ----
    # A peer writes this rank's counters, rows and rowmap only once
    # this launch has started (see the module docstring).
    if (bid == fx.Int32(0)) & (wave == fx.Int32(0)):
        if lane < fx.Int32(npes):
            comm_ops.store_i32_global_system_monotonic(
                peer_va(lane, off_ctl) + fx.Int64(ST_OFF + rank * FLAG_STRIDE),
                gen,
            )
        poll_peer_flags(ST_OFF, 8)
        open_gates(GATE_S, GATE_STRIDE)


@device_fn
def produce(kc, n_mine):
    """1. PRODUCE; returns (routes this wave produced, this rank's routes)."""
    CAP, DEST_STRIDE, E, EPR, EPR_PAD = kc.CAP, kc.DEST_STRIDE, kc.E, kc.EPR, kc.EPR_PAD
    G, GATE_S, H, HS, K, NG = kc.G, kc.GATE_S, kc.H, kc.HS, kc.K, kc.NG
    PAYLOAD, PH, ROW, RT_BASE = kc.PAYLOAD, kc.PH, kc.ROW, kc.RT_BASE
    SRC_OFF, TRACE, addr_ctl, lane = kc.SRC_OFF, kc.TRACE, kc.addr_ctl, kc.lane
    off_ctl, off_disp, off_rowmap, par = kc.off_ctl, kc.off_disp, kc.off_rowmap, kc.par
    peer_va, rank, rsrc_ids, rsrc_wts = kc.peer_va, kc.rank, kc.rsrc_ids, kc.rsrc_wts
    rsrc_x, stamp, tok, wait_gate = kc.rsrc_x, kc.stamp, kc.tok, kc.wait_gate
    wave_gid = kc.wave_gid
    # One wave per route (t, k): the token's quant is redone per route
    # (cheap next to the remote round trip) so the routes go out in
    # parallel. Its lane 0 reserves the destination slot first; the
    # atomic is in flight across the quant.
    n_routes = tok * fx.Int32(K)
    for r in range(wave_gid, n_routes, G * 4):
        n_mine = n_mine + fx.Int32(1)
        t = r // fx.Int32(K)
        gid = fx.Int32(
            buffer_load(rsrc_ids, r, vec_width=1, dtype=T.i32, is_scalar=True)
        )
        w_bits = fx.Int32(
            buffer_load(rsrc_wts, r, vec_width=1, dtype=T.i32, is_scalar=True)
        )
        valid = (gid >= fx.Int32(0)) & (gid < fx.Int32(E if PH & 1 else 0))
        gid_s = fx.Int32(valid.select(gid, fx.Int32(0)))
        dst = gid_s // fx.Int32(EPR)
        le = gid_s - dst * fx.Int32(EPR)
        live0 = valid & (lane == fx.Int32(0))
        # The row loads go out before the slot atomic: loads retire in
        # issue order, so behind the remote round trip the quant could
        # not start until the atomic returned.
        raws = []
        for jj in range_constexpr(NG):
            in_dw = t * fx.Int32(H // 2) + (lane + fx.Int32(jj * WAVE)) * fx.Int32(16)
            for c in range_constexpr(4):
                raws.append(
                    buffer_load(
                        rsrc_x,
                        in_dw + fx.Int32(4 * c),
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
        # Branch-free buffer atomic: idle lanes address past the
        # resource and touch no memory (an add of 0 by each idle lane
        # on one scratch word serialized ~1700 atomics per launch), and
        # the return is waited on only at its first use.
        # Every peer has started this launch (the row loads above
        # are local and go out first).
        if const_expr(HS):
            wait_gate(GATE_S, GATE_STRIDE, 1)
        rsrc_run = create_buffer_resource_from_addr(
            peer_va(dst, off_ctl), num_records_bytes=SRC_OFF
        )
        if const_expr(TRACE):
            t_issue = _now()
        off_l = _buffer_atomic_add_ret(
            rsrc_run,
            fx.Int32(
                live0.select(
                    (par * fx.Int32(EPR_PAD) + le) * fx.Int32(RUN_STRIDE * 4),
                    fx.Int32(_OOB),
                )
            ),
            1,
            SCOPE_SYS,
        )
        if const_expr(TRACE):
            rocdl.s_wait_loadcnt(0)
            t_back = _now()
            if (lane == fx.Int32(0)) & (r < fx.Int32(RT_N)):
                rt_va = addr_ctl + fx.Int64(RT_BASE * 4) + fx.Int64(r * fx.Int32(32))
                _store_i64_sys(rt_va, t_issue)
                _store_i64_sys(rt_va + fx.Int64(8), t_back)
                _store_i64_sys(rt_va + fx.Int64(24), fx.Int64(dst))
        # Lane l owns 32-column groups l + 32 * jj: 64 B of bf16 in,
        # 16 B of fp4 and one e8m0 byte out (compact_plan._wire_quant).
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
            e8m0 = emit_mx_e8m0_scale(amax, mode=_MX_ROUND_UP, dtype=_MxDtype.FP4_E2M1)
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
        # Lane q < 8 packs the e8m0 bytes of groups 4q .. 4q+3 of each
        # 32-group slab into one dword.
        sc_dw = []
        for jj in range_constexpr(NG):
            w_ = fx.Int32(0)
            for b in range_constexpr(4):
                src = ((lane & fx.Int32(7)) * fx.Int32(4) + fx.Int32(b)) * fx.Int32(4)
                w_ = w_ | (
                    fx.Int32(ds_bpermute(T.i32, src, scales[jj])) << fx.Int32(8 * b)
                )
            sc_dw.append(w_)
        off = _uniform(readlane(T.i32, arith.unwrap(off_l), 0))
        if const_expr(TRACE):
            stamp(8)
        if valid & (off < fx.Int32(CAP)):
            row = le * fx.Int32(CAP) + off
            # Sized resources: a masked buffer store sends its idle
            # lanes to offset 0x7FFFFFFF, which must be OOB.
            rsrc_row = create_buffer_resource_from_addr(
                peer_va(dst, off_disp) + fx.Int64(row) * fx.Int64(ROW),
                num_records_bytes=ROW,
            )
            for jj in range_constexpr(NG):
                buffer_store(
                    fx.Vector.from_elements(
                        packed[4 * jj : 4 * jj + 4], dtype=fx.Int32
                    ),
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
            if const_expr(TRACE):  # noqa: SIM102 - preserve DSL staging
                if (lane == fx.Int32(0)) & (r < fx.Int32(RT_N)):
                    _store_i64_sys(
                        addr_ctl
                        + fx.Int64(RT_BASE * 4)
                        + fx.Int64(r * fx.Int32(32))
                        + fx.Int64(16),
                        _now(),
                    )
            meta = fx.Int32(rank * DEST_STRIDE) + r
            buffer_store(
                fx.Vector.from_elements([meta, w_bits], dtype=fx.Int32),
                rsrc_rm,
                fx.Int32(0),
                mask=lane == fx.Int32(0),
                cache_modifier=SCOPE_SYS,
            )
        if valid & (off >= fx.Int32(CAP)) & (lane == fx.Int32(0)):
            comm_ops.atomic_add_agent(addr_ctl + fx.Int64(CTL_ERR * 4), fx.Int32(1))
    return n_mine, n_routes


@device_fn
def signal_reserved(kc, n_mine, n_routes):
    """EARLY: tell every destination its counters from this rank are final."""
    RS_OFF, addr_ctl, bid, gen, lane = kc.RS_OFF, kc.addr_ctl, kc.bid, kc.gen, kc.lane
    npes, off_ctl, par, peer_va, rank = kc.npes, kc.off_ctl, kc.par, kc.peer_va, kc.rank
    wave = kc.wave
    # Every slot atomic of this wave has returned (its rows used
    # the offsets), so the peers' counters hold this wave's routes:
    # the wave completing the count tells every destination its
    # counters from this rank are final, before any row retires.
    res_before = fx.Int32(0)
    if (n_mine > fx.Int32(0)) & (lane == fx.Int32(0)):
        res_before = fx.Int32(
            comm_ops.atomic_add_agent(
                addr_ctl + fx.Int64(CTL_TALLY * 4) + fx.Int64(par) * 128,
                n_mine,
            )
        )
    res_before = _uniform(readlane(T.i32, arith.unwrap(res_before), 0))
    res_last = (n_mine > fx.Int32(0)) & (res_before + n_mine == n_routes)
    res_last = res_last | (
        (n_routes == fx.Int32(0)) & (bid == fx.Int32(0)) & (wave == fx.Int32(0))
    )
    if res_last & (lane < fx.Int32(npes)):
        comm_ops.store_i32_global_system_monotonic(
            peer_va(lane, off_ctl) + fx.Int64(RS_OFF + rank * FLAG_STRIDE),
            gen,
        )


@device_fn
def prefetch_w1(kc):
    """Read the W1 slice this CTA's first item most likely needs."""
    EPR, H, N1, NT1, PREF_B = kc.EPR, kc.H, kc.N1, kc.NT1, kc.PREF_B
    addr_ctl, arg_w1, bid, i32_rows = kc.addr_ctl, kc.arg_w1, kc.bid, kc.i32_rows
    tid, tile_n, tok = kc.tid, kc.tile_n, kc.tok
    # While the plan is pending HBM is idle: read the W1 slice this
    # CTA's first item most likely needs (m-tile bid // NT1 is
    # expert bid // NT1 when every expert before it has rows), so
    # that item finds it in cache. Real loads, PREF_DEPTH x 16 B in
    # flight per lane, folded into one word that is stored only on
    # an impossible value (global_prefetch hints measured no gain).
    e_p = bid // fx.Int32(NT1)
    n_p = bid - e_p * fx.Int32(NT1)
    if (bid != fx.Int32(0)) & (tok >= fx.Int32(PREF_MIN_TOK)) & (e_p < fx.Int32(EPR)):
        w1_base = fx.Int64(fx.ptrtoint(fx.recast_iter(fx.Int8, arg_w1)))
        rsrc_w1p = create_buffer_resource_from_addr(
            w1_base
            + fx.Int64(e_p * fx.Int32(N1) + n_p * fx.Int32(tile_n)) * fx.Int64(H // 2)
        )
        acc_p = fx.Int32(0)
        for p_off in range(tid * fx.Int32(16), PREF_B, 4 * WAVE * 16 * PREF_DEPTH):
            vs = [
                fx.Vector(
                    buffer_load(
                        rsrc_w1p,
                        (p_off + fx.Int32(q * 4 * WAVE * 16)) // fx.Int32(4),
                        vec_width=4,
                        dtype=T.i32,
                    )
                )
                for q in range_constexpr(PREF_DEPTH)
            ]
            for v4 in vs:
                acc_p = acc_p ^ fx.Int32(v4[0])
        if acc_p == i32_rows + fx.Int32(0x5A5A5A5A):
            comm_ops.atomic_add_agent(addr_ctl + fx.Int64(CTL_ERR * 4), fx.Int32(0))
