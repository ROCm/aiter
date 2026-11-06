# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric-arena addressing, epoch-flag readiness and the watchdog for the EP kernel.

Ported from :mod:`mega_moe_tp.communication` (peer / arena resource builders, the
sticky-error watchdog) with EP-specific readiness flags replacing TP's reduce-scatter /
all-gather ones:

* ``dispatch_*`` -- a dispatch-send CTA on a source rank adds its contribution to the
  expert-owner rank's arrival counter; the last arriver zeroes the counter and stamps
  the paired ready flag with the launch epoch. The owner's compute waves
  ``dispatch_wait_meta`` (spin until the ready flag reaches the epoch, wrap-safe).
* ``combine_*`` -- the mirror for GEMM2 partials scattered back to the source rank.

The device code reaches arena peers through ``peer_rs`` / ``own_rs`` (raw buffer
resources at ``peer_base + off_region``) exactly like the TP kernel.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from ..kernels_common import ceildiv
from .common import (
    ACQ_REL,
    AUX_SYS,
    V4I,
    alive,
    before,
    bf16x8_to_f32,
    bld,
    bst,
    g_add_agent,
    g_ld_sys,
    g_or_agent,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_ld_i32,
    lds_st,
    now,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    uni,
    wait_vm,
)
from .mega_moe_ep_config import (
    C_SDL,
    CTRL_ERR,
    ERR_COMBINE,
    ERR_COMPUTE,
    ERR_DISPATCH,
    FLAG_COMB_CNT,
    FLAG_DISPM,
    FLAG_DISPM_CNT,
    FLAG_INIT,
    FLAG_INIT_CNT,
    FLAG_PAYC,
    FLAG_SBD,
    FLAG_SBM,
    FLAG_SBP,
    FLAG_SRCM,
    GBAR_CNT,
    GBAR_GEN,
    GBAR_X,
    HVT,
    MAGG_G,
    MAGG_X,
    MAX_NPES,
    N_XCD,
    NCTA_MAX,
    SBX_SCNT,
    SBX_TB,
    SD_BASE,
    SD_CNT,
    SD_DUM,
    SD_GMAX,
    SD_LAND,
    SD_QP,
    SD_QS,
    SD_RD,
    SD_SELF,
    SD_TURN,
    KernelCtx,
)

__all__ = ["build_communication"]

# payload_send_ovl: rows per wave step whose loads are all issued before the step's remote
# stores.  gfx9's vmcnt counts loads AND stores in issue order, so a load issued after a
# remote (sc0 sc1) store waits for that store's xGMI round trip: one round-trip wait per
# step instead of one per 16-B chunk per owner.
DISP_TB = 6


# --- fp8 per-8 blockwise P2P codec for the combine send (halves the combine a2a traffic) ---
# Per-8-element E8M0 + fp8-e4m3 so the quant/decode stays WITHIN the existing 8-per-lane /
# 8-per-group combine loops (no 32-per-lane restructure -> no parallelism loss).  Finer than
# MXFP8's 1x32, so precision is at least as good (V2 validates 1x32 fp8 P2P for a4w4).
def _e8m0_scale(local_max):
    working = (local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    exponent = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(exponent + fx.Int32(1), exponent)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, scale


def _quantize_group8(v8):
    """8 f32 (list or vector) -> (E8M0 exponent, 2 i32 words = 8 fp8-e4m3), one per-8 scale."""
    local_max = fx.Float32(1e-10)
    for i in range_constexpr(8):
        local_max = local_max.maximumf(v8[i].maximumf(fx.Float32(0.0) - v8[i]))
    e8m0, scale = _e8m0_scale(local_max)
    w0 = rocdl.cvt_pk_fp8_f32(T.i32, v8[0] * scale, v8[1] * scale, fx.Int32(0), 0)
    w0 = rocdl.cvt_pk_fp8_f32(T.i32, v8[2] * scale, v8[3] * scale, w0, 1)
    w1 = rocdl.cvt_pk_fp8_f32(T.i32, v8[4] * scale, v8[5] * scale, fx.Int32(0), 0)
    w1 = rocdl.cvt_pk_fp8_f32(T.i32, v8[6] * scale, v8[7] * scale, w1, 1)
    return e8m0, w0, w1


def _mx_e8m0(amax, fp8):
    """mega_moe.quant's 1x32 MX scale (per_1x32_mx_quant, bit-identical) of a group's
    abs max (folded from 1e-10)."""
    inv = 0x3B124925 if fp8 else 0x3E2AAAAB  # 1 / max_pos (448 / 6)
    working = (amax * fx.Int32(inv).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    exponent = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(exponent + fx.Int32(1), exponent)
    return (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)


def _mx_pack(v, e8m0, fp8):
    """f32 values of one MX group -> i32 words (4 fp8-e4m3 / 8 fp4 each), as
    per_1x32_mx_quant packs them."""
    words = []
    if fp8:
        scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
        for w in range_constexpr(len(v) // 4):
            b = w * 4
            p = rocdl.cvt_pk_fp8_f32(
                T.i32, v[b] * scale, v[b + 1] * scale, fx.Int32(0), 0
            )
            words.append(
                rocdl.cvt_pk_fp8_f32(T.i32, v[b + 2] * scale, v[b + 3] * scale, p, 1)
            )
    else:
        dscale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
        for w in range_constexpr(len(v) // 8):
            p = fx.Int32(0)
            for pair in range_constexpr(4):
                idx = w * 8 + pair * 2
                p = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32, p, v[idx], v[idx + 1], dscale, pair
                )
            words.append(p)
    return words


def _decode_group8(w0, w1, e8m0):
    """2 i32 words (8 fp8) + E8M0 -> 8 f32 (dequantized)."""
    scale = (fx.Int32(e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    vals = []
    for word in (w0, w1):
        for half in range_constexpr(2):
            pair = fx.Vector(
                rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16), as_ir_value(word), as_ir_value(scale), bool(half)
                )
            ).to(fx.Float32)
            vals.extend((pair[0], pair[1]))
    return vals


def build_communication(kc: KernelCtx) -> dict:
    """The arena / flag / watchdog device functions of one instance (added to kc)."""
    const_expr = kc.get("const_expr")[0]
    TOPK, E, H, a8, EPR = kc.get("TOPK E H a8 EPR")
    L_RIX, L_CTL = kc.get(
        "L_RIX L_CTL"
    )  # the compute wave's gathered recv-slot list (per unit)
    ROW_BYTES = H if a8 else H // 2  # quantized activation bytes per row (FP8 vs FP4)
    N_VEC = ROW_BYTES // 16  # 16-byte (V4I) copy units per payload row
    SCALE_I32 = H // 128  # i32 E8M0-scale units per row (H//32 bytes / 4)
    PART_RB = (
        H + H // 8
    )  # combine ``part`` row bytes: H fp8-e4m3 then H/8 per-8 E8M0 scale
    SENTINEL = EPR  # recv_le empty-slot marker (masked by the planner: >= EPR)
    SBX_INV = (
        1 << SBX_TB
    ) - 1  # SBX: value bits of a scrubbed (never valid) routing word
    NW, NT, NTT = kc.get("NW NT NTT")
    # combine arrivals count GEMM2 n-tiles: a row is done once all H // TILE_N2 of its
    # n-tiles landed.
    COMB_W = H // kc.get("TILE_N2")[0]
    SBXP = kc.get("SBXP")[0]
    SDMAD = kc.get("SDMAD")[0]
    OWS = kc.get("OWS")[0]
    PB, PBK, PBT = kc.get("PB PBK PBT")
    if SDMAD:
        from .sdma import sdma_externs

        sd_put, sd_slot = sdma_externs()

    # --- arena addressing --------------------------------------------------------
    def ctrl_at(a, idx):
        return fx.Int64(a["ctrl"]) + fx.Int64(i32(idx) * i32(4))

    def peer_sel(a, p):
        v = fx.Int64(a["peer"][0])
        for j in range_constexpr(1, MAX_NPES):
            v = (p == i32(j)).select(fx.Int64(a["peer"][j]), v)
        return v

    def peer_rs(a, p, key, nbytes=None):
        # p (source rank) is a runtime value: select the peer base with peer_sel, never
        # index the Python peer list.
        return rsrc(peer_sel(a, p) + fx.Int64(a[key]), nbytes)

    def own_rs(a, key, nbytes=None):
        return rsrc(a["mine"] + fx.Int64(a[key]), nbytes)

    def _flag(a, base, parity, tile):
        return fx.Int64((i32(base) + parity * i32(NCTA_MAX) + tile) * i32(4))

    def peer_flag(a, p, base, parity, tile):
        return peer_sel(a, p) + fx.Int64(a["off_flag"]) + _flag(a, base, parity, tile)

    def own_flag(a, base, parity, tile):
        return a["mine"] + fx.Int64(a["off_flag"]) + _flag(a, base, parity, tile)

    # --- watchdog (sticky error word in the global ctrl tensor) ------------------
    def report(a, code):
        g_or_agent(ctrl_at(a, CTRL_ERR), code)

    @flyc.jit
    def _report_if(a, bad, code):
        if bad:
            report(a, code)

    @flyc.jit
    def poll_zero(fn, a=None, lane=None, err=0):
        pend = fn()
        t0 = now()
        while (pend != i32(0)) & alive(t0):
            rocdl.s_sleep(1)
            pend = fn()
        if const_expr(err != 0):
            _report_if(a, (pend != i32(0)) & (lane == i32(0)), err)

    @flyc.jit
    def spin_sys_ge(addr, target, a=None, err=0):
        cur = poll_sys_ge(addr, target)
        if const_expr(a is not None):
            _report_if(a, before(cur, target), err)

    # --- device grid barrier (all CTAs of THIS rank, between GEMM passes) --------
    @flyc.jit
    def grid_bar(a, tid):
        """Cross-CTA handshake over all ``grid_dim.x`` CTAs of this rank's GPU, via an
        agent-scope counter + generation in the local ctrl tensor (device-local; NOT
        cross-rank). Mirrors the LDS ``cbar``: tid0 of each CTA arrives, the last one
        zeroes the counter and bumps the generation to release the spinners.

        tid0 ONLY -- there is deliberately no ``gpu.barrier()`` here: this is called from
        the compute band (tid<NT) while the dispatch/idle bands are elsewhere, so a block
        barrier would deadlock. The caller wraps this with ``cbar`` (compute-wave barrier)
        to hold the other compute lanes until tid0's handshake returns."""
        if tid == i32(0):
            cnt_a = ctrl_at(a, GBAR_CNT)
            gen_a = ctrl_at(a, GBAR_GEN)
            g = g_ld_sys(gen_a)
            # two levels (the XCD's CTAs, then one per XCD): 256 arrivals on one counter
            # serialize
            nblk = i32(gpu.grid_dim.x)
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            xa = ctrl_at(a, i32(GBAR_X) + x)
            # each counter reset must land before the generation that lets a CTA re-arrive
            if g_add_agent(xa, i32(1)) == (nblk - i32(1) - x) // i32(N_XCD):
                g_st_sys(xa, i32(0))
                wait_vm(0)
                if g_add_agent(cnt_a, i32(1)) == fx.min(nblk, i32(N_XCD)) - i32(1):
                    g_st_sys(cnt_a, i32(0))
                    wait_vm(0)
                    g_st_sys(gen_a, g + i32(1))
            spin_sys_ge(gen_a, g + i32(1), a, ERR_COMPUTE)

    # --- EP dispatch / combine readiness -----------------------------------------
    @flyc.jit
    def dispatch_arrive_meta(a, p, parity, tile, delta, expected):
        """Arrival for the META (routing-only) dispatch phase: the
        owner can start planning (pre_pos / dyn_plan / gather) once every sender's routing
        has landed, while the big payload a2a is still in flight."""
        old = g_add_agent(peer_flag(a, p, FLAG_DISPM_CNT, parity, tile), delta)
        if old + delta == expected:
            g_st_sys(peer_flag(a, p, FLAG_DISPM_CNT, parity, tile), i32(0))
            g_st_sys(peer_flag(a, p, FLAG_DISPM, parity, tile), a["epoch"])

    @flyc.jit
    def dispatch_wait_meta(a, parity, tile):
        spin_sys_ge(own_flag(a, FLAG_DISPM, parity, tile), a["epoch"], a, ERR_DISPATCH)

    @flyc.jit
    def init_arrive(a, p, parity, expected):
        old = g_add_agent(peer_flag(a, p, FLAG_INIT_CNT, parity, i32(0)), i32(1))
        if old + i32(1) == expected:
            g_st_sys(peer_flag(a, p, FLAG_INIT_CNT, parity, i32(0)), i32(0))
            g_st_sys(peer_flag(a, p, FLAG_INIT, parity, i32(0)), a["epoch"])

    @flyc.jit
    def barrier_init(L, tid, a):
        """Cross-rank init barrier: recv_le is host-filled on this stream before the
        launch, so "this rank is ready" is a per-RANK fact -- only block 0 arrives (one
        signal per rank, expected=npes), while every CTA waits the stamp. That is an
        npes-way barrier instead of a (#CU * npes)-way single-counter atomic storm."""
        del L
        parity = a["epoch"] & i32(1)
        npes = a["npes"]
        if (i32(gpu.block_id("x")) == i32(0)) & (tid < npes):  # lane p -> peer p
            init_arrive(a, tid, parity, npes)
        if tid == i32(0):
            spin_sys_ge(
                own_flag(a, FLAG_INIT, parity, i32(0)), a["epoch"], a, ERR_DISPATCH
            )
        gpu.barrier()

    @flyc.jit
    def combine_arrive(a, p, parity, delta):
        """Count ``delta`` landed GEMM2 partial units at source rank ``p`` (only the
        source knows its m, so it alone tests completion -- combine_wait).  SBX
        completes on FLAG_SBD instead."""
        if const_expr(not SBXP):
            g_add_agent(peer_flag(a, p, FLAG_COMB_CNT, parity, i32(0)), delta)

    @flyc.jit
    def combine_wait(a, parity):
        """Every CTA waits all m*TOPK*COMB_W partial units of this rank's tokens."""
        spin_sys_ge(
            own_flag(a, FLAG_COMB_CNT, parity, i32(0)),
            a["m"] * i32(TOPK * COMB_W),
            a,
            ERR_COMBINE,
        )

    # --- dispatch send (a2a scatter of this rank's tokens to expert owners) -------
    @flyc.jit
    def init_recv(L, tid, a):
        """Sentinel this rank's received-routing region so the planner masks the
        (token, route) pairs no sender fills this launch. Runs before dispatch; a
        cross-rank barrier (roles_entry) orders it before any peer writes recv_le."""
        n = a["npes"] * a["mtpr"] * i32(TOPK)
        rle = own_rs(a, "off_recv", n * i32(4))
        for s_ in range(tid, n, i32(NT)):
            bst(i32(SENTINEL), rle, i32(s_) * i32(4), 0, 0)
        # commit the sentinels to HBM before barrier_init signals peers -- otherwise an
        # in-flight sentinel can land AFTER a peer's post-barrier dispatch write and clobber it.
        wait_vm(0)

    @flyc.jit
    def dispatch_route_all(L, tid, a):
        """Scatter each local (token, route)'s local expert id to its expert-owner rank's
        ``recv_le``, indexed by ``source_key = rank*mtpr + token`` (bounded by npes*mtpr,
        overflow-proof), then signal ``dispatch_arrive_meta``: the owner plans (pre_pos /
        dyn_plan / gather, all routing-only) while the payload (payload_send_ovl) is still
        in flight.  The top-k weights stay at the source (applied in combine_final).  All NW
        waves of a CTA send (wave w of CTA b is virtual sender b*NW + w of nblk*NW); the CTA
        arrives once, after a block barrier, so all NT threads must call it."""
        m, mtpr, rank, epr, npes = a["m"], a["mtpr"], a["rank"], a["epr"], a["npes"]
        parity = a["epoch"] & i32(1)
        nblk = i32(gpu.grid_dim.x)
        rids = rsrc(a["ids"], m * i32(TOPK) * i32(4))
        lane = tid % i32(64)
        sid = i32(gpu.block_id("x")) * i32(NW) + tid // i32(64)  # virtual sender
        nsnd = nblk * i32(NW)
        # one (token, route) per LANE, so a wave keeps 64 remote stores in flight per step.
        # The buffer resource must be wave-uniform (rsrc readfirstlanes the base), so each
        # lane's store is issued under the static owner loop ``owner == p``.
        r_recv = [peer_rs(a, i32(p), "off_recv") for p in range(MAX_NPES)]
        for wk_ in range(sid * i32(64) + lane, m * i32(TOPK), nsnd * i32(64)):
            wk = i32(wk_)
            t = wk // i32(TOPK)
            r = wk - t * i32(TOPK)
            e = fx.Int32(bld(rids, wk * i32(4), 0, T.i32, 0))
            valid = (e >= i32(0)) & (e < i32(E))
            owner = e // epr
            slot = ((rank * mtpr + t) * i32(TOPK) + r) * i32(4)
            for p in range_constexpr(MAX_NPES):
                if valid & (owner == i32(p)):
                    bst(e - owner * epr, r_recv[p], slot, 0, AUX_SYS)
            if valid == fx.Boolean(False):
                # dropped/invalid route: this rank is the source, so pre-count it into
                # this rank's single combine counter now -- the GEMM2 tiles only arrive
                # for valid routes later, and combine_wait expects m*TOPK total.
                combine_arrive(a, rank, parity, i32(COMB_W))
        if (i32(gpu.block_id("x")) == i32(0)) & (tid < npes):
            # this rank's m (owner p's routing window from it), before the META arrival
            g_st_sys(peer_flag(a, tid, FLAG_SRCM, parity, rank), m)
        wait_vm(0)  # drain routing stores before signalling META-ready
        gpu.barrier()  # every wave's stores drained before the CTA arrives
        # fold the CTAs' arrivals per XCD, then per rank: the last CTA arrives once at
        # every owner (lane p -> owner p) -- nblk remote atomics on one owner counter
        # serialize.  L_RIX is free until the gather.
        if tid == i32(0):
            lds_st(L, L_RIX, i32(0))
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            xa = ctrl_at(a, i32(MAGG_X) + x)
            if g_add_agent(xa, i32(1)) == (nblk - i32(1) - x) // i32(N_XCD):
                g_st_sys(xa, i32(0))
                ga = ctrl_at(a, MAGG_G)
                if g_add_agent(ga, i32(1)) == fx.min(nblk, i32(N_XCD)) - i32(1):
                    g_st_sys(ga, i32(0))
                    lds_st(L, L_RIX, i32(1))
        gpu.barrier()
        if (lds_ld_i32(L, L_RIX) == i32(1)) & (tid < npes):
            dispatch_arrive_meta(a, tid, parity, i32(0), i32(1), npes)

    # --- payload send (expert-ordered rounds, per-expert landed-route counts) -------
    OVL_LB = ceildiv(EPR, N_XCD)  # rounds: one expert per owner XCD block each
    # L_C ints: per (owner, local expert[, token bucket]) routes sent this round
    OVL_HIST = MAX_NPES * EPR * PBK
    OVL_LCNT = OVL_HIST  # L_C int: this token round's send-list length
    OVL_LIST = OVL_HIST + 1  # L_C ints: the send list, token * 8 + owner
    L_C = kc.get("L_C")[0]  # the GEMM2 C-tile: free until the sender joins the GEMMs
    assert EPR <= NCTA_MAX  # FLAG_PAYC: one counter per local expert per parity
    assert (OVL_LIST + NT * min(MAX_NPES, TOPK)) * 4 <= 16 * kc.get("TILE_N")[0] * 4

    def _rounds(a, tc):
        """Per owner p, the token's earliest round among its experts on p (OVL_LB: none);
        every expert's rows are then all sent by its own round le % OVL_LB."""
        epr = a["epr"]
        rids = rsrc(a["ids"], a["m"] * i32(TOPK) * i32(4))
        es = [
            fx.Int32(bld(rids, (tc * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0))
            for r in range(TOPK)
        ]
        vs = [(e >= i32(0)) & (e < i32(E)) for e in es]
        ows = [e // epr for e in es]
        les = [es[r] - ows[r] * epr for r in range(TOPK)]
        mns = []
        for p in range(MAX_NPES):
            mn = i32(OVL_LB)
            for r in range(TOPK):
                rr = les[r] % i32(OVL_LB)
                mn = (vs[r] & (ows[r] == i32(p)) & (rr < mn)).select(rr, mn)
            mns.append(mn)
        return mns

    def sd_at(a, idx):
        return fx.Int64(a["sdst"]) + fx.Int64(i32(idx) * i32(4))

    SD_NH = MAX_NPES * SD_GMAX

    @flyc.jit
    def sd_rows(L, tid, a):
        """SDMA dispatch: the compacted row ``off(p, g) + pos`` of each (token, owner p)
        pair in this rank's stream to p -- round g = the token's earliest round on p, rows
        round-major -- to ``spos[t * 8 + p]`` (local; other XCDs' CTAs pack the later rounds,
        so system-scope) and owner p's ``recv_pos``.  Per-CTA
        LDS counts take one atomic per (owner, round), then a grid barrier gives the round
        totals.  All NT threads; the stores drain with dispatch_route_all's."""
        m, mtpr, rank = a["m"], a["mtpr"], a["rank"]
        rsp = rsrc(a["spos"])
        nblk = i32(gpu.grid_dim.x)
        b = i32(gpu.block_id("x"))
        t_lo = b * m // nblk
        t_hi = (b + i32(1)) * m // nblk
        if tid == i32(0):  # no payload round seen landed yet
            lds_st(L, L_CTL + i32(C_SDL * 4), a["epoch"] * i32(SD_GMAX))
        # L_C: [0, NH) this CTA's counts, [NH, 2 NH) its base, [2 NH, 3 NH) off(p, g)
        for k_ in range(tid, i32(SD_NH), i32(NT)):
            lds_st(L, L_C + i32(k_) * i32(4), i32(0))
        gpu.barrier()
        for r0_ in range(t_lo, t_hi, i32(NT)):
            t = i32(r0_) + tid
            if t < t_hi:
                mns = _rounds(a, t)
                for p in range_constexpr(MAX_NPES):
                    if mns[p] < i32(OVL_LB):
                        pos = lds_atomic_add(
                            L, L_C + (i32(p * SD_GMAX) + mns[p]) * i32(4), 1, ACQ_REL
                        )
                        bst(
                            (mns[p] << i32(24)) | pos,
                            rsp,
                            (t * i32(8) + i32(p)) * i32(4),
                            0,
                            AUX_SYS,
                        )
                    if mns[p] == i32(OVL_LB):
                        bst(i32(-1), rsp, (t * i32(8) + i32(p)) * i32(4), 0, AUX_SYS)
        gpu.barrier()
        if tid < i32(SD_NH):
            c = lds_ld_i32(L, L_C + tid * i32(4))
            if c > i32(0):
                lds_st(
                    L,
                    L_C + (i32(SD_NH) + tid) * i32(4),
                    g_add_agent(sd_at(a, i32(SD_CNT) + tid), c),
                )
        wait_vm(0)
        gpu.barrier()
        grid_bar(a, tid)
        gpu.barrier()
        if tid < i32(SD_NH):
            p = tid // i32(SD_GMAX)
            g = tid - p * i32(SD_GMAX)
            off = i32(0)
            for gg in range_constexpr(SD_GMAX - 1):
                v = g_ld_sys(sd_at(a, p * i32(SD_GMAX) + i32(gg + SD_CNT)))
                off = off + (i32(gg) < g).select(v, i32(0))
            lds_st(L, L_C + (i32(2 * SD_NH) + tid) * i32(4), off)
        gpu.barrier()
        r_rpos = [peer_rs(a, i32(p), "off_rpos") for p in range(MAX_NPES)]
        for r0_ in range(t_lo, t_hi, i32(NT)):
            t = i32(r0_) + tid
            if t < t_hi:
                for p in range_constexpr(MAX_NPES):
                    w = fx.Int32(
                        bld(rsp, (t * i32(8) + i32(p)) * i32(4), 0, T.i32, AUX_SYS)
                    )
                    if w >= i32(0):
                        k = i32(p * SD_GMAX) + w.shrui(i32(24))
                        row = (
                            lds_ld_i32(L, L_C + (i32(2 * SD_NH) + k) * i32(4))
                            + lds_ld_i32(L, L_C + (i32(SD_NH) + k) * i32(4))
                            + (w & i32(0xFFFFFF))
                        )
                        bst(row, rsp, (t * i32(8) + i32(p)) * i32(4), 0, AUX_SYS)
                        bst(row, r_rpos[p], (rank * mtpr + t) * i32(4), 0, AUX_SYS)
        gpu.barrier()

    @flyc.jit
    def sd_post(L, tid, a, g):
        """The last CTA done packing round g (all its threads): in round order, one put per
        owner of the round's rows and one of their scales (a 16-B dummy put if the round
        has none: the owner counts puts), then marks this rank's own rows of the round.
        """
        rank, mtpr = a["rank"], a["mtpr"]
        if tid == i32(0):
            spin_sys_ge(sd_at(a, SD_TURN), g, a, ERR_DISPATCH)
        gpu.barrier()
        if (tid < a["npes"]) & (tid != rank):
            p = tid
            off = i32(0)
            for gg in range_constexpr(SD_GMAX - 1):
                v = g_ld_sys(sd_at(a, p * i32(SD_GMAX) + i32(gg + SD_CNT)))
                off = off + (i32(gg) < g).select(v, i32(0))
            cnt = g_ld_sys(sd_at(a, p * i32(SD_GMAX) + g + i32(SD_CNT)))
            pb = peer_sel(a, p)
            mine = a["mine"]
            if cnt > i32(0):
                srow = fx.Int64(p * mtpr + off)
                drow = fx.Int64(rank * mtpr + off)
                sd_put(
                    a["dc"],
                    p,
                    pb + fx.Int64(a["off_payload"]) + drow * fx.Int64(ROW_BYTES),
                    mine + fx.Int64(a["off_stage"]) + srow * fx.Int64(ROW_BYTES),
                    fx.Int64(cnt) * fx.Int64(ROW_BYTES),
                    i32(SD_QP),
                )
                sd_put(
                    a["dc"],
                    p,
                    pb + fx.Int64(a["off_pscale"]) + drow * fx.Int64(SCALE_I32 * 4),
                    mine + fx.Int64(a["off_sstage"]) + srow * fx.Int64(SCALE_I32 * 4),
                    fx.Int64(cnt) * fx.Int64(SCALE_I32 * 4),
                    i32(SD_QS),
                )
            if cnt == i32(0):
                dd = pb + fx.Int64(a["off_sdum"]) + fx.Int64(rank * i32(SD_DUM))
                sd_put(
                    a["dc"],
                    p,
                    dd,
                    mine + fx.Int64(a["off_sdum"]),
                    fx.Int64(16),
                    i32(SD_QP),
                )
                sd_put(
                    a["dc"],
                    p,
                    dd + fx.Int64(32),
                    mine + fx.Int64(a["off_sdum"]),
                    fx.Int64(16),
                    i32(SD_QS),
                )
        wait_vm(0)
        gpu.barrier()
        if tid == i32(0):
            g_st_sys(sd_at(a, SD_SELF), a["epoch"] * i32(SD_GMAX) + g + i32(1))
            if g == i32(OVL_LB - 1):  # every CTA packed every round: re-zero
                for k in range_constexpr(SD_NH + SD_GMAX):
                    g_st_sys(sd_at(a, SD_CNT + k), i32(0))
                g_st_sys(sd_at(a, SD_TURN), i32(0))
            if g != i32(OVL_LB - 1):
                g_st_sys(sd_at(a, SD_TURN), g + i32(1))

    @flyc.jit
    def sd_wait(a, g):
        """(One thread) every source's payload rounds [0, g] landed here."""
        rank = a["rank"]
        for s_ in range_constexpr(MAX_NPES):
            s = i32(s_)
            if s < a["npes"]:
                if s == rank:
                    spin_sys_ge(
                        sd_at(a, SD_SELF),
                        a["epoch"] * i32(SD_GMAX) + g + i32(1),
                        a,
                        ERR_DISPATCH,
                    )
                if s != rank:
                    for qi in range_constexpr(2):
                        tgt = (
                            g_ld_sys(sd_at(a, i32(SD_BASE + 2 * s_ + qi))) + g + i32(1)
                        )
                        spin_sys_ge(
                            fx.Int64(sd_slot(a["dc"], s, i32((SD_QP, SD_QS)[qi]))),
                            tgt,
                            a,
                            ERR_DISPATCH,
                        )

    @flyc.jit
    def sd_wait_c(L, a, g):
        """(tid0) sd_wait through this CTA's C_SDL and the rank's SD_LAND: a CTA polls the
        sources only for a round no CTA has seen landed yet."""
        tag = a["epoch"] * i32(SD_GMAX) + g + i32(1)
        if before(lds_ld_i32(L, L_CTL + i32(C_SDL * 4)), tag):
            if before(g_ld_sys(sd_at(a, SD_LAND)), tag):
                sd_wait(a, g)
                if before(g_ld_sys(sd_at(a, SD_LAND)), tag):
                    g_st_sys(sd_at(a, SD_LAND), tag)
            lds_st(L, L_CTL + i32(C_SDL * 4), tag)

    @flyc.jit
    def sd_finish(a):
        """(One thread, the last CTA out) every put of this launch landed; advance the
        per-(source, queue) signal bases by its OVL_LB puts."""
        sd_wait(a, i32(OVL_LB - 1))
        for s_ in range_constexpr(MAX_NPES):
            if (i32(s_) < a["npes"]) & (i32(s_) != a["rank"]):
                for qi in range_constexpr(2):
                    k = SD_BASE + 2 * s_ + qi
                    g_st_sys(sd_at(a, k), g_ld_sys(sd_at(a, k)) + i32(OVL_LB))

    @flyc.jit
    def payload_send_ovl(L, tid, a, s, ns, g_lo, g_hi, po, skip):
        """Sender ``s`` of ``ns`` sends its token slice's payload rows (once per
        (token, owner)) of rounds [g_lo, g_hi); round g takes the sends whose smallest
        ``le % OVL_LB`` over the token's local experts on that owner is g, so local expert
        le is complete after round le % OVL_LB -- about one expert per owner XCD per round,
        in the order the owner's queue takes them (_dynq_passes_nw8).  Each round ends with
        a drain and a flush of the per-(owner, local expert) route counts into the owner's
        FLAG_PAYC (every route of the token to that owner counts with the send).  OWS:
        ``po`` >= 0 sends owner po's rows only; owners in the bit mask ``skip`` are left
        to their own senders."""
        m, mtpr, rank, epr, npes = a["m"], a["mtpr"], a["rank"], a["epr"], a["npes"]
        parity = a["epoch"] & i32(1)
        rids = rsrc(a["ids"], m * i32(TOPK) * i32(4))
        rx = rsrc(a["x"])
        rxs = rsrc(a["xs"])
        lane = tid % i32(64)
        wv = tid // i32(64)
        rsp = rsrc(a["spos"])
        t_lo = s * m // ns
        t_hi = (s + i32(1)) * m // ns
        tbs = [(t_lo, t_hi)]
        if const_expr(PB):  # sender s's slice of each token bucket
            tbs = []
            # nb drains of PB / nb buckets each (nb = min(PB, pow2 <= m // PBT), >= 1)
            lg = i32(0)
            for k in range_constexpr(1, PB.bit_length()):
                lg = (m >= i32(PBT << k)).select(i32(k), lg)
            per = i32(PB) >> lg
            for jb in range_constexpr(PB):
                b_lo = (fx.min(i32(jb) * per, i32(PB)) * m + i32(PB - 1)) // i32(PB)
                b_n = (fx.min(i32(jb + 1) * per, i32(PB)) * m + i32(PB - 1)) // i32(
                    PB
                ) - b_lo
                tbs.append((b_lo + s * b_n // ns, b_lo + (s + i32(1)) * b_n // ns))
        NJ = (N_VEC + 63) // 64
        NS = (SCALE_I32 + 63) // 64
        for k_ in range(tid, i32(OVL_LIST), i32(NT)):
            lds_st(L, L_C + i32(k_) * i32(4), i32(0))
        gpu.barrier()
        for g_ in range(g_lo, g_hi, i32(1)):
            g = i32(g_)
            for jb in range_constexpr(PBK):  # the round's token buckets
                for r0_ in range(tbs[jb][0], tbs[jb][1], i32(NT)):
                    t = i32(r0_) + tid
                    live = t < tbs[jb][1]
                    tc = fx.min(t, m - i32(1))
                    es = [
                        fx.Int32(
                            bld(rids, (tc * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0)
                        )
                        for r in range(TOPK)
                    ]
                    vs = [(e >= i32(0)) & (e < i32(E)) for e in es]
                    ows = [e // epr for e in es]
                    les = [es[r] - ows[r] * epr for r in range(TOPK)]
                    tj = tc * i32(PB) // m if PB else None  # the token's bucket
                    for p in range_constexpr(MAX_NPES):
                        # the earliest round among the token's experts on p: every expert's
                        # rows are then all sent by its own round le % OVL_LB
                        mn = i32(OVL_LB)
                        for r in range_constexpr(TOPK):
                            rr = les[r] % i32(OVL_LB)
                            mn = (vs[r] & (ows[r] == i32(p)) & (rr < mn)).select(rr, mn)
                        if const_expr(
                            OWS
                        ):  # sender po >= 0: owner po only; skip: owners whose
                            # senders take these rounds
                            mn = ((po < i32(0)) | (po == i32(p))).select(
                                mn, i32(OVL_LB)
                            )
                            mn = (((skip >> i32(p)) & i32(1)) == i32(0)).select(
                                mn, i32(OVL_LB)
                            )
                        if live & (mn == g):
                            pos = lds_atomic_add(L, L_C + i32(OVL_LCNT * 4), 1, ACQ_REL)
                            lds_st(
                                L,
                                L_C + (i32(OVL_LIST) + pos) * i32(4),
                                t * i32(8) + i32(p),
                            )
                            for r in range_constexpr(0 if SDMAD else TOPK):
                                if vs[r] & (ows[r] == i32(p)):
                                    lds_atomic_add(
                                        L,
                                        L_C
                                        + (
                                            (i32(p * EPR) + les[r]) * i32(PB) + tj
                                            if PB
                                            else i32(p * EPR) + les[r]
                                        )
                                        * i32(4),
                                        1,
                                        ACQ_REL,
                                    )
                    gpu.barrier()
                    n = lds_ld_i32(L, L_C + i32(OVL_LCNT * 4))
                    # each wave copies DISP_TB list rows per step, loads first (DISP_TB)
                    for b_ in range(wv * i32(DISP_TB), n, i32(NW * DISP_TB)):
                        b = i32(b_)
                        ents, rows, scs, rws = [], [], [], []
                        for k in range_constexpr(DISP_TB):
                            ent = lds_ld_i32(
                                L,
                                L_C
                                + (i32(OVL_LIST) + fx.min(b + i32(k), n - i32(1)))
                                * i32(4),
                            )
                            tk = ent.shrui(i32(3))
                            ents.append(ent)
                            if const_expr(
                                SDMAD
                            ):  # spos[t * 8 + owner]: the compacted row
                                rws.append(
                                    fx.Int32(bld(rsp, ent * i32(4), 0, T.i32, AUX_SYS))
                                )
                            rows.append(
                                [
                                    bld(
                                        rx,
                                        tk * i32(ROW_BYTES)
                                        + fx.min(lane + i32(j * 64), i32(N_VEC - 1))
                                        * i32(16),
                                        0,
                                        V4I,
                                        0,
                                    )
                                    for j in range(NJ)
                                ]
                            )
                            scs.append(
                                [
                                    bld(
                                        rxs,
                                        (
                                            tk * i32(SCALE_I32)
                                            + fx.min(
                                                lane + i32(j * 64), i32(SCALE_I32 - 1)
                                            )
                                        )
                                        * i32(4),
                                        0,
                                        T.i32,
                                        0,
                                    )
                                    for j in range(NS)
                                ]
                            )
                        wait_vm(0)
                        for k in range_constexpr(DISP_TB):
                            if b + i32(k) < n:
                                source_key = rank * mtpr + ents[k].shrui(i32(3))
                                owner = ents[k] & i32(7)
                                if const_expr(
                                    SDMAD
                                ):  # own payload, else the owner's stage rows
                                    source_key = owner * mtpr + rws[k]
                                    own = owner == rank
                                    dpay = rsrc(
                                        a["mine"]
                                        + own.select(
                                            fx.Int64(a["off_payload"]),
                                            fx.Int64(a["off_stage"]),
                                        )
                                    )
                                    dsc = rsrc(
                                        a["mine"]
                                        + own.select(
                                            fx.Int64(a["off_pscale"]),
                                            fx.Int64(a["off_sstage"]),
                                        )
                                    )
                                else:
                                    dpay = peer_rs(a, owner, "off_payload")
                                    dsc = peer_rs(a, owner, "off_pscale")
                                for j in range_constexpr(NJ):
                                    u = lane + i32(j * 64)
                                    if u < i32(N_VEC):
                                        bst(
                                            rows[k][j],
                                            dpay,
                                            source_key * i32(ROW_BYTES) + u * i32(16),
                                            0,
                                            AUX_SYS,
                                        )
                                for j in range_constexpr(NS):
                                    u = lane + i32(j * 64)
                                    if u < i32(SCALE_I32):
                                        bst(
                                            scs[k][j],
                                            dsc,
                                            (source_key * i32(SCALE_I32) + u) * i32(4),
                                            0,
                                            AUX_SYS,
                                        )
                    gpu.barrier()
                    if tid == i32(0):
                        lds_st(L, L_C + i32(OVL_LCNT * 4), i32(0))
                    gpu.barrier()
                wait_vm(0)  # this round's rows landed before their routes are counted
                gpu.barrier()
                if const_expr(SDMAD):  # the round's last CTA posts it
                    if tid == i32(0):
                        old = g_add_agent(sd_at(a, i32(SD_RD) + g), i32(1))
                        lds_st(L, L_C, (old == ns - i32(1)).select(i32(1), i32(0)))
                    gpu.barrier()
                    if lds_ld_i32(L, L_C) == i32(1):
                        sd_post(L, tid, a, g)
                    gpu.barrier()
                else:
                    for k_ in range(tid, npes * i32(EPR * PBK), i32(NT)):
                        k = i32(k_)
                        c = lds_ld_i32(L, L_C + k * i32(4))
                        if c > i32(0):
                            p = k // i32(EPR * PBK)
                            slot = k - p * i32(EPR * PBK)  # (PB: le * PB + bucket)
                            g_add_agent(peer_flag(a, p, FLAG_PAYC, parity, slot), c)
                            lds_st(L, L_C + k * i32(4), i32(0))
                    gpu.barrier()

    @flyc.jit
    def payload_hot0(L, tid, a):
        """Owners (bit mask) taking over HVT% of a strided NT-token sample's round-0
        (token, owner) sends, minus the sample's most-loaded owner (by routes).  Every
        CTA of the rank samples the same tokens."""
        m, epr = a["m"], a["epr"]
        rids = rsrc(a["ids"], m * i32(TOPK) * i32(4))
        if tid < i32(2 * MAX_NPES):
            lds_st(L, L_C + tid * i32(4), i32(0))
        gpu.barrier()
        t = (tid * m) // i32(NT)
        es = [
            fx.Int32(bld(rids, (t * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0))
            for r in range(TOPK)
        ]
        vs = [(e >= i32(0)) & (e < i32(E)) & (t < m) for e in es]
        ows = [e // epr for e in es]
        for p in range_constexpr(MAX_NPES):
            hit = i32(0)
            nr = i32(0)
            for r in range_constexpr(TOPK):
                on = vs[r] & (ows[r] == i32(p))
                hit = hit | (
                    on & ((es[r] - ows[r] * epr) % i32(OVL_LB) == i32(0))
                ).select(i32(1), i32(0))
                nr = nr + on.select(i32(1), i32(0))
            if hit != i32(0):
                lds_atomic_add(L, L_C + i32(p * 4), 1, ACQ_REL)
            if nr != i32(0):
                lds_atomic_add(L, L_C + i32((MAX_NPES + p) * 4), nr, ACQ_REL)
        gpu.barrier()
        hv = i32(0)
        for p in range_constexpr(MAX_NPES):
            c = lds_ld_i32(L, L_C + i32(p * 4))
            hv = hv | (c * i32(100) > i32(HVT * NT)).select(i32(1 << p), i32(0))
        bm = i32(-1)
        bp = i32(0)
        for p in range_constexpr(MAX_NPES):
            c = lds_ld_i32(L, L_C + i32((MAX_NPES + p) * 4))
            bp = (c > bm).select(i32(p), bp)
            bm = fx.max(c, bm)
        hv = hv & (i32(-1) ^ (i32(1) << bp))
        gpu.barrier()
        return uni(hv)

    @flyc.jit
    def payload_wait(a, le, need):
        """Block until ``need`` routes to local expert ``le`` have their payload (SDMA:
        until every source's rounds up to le % OVL_LB landed)."""
        if const_expr(SDMAD):
            sd_wait(a, le % i32(OVL_LB))
        else:
            spin_sys_ge(
                own_flag(a, FLAG_PAYC, a["epoch"] & i32(1), le), need, a, ERR_DISPATCH
            )

    def payload_reset(a):
        """Zero this launch parity's FLAG_PAYC (every route has been counted and
        every waiter has passed).  SDMA: wait the last puts, advance the signal bases.
        """
        if const_expr(SDMAD):
            sd_finish(a)
            return
        for e in range_constexpr(EPR * PBK):
            g_st_sys(own_flag(a, FLAG_PAYC, a["epoch"] & i32(1), i32(e)), i32(0))

    # --- SBX small-batch protocol (see mega_moe_ep_config.sbx_on) -------------------
    @flyc.jit
    def sbx_send(L, tid, a):
        """CTA b sends to owner d = b % npes: the first of d's CTAs its header
        (FLAG_SBM = tag | m) and whole routing block (tag | local expert or SENTINEL per
        route), all tagged so neither waits a drain; then all of d's CTAs split the
        tokens routed to d, 1x32-MX quantize their bf16 rows (bit-identical to the
        host's per_1x32_mx_quant) and send them; the last one done stamps FLAG_SBP."""
        del L
        m, mtpr, rank, epr, npes = a["m"], a["mtpr"], a["rank"], a["epr"], a["npes"]
        nblk = i32(gpu.grid_dim.x)
        bid = i32(gpu.block_id("x"))
        d = bid % npes
        k = bid // npes
        nk = (nblk - i32(1) - d) // npes + i32(1)
        rids = rsrc(a["ids"], m * i32(TOPK) * i32(4))
        rx = rsrc(a["x"])
        if k == i32(0):
            tag = sbx_tag(a)
            if tid == i32(0):
                g_st_sys(peer_flag(a, d, FLAG_SBM, i32(0), rank), tag | m)
            rrecv = peer_rs(a, d, "off_recv")
            for c_ in range(tid, m * i32(TOPK), i32(NT)):
                c = i32(c_)
                e = fx.Int32(bld(rids, c * i32(4), 0, T.i32, 0))
                mine = (e >= i32(0)) & (e < i32(E)) & (e // epr == d)
                bst(
                    tag | mine.select(e - d * epr, i32(SENTINEL)),
                    rrecv,
                    (rank * mtpr * i32(TOPK) + c) * i32(4),
                    0,
                    AUX_SYS,
                )
        dpay = peer_rs(a, d, "off_payload")
        dsc = peer_rs(a, d, "off_pscale")
        lane = tid % i32(64)
        nsnd = nk * i32(NW)
        # one wave-uniform token per step from the bf16 rows, in H / 1024 passes of
        # 16 elements per lane (half an MX group; lanes l, l ^ 1 share its max), so
        # the remote payload stores are lane-contiguous; lane 8j stores scale word j
        QB = ROW_BYTES // (H // 16)  # payload bytes per lane per pass
        c64 = i32(64)
        for t_ in range(k * i32(NW) + tid // i32(64), m, nsnd):
            t = i32(t_)
            anyr = fx.Boolean(False)
            for r in range_constexpr(TOPK):
                e = fx.Int32(bld(rids, (t * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0))
                anyr = anyr | ((e >= i32(0)) & (e < i32(E)) & (e // epr == d))
            if anyr:
                raw = [
                    bld(
                        rx,
                        t * i32(2 * H) + lane * i32(32),
                        i32(c // 2 * 2048 + c % 2 * 16),
                        V4I,
                        0,
                    )
                    for c in range(2 * (H // 1024))
                ]
                sk = rank * mtpr + t
                for p in range_constexpr(H // 1024):
                    v = bf16x8_to_f32(raw[2 * p]) + bf16x8_to_f32(raw[2 * p + 1])
                    lm = fx.Float32(1e-10)
                    for x in v:
                        lm = lm.maximumf(x.maximumf(fx.Float32(0.0) - x))
                    e8 = _mx_e8m0(lm.maximumf(lm.shuffle_xor(i32(1), c64)), a8)
                    words = _mx_pack(v, e8, a8)
                    bst(
                        fx.Vector.from_elements(words, fx.Int32),
                        dpay,
                        sk * i32(ROW_BYTES) + lane * i32(QB),
                        i32(p * 64 * QB),
                        AUX_SYS,
                    )
                    sc = (
                        e8
                        | (e8.shuffle_xor(i32(2), c64) << i32(8))
                        | (e8.shuffle_xor(i32(4), c64) << i32(16))
                        | (e8.shuffle_xor(i32(6), c64) << i32(24))
                    )
                    if lane % i32(8) == i32(0):
                        bst(
                            sc,
                            dsc,
                            sk * i32(H // 32) + lane // i32(2),
                            i32(p * 32),
                            AUX_SYS,
                        )
        wait_vm(0)
        gpu.barrier()
        if tid == i32(0):
            ca = ctrl_at(a, i32(SBX_SCNT) + d)
            if g_add_agent(ca, i32(1)) == nk - i32(1):
                g_st_sys(ca, i32(0))
                g_st_sys(peer_flag(a, d, FLAG_SBP, i32(0), rank), a["epoch"])

    def sbx_tag(a):
        """This launch's tag in the high bits of the SBX header / routing words (the
        low SBX_TB bits hold m or the local expert)."""
        return (a["epoch"] & i32((1 << (32 - SBX_TB)) - 1)) << i32(SBX_TB)

    @flyc.jit
    def sbx_wait(tid, a, base, err):
        """Lane p < npes waits peer p's stamp ``base`` (call under a CTA barrier)."""
        if tid < a["npes"]:
            spin_sys_ge(own_flag(a, base, i32(0), tid), a["epoch"], a, err)

    @flyc.jit
    def sbx_tagged(addr, tag, a):
        """Poll a tagged SBX word at ``addr`` until it carries ``tag``."""
        v = g_ld_sys(addr)
        t0 = now()
        while ((v & i32(-(1 << SBX_TB))) != tag) & alive(t0):
            rocdl.s_sleep(1)
            v = g_ld_sys(addr)
        _report_if(a, (v & i32(-(1 << SBX_TB))) != tag, ERR_DISPATCH)
        return v

    def _sbx_stale(v, tag):
        return ((v & i32(-(1 << SBX_TB))) != tag) | ((v & i32(SBX_INV)) == i32(SBX_INV))

    @flyc.jit
    def sbx_word(a, rle, idx):
        """Received routing word idx (local expert, tag stripped): the cached load, else
        re-read it uncached until the sender's tagged store lands."""
        v = fx.Int32(bld(rle, idx * i32(4), 0, T.i32, 0))
        addr = a["recv_le"] + fx.Int64(idx * i32(4))
        t0 = now()
        while _sbx_stale(v, a["sbtag"]) & alive(t0):
            v = g_ld_sys(addr)
        _report_if(a, _sbx_stale(v, a["sbtag"]), ERR_DISPATCH)
        return v & i32((1 << SBX_TB) - 1)

    @flyc.jit
    def sbx_scrub(a):
        """The owner's last CTA out (every routing word read): at tag 2 -- once per tag
        period, and the first launch after the epoch's i32 wrap -- reset the whole routing
        region to SBX_INV words (drained before FLAG_SBD lets the senders write again), so
        no word outlives a tag period and matches a later launch's tag."""
        if sbx_tag(a) == i32(2 << SBX_TB):
            r = rsrc(a["recv_le"])
            for c_ in range(i32(0), a["nrecv"], i32(1)):
                bst(i32(-1), r, i32(c_) * i32(4), 0, AUX_SYS)
            wait_vm(0)

    @flyc.jit
    def sbx_wait_hdr(tid, a):
        """Lane p < npes waits peer p's header (call under a CTA barrier)."""
        if tid < a["npes"]:
            sbx_tagged(own_flag(a, FLAG_SBM, i32(0), tid), sbx_tag(a), a)

    def src_windows(a):
        """After the META wait (SBX: sbx_wait_hdr): per-source routing-window prefix
        o[0..MAX_NPES] (source p's m_p * TOPK routes are window entries [o[p], o[p+1])).
        """
        parity = a["epoch"] & i32(1)
        o = [i32(0)]
        for p in range_constexpr(MAX_NPES):
            if const_expr(SBXP):
                mp = uni(g_ld_sys(own_flag(a, FLAG_SBM, i32(0), i32(p)))) & i32(SBX_INV)
            else:
                mp = uni(g_ld_sys(own_flag(a, FLAG_SRCM, parity, i32(p))))
            o.append(o[-1] + (i32(p) < a["npes"]).select(mp * i32(TOPK), i32(0)))
        return o

    @flyc.jit
    def sbx_done(a):
        """The owner's last CTA out of the tile queue (every part store drained)."""
        for p in range_constexpr(MAX_NPES):
            if i32(p) < a["npes"]:
                g_st_sys(peer_flag(a, i32(p), FLAG_SBD, i32(0), a["rank"]), a["epoch"])

    # --- combine (source-side weighted sum of the GEMM2 epilogue's scatter) ----------
    @flyc.jit
    def combine_final(L, tid, a):
        """Source-side finish: for each local token wait its top-k partials, then
        out[t] = sum over the top-k routes of part[t*topk+r] * wts[t,r]. The top-k
        weight is applied here, so dispatch / compute never carry weights. Runs on the
        whole block in the finish phase.  Invalid (-1 / out-of-range) routes were
        pre-counted at dispatch and are dropped here."""
        m = a["m"]
        parity = a["epoch"] & i32(1)
        rpart = own_rs(a, "off_part")
        rwts = rsrc(a["wts"], m * i32(TOPK) * i32(4))
        rids = rsrc(a["ids"], m * i32(TOPK) * i32(4))
        rout = rsrc(a["out"])
        HV = H // 8
        nblk = i32(gpu.grid_dim.x)
        # one per-rank combine barrier: wait until all m*TOPK partials for this rank's
        # tokens have been scattered back (single tile-0 counter, no per-token flags).
        if (i32(gpu.block_id("x")) == i32(0)) & (tid == i32(0)):
            # the next launch's counter: past this launch's META / header wait every peer
            # finished the previous launch, and none arrives for the next one before it
            # sees this rank's next META
            g_st_sys(own_flag(a, FLAG_COMB_CNT, parity ^ i32(1), i32(0)), i32(0))
        if const_expr(SBXP):
            sbx_wait(tid, a, FLAG_SBD, ERR_COMBINE)
        elif tid == i32(0):
            combine_wait(a, parity)
        gpu.barrier()
        for t_ in range(i32(gpu.block_id("x")), m, nblk):
            t = i32(t_)
            for g_ in range(tid, i32(HV), i32(NTT)):
                g = i32(g_)
                acc = [fx.Float32(0.0) for _ in range(8)]
                for r in range_constexpr(TOPK):
                    e = fx.Int32(
                        bld(rids, (t * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0)
                    )
                    w = fx.Int32(
                        bld(rwts, (t * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0)
                    ).bitcast(fx.Float32)
                    # drop invalid routes (their part slot was never scattered)
                    w = ((e >= i32(0)) & (e < i32(E))).select(w, fx.Float32(0.0))
                    prow = (t * i32(TOPK) + i32(r)) * i32(PART_RB)
                    w0 = fx.Int32(bld(rpart, prow + g * i32(8), 0, T.i32, 0))
                    w1 = fx.Int32(bld(rpart, prow + g * i32(8) + i32(4), 0, T.i32, 0))
                    e8 = fx.Int32(bld(rpart, prow + i32(H) + g, 0, T.i8, 0)) & i32(0xFF)
                    fv = _decode_group8(w0, w1, e8)
                    for i in range_constexpr(8):
                        acc[i] = acc[i] + w * fv[i]
                bst(pack_bf16x8(acc), rout, (t * i32(HV) + g) * i32(16), 0, 0)

    return locals()
