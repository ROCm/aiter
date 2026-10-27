# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric-arena addressing, epoch-flag readiness and the watchdog for the EP kernel.

Ported from :mod:`mega_moe_tp.communication` (peer / arena resource builders, the
sticky-error watchdog) with EP-specific readiness flags replacing TP's reduce-scatter /
all-gather ones:

* ``dispatch_*`` -- a dispatch-send wave on a source rank adds its contribution to the
  expert-owner rank's per-receive-tile arrival counter; the last arriver zeroes the
  counter and stamps the paired ready flag with the launch epoch. The owner's compute
  waves ``dispatch_wait`` (spin until the ready flag reaches the epoch, wrap-safe).
* ``combine_*`` -- the mirror for GEMM2 partials scattered back to the source rank.

The device code reaches arena peers through ``peer_rs`` / ``own_rs`` (raw buffer
resources at ``peer_base + off_region``) exactly like the TP kernel.
"""

from __future__ import annotations

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .common import (
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
    lds_ld_i32,
    now,
    pack_bf16x8,
    poll_sys_ge,
    rsrc,
    wait_vm,
)
from .mega_moe_ep_config import (
    CTRL_ERR,
    ERR_COMBINE,
    ERR_COMPUTE,
    ERR_DISPATCH,
    FLAG_COMB_CNT,
    FLAG_COMBINE,
    FLAG_DISP_CNT,
    FLAG_DISPATCH,
    FLAG_DISPM,
    FLAG_DISPM_CNT,
    FLAG_INIT,
    FLAG_INIT_CNT,
    GBAR_CNT,
    GBAR_GEN,
    MAX_NPES,
    NCTA_MAX,
    KernelCtx,
)

__all__ = ["build_communication"]

# Device-phase profiling (matches mega_moe_ep_kernel._PROF): slot 12 = post combine_wait.
_PROF = bool(int(os.environ.get("EP_PROF_DEV", "0")))
_PROF_SLOTS = 19  # 13..18: per-n-tile GEMM split (gemm.py _pmark)

# dispatch_send PASS B: tokens per wave step whose loads are all issued before the
# step's remote stores (see the batching note there).
DISP_TB = 4


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
    RG = kc.get("RG")[0]  # gout row tile per CTA (combine_send reads this CTA's rows)
    L_RIX = kc.get("L_RIX")[0]  # the compute wave's gathered recv-slot list (per unit)
    ROW_BYTES = H if a8 else H // 2  # quantized activation bytes per row (FP8 vs FP4)
    N_VEC = ROW_BYTES // 16  # 16-byte (V4I) copy units per payload row
    SCALE_I32 = H // 128  # i32 E8M0-scale units per row (H//32 bytes / 4)
    PART_RB = H + H // 8  # combine ``part`` row bytes: H fp8-e4m3 then H/8 per-8 E8M0 scale
    SENTINEL = EPR  # recv_le empty-slot marker (masked by the planner: >= EPR)
    NW, NT, NTT = kc.get("NW NT NTT")  # per-launch warp-spec width (not the module globals)

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
        to hold the other compute lanes until tid0's handshake returns (see schedule.gbar)."""
        if tid == i32(0):
            cnt_a = ctrl_at(a, GBAR_CNT)
            gen_a = ctrl_at(a, GBAR_GEN)
            g = g_ld_sys(gen_a)
            old = g_add_agent(cnt_a, i32(1))
            if old == i32(gpu.grid_dim.x) - i32(1):
                g_st_sys(cnt_a, i32(0))
                g_st_sys(gen_a, g + i32(1))
            else:
                spin_sys_ge(gen_a, g + i32(1), a, ERR_COMPUTE)

    # --- EP dispatch / combine readiness -----------------------------------------
    @flyc.jit
    def dispatch_arrive(a, p, parity, tile, delta, expected):
        """Source wave adds ``delta`` to owner ``p``'s receive-tile arrival counter;
        the last arriver zeroes it and stamps the ready flag with the launch epoch."""
        old = g_add_agent(peer_flag(a, p, FLAG_DISP_CNT, parity, tile), delta)
        if old + delta == expected:
            g_st_sys(peer_flag(a, p, FLAG_DISP_CNT, parity, tile), i32(0))
            g_st_sys(peer_flag(a, p, FLAG_DISPATCH, parity, tile), a["epoch"])

    @flyc.jit
    def dispatch_wait(a, parity, tile):
        spin_sys_ge(
            own_flag(a, FLAG_DISPATCH, parity, tile), a["epoch"], a, ERR_DISPATCH
        )

    @flyc.jit
    def dispatch_arrive_meta(a, p, parity, tile, delta, expected):
        """Mirror of ``dispatch_arrive`` for the META (routing-only) dispatch phase: the
        owner can start planning (pre_pos / dyn_plan / gather) once every sender's routing
        has landed, while the big payload a2a is still in flight."""
        old = g_add_agent(peer_flag(a, p, FLAG_DISPM_CNT, parity, tile), delta)
        if old + delta == expected:
            g_st_sys(peer_flag(a, p, FLAG_DISPM_CNT, parity, tile), i32(0))
            g_st_sys(peer_flag(a, p, FLAG_DISPM, parity, tile), a["epoch"])

    @flyc.jit
    def dispatch_wait_meta(a, parity, tile):
        spin_sys_ge(
            own_flag(a, FLAG_DISPM, parity, tile), a["epoch"], a, ERR_DISPATCH
        )

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
        if (i32(gpu.block_id("x")) == i32(0)) & (tid == i32(0)):
            for p in range_constexpr(MAX_NPES):
                if i32(p) < npes:
                    init_arrive(a, i32(p), parity, npes)
        if tid == i32(0):
            spin_sys_ge(
                own_flag(a, FLAG_INIT, parity, i32(0)), a["epoch"], a, ERR_DISPATCH
            )
        gpu.barrier()

    @flyc.jit
    def combine_arrive(a, p, parity, tile, delta, expected):
        """Mirror of ``dispatch_arrive`` for GEMM2 partials scattered back to source."""
        old = g_add_agent(peer_flag(a, p, FLAG_COMB_CNT, parity, tile), delta)
        if old + delta == expected:
            g_st_sys(peer_flag(a, p, FLAG_COMB_CNT, parity, tile), i32(0))
            g_st_sys(peer_flag(a, p, FLAG_COMBINE, parity, tile), a["epoch"])

    @flyc.jit
    def combine_wait(a, parity, tile):
        spin_sys_ge(own_flag(a, FLAG_COMBINE, parity, tile), a["epoch"], a, ERR_COMBINE)

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
    def dispatch_send(L, lane, a):
        """One wave: scatter each local (token, route) to its expert-owner rank's arena,
        indexed by ``source_key = rank*mtpr + token`` (bounded by npes*mtpr,
        overflow-proof). The top-k weights stay at the source (applied in combine_final).

        Two phases so dispatch overlaps GEMM1: PASS A scatters only the tiny routing
        (recv_le, 1 int/route) + signals ``dispatch_arrive_meta`` -- the owner then plans
        (pre_pos / dyn_plan / gather, all routing-only) while PASS B's big payload a2a is
        still in flight. PASS B scatters the activation + scale and signals the full
        ``dispatch_arrive`` (payload ready) that GEMM1 waits on. Each phase is an
        all-to-all barrier (npes*nblk arrivals per owner)."""
        m, mtpr, rank, epr, npes = a["m"], a["mtpr"], a["rank"], a["epr"], a["npes"]
        parity = a["epoch"] & i32(1)
        nblk = i32(gpu.grid_dim.x)
        rids = rsrc(a["ids"], m * i32(TOPK) * i32(4))
        rx = rsrc(a["x"])
        rxs = rsrc(a["xs"])
        bid = i32(gpu.block_id("x"))
        # PASS A -- routing only (recv_le, 1 int/route): one (token, route) per LANE, so a
        # wave keeps 64 remote stores in flight per step.  The buffer resource must be
        # wave-uniform (rsrc readfirstlanes the base), so each lane's store is issued under
        # the static owner loop ``owner == p``.
        r_recv = [peer_rs(a, i32(p), "off_recv") for p in range(MAX_NPES)]
        for wk_ in range(bid * i32(64) + lane, m * i32(TOPK), nblk * i32(64)):
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
                # this rank's single combine counter now -- combine_send only arrives for
                # valid routes later, and combine_wait expects m*TOPK total.
                combine_arrive(a, rank, parity, i32(0), i32(1), m * i32(TOPK))
        wait_vm(0)  # drain routing stores before signalling META-ready
        if lane == i32(0):
            for p in range_constexpr(MAX_NPES):
                if i32(p) < npes:
                    dispatch_arrive_meta(a, i32(p), parity, i32(0), i32(1), npes * nblk)
        # PASS B -- the activation payload + E8M0 scale (the bulk of the a2a).  Iterate per
        # TOKEN (not per token-route) and DEDUP owner ranks across the token's TOPK routes:
        # the payload destination is source_key=rank*mtpr+t (per token, NOT per expert), so
        # two routes to experts on the SAME owner would scatter the identical payload to the
        # identical slot twice.  Sending once per (token, owner) cuts the payload a2a traffic
        # ~45% (TOPK=6 routes -> ~3.3 unique owners @ world=4).  ``seen`` is a per-token
        # npes-bit mask; it is updated UNCONDITIONALLY (bit=0 for invalid routes, so no
        # pollution) to avoid a conditional-reassignment scf tangle.
        #
        # Load/store batching: gfx9's vmcnt counts loads AND stores in issue order, so a
        # load issued after a remote (sc0 sc1) store waits for that store's xGMI round
        # trip.  Each step therefore issues ALL loads of DISP_TB tokens (ids, payload row,
        # scale; out-of-range tokens load a clamped row and skip the stores) up front,
        # waits once, then issues the stores to every owner with no wait in between --
        # one round-trip wait per step instead of one per 16-B chunk per owner.
        NJ = (N_VEC + 63) // 64
        NS = (SCALE_I32 + 63) // 64
        for t0_ in range(bid, m, nblk * i32(DISP_TB)):
            t0 = i32(t0_)
            toks, ids, rows, scs = [], [], [], []
            for k in range_constexpr(DISP_TB):
                t = t0 + i32(k) * nblk
                tc = fx.min(t, m - i32(1))
                toks.append(t)
                ids.append([
                    fx.Int32(bld(rids, (tc * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0))
                    for r in range(TOPK)
                ])
                rows.append([
                    bld(rx, tc * i32(ROW_BYTES)
                        + fx.min(lane + i32(j * 64), i32(N_VEC - 1)) * i32(16), 0, V4I, 0)
                    for j in range(NJ)
                ])
                scs.append([
                    bld(rxs, (tc * i32(SCALE_I32)
                              + fx.min(lane + i32(j * 64), i32(SCALE_I32 - 1))) * i32(4),
                        0, T.i32, 0)
                    for j in range(NS)
                ])
            # explicit drain: past the per-owner branches the waitcnt pass can't count the
            # stores issued and would insert partial waits that block on remote stores.
            wait_vm(0)
            for k in range_constexpr(DISP_TB):
                t = toks[k]
                if t < m:
                    source_key = rank * mtpr + t
                    seen = i32(0)
                    for r in range_constexpr(TOPK):
                        e = ids[k][r]
                        valid = (e >= i32(0)) & (e < i32(E))
                        owner = e // epr
                        bit = valid.select(i32(1) << (owner & i32(7)), i32(0))  # &7: shift-safe
                        do_send = valid & ((seen & bit) == i32(0))  # valid, owner not yet sent
                        seen = seen | bit
                        if do_send:
                            dpay = peer_rs(a, owner, "off_payload")
                            dsc = peer_rs(a, owner, "off_pscale")
                            for j in range_constexpr(NJ):
                                u = lane + i32(j * 64)
                                if u < i32(N_VEC):
                                    bst(rows[k][j], dpay,
                                        source_key * i32(ROW_BYTES) + u * i32(16), 0, AUX_SYS)
                            for j in range_constexpr(NS):
                                u = lane + i32(j * 64)
                                if u < i32(SCALE_I32):
                                    bst(scs[k][j], dsc,
                                        (source_key * i32(SCALE_I32) + u) * i32(4), 0, AUX_SYS)
        wait_vm(0)
        if lane == i32(0):
            for p in range_constexpr(MAX_NPES):
                if i32(p) < npes:
                    # one arrival per (sender rank, CTA): owner stamps at npes*nblk.
                    dispatch_arrive(a, i32(p), parity, i32(0), i32(1), npes * nblk)

    # --- combine (GEMM2 epilog scatter back to source + source-side weighted sum) --
    @flyc.jit
    def combine_send(L, tid, a, src_out, e, nrows):
        """Combine pass (per echunk ``e``): scatter the echunk's ``nrows`` output rows
        (row k's recv slot = L_RIX[k], its H bf16 output at ``src_out`` row e*RG+k) back
        to each row's SOURCE rank's ``part`` region at (source_token*topk + route), then
        signal that source (combine_arrive). ``src_out`` is the shared gout; ``nrows`` the
        echunk's gathered row count (L_RIX holds the recv-slot ids from the gather pass).

        Runs on ALL ``NW`` compute waves (not one): the scatter is a latency-bound
        cross-rank P2P stream, so striding the rows across the waves (each a 64-lane
        row scatter) multiplies the outstanding writes to better fill xGMI."""
        mtpr = a["mtpr"]
        m = a["m"]
        parity = a["epoch"] & i32(1)
        HV = H // 8
        r_rowidx = rsrc(a["rowidx"])  # echunk e's recv-slot ids (gather pass, global)
        wave = tid // i32(64)
        lane = tid % i32(64)
        for k_ in range(wave, nrows, i32(NW)):
            k = i32(k_)
            idx = fx.Int32(bld(r_rowidx, (e * i32(RG) + k) * i32(4), 0, T.i32, 0))
            sk = idx // i32(TOPK)
            route = idx - sk * i32(TOPK)
            src_rank = sk // mtpr
            src_tok = sk - src_rank * mtpr
            slot = src_tok * i32(TOPK) + route
            dpart = peer_rs(a, src_rank, "off_part")
            row_v4 = (e * i32(RG) + k) * i32(HV)  # this gout row's base V4I index
            prow = slot * i32(PART_RB)  # this slot's byte base in the fp8 part
            # per-8 fp8: same 64-lane g-loop; each lane reads its 8 bf16, quantizes to 8 fp8
            # (w0,w1) + 1 E8M0.  fp8/E8M0 stores are contiguous across lanes -> coalesced.
            for g_ in range(lane, i32(HV), i32(64)):
                g = i32(g_)
                fv = bf16x8_to_f32(bld(src_out, (row_v4 + g) * i32(16), 0, V4I, 0))
                e8m0, w0, w1 = _quantize_group8(fv)
                bst(fx.Int32(w0), dpart, prow + g * i32(8), 0, AUX_SYS)
                bst(fx.Int32(w1), dpart, prow + g * i32(8) + i32(4), 0, AUX_SYS)
                bst(fx.Int32(e8m0).to(fx.Int8), dpart, prow + i32(H) + g, 0, AUX_SYS)
            if lane == i32(0):
                combine_arrive(a, src_rank, parity, i32(0), i32(1), m * i32(TOPK))
        wait_vm(0)

    @flyc.jit
    def combine_final(L, tid, a):
        """Source-side finish: for each local token wait its top-k partials, then
        out[t] = sum over the top-k routes of part[t*topk+r] * wts[t,r]. The top-k
        weight is applied here, so dispatch / compute never carry weights. Runs on the
        whole block in the finish phase; expected=topk assumes all routes valid (the
        -1 / mask_invalid_slots case is a follow-up)."""
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
        if tid == i32(0):
            combine_wait(a, parity, i32(0))
        gpu.barrier()
        if const_expr(_PROF):  # noqa: SIM102 - compile-time gate, not dynamic
            if tid == i32(0):  # post combine_wait (slot 12)
                bst(now(), rsrc(a["dbg"]),
                    (i32(gpu.block_id("x")) * i32(_PROF_SLOTS) + i32(12)) * i32(8), 0, 0)
        for t_ in range(i32(gpu.block_id("x")), m, nblk):
            t = i32(t_)
            for g_ in range(tid, i32(HV), i32(NTT)):
                g = i32(g_)
                acc = [fx.Float32(0.0) for _ in range(8)]
                for r in range_constexpr(TOPK):
                    e = fx.Int32(bld(rids, (t * i32(TOPK) + i32(r)) * i32(4), 0, T.i32, 0))
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
