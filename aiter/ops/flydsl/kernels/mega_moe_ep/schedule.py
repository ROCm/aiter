# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""In-kernel dispatch planner for the EP MegaMoE kernel.

* the active-expert space is this rank's ``EPR`` *local* experts (bitmap ``LDS_ACTIVE``,
  per-expert counts ``LDS_EXPERT_ROWS``);
* the routing scanned is the dispatched ``(token, expert)`` pairs this rank *received*
  -- ``a["recv_expert"]`` (received pair -> local expert id) over ``a["n_recv"]``
  pairs -- which the dispatch fills;
* a local expert routed by ``R`` received tokens runs as ``ceil(R / TILE_M)`` row chunks
  (echunks), whose n-tiles the CTAs of the owning XCD pull from a tile queue.

``compute_units`` runs the whole compute side of one launch: the dispatch sends, the
position pre-pass and plan, the per-echunk received-row gather (``gather_recv_chunk``)
and the per-XCD GEMM1 -> GEMM2 tile queue.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import ceildiv
from .common import (
    AUX_SYS,
    _ctpop,
    bld,
    bst,
    g_add_agent,
    g_ld_sys,
    g_or_agent,
    g_st_sys,
    i32,
    lds_atomic_add,
    lds_atomic_or,
    lds_ld_i32,
    lds_st,
    rsrc,
    wait_vm,
    wave_rank,
)
from .mega_moe_ep_config import (
    CTL_N_ACTIVE,
    CTL_N_CHUNKS,
    CTL_PREV_COUNT,
    CTL_QUEUE_INDEX,
    CTL_WALK_START,
    EPOCH_SLOT,
    ERR_COMPUTE,
    ERR_DISPATCH,
    FLAG_SMALL_PAYLOAD,
    GEMM1_DONE,
    GRID_BAR_XCD_COUNT,
    GRID_BAR_XCD_GEN,
    LARGE_M,
    MAX_RANKS,
    N_XCD,
    QUEUE_EXITED,
    QUEUE_NEXT,
    SENDERS_PER_XCD,
    UPFRONT_ROUNDS,
    UPFRONT_ROUNDS_LARGE_M,
    XCC_OFFSETS,
    KernelCtx,
)

__all__ = ["build_schedule"]


def _hw_xcc_id():
    return fx.Int32(
        _llvm.InlineAsmOp(
            T.i32,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 4)",
            "=s",
            has_side_effects=True,
        ).result
    )


def build_schedule(kc: KernelCtx) -> dict:
    """The dispatch-planner device functions of one instance (added to kc)."""
    EPR, TOPK = kc.get("EPR TOPK")
    LDS_CTRL, LDS_GROUP_START, LDS_EXPERT_ROWS, LDS_ACTIVE = kc.get(
        "LDS_CTRL LDS_GROUP_START LDS_EXPERT_ROWS LDS_ACTIVE"
    )
    LDS_GATHER, LDS_WAVE_SUM, BITMAP_WORDS = kc.get(
        "LDS_GATHER LDS_WAVE_SUM BITMAP_WORDS"
    )
    GEMM2_DONE, CHUNK_COUNT_BITS, LDS_CHUNK_GROUP, CHUNK_GROUP_WORDS = kc.get(
        "GEMM2_DONE CHUNK_COUNT_BITS LDS_CHUNK_GROUP CHUNK_GROUP_WORDS"
    )
    const_expr, TILE_M, N_TILES = kc.get("const_expr TILE_M N_TILES")
    HW_BARRIER = kc.get("HW_BARRIER")[0]
    N_TILES2 = kc.get("N_TILES2")[0]
    # The inline-A GEMMs (gemm.build_gemm) and the combine epilogue
    # (communication.build_communication) are added to kc before build_schedule.
    gemm1, gemm2_scatter = kc.get("gemm1 gemm2_scatter")
    cta_barrier = kc.get("cta_barrier")[0]
    combine_arrive_rows = kc.get("combine_arrive_rows")[0]
    PULL_AHEAD = kc.get("PULL_AHEAD")[0]
    TOKEN_BUCKETS, BUCKET_COUNTERS, LDS_BUCKET_END, HOT_SUBGROUP_CHUNKS = kc.get(
        "TOKEN_BUCKETS BUCKET_COUNTERS LDS_BUCKET_END HOT_SUBGROUP_CHUNKS"
    )
    SMALL_BATCH, small_send, small_wait = kc.get("SMALL_BATCH small_send small_wait")
    source_windows, small_done, small_scrub = kc.get(
        "source_windows small_done small_scrub"
    )
    small_wait_header, small_tag = kc.get("small_wait_header small_tag")
    small_routing_word = kc.get("small_routing_word")[0]
    dispatch_route_all, payload_send = kc.get("dispatch_route_all payload_send")
    hot_owners = kc.get("hot_owners")[0]
    payload_wait, payload_reset = kc.get("payload_wait payload_reset")
    OWNER_SENDERS = kc.get("OWNER_SENDERS")[0]
    report = kc.get("report")[0]
    dispatch_wait_meta = kc.get("dispatch_wait_meta")[0]
    grid_barrier = kc.get("grid_barrier")[0]
    ctrl_at, spin_sys_ge = kc.get("ctrl_at spin_sys_ge")
    NW, NT = kc.get("NW NT")
    # The plan hand-off (expert_count, expert_recv_slots) goes through memory with
    # system-scope accesses and each XCD gathers only its own tile-queue echunks, so
    # neither barrier before the GEMMs needs the agent-scope L2 writeback / invalidate
    # (needs every bid % N_XCD class on one hardware XCC, checked in _tile_queue).
    AUX_PLAN = AUX_SYS
    # Small-batch variant: every CTA counts the whole routing window into LDS itself and
    # each XCD buckets the rows of its own expert-aligned echunk range with an
    # XCD-private position counter, so the plan needs neither the grid barrier nor
    # expert_recv_slots / gather. Small batches up to LOCAL_ROUNDS NT-route rounds, or
    # when the expert-snapped split's largest XCD range exceeds LOCAL_IMB/8 of the even
    # share (hot experts; LOCAL_IMB_ANY/8 beyond LOCAL_IMB_ROUNDS rounds, where the
    # per-CTA route scan outweighs a mild imbalance): small_local, every CTA ranks its
    # XCD's routes itself on the even (unsnapped) split, so a hot expert's echunks may
    # span XCDs (small_bucket would leave a hot expert to one XCD).
    LOCAL_ROUNDS, LOCAL_IMB_ROUNDS, LOCAL_IMB, LOCAL_IMB_ANY = 8, 24, 10, 11
    HOT_SPLIT, LDS_SEGMENTS = kc.get("HOT_SPLIT LDS_SEGMENTS")
    # LDS_SEGMENTS words
    SEG_START, SEG_EXPERT, SEG_CHUNK0 = 4, 4 + EPR + 1, 4 + 2 * EPR + 1

    def _seg_addr(base, i):
        return LDS_SEGMENTS + (i32(base) + i) * i32(4)

    def _group_start(L, j):
        """Start of echunk group j: active expert j, or (HOT_SPLIT) this XCD's segment
        j."""
        return lds_ld_i32(
            L, _seg_addr(SEG_START, j) if HOT_SPLIT else LDS_GROUP_START + j * i32(4)
        )

    def _n_groups(L):
        return lds_ld_i32(
            L, LDS_SEGMENTS + 8 if HOT_SPLIT else LDS_CTRL + CTL_N_ACTIVE * 4
        )

    def _first_chunk(L):
        """First echunk LDS_CHUNK_GROUP covers: 0, or (HOT_SPLIT) this XCD's range
        start."""
        return lds_ld_i32(L, LDS_SEGMENTS) if HOT_SPLIT else i32(0)

    @flyc.jit
    def xcd_barrier(L, tid, a):
        """Barrier over this XCD's CTAs (bid % N_XCD).  Their drained stores are in the
        shared L2, so no L2 writeback / invalidate."""
        wait_vm(0)
        cta_barrier(L, tid)
        if tid == i32(0):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            ca = ctrl_at(a, i32(GRID_BAR_XCD_COUNT) + x)
            ga = ctrl_at(a, i32(GRID_BAR_XCD_GEN) + x)
            cx = (i32(gpu.grid_dim.x) - i32(1) - x) // i32(N_XCD) + i32(1)
            g = g_ld_sys(ga)
            if g_add_agent(ca, i32(1)) == cx - i32(1):
                g_st_sys(ca, i32(0))
                wait_vm(0)  # the reset lands before the generation lets a CTA re-arrive
                g_st_sys(ga, g + i32(1))
            spin_sys_ge(ga, g + i32(1), a, ERR_COMPUTE)
        cta_barrier(L, tid)

    @flyc.jit
    def echunk_wait(a, lane0, slot, n):
        """Block until ctrl[slot] (an echunk's tile counter) reaches n.  ``lane0`` polls;
        the caller syncs the rest of its waves."""
        if lane0:
            spin_sys_ge(ctrl_at(a, slot), n, a, ERR_COMPUTE)

    def masked(le):
        # invalid / foreign slot: le >= EPR (and -1 wraps to a huge unsigned value)
        return fx.Int32(le).bitcast(fx.Uint32) >= fx.Uint32(EPR)

    def _search_group(L, q):
        """Active index j of echunk q: binary search of the active-expert chunk prefix
        LDS_GROUP_START (_chunk_table) for the last start <= q."""
        nact = _n_groups(L)
        j = i32(0)
        for b in range_constexpr((EPR - 1).bit_length() - 1, -1, -1):
            cand = j + i32(1 << b)
            st = _group_start(L, fx.min(cand, i32(EPR)))
            j = ((cand < nact) & (st <= q)).select(cand, j)
        return j

    @flyc.jit
    def _chunk_group(L, q):
        j = i32(0)
        r = q - _first_chunk(L)
        if r < i32(4 * CHUNK_GROUP_WORDS):
            w = lds_ld_i32(L, LDS_CHUNK_GROUP + r.shrui(i32(2)) * i32(4))
            j = w.shrui((r & i32(3)) * i32(8)) & i32(0xFF)
        else:
            j = _search_group(L, q)
        return j

    @flyc.jit
    def _fill_chunk_group(L, tid):
        """LDS_CHUNK_GROUP[q] = active index of echunk q (q < 4*CHUNK_GROUP_WORDS): one
        search per word, then j advances by at most one per echunk (every active expert
        has >= 1 chunk)."""
        if const_expr(HOT_SPLIT):
            nq = lds_ld_i32(L, LDS_SEGMENTS + 4) - _first_chunk(L)
        else:
            nq = lds_ld_i32(L, LDS_CTRL + CTL_N_CHUNKS * 4)
        nw = fx.min(ceildiv(nq, i32(4)), i32(CHUNK_GROUP_WORDS))
        for w_ in range(tid, nw, i32(NT)):
            w = i32(w_)
            q = _first_chunk(L) + w * i32(4)
            j = _search_group(L, q)
            word = j
            for k in range_constexpr(1, 4):
                nxt = _group_start(L, j + i32(1))
                j = (nxt <= q + i32(k)).select(j + i32(1), j)
                word = word | (j << i32(8 * k))
            lds_st(L, LDS_CHUNK_GROUP + w * i32(4), word)

    def chunk_info(L, q):
        """(expert, c, ce, c0) of echunk q: chunk c of its group's ce chunks, the group
        starting at the expert's chunk c0 (the group is the expert's chunks, or
        HOT_SPLIT this XCD's segment of them)."""
        j = _chunk_group(L, q) if CHUNK_GROUP_WORDS else _search_group(L, q)
        s0 = _group_start(L, j)
        s1 = _group_start(L, j + i32(1))
        if const_expr(HOT_SPLIT):
            return (
                lds_ld_i32(L, _seg_addr(SEG_EXPERT, j)),
                q - s0,
                s1 - s0,
                lds_ld_i32(L, _seg_addr(SEG_CHUNK0, j)),
            )
        expert = lds_ld_i32(L, LDS_ACTIVE + (i32(2 * BITMAP_WORDS) + j) * i32(4))
        return expert, q - s0, s1 - s0, i32(0)

    def chunk_rows(L, expert, c):
        """Valid rows of chunk c of ``expert`` (the last chunk may be partial)."""
        return fx.min(
            lds_ld_i32(L, LDS_EXPERT_ROWS + expert * i32(4)) - c * i32(TILE_M),
            i32(TILE_M),
        )

    def _active_index(L, e):
        """Active index of (active) local expert e (its LDS_GROUP_START slot)."""
        wv = lds_ld_i32(L, LDS_ACTIVE + (e >> i32(5)) * i32(4))
        low = (i32(1) << (e & i32(31))) - i32(1)
        return lds_ld_i32(
            L, LDS_ACTIVE + (i32(BITMAP_WORDS) + (e >> i32(5))) * i32(4)
        ) + _ctpop(wv & low)

    def _snap_to_group(L, b, nch):
        """Echunk boundary b snapped to the nearer start of the expert group holding it."""
        _, c, ce, _ = chunk_info(L, b)
        s0 = b - c
        return fx.min(fx.max(((c + c) <= ce).select(s0, s0 + ce), i32(0)), nch)

    def _xcd_chunks(L, nch, x, a):
        """XCD x's tile-queue echunks [lo, hi): an even split of nch; small-batch
        outside small_local snaps the bounds to expert groups so one XCD owns all of an
        expert's echunks (small_bucket's XCD-private positions need it; small_local's
        deterministic ones do not)."""
        if const_expr(HOT_SPLIT):  # this XCD's (x's) range, from _hot_segments
            return lds_ld_i32(L, LDS_SEGMENTS), lds_ld_i32(L, LDS_SEGMENTS + 4)
        lo = x * nch // i32(N_XCD)
        hi = (x + i32(1)) * nch // i32(N_XCD)
        if SMALL_BATCH:
            lo_s = (x == i32(0)).select(
                i32(0), lds_ld_i32(L, LDS_WAVE_SUM + x * i32(4))
            )
            hi_s = (x == i32(N_XCD - 1)).select(
                nch, lds_ld_i32(L, LDS_WAVE_SUM + (x + i32(1)) * i32(4))
            )
            split = a["use_local"]
            lo = split.select(lo, lo_s)
            hi = split.select(hi, hi_s)
        return lo, hi

    @flyc.jit
    def _small_use_local(L, tid, a, nch):
        """small_local for batches up to LOCAL_ROUNDS rounds, or when the snapped
        split's largest XCD range exceeds LOCAL_IMB/8 of nch/N_XCD (LOCAL_IMB_ANY/8
        beyond LOCAL_IMB_ROUNDS rounds); identical on every CTA (from LDS_GROUP_START).
        LDS_WAVE_SUM[x] = snapped bound x (0 < x < N_XCD), for _xcd_chunks."""
        few_rounds = _window_len(a) <= i32(LOCAL_ROUNDS * NT)
        if (tid >= i32(1)) & (tid < i32(N_XCD)):
            lds_st(
                L,
                LDS_WAVE_SUM + tid * i32(4),
                _snap_to_group(L, tid * nch // i32(N_XCD), nch),
            )
        cta_barrier(L, tid)
        prev, widest = i32(0), i32(0)
        for x in range_constexpr(1, N_XCD + 1):
            b = nch if x == N_XCD else lds_ld_i32(L, LDS_WAVE_SUM + i32(x * 4))
            widest = fx.max(widest, b - prev)
            prev = b
        imbalanced = widest * i32(8 * N_XCD) > nch * i32(LOCAL_IMB)
        imbalanced_any = widest * i32(8 * N_XCD) > nch * i32(LOCAL_IMB_ANY)
        return (
            few_rounds
            | ((_window_len(a) <= i32(LOCAL_IMB_ROUNDS * NT)) & imbalanced)
            | imbalanced_any
        )

    def _wave_prefix(v, bits):
        pre, tot = i32(0), i32(0)
        for b in range_constexpr(bits):
            below, cnt = wave_rank(((v >> i32(b)) & i32(1)) == i32(1))
            pre = pre + (below << i32(b))
            tot = tot + (cnt << i32(b))
        return pre, tot

    @flyc.jit
    def _bitmap_prefix(L, tid):
        if tid == i32(0):
            acc = i32(0)
            for w in range_constexpr(BITMAP_WORDS):
                lds_st(L, LDS_ACTIVE + i32((BITMAP_WORDS + w) * 4), acc)
                acc = acc + _ctpop(lds_ld_i32(L, LDS_ACTIVE + i32(w * 4)))
            lds_st(L, LDS_CTRL + CTL_N_ACTIVE * 4, acc)

    @flyc.jit
    def _add_active(L, e):
        if e < i32(EPR):
            wv = lds_ld_i32(L, LDS_ACTIVE + (e >> i32(5)) * i32(4))
            if ((wv >> (e & i32(31))) & i32(1)) != i32(0):
                low = (i32(1) << (e & i32(31))) - i32(1)
                rank = lds_ld_i32(
                    L, LDS_ACTIVE + (i32(BITMAP_WORDS) + (e >> i32(5))) * i32(4)
                ) + _ctpop(wv & low)
                lds_st(L, LDS_ACTIVE + (i32(2 * BITMAP_WORDS) + rank) * i32(4), e)

    @flyc.jit
    def _chunk_table(L, tid):
        nact = lds_ld_i32(L, LDS_CTRL + CTL_N_ACTIVE * 4)
        w = tid // i32(64)
        lane = tid % i32(64)
        base = i32(0)
        for rnd in range_constexpr(ceildiv(EPR, NT)):
            j = tid + i32(rnd * NT)
            live = j < nact
            e = live.select(
                lds_ld_i32(
                    L,
                    LDS_ACTIVE
                    + (i32(2 * BITMAP_WORDS) + fx.min(j, i32(EPR - 1))) * i32(4),
                ),
                i32(0),
            )
            r = lds_ld_i32(L, LDS_EXPERT_ROWS + e * i32(4))
            # Balanced gather: position // TILE_M is an exact partition of the expert's
            # 0..R_e-1 rows, so a chunk never exceeds TILE_M (no row drop) and only the
            # expert's last chunk is partial.
            ce = fx.max(ceildiv(r, i32(TILE_M)), i32(1))
            ce = live.select(ce, i32(0))
            pre, tot = _wave_prefix(ce, CHUNK_COUNT_BITS)
            if lane == i32(0):
                lds_st(L, LDS_WAVE_SUM + w * i32(4), tot)
            cta_barrier(L, tid)
            o, total = base + pre, i32(0)
            for v in range_constexpr(NW):
                sv = lds_ld_i32(L, LDS_WAVE_SUM + i32(v * 4))
                o = o + (i32(v) < w).select(sv, i32(0))
                total = total + sv
            base = base + total
            if live:
                lds_st(L, LDS_GROUP_START + j * i32(4), o)
            cta_barrier(L, tid)
        if tid == i32(0):
            lds_st(L, LDS_GROUP_START + nact * i32(4), base)
            lds_st(L, LDS_CTRL + CTL_N_CHUNKS * 4, base)

    EXPERTS_PER_LANE = ceildiv(EPR, 64)  # _hot_segments: active experts per lane
    LANE_SLOTS = tuple(range(EXPERTS_PER_LANE))

    def _lane_groups(tid):
        if EXPERTS_PER_LANE == 1:
            return [tid]
        return [tid * i32(EXPERTS_PER_LANE) + i32(r) for r in LANE_SLOTS]

    def _lane_prefix(vs, bits):
        """Exclusive prefix over j = lane * EXPERTS_PER_LANE + r of wave 0's
        EXPERTS_PER_LANE values per lane (each < 2**bits), and the wave total."""
        if EXPERTS_PER_LANE == 1:
            pre, tot = _wave_prefix(vs[0], bits)
            return [pre], tot
        s = vs[0]
        for v in vs[1:]:
            s = s + v
        pre, tot = _wave_prefix(s, bits + (EXPERTS_PER_LANE - 1).bit_length())
        out = [pre]
        for v in vs[:-1]:
            out.append(out[-1] + v)
        return out, tot

    def _lane_rank(sels):
        """``_lane_prefix`` of the predicates ``sels`` (rank among the set ones, count)."""
        if EXPERTS_PER_LANE == 1:
            b, n = wave_rank(sels[0])
            return [b], n
        return _lane_prefix([s.select(i32(1), i32(0)) for s in sels], 1)

    @flyc.jit
    def _hot_segments(L, tid, nch):
        """HOT_SPLIT (wave 0, lane l = active experts l EXPERTS_PER_LANE .. l
        EXPERTS_PER_LANE + EXPERTS_PER_LANE - 1): the echunk list reordered XCD-major --
        XCD x's range [X0, X1) holds chunks [x ce / 8, (x + 1) ce / 8) of every hot
        expert (ce * 64 > HOT_SPLIT * nch) and the even share x of the other experts'
        chunks (in expert order, as without HOT_SPLIT), as one segment per expert
        ordered by (payload round le % PAYLOAD_ROUNDS, expert), so every XCD starts on
        the earliest-sent rows."""
        if tid < i32(64):
            x = i32(gpu.block_id("x")) % i32(N_XCD)
            nact = lds_ld_i32(L, LDS_CTRL + CTL_N_ACTIVE * 4)
            R = LANE_SLOTS
            js = _lane_groups(tid)
            live = [j < nact for j in js]
            jj = [fx.min(j, i32(EPR - 1)) for j in js]
            s0 = [lds_ld_i32(L, LDS_GROUP_START + j * i32(4)) for j in jj]
            ce = [
                live[r].select(
                    lds_ld_i32(L, LDS_GROUP_START + (jj[r] + i32(1)) * i32(4)) - s0[r],
                    i32(0),
                )
                for r in R
            ]
            expert = [
                lds_ld_i32(L, LDS_ACTIVE + (i32(2 * BITMAP_WORDS) + j) * i32(4))
                for j in jj
            ]
            hot = [c * i32(64) > nch * i32(HOT_SPLIT) for c in ce]
            hc = [hot[r].select(ce[r], i32(0)) for r in R]
            cc = [hot[r].select(i32(0), ce[r]) for r in R]
            cpre, ctot = _lane_prefix(cc, CHUNK_COUNT_BITS)
            # TOKEN_BUCKETS: hot chunks x, x + 8, ... (as many as c % 8 == x)
            if const_expr(TOKEN_BUCKETS):
                hlo = [(h // i32(N_XCD)) * x + fx.min(h % i32(N_XCD), x) for h in hc]
            else:
                hlo = [x * h // i32(N_XCD) for h in hc]
            k0 = x * ctot // i32(N_XCD)
            k1 = (x + i32(1)) * ctot // i32(N_XCD)
            lo = [hot[r].select(hlo[r], fx.max(k0 - cpre[r], i32(0))) for r in R]
            if const_expr(TOKEN_BUCKETS):
                hhi = [
                    hlo[r]
                    + hc[r] // i32(N_XCD)
                    + (x < hc[r] % i32(N_XCD)).select(i32(1), i32(0))
                    for r in R
                ]
            else:
                hhi = [(x + i32(1)) * h // i32(N_XCD) for h in hc]
            hi = [hot[r].select(hhi[r], fx.min(k1 - cpre[r], cc[r])) for r in R]
            ln = [fx.max(hi[r] - lo[r], i32(0)) for r in R]
            _, x0h = _lane_prefix(hlo, CHUNK_COUNT_BITS)
            x0 = x0h + k0
            emit = [n_ > i32(0) for n_ in ln]
            st, k = [i32(0)] * EXPERTS_PER_LANE, [i32(0)] * EXPERTS_PER_LANE
            acc, kacc = i32(0), i32(0)
            for g in range_constexpr(PAYLOAD_ROUNDS):
                sel = [emit[r] & (expert[r] % i32(PAYLOAD_ROUNDS) == i32(g)) for r in R]
                p, t = _lane_prefix(
                    [sel[r].select(ln[r], i32(0)) for r in R], CHUNK_COUNT_BITS
                )
                b, n = _lane_rank(sel)
                st = [sel[r].select(acc + p[r], st[r]) for r in R]
                k = [sel[r].select(kacc + b[r], k[r]) for r in R]
                acc = acc + t
                kacc = kacc + n
            for r in range_constexpr(EXPERTS_PER_LANE):
                # TOKEN_BUCKETS: hot c0 = -1 - x (_expert_chunk)
                c0w = hot[r].select(i32(-1) - x, lo[r]) if TOKEN_BUCKETS else lo[r]
                if emit[r]:
                    lds_st(L, _seg_addr(SEG_START, k[r]), x0 + st[r])
                    lds_st(L, _seg_addr(SEG_EXPERT, k[r]), expert[r])
                    lds_st(L, _seg_addr(SEG_CHUNK0, k[r]), c0w)
            if tid == i32(0):
                lds_st(L, LDS_SEGMENTS, x0)
                lds_st(L, LDS_SEGMENTS + 4, x0 + acc)
                lds_st(L, LDS_SEGMENTS + 8, kacc)
                lds_st(L, _seg_addr(SEG_START, kacc), x0 + acc)

    @flyc.jit
    def assign_positions(tid, a):
        """Balanced-gather pre-pass (pre-sort): give every received (token,route) pair a
        GLOBAL per-local-expert position (0..R_e-1 rank) via an agent-scope atomic
        counter (``expert_count``, host-zeroed each launch) and bucket its recv slot
        straight into ``expert_recv_slots[le*cap + pos]`` (cap = n_ranks*mtpr: a token
        routes to an expert at most once).  Strided across ALL CTAs x threads so each
        recv slot is visited exactly once; the gather then reads a chunk's TILE_M rows
        by position -- O(TILE_M), no rescan of the n_ranks x m x topk window per echunk.
        One atomic per (wave, local expert): a hot expert's routes would otherwise
        serialize on one address."""
        r_recv_expert = rsrc(a["recv_expert"], a["n_recv"] * i32(4))
        cap = a["n_ranks"] * a["mtpr"]
        r_expert_slots = rsrc(a["expert_recv_slots"], i32(EPR) * cap * i32(4))
        expert_count = fx.Int64(a["expert_count"])
        base = i32(gpu.block_id("x")) * i32(NT) + tid
        grid_stride = i32(gpu.grid_dim.x) * i32(NT)
        # wave-uniform trip count (full waves for the match ballots); each group of
        # same-expert lanes takes one atomic of its size at its lowest lane
        lane = tid % i32(64)
        n = _window_len(a)
        for c0_ in range(base - lane, n, grid_stride):
            c = i32(c0_) + lane
            idx, le = _routing_entry(a, r_recv_expert, fx.min(c, n - i32(1)))
            own = (c < n) & (masked(le) == fx.Boolean(False))
            # key (expert, token bucket), bucket-major rows
            if const_expr(TOKEN_BUCKETS):
                jb = _token_bucket(a, fx.min(c, n - i32(1)))
                key = own.select(le * i32(TOKEN_BUCKETS) + jb, i32(EPR * TOKEN_BUCKETS))
                below, cnt, lead = _match_lead(
                    key, lane, (EPR * TOKEN_BUCKETS).bit_length()
                )
            else:
                key = own.select(le, i32(EPR))
                below, cnt, lead = _match_lead(key, lane, EPR.bit_length())
            pos_base = i32(0)
            if own & (below == i32(0)):
                pos_base = g_add_agent(expert_count + fx.Int64(key) * fx.Int64(4), cnt)
            pos_base = gpu.shuffle(pos_base, lead, 64, mode="idx")
            if own:
                if const_expr(TOKEN_BUCKETS):
                    bst(
                        idx,
                        r_expert_slots,
                        (le * cap + jb * _bucket_cap(a) + pos_base + below) * i32(4),
                        0,
                        AUX_PLAN,
                    )
                else:
                    bst(
                        idx,
                        r_expert_slots,
                        (le * cap + pos_base + below) * i32(4),
                        0,
                        AUX_PLAN,
                    )

    def _epoch_bump(a):
        # skip 0 (the zeroed flags would pass its waits) by 2: the parity banks alternate
        e1 = a["epoch"] + i32(1)
        g_st_sys(ctrl_at(a, EPOCH_SLOT), (e1 == i32(0)).select(i32(2), e1))

    def _expert_routes(L, r_expert_count, e):
        """Routes of local expert e: the small-batch kernel's own LDS count, else
        assign_positions's global expert_count."""
        if SMALL_BATCH:
            return lds_ld_i32(L, LDS_EXPERT_ROWS + e * i32(4))
        if TOKEN_BUCKETS:  # sum the buckets, keep their inclusive prefix
            cnt = i32(0)
            for jb in range_constexpr(TOKEN_BUCKETS):
                cnt = cnt + fx.Int32(
                    bld(
                        r_expert_count,
                        (e * i32(TOKEN_BUCKETS) + i32(jb)) * i32(4),
                        0,
                        T.i32,
                        AUX_PLAN,
                    )
                )
                lds_st(
                    L, LDS_BUCKET_END + (e * i32(TOKEN_BUCKETS) + i32(jb)) * i32(4), cnt
                )
        else:
            cnt = fx.Int32(bld(r_expert_count, e * i32(4), 0, T.i32, AUX_PLAN))
        lds_st(L, LDS_EXPERT_ROWS + e * i32(4), cnt)  # distinct e per thread: no race
        return cnt

    def _bucket_cap(a):
        """expert_recv_slots entries per (expert, token bucket) (kernel_config asserts
        TOKEN_BUCKETS | mtpr)."""
        return a["n_ranks"] * (a["mtpr"] // i32(TOKEN_BUCKETS))

    def _token_bucket(a, c):
        """Token bucket of routing-window entry c (its source's
        token * TOKEN_BUCKETS // m)."""
        o = a["window_offsets"]
        base, nxt = i32(0), o[1]
        for j in range_constexpr(1, MAX_RANKS):
            ge = c >= o[j]
            base = ge.select(o[j], base)
            nxt = ge.select(o[j + 1], nxt)
        return (
            ((c - base) // i32(TOPK))
            * i32(TOKEN_BUCKETS)
            // ((nxt - base) // i32(TOPK))
        )

    def _bucket_position(L, expert, q):
        """(bucket, position in it) of the expert's bucket-major row q."""
        jb, bs = i32(0), i32(0)
        for k in range_constexpr(1, TOKEN_BUCKETS):
            P = lds_ld_i32(
                L, LDS_BUCKET_END + (expert * i32(TOKEN_BUCKETS) + i32(k - 1)) * i32(4)
            )
            ge = q >= P
            jb = ge.select(i32(k), jb)
            bs = ge.select(P, bs)
        return jb, q - bs

    def _window_len(a):
        """Routing-window size: every source's own m_p * topk (source_windows
        prefix)."""
        return a["window_offsets"][MAX_RANKS]

    def _recv_slot(a, c):
        """Recv slot of routing-window entry c (source p's block [o[p], o[p+1]))."""
        o = a["window_offsets"]
        p, base = i32(0), i32(0)
        for j in range_constexpr(1, MAX_RANKS):
            ge = c >= o[j]
            p = ge.select(i32(j), p)
            base = ge.select(o[j], base)
        return p * (a["mtpr"] * i32(TOPK)) + (c - base)

    def _routing_entry(a, r_recv_expert, c):
        """Routing entry c of the window: (recv slot idx, local expert)."""
        idx = _recv_slot(a, c)
        if const_expr(SMALL_BATCH):
            return idx, small_routing_word(a, r_recv_expert, idx)
        return idx, fx.Int32(bld(r_recv_expert, idx * i32(4), 0, T.i32, 0))

    @flyc.jit
    def small_count(L, tid, a):
        """Small batch: LDS_EXPERT_ROWS[e] = routes of local expert e, from this CTA's
        own scan."""
        for e_ in range(tid, i32(EPR), i32(NT)):
            lds_st(L, LDS_EXPERT_ROWS + i32(e_) * i32(4), i32(0))
        cta_barrier(L, tid)
        r_recv_expert = rsrc(a["recv_expert"], a["n_recv"] * i32(4))
        n = _window_len(a)
        for c_ in range(tid, n, i32(NT)):
            _, le = _routing_entry(a, r_recv_expert, i32(c_))
            if masked(le) == fx.Boolean(False):
                lds_atomic_add(L, LDS_EXPERT_ROWS + le * i32(4), 1)
        cta_barrier(L, tid)

    @flyc.jit
    def small_bucket(L, tid, a, lo, hi):
        """Small-batch gather: XCD x owns every expert whose echunks start in [lo, hi);
        its CTAs give each of those routes a position from an XCD-private counter
        (expert_count + x*EPR) and publish it straight to chunk_recv_slot /
        chunk_payload_row, then zero the partial chunks' pad rows.  Readers are the
        XCD's own GEMM tiles (xcd_barrier)."""
        bid = i32(gpu.block_id("x"))
        x = bid % i32(N_XCD)
        cx = (i32(gpu.grid_dim.x) - i32(1) - x) // i32(N_XCD) + i32(1)
        k0 = (bid // i32(N_XCD)) * i32(NT) + tid
        xs = cx * i32(NT)
        xcd_expert_count = fx.Int64(a["expert_count"]) + fx.Int64(x * i32(EPR * 4))
        r_payload_row = rsrc(a["chunk_payload_row"])
        r_recv_slot = rsrc(a["chunk_recv_slot"])
        r_recv_expert = rsrc(a["recv_expert"], a["n_recv"] * i32(4))
        n = _window_len(a)
        for c_ in range(k0, n, xs):
            idx, le = _routing_entry(a, r_recv_expert, i32(c_))
            if masked(le) == fx.Boolean(False):
                s0 = lds_ld_i32(L, LDS_GROUP_START + _active_index(L, le) * i32(4))
                if (s0 >= lo) & (s0 < hi):
                    pos = g_add_agent(
                        xcd_expert_count + fx.Int64(le) * fx.Int64(4), i32(1)
                    )
                    off = (s0 * i32(TILE_M) + pos) * i32(4)
                    bst(idx, r_recv_slot, off, 0, 0)
                    bst(idx // i32(TOPK), r_payload_row, off, 0, 0)
        _zero_pad_rows(L, lo, hi, k0, xs, r_recv_slot, r_payload_row)

    @flyc.jit
    def _zero_pad_rows(L, lo, hi, k0, xs, r_recv_slot, r_payload_row):
        """Zero the pad rows of the partial last chunks of echunks [lo, hi)."""
        for t_ in range(k0, (hi - lo) * i32(TILE_M), xs):
            t = i32(t_)
            ei = lo + t // i32(TILE_M)
            r = t - (t // i32(TILE_M)) * i32(TILE_M)
            expert, cc, _, _ = chunk_info(L, ei)
            if r >= chunk_rows(L, expert, cc):
                off = (ei * i32(TILE_M) + r) * i32(4)
                bst(i32(0), r_recv_slot, off, 0, 0)
                bst(i32(0), r_payload_row, off, 0, 0)

    def _popcount64(m):
        return _ctpop(i32(m & fx.Int64(0xFFFFFFFF))) + _ctpop(i32(m >> fx.Int64(32)))

    def _match_mask(key, bits):
        m = fx.Int64(-1)
        for b in range_constexpr(bits):
            hit = ((key >> i32(b)) & i32(1)) == i32(1)
            bal = fx.Int64(rocdl.ballot(T.i64, hit))
            m = m & hit.select(bal, bal ^ fx.Int64(-1))
        return m

    def _match_rank(key, lane, bits):
        """(lanes below with the same key, lanes with the same key) of a fully active
        wave; key < 2**bits."""
        m = _match_mask(key, bits)
        lt = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        return _popcount64(m & lt), _popcount64(m)

    def _match_lead(key, lane, bits):
        """_match_rank plus the lowest lane with the same key."""
        m = _match_mask(key, bits)
        lt = (fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1)
        lo = i32(m & fx.Int64(0xFFFFFFFF))
        hi = i32(m >> fx.Int64(32))
        lead = (lo != i32(0)).select(
            _ctpop((lo & (i32(0) - lo)) - i32(1)),
            i32(32) + _ctpop((hi & (i32(0) - hi)) - i32(1)),
        )
        return _popcount64(m & lt), _popcount64(m), lead

    @flyc.jit
    def small_local(L, tid, a, lo, hi):
        """Small-batch gather for tiny or imbalanced batches (_small_use_local): every
        CTA of XCD x ranks the routes of its experts itself, in routing order (wave
        match-rank + cross-wave / cross-round prefix in LDS), so all of them compute the
        SAME chunk_recv_slot / chunk_payload_row.  Each writes the whole table
        (identical values) and reads back its own drained stores: no position atomics,
        no XCD barrier."""
        r_payload_row = rsrc(a["chunk_payload_row"])
        r_recv_slot = rsrc(a["chunk_recv_slot"])
        r_recv_expert = rsrc(a["recv_expert"], a["n_recv"] * i32(4))
        n = _window_len(a)
        w = tid // i32(64)
        # per-wave key counts | running per-expert base
        wave_counts = LDS_C_TILE
        expert_base = LDS_C_TILE + NW * EPR * 4
        for e_ in range(tid, i32(EPR), i32(NT)):
            lds_st(L, expert_base + i32(e_) * i32(4), i32(0))
        for rnd_ in range(i32(0), ceildiv(n, i32(NT)), i32(1)):
            c = i32(rnd_) * i32(NT) + tid
            idx, le = _routing_entry(a, r_recv_expert, fx.min(c, n - i32(1)))
            own = (c < n) & (masked(le) == fx.Boolean(False))
            j = _active_index(L, own.select(le, i32(0)))
            s0 = lds_ld_i32(L, LDS_GROUP_START + j * i32(4))
            # every expert with an echunk in [lo, hi): its routes rank alike on each XCD
            # holding part of it, and each XCD writes only its own echunks' rows
            own = (
                own
                & (s0 < hi)
                & (lds_ld_i32(L, LDS_GROUP_START + (j + i32(1)) * i32(4)) > lo)
            )
            key = own.select(le, i32(EPR))
            for t_ in range(tid, i32(NW * EPR), i32(NT)):
                lds_st(L, wave_counts + i32(t_) * i32(4), i32(0))
            cta_barrier(L, tid)
            below, cnt = _match_rank(key, tid % i32(64), EPR.bit_length())
            if own & (below == i32(0)):
                lds_st(L, wave_counts + (w * i32(EPR) + key) * i32(4), cnt)
            cta_barrier(L, tid)
            if own:
                pos = lds_ld_i32(L, expert_base + key * i32(4)) + below
                for v in range_constexpr(NW - 1):
                    wv = lds_ld_i32(L, wave_counts + (i32(v * EPR) + key) * i32(4))
                    pos = pos + (i32(v) < w).select(wv, i32(0))
                off = (s0 * i32(TILE_M) + pos) * i32(4)
                row = s0 * i32(TILE_M) + pos
                if (row >= lo * i32(TILE_M)) & (row < hi * i32(TILE_M)):
                    bst(idx, r_recv_slot, off, 0, 0)
                    bst(idx // i32(TOPK), r_payload_row, off, 0, 0)
            cta_barrier(L, tid)
            for e_ in range(tid, i32(EPR), i32(NT)):
                e = i32(e_)
                s = lds_ld_i32(L, expert_base + e * i32(4))
                for v in range_constexpr(NW):
                    s = s + lds_ld_i32(L, wave_counts + (i32(v * EPR) + e) * i32(4))
                lds_st(L, expert_base + e * i32(4), s)
            cta_barrier(L, tid)
        _zero_pad_rows(L, lo, hi, tid, i32(NT), r_recv_slot, r_payload_row)
        wait_vm(0)
        cta_barrier(L, tid)

    @flyc.jit
    def plan(L, tid, a):
        """Per-local-expert bitmap + counts -> active list -> row-chunk table (+ the hot
        expert segment table if HOT_SPLIT, the chunk -> group fill if
        CHUNK_GROUP_WORDS).  The per-expert counts come straight from assign_positions's
        global ``expert_count`` (one value per local expert; assign_positions already
        did the agent-atomic scan of recv_expert) -- so plan reads only EPR ints instead
        of re-scanning the whole n_ranks x m x topk routing window (which otherwise
        contends for HBM with the overlapped payload a2a).  The grid_barrier before this
        publishes expert_count."""
        r_expert_count = rsrc(
            fx.Int64(a["expert_count"]), i32(EPR * BUCKET_COUNTERS) * i32(4)
        )
        for e_ in range(tid, i32(EPR), i32(NT)):
            e = i32(e_)
            cnt = _expert_routes(L, r_expert_count, e)
            if cnt > i32(0):
                lds_atomic_or(  # bitmap word shared by 32 experts: atomic
                    L, LDS_ACTIVE + (e >> i32(5)) * i32(4), i32(1) << (e & i32(31))
                )
        cta_barrier(L, tid)
        _bitmap_prefix(L, tid)
        cta_barrier(L, tid)
        for it in range_constexpr(ceildiv(EPR, NT)):
            _add_active(L, tid + i32(it * NT))
        cta_barrier(L, tid)
        _chunk_table(L, tid)
        cta_barrier(L, tid)
        if const_expr(HOT_SPLIT):
            _hot_segments(L, tid, lds_ld_i32(L, LDS_CTRL + CTL_N_CHUNKS * 4))
            cta_barrier(L, tid)
        if const_expr(CHUNK_GROUP_WORDS):
            _fill_chunk_group(L, tid)
            cta_barrier(L, tid)

    @flyc.jit
    def gather_recv_chunk(L, tid, a, expert, cc):
        """Collect echunk (``expert``, chunk ``cc``)'s payload rows into LDS_GATHER:
        chunk cc is exactly positions [cc*TILE_M, min((cc+1)*TILE_M, R_e)) of the
        expert's pre-sorted bucket ``expert_recv_slots[expert*cap + pos]``
        (assign_positions) -- deterministic; only the expert's last chunk is partial.
        LDS_GATHER stores the recv slot idx (= source_key*topk + route); the GEMM reads
        payload row idx//topk and combine recovers route/source from idx."""
        cap = a["n_ranks"] * a["mtpr"]
        r_expert_slots = rsrc(a["expert_recv_slots"], i32(EPR) * cap * i32(4))
        lo = cc * i32(TILE_M)
        R = fx.min(lds_ld_i32(L, LDS_EXPERT_ROWS + expert * i32(4)) - lo, i32(TILE_M))
        for r_ in range(tid, R, i32(NT)):
            r = i32(r_)
            if const_expr(TOKEN_BUCKETS):
                jb, pos = _bucket_position(L, expert, lo + r)
                idx = fx.Int32(
                    bld(
                        r_expert_slots,
                        (expert * cap + jb * _bucket_cap(a) + pos) * i32(4),
                        0,
                        T.i32,
                        AUX_PLAN,
                    )
                )
            else:
                idx = fx.Int32(
                    bld(
                        r_expert_slots,
                        (expert * cap + lo + r) * i32(4),
                        0,
                        T.i32,
                        AUX_PLAN,
                    )
                )
            lds_st(L, LDS_GATHER + r * i32(4), idx)
        cta_barrier(L, tid)
        return R

    # XCD x's CTAs pull tiles i = 0, 1, ... of its echunks [chunk_lo, chunk_lo +
    # xcd_chunks) from ctrl[QUEUE_NEXT + x]: i < xcd_chunks * N_TILES is a GEMM1 tile
    # (_queue_tile), then GEMM2 the same way. Every GEMM1 tile is pulled before any
    # GEMM2 tile and all CTAs are resident, so a GEMM2 tile's GEMM1_DONE wait always
    # resolves.  Each CTA ends on exactly one pull past the end, so the pull that
    # returns end + #CTAs-on-the-XCD - 1 is the last: it resets.
    @flyc.jit
    def _queue_arrive(L, tid, a, slot):
        # XCD-local: drained stores are in the shared L2
        wait_vm(0)
        cta_barrier(L, tid)
        if tid == i32(0):
            lds_st(
                L, LDS_CTRL + CTL_PREV_COUNT * 4, g_add_agent(ctrl_at(a, slot), i32(1))
            )
        cta_barrier(L, tid)

    @flyc.jit
    def _queue_pull(L, tid, qa):
        if tid == i32(0):
            lds_st(L, LDS_CTRL + CTL_QUEUE_INDEX * 4, g_add_agent(qa, i32(1)))
        cta_barrier(L, tid)

    @flyc.jit
    def _queue_arrive_pull(L, tid, a, slot, qa):
        """_queue_arrive(slot) then _queue_pull(qa); the XCD-local form issues both
        atomics back to back, so they cost one round trip and one cta_barrier."""
        wait_vm(0)
        cta_barrier(L, tid)
        if tid == i32(0):
            c = g_add_agent(ctrl_at(a, slot), i32(1))
            q = g_add_agent(qa, i32(1))
            lds_st(L, LDS_CTRL + CTL_PREV_COUNT * 4, c)
            lds_st(L, LDS_CTRL + CTL_QUEUE_INDEX * 4, q)
        cta_barrier(L, tid)

    # payload send rounds (communication.payload_send)
    PAYLOAD_ROUNDS = ceildiv(EPR, N_XCD)
    LDS_C_TILE = kc.get("LDS_C_TILE")[0]
    assert (NW + 1) * EPR * 4 <= 16 * kc.get("TILE_N")[
        0
    ] * 4  # small_local's LDS_C_TILE scratch

    def _upfront_rounds(a):
        """Payload rounds every CTA sends up front (the rest go to the SENDERS_PER_XCD
        senders)."""
        return (a["m"] >= i32(LARGE_M)).select(
            i32(min(UPFRONT_ROUNDS_LARGE_M, PAYLOAD_ROUNDS)),
            i32(min(UPFRONT_ROUNDS, PAYLOAD_ROUNDS)),
        )

    def _pass_tile(L, j, n_tiles, chunk_lo, xcd_chunks):
        """(echunk, n-tile, expert, chunk) of tile j of a pass over the XCD's echunks
        [chunk_lo, chunk_lo + xcd_chunks)."""
        # expert-major: a group (an expert's echunks in the range, n_g of them) spans
        # n_tiles * n_g tiles, so tile j lies in echunk chunk_lo + j // n_tiles's group;
        # within it n-tile-major, echunk minor
        q = chunk_lo + j // n_tiles
        expert, c, ce, c0 = chunk_info(L, q)
        g0 = q - c
        gs = fx.max(g0, chunk_lo)
        ng = fx.min(g0 + ce, chunk_lo + xcd_chunks) - gs
        jj = j - n_tiles * (gs - chunk_lo)
        # a hot segment runs as TOKEN_BUCKETS sub-groups in send order
        if TOKEN_BUCKETS:
            hot = c0 < i32(0)
            # 2**lg sub-groups (<= TOKEN_BUCKETS) of >= HOT_SUBGROUP_CHUNKS echunks each
            lg = i32(0)
            for k in range_constexpr(1, TOKEN_BUCKETS.bit_length()):
                lg = (ng >= i32(HOT_SUBGROUP_CHUNKS << k)).select(i32(k), lg)
            sub_starts = [
                (i32(k) < (i32(1) << lg)).select((i32(k) * ng) >> lg, ng)
                for k in range(1, TOKEN_BUCKETS)
            ]
            sub_lo, sub_hi = i32(0), ng
            for k in range_constexpr(1, TOKEN_BUCKETS):
                sub_lo = (hot & (jj >= n_tiles * sub_starts[k - 1])).select(
                    sub_starts[k - 1], sub_lo
                )
            for k in range_constexpr(TOKEN_BUCKETS - 1, 0, -1):
                sub_hi = (hot & (jj < n_tiles * sub_starts[k - 1])).select(
                    sub_starts[k - 1], sub_hi
                )
            jj = jj - n_tiles * sub_lo
            sub_len = sub_hi - sub_lo
            nt = jj // sub_len
            e = gs + sub_lo + jj - nt * sub_len
            return e, nt, expert, _expert_chunk(c0, e - g0)
        nt = jj // ng
        e = gs + jj - nt * ng
        return e, nt, expert, c0 + e - g0

    def _expert_chunk(c0, c):
        """Expert chunk of segment chunk c (c0 < 0: hot, XCD -1 - c0's
        strided share)."""
        return (c0 >= i32(0)).select(c0 + c, i32(-1) - c0 + c * i32(N_XCD))

    def _queue_tile(L, i, gemm2_start, chunk_lo, xcd_chunks):
        """(is GEMM2, echunk, n-tile, expert, chunk) of queue index i (< queue_len)."""
        g2 = i >= gemm2_start
        n_tiles = g2.select(i32(N_TILES2), i32(N_TILES))
        j = g2.select(i - gemm2_start, i)
        # re-read per tile: a value live across the GEMM spills in its k-loop
        j = j + n_tiles * lds_ld_i32(L, LDS_CTRL + CTL_WALK_START * 4)
        j = (j >= n_tiles * xcd_chunks).select(j - n_tiles * xcd_chunks, j)
        e, ntile, expert, cc = _pass_tile(L, j, n_tiles, chunk_lo, xcd_chunks)
        return g2, e, ntile, expert, cc

    @flyc.jit
    def _walk_start(L, tid, chunk_lo, xcd_chunks):
        """CTL_WALK_START = the first whole group (echunk offset from chunk_lo) whose
        expert is sent in round 0 (le % PAYLOAD_ROUNDS == 0), else 0; the walk starts
        there and wraps, so the groups come in send-round order."""
        if const_expr(HOT_SPLIT):  # the segments are in round order already
            if tid == i32(0):
                lds_st(L, LDS_CTRL + CTL_WALK_START * 4, i32(0))
        else:
            if tid == i32(0):
                nact = lds_ld_i32(L, LDS_CTRL + CTL_N_ACTIVE * 4)
                r0 = xcd_chunks
                for j in range_constexpr(EPR):
                    s = lds_ld_i32(L, LDS_GROUP_START + i32(j * 4)) - chunk_lo
                    ex = lds_ld_i32(L, LDS_ACTIVE + i32((2 * BITMAP_WORDS + j) * 4))
                    hit = (
                        (i32(j) < nact)
                        & (s >= i32(0))
                        & (s < xcd_chunks)
                        & (r0 == xcd_chunks)
                    )
                    r0 = (hit & (ex % i32(PAYLOAD_ROUNDS) == i32(0))).select(s, r0)
                lds_st(
                    L,
                    LDS_CTRL + CTL_WALK_START * 4,
                    (r0 == xcd_chunks).select(i32(0), r0),
                )
        gpu.barrier()

    @flyc.jit
    def _tile_queue(L, tid, a, nch, nblk):
        x = i32(gpu.block_id("x")) % i32(N_XCD)
        if tid == i32(0):
            g_or_agent(
                ctrl_at(a, i32(XCC_OFFSETS)),
                i32(1) << ((_hw_xcc_id() - x) & i32(N_XCD - 1)),
            )
        if const_expr(not SMALL_BATCH):
            # the first SENDERS_PER_XCD CTAs of each XCD send the remaining rounds, then
            # join
            ns = fx.min(i32(SENDERS_PER_XCD * N_XCD), nblk)
            bid = i32(gpu.block_id("x"))
            n_ranks = a["n_ranks"]
            if bid < ns:
                # sender bid: owner bid % n_ranks (+ its deferred up-front rounds)
                if const_expr(OWNER_SENDERS):
                    po = bid % n_ranks
                    g_lo = _upfront_rounds(a)
                    g_lo = (((a["hot_owners"] >> po) & i32(1)) == i32(1)).select(
                        i32(0), g_lo
                    )
                    payload_send(
                        L,
                        tid,
                        a,
                        bid // n_ranks,
                        (ns - i32(1) - po) // n_ranks + i32(1),
                        g_lo,
                        i32(PAYLOAD_ROUNDS),
                        po,
                        i32(0),
                    )
                else:
                    payload_send(
                        L,
                        tid,
                        a,
                        bid,
                        ns,
                        _upfront_rounds(a),
                        i32(PAYLOAD_ROUNDS),
                        i32(-1),
                        i32(0),
                    )
        cx = (nblk - i32(1) - x) // i32(N_XCD) + i32(1)
        chunk_lo, chunk_hi = _xcd_chunks(L, nch, x, a)
        xcd_chunks = chunk_hi - chunk_lo
        # queue: GEMM1 tiles [0, gemm2_start), GEMM2 tiles [gemm2_start, queue_len)
        gemm2_start = xcd_chunks * i32(N_TILES)
        queue_len = gemm2_start + xcd_chunks * i32(N_TILES2)
        _walk_start(L, tid, chunk_lo, xcd_chunks)
        qa = ctrl_at(a, i32(QUEUE_NEXT) + x)
        _queue_pull(L, tid, qa)
        i = lds_ld_i32(L, LDS_CTRL + CTL_QUEUE_INDEX * 4)
        # PULL_AHEAD: the next pull runs between this tile's K-loop and its epilogue (a
        # CTA holding an unstarted GEMM1 tile is itself on a GEMM1 tile, which never
        # waits)
        claim = None
        if const_expr(PULL_AHEAD):

            def claim():
                _queue_pull(L, tid, qa)

        while i < queue_len:
            g2, e, ntile, expert, cc = _queue_tile(
                L, i, gemm2_start, chunk_lo, xcd_chunks
            )
            if g2:
                rows = chunk_rows(L, expert, cc)
                echunk_wait(a, tid == i32(0), i32(GEMM1_DONE) + e, i32(N_TILES))
                cta_barrier(L, tid)
                gemm2_scatter(L, tid, a, expert, e, rows, ntile, ntile + i32(1), claim)
                if const_expr(PULL_AHEAD):
                    _queue_arrive(L, tid, a, i32(GEMM2_DONE) + e)
                else:
                    _queue_arrive_pull(L, tid, a, i32(GEMM2_DONE) + e, qa)
                # the echunk's last GEMM2 tile: every part store of e is drained and every
                # GEMM1_DONE waiter of e has passed
                if lds_ld_i32(L, LDS_CTRL + CTL_PREV_COUNT * 4) == i32(N_TILES2 - 1):
                    if tid == i32(0):
                        g_st_sys(ctrl_at(a, i32(GEMM2_DONE) + e), i32(0))
                        g_st_sys(ctrl_at(a, i32(GEMM1_DONE) + e), i32(0))
                    if const_expr(not SMALL_BATCH):
                        combine_arrive_rows(tid, a, e, rows, i32(N_TILES2))
            else:
                if const_expr(not SMALL_BATCH):
                    if tid == i32(0):
                        # the buckets of rows [r_lo, r_hi)
                        if const_expr(TOKEN_BUCKETS):
                            r_lo = cc * i32(TILE_M)
                            r_hi = r_lo + chunk_rows(L, expert, cc)
                            prev = i32(0)
                            for jb in range_constexpr(TOKEN_BUCKETS):
                                P = lds_ld_i32(
                                    L,
                                    LDS_BUCKET_END
                                    + (expert * i32(TOKEN_BUCKETS) + i32(jb)) * i32(4),
                                )
                                if (prev < r_hi) & (P > r_lo):
                                    payload_wait(
                                        a,
                                        expert * i32(TOKEN_BUCKETS) + i32(jb),
                                        P - prev,
                                    )
                                prev = P
                        else:
                            payload_wait(
                                a,
                                expert,
                                lds_ld_i32(L, LDS_EXPERT_ROWS + expert * i32(4)),
                            )
                    cta_barrier(L, tid)
                gemm1(
                    L,
                    tid,
                    a,
                    expert,
                    e,
                    ntile,
                    ntile + i32(1),
                    chunk_rows(L, expert, cc),
                    claim,
                )
                if const_expr(PULL_AHEAD):
                    _queue_arrive(L, tid, a, i32(GEMM1_DONE) + e)
                else:
                    _queue_arrive_pull(L, tid, a, i32(GEMM1_DONE) + e, qa)
            i = lds_ld_i32(L, LDS_CTRL + CTL_QUEUE_INDEX * 4)
        if tid == i32(0):
            if i == queue_len + cx - i32(1):
                g_st_sys(qa, i32(0))
            # this CTA's XCC_OFFSETS bit lands before its QUEUE_EXITED arrival
            wait_vm(0)
            # the last CTA out: every expert's routes were counted and waited
            if g_add_agent(ctrl_at(a, i32(QUEUE_EXITED)), i32(1)) == nblk - i32(1):
                g_st_sys(ctrl_at(a, i32(QUEUE_EXITED)), i32(0))
                xmap = g_add_agent(ctrl_at(a, i32(XCC_OFFSETS)), i32(0))
                g_st_sys(ctrl_at(a, i32(XCC_OFFSETS)), i32(0))
                if (xmap & (xmap - i32(1))) != i32(0):
                    report(a, ERR_COMPUTE)
                if const_expr(SMALL_BATCH):
                    small_scrub(a)
                    small_done(a)
                    _epoch_bump(a)  # every CTA has read the epoch
                else:
                    payload_reset(a)

    @flyc.jit
    def compute_units(L, tid, a):
        """Dispatch, plan, GATHER, then the per-XCD GEMM1 -> GEMM2 tile queue
        (_tile_queue, whose GEMM2 epilogue scatters straight to the source ranks). The
        gather publishes each row's recv-slot (``chunk_recv_slot``) + payload-row
        (``chunk_payload_row``) to GLOBAL for the XCD's GEMM tiles."""
        # All 8 waves compute MFMA, so the CTA does its a2a scatter HERE on all 8 waves,
        # then all of them run the planner + inline-A GEMM.
        if const_expr(SMALL_BATCH):
            small_send(tid, a)
        else:  # the rest of the payload rounds later, by the senders (_tile_queue)
            dispatch_route_all(L, tid, a)
            a["hot_owners"] = (
                hot_owners(L, tid, a) if const_expr(OWNER_SENDERS) else i32(0)
            )
            payload_send(
                L,
                tid,
                a,
                i32(gpu.block_id("x")),
                i32(gpu.grid_dim.x),
                i32(0),
                _upfront_rounds(a),
                i32(-1),
                a["hot_owners"],
            )
        gpu.barrier()
        if const_expr(SMALL_BATCH):
            small_wait_header(tid, a)
            cta_barrier(L, tid)
            a["tag"] = small_tag(a)
        else:
            if tid == i32(0):
                # wait ROUTING only (fast)
                dispatch_wait_meta(a, a["epoch"] & i32(1), i32(0))
            cta_barrier(L, tid)
        a["window_offsets"] = source_windows(a)
        # balanced-gather pre-pass: assign global per-expert positions, then a true
        # cross-CTA barrier (the scan is distributed) publishes expert_recv_slots to
        # every CTA. All of assign_positions / plan / PASS0-gather read only the routing
        # (recv_expert), so they run NOW -- overlapping the big payload a2a still in
        # flight -- and each GEMM1 tile waits for its expert's payload just before it
        # needs it.
        if const_expr(SMALL_BATCH):
            small_count(L, tid, a)
        else:
            assign_positions(tid, a)
            # assign_positions's atomics returned and its expert_recv_slots stores went
            # to memory
            wait_vm(0)
            cta_barrier(L, tid)
            grid_barrier(a, tid)
            cta_barrier(L, tid)
            # every CTA has read it
            if (i32(gpu.block_id("x")) == i32(0)) & (tid == i32(0)):
                _epoch_bump(a)
        plan(L, tid, a)
        nch = lds_ld_i32(L, LDS_CTRL + CTL_N_CHUNKS * 4)
        nblk = i32(gpu.grid_dim.x)
        r_payload_row = rsrc(a["chunk_payload_row"])
        r_recv_slot = rsrc(a["chunk_recv_slot"])
        # --- PASS 0 (gather): walk this CTA's echunks, publish per-row recv-slot
        #     (chunk_recv_slot, for combine) + payload-row (chunk_payload_row, for
        #     GEMM1's A-gather) to global.
        if const_expr(SMALL_BATCH):
            a["use_local"] = _small_use_local(L, tid, a, nch)
            x_lo, x_hi = _xcd_chunks(L, nch, i32(gpu.block_id("x")) % i32(N_XCD), a)
            if a["use_local"]:
                small_local(L, tid, a, x_lo, x_hi)
            else:
                small_bucket(L, tid, a, x_lo, x_hi)
                xcd_barrier(L, tid, a)
                # the XCD's position atomics returned: re-zero
                if i32(gpu.block_id("x")) < i32(N_XCD):
                    for e_ in range(tid, i32(EPR), i32(NT)):
                        g_st_sys(
                            fx.Int64(a["expert_count"])
                            + fx.Int64(
                                (i32(gpu.block_id("x")) * i32(EPR) + i32(e_)) * i32(4)
                            ),
                            i32(0),
                        )
        else:
            # XCD x gathers its tile-queue echunks (_xcd_chunks)
            gx = i32(gpu.block_id("x")) % i32(N_XCD)
            x_lo, g_hi = _xcd_chunks(L, nch, gx, a)
            g_lo = x_lo + i32(gpu.block_id("x")) // i32(N_XCD)
            g_st = (nblk - i32(1) - gx) // i32(N_XCD) + i32(1)
            for ei_ in range(g_lo, g_hi, g_st):
                ei = i32(ei_)
                expert, cc, _, c0 = chunk_info(L, ei)
                expert_chunk = _expert_chunk(c0, cc) if TOKEN_BUCKETS else c0 + cc
                # -> LDS_GATHER, R; ends in cta_barrier
                R = gather_recv_chunk(L, tid, a, expert, expert_chunk)
                for r_ in range(tid, i32(TILE_M), i32(NT)):
                    r = i32(r_)
                    idx = (r < R).select(lds_ld_i32(L, LDS_GATHER + r * i32(4)), i32(0))
                    off = (ei * i32(TILE_M) + r) * i32(4)
                    bst(idx, r_recv_slot, off, 0, 0)
                    # pad rows gather the echunk's first row: the tile waits only its
                    # own rows' payload buckets, and a cached read of a row not yet
                    # landed would go stale for the tile that later needs it
                    row = (r < R).select(idx, lds_ld_i32(L, LDS_GATHER))
                    source_key = row // i32(TOPK)
                    bst(source_key, r_payload_row, off, 0, 0)
                # done reading LDS_GATHER before the next echunk's gather overwrites it
                cta_barrier(L, tid)
        # GEMM tiles read only their own XCD's gather (small-batch: synced above)
        if const_expr(not SMALL_BATCH):
            xcd_barrier(L, tid, a)
        # payload wait: only NOW does the kernel need the dispatched activation (GEMM1's
        # A-gather). The planning + gather above overlapped the payload a2a, so this spin
        # is short (payload mostly landed). After it, the rows for every echunk are live.
        # (Large kernel: per expert, in the tile queue.)
        if const_expr(SMALL_BATCH):
            small_wait(tid, a, FLAG_SMALL_PAYLOAD, ERR_DISPATCH)
        cta_barrier(L, tid)
        _tile_queue(L, tid, a, nch, nblk)

    return locals()
