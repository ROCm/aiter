# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 EP8 MoE communication kernels of the mega MoE: routing plan + count exchange (ep_prep, ep_plan) and
the combine completion barrier (ep_finish). Peers are MORI symmetric-heap buffers, passed as int64 tables of their
addresses. The call epoch lives in device memory (`epoch` [1], local int32): ep_prep and ep_finish use epoch[0] + 1
and ep_finish stores it, so the calls can be captured in a CUDA graph. Routes whose expert id is out of
[0, E_GLOBAL) are not routed (padding rows and slots, idle ranks).

Workspace `ws` (symmetric, int32): counts [NPES][E_GLOBAL] (row s written by rank s), then flags [3][NPES]: count
row / dispatch (unused) / combine of rank s finished, each holding the call's epoch, then arrived [s_cap / BLOCK_M]:
routes already written into each sorted block here (added by the sources' producers).
Schedule workspace `sched` (local, int32): [0, 8) task tickets of the per-XCD queues, [SCHED_EXPECTED + b] routes
sorted block b receives, [SCHED_EXPECTED + s_cap / BLOCK_M + b] GateUp tiles finished of block b,
[SCHED_EXPECTED + 2 s_cap / BLOCK_M + b] set once the mega kernel transposed block b's A scales.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ...tensor_shim import _run_compiled, current_stream_arg, ptr_arg
from ..atomic import _atomic_add_at
from ..host import launcher
from ..hw import _sleep
from ..mem import _pointer
from .util import _load_i32, _rsrc_of

NPES = 8
BLOCK_M = 256
SCHED_EXPECTED = 16
# Send-order keys: a route's block position in its destination's queue (the last key: later or dropped).
MAX_KEYS = 64
# Routes per thread with their loads issued together (ep_prep); ep_plan slices of up to 256 * SLICE_PER_THREAD.
ROUTE_BATCH = 8
SLICE_PER_THREAD = 4
PLAN_TOTAL = 0
# send_info: destination row in bits [0, 19), global expert above.
ROW_BITS = 19


def row_bits(e_global, wide):
    """send_info's row bits: ROW_BITS, or with wide receive rows (past 2 GiB,
    LargeMRuntime.wide) every bit the global expert and the sign (-1: not sent) leave.
    """
    return 31 - (e_global - 1).bit_length() if wide else ROW_BITS


PREP_THREADS = 1024
PLAN_THREADS = 256
FINISH_THREADS = 64


def layout(E_LOCAL):
    """The int32 offsets of the buffers with E_LOCAL experts per rank (E_GLOBAL = NPES * E_LOCAL):
    ws: counts [NPES][E_GLOBAL], flags CFLAG / DFLAG / FFLAG [NPES] each, ARRIVED [blocks].
    ep_prep -> ep_plan tables (PLAN_INTS): totals [E_GLOBAL], fewest routes from one source [E_GLOBAL], expert
    bases [E_GLOBAL + NPES], first block of each destination's 8 mega-kernel queues and its block count [NPES][9],
    first send position of each key [MAX_KEYS], and the per-key cursors ep_plan's workgroups reserve their ranges
    with [MAX_KEYS].
    ep_prep's LDS (P_INTS): its own route counts, all count rows, expert bases (+ rows per destination), this rank's
    rows before its own per expert, totals, fewest per source, queue starts, routes per send key.
    ep_plan's LDS (L_INTS): the plan tables, then routes and first send position per key of this workgroup.
    """
    E_GLOBAL = NPES * E_LOCAL
    o = {"E_GLOBAL": E_GLOBAL, "CFLAG": NPES * E_GLOBAL}
    o["DFLAG"] = o["CFLAG"] + NPES
    o["FFLAG"] = o["DFLAG"] + NPES
    o["ARRIVED"] = o["FFLAG"] + NPES
    o["PLAN_MIN"] = E_GLOBAL
    o["PLAN_BASE"] = 2 * E_GLOBAL
    o["PLAN_QSTART"] = 3 * E_GLOBAL + NPES
    o["PLAN_KEY_BASE"] = o["PLAN_QSTART"] + NPES * 9
    o["PLAN_KEY_CURSOR"] = o["PLAN_KEY_BASE"] + MAX_KEYS
    o["PLAN_INTS"] = o["PLAN_KEY_CURSOR"] + MAX_KEYS
    o["P_HIST"] = 0
    o["P_COUNTS"] = E_GLOBAL
    o["P_BASE"] = o["P_COUNTS"] + NPES * E_GLOBAL
    o["P_SBASE"] = o["P_BASE"] + E_GLOBAL + NPES
    o["P_TOTAL"] = o["P_SBASE"] + E_GLOBAL
    o["P_MIN"] = o["P_TOTAL"] + E_GLOBAL
    o["P_QSTART"] = o["P_MIN"] + E_GLOBAL
    o["P_KEY_HIST"] = o["P_QSTART"] + NPES * 9
    o["P_INTS"] = o["P_KEY_HIST"] + MAX_KEYS
    o["L_KEY_COUNT"] = o["PLAN_INTS"]
    o["L_KEY_NEXT"] = o["L_KEY_COUNT"] + MAX_KEYS
    o["L_INTS"] = o["L_KEY_NEXT"] + MAX_KEYS
    return o


def _addr(ptr, index, elem_bytes=4):
    """i64 byte address of element `index` of a global pointer."""
    return fx.Int64(fx.ptrtoint(ptr)) + fx.Int64(index) * elem_bytes


def _ordering(name):
    """fx.AtomicOrdering of None (not atomic), "monotonic", "acquire" or "release"."""
    return (
        fx.AtomicOrdering.NotAtomic
        if name is None
        else getattr(fx.AtomicOrdering, name.capitalize())
    )


def _load(addr, ordering=None):
    """i32 at an i64 global address; an ordering makes it atomic at system scope."""
    return fx.generic_load(
        _pointer(addr, fx.Int32, 4), memory_order=_ordering(ordering)
    )


def _store(addr, value, ordering=None):
    fx.generic_store(
        _pointer(addr, fx.Int32, 4), fx.Int32(value), memory_order=_ordering(ordering)
    )


def _load_i64(addr):
    return fx.generic_load(_pointer(addr, fx.Int64, 8))


def _lds_ptr(lds, index):
    return fx.inttoptr(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4),
        lds + fx.Int32(index) * 4,
    )


def _lds_load(lds, index):
    """i32 at element `index` of an LDS array at byte address `lds`."""
    return fx.generic_load(_lds_ptr(lds, index))


def _lds_store(lds, index, value):
    fx.generic_store(_lds_ptr(lds, index), fx.Int32(value))


def _lds_add(lds, index, value):
    """LDS fetch-add; returns the old value."""
    return fx.atomic_add(_lds_ptr(lds, index), fx.Int32(value), syncscope="workgroup")


def _lds_base(storage):
    return fx.Int32(fx.ptrtoint(storage.words.ptr))


def _sync_threads():
    gpu.barrier()


def _fence_system():
    fx.memory_fence()


def _min(a, b):
    return (a < b).select(a, b)


def _div_pow2(x, d):
    """x // d for a power-of-two d as one arithmetic shift, which rounds toward -inf like //. FlyDSL expands // on an
    Int32 into a signed division plus a floor fix-up (about 1 us of ep_prep's tables in its loops over LDS values).
    """
    return x >> (d.bit_length() - 1)


def _round_block(rows):
    """rows rounded up to whole blocks (a mask, see _div_pow2)."""
    return (rows + (BLOCK_M - 1)) & -BLOCK_M


@flyc.jit
def _wait_eq_sys(addr, value):
    seen = _load(addr, "acquire")
    while seen != value:
        _sleep(1)
        seen = _load(addr, "acquire")


def _signal_and_wait(ws_peers, ws, flag, me, epoch, peer):
    """Set flag[me] = epoch on `peer` (release), then wait until this rank's flag[peer] holds epoch."""
    _store(
        _load_i64(_addr(ws_peers, peer, 8)) + fx.Int64(flag + me) * 4, epoch, "release"
    )
    _wait_eq_sys(_addr(ws, flag + peer), epoch)


def _valid_expert(g, E_GLOBAL):
    """Whether global expert id g routes anywhere (padding rows and slots carry ids outside [0, E_GLOBAL))."""
    return (g >= 0) & (g < E_GLOBAL)


def _mega_row_fn(E_GLOBAL, P_COUNTS):
    """_mega_row for E_GLOBAL experts (P_COUNTS: ep_prep's LDS count rows)."""

    @flyc.jit
    def _mega_row(g, idx, me, fewest, base, lds=None, ws=None):
        """Mega-kernel row of this rank's idx-th route to global expert g: source s's i-th route sits at virtual
        position 8 i + s (compacted where sources run out). Counts [NPES][E_GLOBAL] from LDS (ep_prep) or the
        workspace (ep_plan); fewest and base are expert g's."""
        pos = idx * NPES + me
        if idx >= fewest:
            pos = fx.Int32(0)
            for s in range_constexpr(NPES):
                if const_expr(lds is None):
                    c = _load(_addr(ws, s * E_GLOBAL + g))
                else:
                    c = _lds_load(lds, P_COUNTS + s * E_GLOBAL + g)
                k = idx + (me > s).select(fx.Int32(1), fx.Int32(0))
                pos = pos + _min(c, k)
        return base + pos

    return _mega_row


def _route_key(g, row, s_cap, lds, qstart, E_LOCAL):
    """Send-order key of a route to `row` of global expert g: the block's position in its destination's queue (the
    [NPES][9] queue starts at LDS element `qstart`); MAX_KEYS - 1 for later positions and dropped routes.
    """
    dest = _div_pow2(g, E_LOCAL) if E_LOCAL & (E_LOCAL - 1) == 0 else g // E_LOCAL
    qs = qstart + dest * 9
    block = _div_pow2(row, BLOCK_M)
    q = (block >= _lds_load(lds, qs + 4)).select(fx.Int32(4), fx.Int32(0))
    q = q + (block >= _lds_load(lds, qs + q + 2)).select(fx.Int32(2), fx.Int32(0))
    q = q + (block >= _lds_load(lds, qs + q + 1)).select(fx.Int32(1), fx.Int32(0))
    key = _min(block - _lds_load(lds, qs + q), fx.Int32(MAX_KEYS - 1))
    return (row >= s_cap).select(fx.Int32(MAX_KEYS - 1), key)


def _compile_ep_prep(E_LOCAL):
    """One workgroup of 1024 threads per rank: count exchange and plan tables (ep_plan then plans the routes).
    route_idx [T*TOPK]: rank of each route among this rank's routes to its global expert (unset for no route).
    send_base [E_GLOBAL]: first destination row of this rank's routes to each global expert.
    num_valid [1]: rows this rank receives (capped at s_cap).
    status [1]: rows needed by this rank (for the capacity check).
    plan [PLAN_INTS]: tables for ep_plan.
    epoch [1]: this call's epoch is epoch[0] + 1 (ep_finish stores it)."""
    o = layout(E_LOCAL)
    E_GLOBAL, CFLAG, ARRIVED = o["E_GLOBAL"], o["CFLAG"], o["ARRIVED"]
    PLAN_MIN, PLAN_BASE, PLAN_QSTART = o["PLAN_MIN"], o["PLAN_BASE"], o["PLAN_QSTART"]
    PLAN_KEY_BASE, PLAN_KEY_CURSOR = o["PLAN_KEY_BASE"], o["PLAN_KEY_CURSOR"]
    P_HIST, P_COUNTS, P_BASE, P_SBASE = (
        o["P_HIST"],
        o["P_COUNTS"],
        o["P_BASE"],
        o["P_SBASE"],
    )
    P_TOTAL, P_MIN, P_QSTART, P_KEY_HIST = (
        o["P_TOTAL"],
        o["P_MIN"],
        o["P_QSTART"],
        o["P_KEY_HIST"],
    )
    P_INTS = o["P_INTS"]
    # A thread per global expert; the count rows in whole passes of the workgroup.
    assert E_GLOBAL + NPES <= PREP_THREADS and NPES * E_GLOBAL % PREP_THREADS == 0
    _mega_row = _mega_row_fn(E_GLOBAL, P_COUNTS)

    @fx.struct
    class Shared:
        words: fx.Array[fx.Int32, P_INTS, 16]

    @flyc.kernel(known_block_size=[PREP_THREADS, 1, 1])
    def ep_prep_kernel(
        pTopkIds: fx.Pointer,
        pRouteIdx: fx.Pointer,
        pWs: fx.Pointer,
        pWsPeers: fx.Pointer,
        pSendBase: fx.Pointer,
        pNumValid: fx.Pointer,
        pStatus: fx.Pointer,
        pSched: fx.Pointer,
        pPlan: fx.Pointer,
        pEpoch: fx.Pointer,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lds = _lds_base(fx.SharedAllocator().allocate(Shared).peek())
        epoch = _load(_addr(pEpoch, 0)) + 1
        blocks = s_cap // BLOCK_M
        if tid < E_GLOBAL:
            _lds_store(lds, P_HIST + tid, 0)
        # Before the count flag: no source adds to `arrived` until it has seen this rank's flag.
        for b in range(tid, blocks, PREP_THREADS):
            _store(_addr(pWs, ARRIVED + fx.Int32(b)), 0)
        _sync_threads()
        # Route numbering, ROUTE_BATCH routes per thread with their loads issued together (0 past the routes).
        topk_rsrc = _rsrc_of(pTopkIds, num_routes * 4)
        for r0 in range(0, num_routes, ROUTE_BATCH * PREP_THREADS):
            first = fx.Int32(r0) + tid
            experts = [
                _load_i32(topk_rsrc, (first + j * PREP_THREADS) * 4)
                for j in range_constexpr(ROUTE_BATCH)
            ]
            for j in range_constexpr(ROUTE_BATCH):
                r = first + j * PREP_THREADS
                if (r < num_routes) & _valid_expert(experts[j], E_GLOBAL):
                    _store(_addr(pRouteIdx, r), _lds_add(lds, P_HIST + experts[j], 1))
        _sync_threads()
        for j in range_constexpr(NPES * E_GLOBAL // PREP_THREADS):
            i = tid + j * PREP_THREADS
            peer_ws = _load_i64(_addr(pWsPeers, i // E_GLOBAL, 8))
            e = i % E_GLOBAL
            _store(
                peer_ws + fx.Int64(me * E_GLOBAL + e) * 4,
                _lds_load(lds, P_HIST + e),
                "monotonic",
            )
        _fence_system()
        _sync_threads()
        if tid < NPES:
            _signal_and_wait(pWsPeers, pWs, CFLAG, me, epoch, tid)
        _sync_threads()
        for j in range_constexpr(NPES * E_GLOBAL // PREP_THREADS):
            i = tid + j * PREP_THREADS
            _lds_store(lds, P_COUNTS + i, _load(_addr(pWs, i), "monotonic"))
        _sync_threads()
        # Per global expert: routes from all sources (total), the fewest from one source, and this rank's rows before
        # its own.
        if tid < E_GLOBAL:
            total = fx.Int32(0)
            before = fx.Int32(0)
            fewest = fx.Int32(1 << 30)
            for s in range_constexpr(NPES):
                c = _lds_load(lds, P_COUNTS + s * E_GLOBAL + tid)
                total = total + c
                before = before + (me > s).select(c, fx.Int32(0))
                fewest = _min(c, fewest)
            _lds_store(lds, P_TOTAL + tid, total)
            _lds_store(lds, P_SBASE + tid, before)
            _lds_store(lds, P_MIN + tid, fewest)
        _sync_threads()
        # Destination d's experts start on 256-row boundaries in expert order.
        if tid < E_GLOBAL:
            d = tid // E_LOCAL
            expert_base = fx.Int32(0)
            for k in range(d * E_LOCAL, tid):
                expert_base = expert_base + _round_block(
                    _lds_load(lds, P_TOTAL + fx.Int32(k))
                )
            _lds_store(lds, P_BASE + tid, expert_base)
            if tid % E_LOCAL == E_LOCAL - 1:
                _lds_store(
                    lds,
                    P_BASE + E_GLOBAL + d,
                    expert_base + _round_block(_lds_load(lds, P_TOTAL + tid)),
                )
            _store(_addr(pSendBase, tid), expert_base + _lds_load(lds, P_SBASE + tid))
        for b in range(tid, blocks, PREP_THREADS):
            _store(_addr(pSched, SCHED_EXPECTED + blocks + fx.Int32(b)), 0)
            _store(_addr(pSched, SCHED_EXPECTED + 2 * blocks + fx.Int32(b)), 0)
        if tid < SCHED_EXPECTED:
            _store(_addr(pSched, tid), 0)
        _sync_threads()
        # Plan tables for ep_plan: totals, fewest routes from one source, expert bases, and the first block of each of
        # destination d's mega-kernel queues (its valid blocks split in eight), then the block count.
        if tid < E_GLOBAL:
            _store(_addr(pPlan, PLAN_TOTAL + tid), _lds_load(lds, P_TOTAL + tid))
            _store(_addr(pPlan, PLAN_MIN + tid), _lds_load(lds, P_MIN + tid))
        if tid < E_GLOBAL + NPES:
            _store(_addr(pPlan, PLAN_BASE + tid), _lds_load(lds, P_BASE + tid))
        if tid < NPES * 9:
            dest_rows = _lds_load(lds, P_BASE + E_GLOBAL + tid // 9)
            qstart = _min(dest_rows, s_cap) // BLOCK_M * (tid % 9) // 8
            _lds_store(lds, P_QSTART + tid, qstart)
            _store(_addr(pPlan, PLAN_QSTART + tid), qstart)
        if tid < MAX_KEYS:
            _lds_store(lds, P_KEY_HIST + tid, 0)
            _store(_addr(pPlan, PLAN_KEY_CURSOR + tid), 0)
        _sync_threads()
        # Routes per send key: ep_plan places every route at its key's base plus its rank within the key. This rank's
        # routes to expert g are idx = 0 .. C[me][g] - 1 and their rows grow with idx, so each block takes a range of
        # idx.
        if tid < E_GLOBAL:
            n_g = _lds_load(lds, P_COUNTS + me * E_GLOBAL + tid)
            fewest_g = _lds_load(lds, P_MIN + tid)
            base_g = _lds_load(lds, P_BASE + tid)
            lo = fx.Int32(0)
            while lo < n_g:
                row = _mega_row(tid, lo, me, fewest_g, base_g, lds=lds)
                end = (_div_pow2(row, BLOCK_M) + 1) * BLOCK_M
                hi = n_g
                if lo < fewest_g:
                    # Rows base + 8 idx + me up to fewest: the first idx at or past the next block.
                    hi = _min(_div_pow2(end - base_g - me + NPES - 1, NPES), fewest_g)
                else:
                    search_lo = lo + 1
                    search_hi = n_g
                    while search_lo < search_hi:
                        mid = _div_pow2(search_lo + search_hi, 2)
                        if _mega_row(tid, mid, me, fewest_g, base_g, lds=lds) >= end:
                            search_hi = mid
                            hi = mid
                        else:
                            search_lo = mid + 1
                hi = _min(hi, n_g)
                _lds_add(
                    lds,
                    P_KEY_HIST + _route_key(tid, row, s_cap, lds, P_QSTART, E_LOCAL),
                    hi - lo,
                )
                lo = hi
        _sync_threads()
        if tid == 0:
            acc = fx.Int32(0)
            for key in range_constexpr(MAX_KEYS):
                _store(_addr(pPlan, PLAN_KEY_BASE + key), acc)
                acc = acc + _lds_load(lds, P_KEY_HIST + key)
            my_rows = _lds_load(lds, P_BASE + E_GLOBAL + me)
            _store(_addr(pNumValid, 0), _min(my_rows, s_cap))
            _store(_addr(pStatus, 0), my_rows)

    @flyc.jit
    def launch(
        pTopkIds: fx.Pointer,
        pRouteIdx: fx.Pointer,
        pWs: fx.Pointer,
        pWsPeers: fx.Pointer,
        pSendBase: fx.Pointer,
        pNumValid: fx.Pointer,
        pStatus: fx.Pointer,
        pSched: fx.Pointer,
        pPlan: fx.Pointer,
        pEpoch: fx.Pointer,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
        stream: fx.Stream,
    ):
        ep_prep_kernel(
            pTopkIds,
            pRouteIdx,
            pWs,
            pWsPeers,
            pSendBase,
            pNumValid,
            pStatus,
            pSched,
            pPlan,
            pEpoch,
            num_routes,
            me,
            s_cap,
        ).launch(grid=(1, 1, 1), block=(PREP_THREADS, 1, 1), stream=stream)

    return launch


def _compile_ep_plan(E_LOCAL, wide=False):
    """Routing plan, one workgroup of 256 threads per slice of this rank's routes:
    send_route / send_info [T*TOPK]: all routes sorted by when the destinations' mega-kernel queues need them (block
    position in the queue, the send key): workgroup j places the routes of slice j at their key's base (ep_prep) plus
    a range reserved with an atomic on the key's cursor, so the order within a key varies from call to call. The
    producers send this array chunk by chunk. info = destination row | global expert << row_bits(E_GLOBAL, wide), -1
    past s_cap.
    Slots without a route (expert id outside [0, E_GLOBAL)) go last, with info -1: ep_prep counted only the routed
    ones into the keys' bases, so they fill [routed, num_routes) and the producers skip them.
    gu_ids [s_cap] / expert_ids [s_cap/256] / sched expected rows per block: this rank's sorted blocks.
    down_ids [s_cap] (symmetric, local view): padding rows set to -1 here; producers fill the valid rows.
    Mega-kernel rows of an expert interleave the sources (_mega_row), so every block holds about 32 routes of each
    source and the first blocks of all queues load all senders evenly."""
    o = layout(E_LOCAL)
    E_GLOBAL, PLAN_MIN, PLAN_BASE = o["E_GLOBAL"], o["PLAN_MIN"], o["PLAN_BASE"]
    PLAN_QSTART, PLAN_KEY_BASE = o["PLAN_QSTART"], o["PLAN_KEY_BASE"]
    PLAN_KEY_CURSOR, PLAN_INTS = o["PLAN_KEY_CURSOR"], o["PLAN_INTS"]
    L_KEY_COUNT, L_KEY_NEXT, L_INTS = o["L_KEY_COUNT"], o["L_KEY_NEXT"], o["L_INTS"]
    _mega_row = _mega_row_fn(E_GLOBAL, o["P_COUNTS"])
    # A closure value, in FlyDSL's cache key (module globals are not).
    info_row_bits = row_bits(E_GLOBAL, wide)

    @fx.struct
    class Shared:
        words: fx.Array[fx.Int32, L_INTS, 16]

    @flyc.kernel(known_block_size=[PLAN_THREADS, 1, 1])
    def ep_plan_kernel(
        pTopkIds: fx.Pointer,
        pRouteIdx: fx.Pointer,
        pPlan: fx.Pointer,
        pWs: fx.Pointer,
        pSendRoute: fx.Pointer,
        pSendInfo: fx.Pointer,
        pGuIds: fx.Pointer,
        pDownIds: fx.Pointer,
        pExpertIds: fx.Pointer,
        pSched: fx.Pointer,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
        num_slices: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        wg = fx.Int32(fx.block_idx.x)
        lds = _lds_base(fx.SharedAllocator().allocate(Shared).peek())
        for j in range_constexpr((PLAN_INTS + PLAN_THREADS - 1) // PLAN_THREADS):
            i = tid + j * PLAN_THREADS
            if i < PLAN_INTS:
                _lds_store(lds, i, _load(_addr(pPlan, i)))
        if tid < MAX_KEYS:
            _lds_store(lds, L_KEY_COUNT + tid, 0)
        _sync_threads()
        per = (num_routes + num_slices - 1) // num_slices
        begin = wg * per
        end = _min(begin + per, num_routes)
        routes, infos, keys, ranks = [], [], [], []
        for j in range_constexpr(SLICE_PER_THREAD):
            route = begin + j * PLAN_THREADS + tid
            info = fx.Int32(0)
            key = fx.Int32(0)
            rank = fx.Int32(0)
            if route < end:
                g = _load(_addr(pTopkIds, route))
                info = fx.Int32(-1)
                key = fx.Int32(MAX_KEYS - 1)
                if _valid_expert(g, E_GLOBAL):
                    row = _mega_row(
                        g,
                        _load(_addr(pRouteIdx, route)),
                        me,
                        _lds_load(lds, PLAN_MIN + g),
                        _lds_load(lds, PLAN_BASE + g),
                        ws=pWs,
                    )
                    info = (row < s_cap).select(
                        row | (g << info_row_bits), fx.Int32(-1)
                    )
                    key = _route_key(g, row, s_cap, lds, PLAN_QSTART, E_LOCAL)
                rank = _lds_add(lds, L_KEY_COUNT + key, 1)
            routes.append(route)
            infos.append(info)
            keys.append(key)
            ranks.append(rank)
        _sync_threads()
        if tid < MAX_KEYS:
            key_routes = _lds_load(lds, L_KEY_COUNT + tid)
            reserved = fx.Int32(0)
            if key_routes > 0:
                reserved = _atomic_add_at(
                    _addr(pPlan, PLAN_KEY_CURSOR + tid), key_routes, "agent"
                )
            _lds_store(
                lds, L_KEY_NEXT + tid, _lds_load(lds, PLAN_KEY_BASE + tid) + reserved
            )
        _sync_threads()
        for j in range_constexpr(SLICE_PER_THREAD):
            if routes[j] < end:
                at = _lds_load(lds, L_KEY_NEXT + keys[j]) + ranks[j]
                _store(_addr(pSendRoute, at), routes[j])
                _store(_addr(pSendInfo, at), infos[j])
        # This rank's blocks, one per workgroup at a time (a thread per row): expert, routes and row ids.
        blocks = _min(_lds_load(lds, PLAN_BASE + E_GLOBAL + me), s_cap) // BLOCK_M
        for b in range(wg, blocks, num_slices):
            block = fx.Int32(b)
            first_row = block * BLOCK_M
            e = fx.Int32(0)
            while (e + 1 < E_LOCAL) & (
                _lds_load(lds, PLAN_BASE + me * E_LOCAL + e + 1) <= first_row
            ):
                e = e + 1
            block_g = me * E_LOCAL + e
            left = _lds_load(lds, PLAN_TOTAL + block_g) - (
                first_row - _lds_load(lds, PLAN_BASE + block_g)
            )
            block_n = _min(left, fx.Int32(BLOCK_M))
            if tid == 0:
                _store(_addr(pExpertIds, block), e)
                _store(_addr(pSched, SCHED_EXPECTED + block), block_n)
            pos = first_row + tid
            _store(_addr(pGuIds, pos), (tid < block_n).select(pos, fx.Int32(-1)))
            if tid >= block_n:
                _store(_addr(pDownIds, pos), -1)

    @flyc.jit
    def launch(
        pTopkIds: fx.Pointer,
        pRouteIdx: fx.Pointer,
        pPlan: fx.Pointer,
        pWs: fx.Pointer,
        pSendRoute: fx.Pointer,
        pSendInfo: fx.Pointer,
        pGuIds: fx.Pointer,
        pDownIds: fx.Pointer,
        pExpertIds: fx.Pointer,
        pSched: fx.Pointer,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
        num_slices: fx.Int32,
        stream: fx.Stream,
    ):
        ep_plan_kernel(
            pTopkIds,
            pRouteIdx,
            pPlan,
            pWs,
            pSendRoute,
            pSendInfo,
            pGuIds,
            pDownIds,
            pExpertIds,
            pSched,
            num_routes,
            me,
            s_cap,
            num_slices,
        ).launch(grid=(num_slices, 1, 1), block=(PLAN_THREADS, 1, 1), stream=stream)

    return launch


def _compile_ep_finish(E_LOCAL):
    """After Down (same stream): tell every rank this rank's remote adds are done, wait for all ranks, then store
    this call's epoch (epoch[0] + 1) for the next call."""
    FFLAG = layout(E_LOCAL)["FFLAG"]

    @flyc.kernel(known_block_size=[FINISH_THREADS, 1, 1])
    def ep_finish_kernel(
        pWs: fx.Pointer, pWsPeers: fx.Pointer, pEpoch: fx.Pointer, me: fx.Int32
    ):
        tid = fx.Int32(fx.thread_idx.x)
        # One wave: every lane reads the epoch before lane 0 stores the next one.
        epoch = _load(_addr(pEpoch, 0)) + 1
        _fence_system()
        if tid < NPES:
            _signal_and_wait(pWsPeers, pWs, FFLAG, me, epoch, tid)
        if tid == 0:
            _store(_addr(pEpoch, 0), epoch)

    @flyc.jit
    def launch(
        pWs: fx.Pointer,
        pWsPeers: fx.Pointer,
        pEpoch: fx.Pointer,
        me: fx.Int32,
        stream: fx.Stream,
    ):
        ep_finish_kernel(pWs, pWsPeers, pEpoch, me).launch(
            grid=(1, 1, 1), block=(FINISH_THREADS, 1, 1), stream=stream
        )

    return launch


def _ptr(tensor):
    return ptr_arg(tensor, fx.Int32)


def ep_prep(
    e_local,
    topk_ids,
    num_routes,
    me,
    epoch,
    route_idx,
    ws,
    ws_peers,
    send_base,
    num_valid,
    s_cap,
    status,
    sched,
    plan,
):
    """e_local: experts per rank; epoch: the int32 [1] epoch buffer (device memory)."""
    _run_compiled(
        launcher(_compile_ep_prep, topk_ids.device.index, E_LOCAL=e_local),
        _ptr(topk_ids),
        _ptr(route_idx),
        _ptr(ws),
        _ptr(ws_peers),
        _ptr(send_base),
        _ptr(num_valid),
        _ptr(status),
        _ptr(sched),
        _ptr(plan),
        _ptr(epoch),
        num_routes,
        me,
        s_cap,
        current_stream_arg(),
    )


def ep_plan(
    e_local,
    num_slices,
    topk_ids,
    route_idx,
    num_routes,
    me,
    s_cap,
    plan,
    ws,
    send_route,
    send_info,
    gu_ids,
    down_ids,
    expert_ids,
    sched,
    wide=False,
):
    _run_compiled(
        launcher(_compile_ep_plan, topk_ids.device.index, E_LOCAL=e_local, wide=wide),
        _ptr(topk_ids),
        _ptr(route_idx),
        _ptr(plan),
        _ptr(ws),
        _ptr(send_route),
        _ptr(send_info),
        _ptr(gu_ids),
        _ptr(down_ids),
        _ptr(expert_ids),
        _ptr(sched),
        num_routes,
        me,
        s_cap,
        num_slices,
        current_stream_arg(),
    )


def ep_finish(e_local, me, epoch, ws, ws_peers):
    """epoch: the int32 [1] epoch buffer ep_prep read; advanced here."""
    _run_compiled(
        launcher(_compile_ep_finish, epoch.device.index, E_LOCAL=e_local),
        _ptr(ws),
        _ptr(ws_peers),
        _ptr(epoch),
        me,
        current_stream_arg(),
    )
