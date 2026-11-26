# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The main path's communication kernels (gfx942): x quant + out zeroing, the routing
plan with the count exchange, and the completion barrier.

Cross-rank synchronization without L2 write-back / invalidation (``buffer_wbl2`` /
``buffer_inv sc0 sc1`` of a system-scope fence): everything a peer reads is written with
system-scope (sc0 sc1, written-through) stores or lives in the uncached MORI heap, every
thread waits for its stores (vmcnt 0) before the barrier that precedes a flag store, and
flags and counts are read with system-scope loads. The mega kernel's remote adds are
complete at its kernel boundary (the finish kernel runs after it on the same stream).

Workspace ``ws`` (symmetric, int32): counts [NPES][E_GLOBAL] (row s written by rank s),
flags [2][NPES] (count row / combine of rank s finished, each holding the call's epoch),
then arrived [s_cap / BLOCK_M] (routes already written into each sorted block here, added
by the sources' producers). The epoch is ``status[STATUS_EPOCH] + 1``, a device counter
of the completed calls that the finish kernel advances: the same on every rank (they make
calls) and right in a CUDA graph replay (a launch argument would be frozen in the graph).
``sched`` (local, int32): ticket counter of mega-kernel queue q at q * QSTRIDE, the routes
block b receives at SCHED_EXPECTED + b, gemm1 tiles finished of block b at
SCHED_EXPECTED + s_cap / BLOCK_M + b. ``plan`` (local, int32): the tables the prep kernel
leaves the plan kernel (PLAN_*), incl. a copy of the count matrix.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .atomic import _latomic_add, _sys_load_at, _sys_store_at
from .cvt import _abs_bits, _amax_floor, _fp8x4_pack
from .hw import _sleep, _u
from .mem import (
    _ld,
    _rsrc,
    _store,
    buffer_store,
    global_typed_ptr,
    lds_typed_ptr,
)
from .protocol import (
    BLOCK_M,
    SCHED_EXPECTED,
    STATUS_EPOCH,
    row_bits,
    ws_layout,
)

# plan (int32): totals [E_GLOBAL], fewest routes from one source [E_GLOBAL], expert bases
# [E_GLOBAL + NPES] ([E_GLOBAL + d] = rows of destination d), first block of each
# destination's 8 mega-kernel queues and its block count [NPES][9], the counts
# [NPES][E_GLOBAL]. The plan kernel stages [0, PLAN_COUNTS) in LDS.
PLAN_TOTAL = 0


def plan_layout(E_LOCAL, NPES):
    """(PLAN_MIN, PLAN_BASE, PLAN_QSTART, PLAN_COUNTS, PLAN_INTS) with ``E_LOCAL``
    experts on each of ``NPES`` ranks: the i32 offsets of the plan tables after the
    totals and their size."""
    E_GLOBAL = NPES * E_LOCAL
    PLAN_MIN = E_GLOBAL
    PLAN_BASE = 2 * E_GLOBAL
    PLAN_QSTART = 3 * E_GLOBAL + NPES
    PLAN_COUNTS = PLAN_QSTART + NPES * 9
    return PLAN_MIN, PLAN_BASE, PLAN_QSTART, PLAN_COUNTS, PLAN_COUNTS + NPES * E_GLOBAL


MAX_KEYS = 64  # send-order keys: a route's block position in its destination's queue
ROUTE_BATCH = 8  # prep kernel: routes per thread with their loads issued together
SLICE_PER_THREAD = 4  # plan kernel: routes per thread (slices of up to 1024 routes)
PLAN_THREADS = 256


@flyc.jit
def _flag_wait(addr, epoch):
    while _sys_load_at(addr) != epoch:
        _sleep(1)


def _scan(v, lane, seg):
    """Inclusive prefix sum of i32 ``v`` over segments of ``seg`` lanes of a wave."""
    pos = lane % fx.Int32(seg)
    for bit in range_constexpr(seg.bit_length() - 1):
        d = 1 << bit
        other = fx.Int32(
            fx.gpu.shuffle_idx(v, fx.max(lane - fx.Int32(d), fx.Int32(0)), 64)
        )
        v = v + (pos >= fx.Int32(d)).select(other, fx.Int32(0))
    return v


# =============================================================================
# x quant + out zeroing
# =============================================================================


def mega_quant_kernel(*, H, QX=1):
    """x bf16 [T, H] -> x8 fp8 [T, H] + xs f32 scales (QX 1: [T, H / 128], one per (token,
    128 columns); QX 0: [T], one per token; max(amax, 2^-117) / 240, 1 for an all-zero
    group / row), and
    out [T, H] bf16 (the combine target) zeroed; quant == 0: only the zeroing. QX 1: one
    wave per (token, 512 columns): a lane holds 8 columns, a 128-column group is 16 lanes;
    QX 0: one wave per token: a lane holds 8 columns of every 512."""
    NT = 256
    CH = H // 512
    G = H // 128
    assert H % 512 == 0 and QX in (0, 1), (H, QX)

    @flyc.kernel(name=f"a8w4_megamoe_quant_h{H}", known_block_size=[NT, 1, 1])
    def kernel(
        x: fx.Int64,
        x8: fx.Int64,
        xs: fx.Int64,
        out: fx.Int64,
        tokens: fx.Int32,
        quant: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx % fx.Int32(64)
        w = fx.Int32(gpu.block_id("x")) * fx.Int32(NT // 64) + _u(tx // fx.Int32(64))
        t = w // fx.Int32(CH)
        c = w - t * fx.Int32(CH)
        if t < tokens:
            u16 = (t * fx.Int32(H) + c * fx.Int32(512)) // fx.Int32(
                8
            ) + lane  # 16 B unit
            buffer_store(
                Vec.filled(4, 0, fx.Int32),
                _rsrc(out, fx.Int64(tokens) * fx.Int64(H * 2)),
                u16 * fx.Int32(4),
            )
            if quant != fx.Int32(0):
                v = Vec(
                    _ld(
                        _rsrc(x, fx.Int64(tokens) * fx.Int64(H * 2)),
                        u16 * fx.Int32(4),
                        4,
                    )
                )
                f = Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)
                am = fx.Int32(0)
                for i in range_constexpr(8):
                    am = fx.max(am, _abs_bits(f[i]))
                for s in range_constexpr(4):
                    am = fx.max(am, am.shuffle_xor(1 << s, 64))
                scale = (am == 0).select(
                    fx.Float32(1.0), _amax_floor(am) / fx.Float32(240.0)
                )
                inv = fx.Float32(1.0) / scale
                q = Vec.from_elements(
                    [
                        _fp8x4_pack(*[fx.Float32(f[4 * h + i]) * inv for i in range(4)])
                        for h in range(2)
                    ],
                    fx.Int32,
                )
                buffer_store(
                    q, _rsrc(x8, fx.Int64(tokens) * fx.Int64(H)), u16 * fx.Int32(2)
                )
                buffer_store(
                    scale,
                    _rsrc(xs, fx.Int64(tokens) * fx.Int64(G * 4)),
                    t * fx.Int32(G) + c * fx.Int32(4) + lane // fx.Int32(16),
                    mask=lane % fx.Int32(16) == fx.Int32(0),
                )

    @flyc.kernel(name=f"a8w4_megamoe_quant_h{H}_xrow", known_block_size=[NT, 1, 1])
    def kernel_row(
        x: fx.Int64,
        x8: fx.Int64,
        xs: fx.Int64,
        out: fx.Int64,
        tokens: fx.Int32,
        quant: fx.Int32,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        lane = tx % fx.Int32(64)
        t = fx.Int32(gpu.block_id("x")) * fx.Int32(NT // 64) + _u(tx // fx.Int32(64))
        if t < tokens:
            u16 = [
                (t * fx.Int32(H) + fx.Int32(c * 512)) // fx.Int32(8) + lane
                for c in range(CH)
            ]
            out_rs = _rsrc(out, fx.Int64(tokens) * fx.Int64(H * 2))
            for c in range_constexpr(CH):
                buffer_store(Vec.filled(4, 0, fx.Int32), out_rs, u16[c] * fx.Int32(4))
            if quant != fx.Int32(0):
                x_rs = _rsrc(x, fx.Int64(tokens) * fx.Int64(H * 2))
                fs = []
                am = fx.Int32(0)
                for c in range_constexpr(CH):
                    v = Vec(_ld(x_rs, u16[c] * fx.Int32(4), 4))
                    f = Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)
                    fc = [fx.Float32(f[i]) for i in range(8)]
                    for e in fc:
                        am = fx.max(am, _abs_bits(e))
                    fs.append(fc)
                for s in range_constexpr(6):
                    am = fx.max(am, am.shuffle_xor(1 << s, 64))
                scale = (am == 0).select(
                    fx.Float32(1.0), _amax_floor(am) / fx.Float32(240.0)
                )
                inv = fx.Float32(1.0) / scale
                x8_rs = _rsrc(x8, fx.Int64(tokens) * fx.Int64(H))
                for c in range_constexpr(CH):
                    q = Vec.from_elements(
                        [
                            _fp8x4_pack(*[fs[c][4 * h + i] * inv for i in range(4)])
                            for h in range(2)
                        ],
                        fx.Int32,
                    )
                    buffer_store(q, x8_rs, u16[c] * fx.Int32(2))
                buffer_store(
                    scale,
                    _rsrc(xs, fx.Int64(tokens) * fx.Int64(4)),
                    t,
                    mask=lane == fx.Int32(0),
                )

    @flyc.jit
    def launch(
        x: fx.Int64,
        x8: fx.Int64,
        xs: fx.Int64,
        out: fx.Int64,
        tokens: fx.Int32,
        quant: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        if const_expr(QX):
            kernel(x, x8, xs, out, tokens, quant).launch(
                grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream
            )
        else:
            kernel_row(x, x8, xs, out, tokens, quant).launch(
                grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream
            )

    return launch


# =============================================================================
# count exchange + plan tables (1 workgroup per rank)
# =============================================================================


def mega_prep_kernel(*, E_LOCAL, NPES):
    """route_idx [T*TOPK]: rank of each route among this rank's routes to its global
    expert (a negative id is no route: padding, not counted, route_idx not written).
    Writes this rank's count row to every rank, waits for all rows, and fills
    ``plan``, num_valid [1] (rows this rank receives, capped at s_cap) and status[1] (rows
    needed). Zeroes this rank's arrived counters (before its count flag: no source adds to
    them before it has seen the flag) and the mega kernel's ticket and done counters.
    ``E_LOCAL``: experts per rank; ``NPES``: ranks."""
    NT = 1024
    E_GLOBAL, CFLAG, _, ARRIVED = ws_layout(E_LOCAL, NPES)
    PLAN_MIN, PLAN_BASE, PLAN_QSTART, PLAN_COUNTS, _ = plan_layout(E_LOCAL, NPES)
    # a destination's experts in one wave: a shuffle scan of each power-of-two segment
    WAVE_SCAN = E_LOCAL & (E_LOCAL - 1) == 0 and E_LOCAL <= 64
    L_HIST = 0
    L_CNT = L_HIST + E_GLOBAL
    L_BASE = L_CNT + NPES * E_GLOBAL
    L_TOT = L_BASE + E_GLOBAL + NPES
    L_MIN = L_TOT + E_GLOBAL
    LDS_INTS = L_MIN + E_GLOBAL
    # the count matrix in passes of NT entries (the last one partial unless NT divides it)
    COUNTS = NPES * E_GLOBAL
    COUNT_PASSES = -(-COUNTS // NT)
    assert E_GLOBAL <= NT and SCHED_EXPECTED <= NT and NPES * 9 <= NT

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_INTS * 4, 16]

    @flyc.kernel(
        name=f"a8w4_megamoe_prep_e{E_LOCAL}" + ("" if NPES == 8 else f"_n{NPES}"),
        known_block_size=[NT, 1, 1],
    )
    def kernel(
        topk_ids: fx.Int64,
        route_idx: fx.Int64,
        ws: fx.Int64,
        ws_tab: fx.Int64,
        num_valid: fx.Int64,
        status: fx.Int64,
        sched: fx.Int64,
        plan: fx.Int64,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
    ):
        lds_i32 = fx.Int32(
            fx.ptrtoint(fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr)
        )
        lds = lds_typed_ptr(lds_i32, T.i32)
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(64)
        epoch = global_typed_ptr(status, T.i32)[fx.Int32(STATUS_EPOCH)] + fx.Int32(1)
        sched_p = global_typed_ptr(sched, T.i32)
        plan_p = global_typed_ptr(plan, T.i32)
        ridx_p = global_typed_ptr(route_idx, T.i32)
        tab = global_typed_ptr(ws_tab, T.i64, align=8)
        blocks_cap = s_cap // fx.Int32(BLOCK_M)
        if tid < fx.Int32(E_GLOBAL):
            _store(lds, fx.Int32(L_HIST) + tid, fx.Int32(0))
        for b in range(tid, blocks_cap, NT):
            _sys_store_at(
                fx.Int64(ws) + fx.Int64((fx.Int32(ARRIVED) + b) * fx.Int32(4)),
                fx.Int32(0),
            )
        gpu.barrier()
        ids_rs = _rsrc(topk_ids, fx.Int64(num_routes) * fx.Int64(4))
        for r0 in range(fx.Int32(0), num_routes, ROUTE_BATCH * NT):
            gs = [
                fx.Int32(
                    _ld(
                        ids_rs,
                        r0 + fx.Int32(j * NT) + tid,
                        mask=r0 + fx.Int32(j * NT) + tid < num_routes,
                    )
                )
                for j in range(ROUTE_BATCH)
            ]
            for j in range_constexpr(ROUTE_BATCH):
                r = r0 + fx.Int32(j * NT) + tid
                if (r < num_routes) & (gs[j] >= fx.Int32(0)):
                    _store(
                        ridx_p, r, _latomic_add(lds_i32, fx.Int32(L_HIST) + gs[j], 1)
                    )
        gpu.barrier()
        for k in range_constexpr(COUNT_PASSES):
            i = tid + fx.Int32(k * NT)
            if i < fx.Int32(COUNTS):
                peer = i // fx.Int32(E_GLOBAL)
                e = i % fx.Int32(E_GLOBAL)
                _sys_store_at(
                    fx.Int64(tab[peer])
                    + fx.Int64((me * fx.Int32(E_GLOBAL) + e) * fx.Int32(4)),
                    lds[fx.Int32(L_HIST) + e],
                )
        # this thread's count row stores and arrived zeroing landed
        rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if tid < fx.Int32(NPES):
            _sys_store_at(
                fx.Int64(tab[tid]) + fx.Int64((fx.Int32(CFLAG) + me) * fx.Int32(4)),
                epoch,
            )
            _flag_wait(
                fx.Int64(ws) + fx.Int64((fx.Int32(CFLAG) + tid) * fx.Int32(4)), epoch
            )
        gpu.barrier()
        for k in range_constexpr(COUNT_PASSES):
            i = tid + fx.Int32(k * NT)
            if i < fx.Int32(COUNTS):
                c = _sys_load_at(fx.Int64(ws) + fx.Int64(i * fx.Int32(4)))
                _store(lds, fx.Int32(L_CNT) + i, c)
                _store(plan_p, fx.Int32(PLAN_COUNTS) + i, c)
        gpu.barrier()
        # per global expert: routes from all sources and the fewest from one source
        if tid < fx.Int32(E_GLOBAL):
            n = fx.Int32(0)
            fewest = fx.Int32(1 << 30)
            for s in range_constexpr(NPES):
                c = lds[fx.Int32(L_CNT + s * E_GLOBAL) + tid]
                n = n + c
                fewest = fx.min(fewest, c)
            _store(lds, fx.Int32(L_TOT) + tid, n)
            _store(lds, fx.Int32(L_MIN) + tid, fewest)
            _store(plan_p, fx.Int32(PLAN_TOTAL) + tid, n)
            _store(plan_p, fx.Int32(PLAN_MIN) + tid, fewest)
        gpu.barrier()
        # destination d's experts start on BLOCK_M-row boundaries in expert order:
        # an exclusive scan of the padded totals over the experts of a destination
        if const_expr(WAVE_SCAN):
            if tid < fx.Int32(E_GLOBAL):
                pad = (
                    (lds[fx.Int32(L_TOT) + tid] + fx.Int32(BLOCK_M - 1))
                    // fx.Int32(BLOCK_M)
                    * fx.Int32(BLOCK_M)
                )
                inc = _scan(pad, lane, E_LOCAL)
                _store(lds, fx.Int32(L_BASE) + tid, inc - pad)
                _store(plan_p, fx.Int32(PLAN_BASE) + tid, inc - pad)
                if tid % fx.Int32(E_LOCAL) == fx.Int32(E_LOCAL - 1):
                    _store(
                        lds, fx.Int32(L_BASE + E_GLOBAL) + tid // fx.Int32(E_LOCAL), inc
                    )
                    _store(
                        plan_p,
                        fx.Int32(PLAN_BASE + E_GLOBAL) + tid // fx.Int32(E_LOCAL),
                        inc,
                    )
        else:
            # a destination's experts span waves: one thread per destination
            if tid < fx.Int32(NPES):
                base = fx.Int32(0)
                for k in range_constexpr(E_LOCAL):
                    g = tid * fx.Int32(E_LOCAL) + fx.Int32(k)
                    _store(lds, fx.Int32(L_BASE) + g, base)
                    _store(plan_p, fx.Int32(PLAN_BASE) + g, base)
                    base = base + (
                        (lds[fx.Int32(L_TOT) + g] + fx.Int32(BLOCK_M - 1))
                        // fx.Int32(BLOCK_M)
                        * fx.Int32(BLOCK_M)
                    )
                _store(lds, fx.Int32(L_BASE + E_GLOBAL) + tid, base)
                _store(plan_p, fx.Int32(PLAN_BASE + E_GLOBAL) + tid, base)
        for b in range(tid, blocks_cap, NT):
            _store(sched_p, fx.Int32(SCHED_EXPECTED) + blocks_cap + b, fx.Int32(0))
        if tid < fx.Int32(SCHED_EXPECTED):
            _store(sched_p, tid, fx.Int32(0))
        gpu.barrier()
        # first block of each of destination d's 8 mega-kernel queues (its valid blocks
        # split in eight), then the block count
        if tid < fx.Int32(NPES * 9):
            d = tid // fx.Int32(9)
            q = tid % fx.Int32(9)
            rows = fx.min(lds[fx.Int32(L_BASE + E_GLOBAL) + d], s_cap)
            _store(
                plan_p,
                fx.Int32(PLAN_QSTART) + tid,
                rows // fx.Int32(BLOCK_M) * q // fx.Int32(8),
            )
        if tid == fx.Int32(0):
            rows = lds[fx.Int32(L_BASE + E_GLOBAL) + me]
            _store(global_typed_ptr(num_valid, T.i32), fx.Int32(0), fx.min(rows, s_cap))
            _store(global_typed_ptr(status, T.i32), fx.Int32(1), rows)

    @flyc.jit
    def launch(
        topk_ids: fx.Int64,
        route_idx: fx.Int64,
        ws: fx.Int64,
        ws_tab: fx.Int64,
        num_valid: fx.Int64,
        status: fx.Int64,
        sched: fx.Int64,
        plan: fx.Int64,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            topk_ids,
            route_idx,
            ws,
            ws_tab,
            num_valid,
            status,
            sched,
            plan,
            num_routes,
            me,
            s_cap,
        ).launch(grid=(1, 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =============================================================================
# routing plan (one workgroup per producer wave of the mega kernel)
# =============================================================================


def mega_plan_kernel(*, E_LOCAL, NPES):
    """send_route / send_info [T*TOPK]: workgroup j sorts routes [j S, (j + 1) S) (S =
    ceil(routes / grid)) by when the destination's mega-kernel queues need them (block
    position in the queue) into the same range; the producer wave j sends that range in
    order. info = destination row | global expert << ROW_BITS, -1 past s_cap and for a
    negative id (padding: sent never).
    expert_ids [s_cap / BLOCK_M] / sched expected rows per block: this rank's sorted
    blocks; down_ids (symmetric, local view): padding rows set to -1 (producers fill the
    valid ones). The rows of an expert interleave the sources: source s's i-th route sits
    at virtual position NPES i + s (compacted where sources run out). ``E_LOCAL``:
    experts per rank; ``NPES``: ranks."""
    NT = PLAN_THREADS
    PLAN_MIN, PLAN_BASE, PLAN_QSTART, PLAN_COUNTS, _ = plan_layout(E_LOCAL, NPES)
    E_GLOBAL = NPES * E_LOCAL
    ROW_BITS = row_bits(E_GLOBAL)
    L_KC = PLAN_COUNTS
    L_KN = L_KC + MAX_KEYS
    LDS_INTS = L_KN + MAX_KEYS

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_INTS * 4, 16]

    @flyc.kernel(
        name=f"a8w4_megamoe_plan_e{E_LOCAL}" + ("" if NPES == 8 else f"_n{NPES}"),
        known_block_size=[NT, 1, 1],
    )
    def kernel(
        topk_ids: fx.Int64,
        route_idx: fx.Int64,
        plan: fx.Int64,
        send_route: fx.Int64,
        send_info: fx.Int64,
        down_ids: fx.Int64,
        expert_ids: fx.Int64,
        sched: fx.Int64,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
    ):
        lds_i32 = fx.Int32(
            fx.ptrtoint(fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr)
        )
        lds = lds_typed_ptr(lds_i32, T.i32)
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        grid = fx.Int32(gpu.grid_dim.x)
        plan_p = global_typed_ptr(plan, T.i32)
        for k in range_constexpr(-(-PLAN_COUNTS // NT)):
            i = tid + fx.Int32(k * NT)
            if i < fx.Int32(PLAN_COUNTS):
                _store(lds, i, plan_p[i])
        if tid < fx.Int32(MAX_KEYS):
            _store(lds, fx.Int32(L_KC) + tid, fx.Int32(0))
        gpu.barrier()
        per = (num_routes + grid - fx.Int32(1)) // grid
        begin = bid * per
        end = fx.min(begin + per, num_routes)
        ids_rs = _rsrc(topk_ids, fx.Int64(num_routes) * fx.Int64(4))
        ridx_rs = _rsrc(route_idx, fx.Int64(num_routes) * fx.Int64(4))
        items = []
        for j in range_constexpr(SLICE_PER_THREAD):
            r = begin + fx.Int32(j * NT) + tid
            ok = r < end
            g = fx.Int32(_ld(ids_rs, r, mask=ok))
            valid = g >= fx.Int32(0)
            g = fx.max(g, fx.Int32(0))  # in-range table reads; the route gets info -1
            idx = fx.Int32(_ld(ridx_rs, r, mask=ok))
            pos = idx * fx.Int32(NPES) + me
            if ok & valid & (idx >= lds[fx.Int32(PLAN_MIN) + g]):  # some source ran out
                p = fx.Int32(0)
                for s in range_constexpr(NPES):
                    c = plan_p[fx.Int32(PLAN_COUNTS + s * E_GLOBAL) + g]
                    p = p + fx.min(
                        c, idx + (fx.Int32(s) < me).select(fx.Int32(1), fx.Int32(0))
                    )
                pos = p
            row = lds[fx.Int32(PLAN_BASE) + g] + pos
            kept = valid & (row < s_cap)
            info = kept.select(row | (g << fx.Int32(ROW_BITS)), fx.Int32(-1))
            # key: the block's position in its destination's queue
            qs = fx.Int32(PLAN_QSTART) + g // fx.Int32(E_LOCAL) * fx.Int32(9)
            block = row // fx.Int32(BLOCK_M)
            q = (block >= lds[qs + fx.Int32(4)]).select(fx.Int32(4), fx.Int32(0))
            q = q + (block >= lds[qs + q + fx.Int32(2)]).select(
                fx.Int32(2), fx.Int32(0)
            )
            q = q + (block >= lds[qs + q + fx.Int32(1)]).select(
                fx.Int32(1), fx.Int32(0)
            )
            key = kept.select(
                fx.min(block - lds[qs + q], fx.Int32(MAX_KEYS - 1)),
                fx.Int32(MAX_KEYS - 1),
            )
            rank = fx.Int32(0)
            if ok:
                rank = _latomic_add(lds_i32, fx.Int32(L_KC) + key, 1)
            items.append((r, ok, info, key, rank))
        gpu.barrier()
        if tid < fx.Int32(MAX_KEYS):  # wave 0: exclusive scan of the key counts
            c = lds[fx.Int32(L_KC) + tid]
            _store(lds, fx.Int32(L_KN) + tid, begin + _scan(c, tid, MAX_KEYS) - c)
        gpu.barrier()
        route_p = global_typed_ptr(send_route, T.i32)
        info_p = global_typed_ptr(send_info, T.i32)
        for r, ok, info, key, rank in items:
            if ok:
                at = lds[fx.Int32(L_KN) + key] + rank
                _store(route_p, at, r)
                _store(info_p, at, info)
        # this rank's blocks, one per workgroup at a time (a thread per row)
        blocks = fx.min(lds[fx.Int32(PLAN_BASE + E_GLOBAL) + me], s_cap) // fx.Int32(
            BLOCK_M
        )
        sched_p = global_typed_ptr(sched, T.i32)
        eids_p = global_typed_ptr(expert_ids, T.i32)
        down_p = global_typed_ptr(down_ids, T.i32)
        for b in range(bid, blocks, grid):
            row0 = b * fx.Int32(BLOCK_M)
            e = fx.Int32(0)
            for k in range_constexpr(1, E_LOCAL):
                e = e + (
                    lds[me * fx.Int32(E_LOCAL) + fx.Int32(PLAN_BASE + k)] <= row0
                ).select(fx.Int32(1), fx.Int32(0))
            g = me * fx.Int32(E_LOCAL) + e
            n = fx.min(
                lds[fx.Int32(PLAN_TOTAL) + g] - (row0 - lds[fx.Int32(PLAN_BASE) + g]),
                fx.Int32(BLOCK_M),
            )
            if tid == fx.Int32(0):
                _store(eids_p, b, e)
                _store(sched_p, fx.Int32(SCHED_EXPECTED) + b, n)
            if (tid < fx.Int32(BLOCK_M)) & (tid >= n):
                _store(down_p, row0 + tid, fx.Int32(-1))

    @flyc.jit
    def launch(
        topk_ids: fx.Int64,
        route_idx: fx.Int64,
        plan: fx.Int64,
        send_route: fx.Int64,
        send_info: fx.Int64,
        down_ids: fx.Int64,
        expert_ids: fx.Int64,
        sched: fx.Int64,
        num_routes: fx.Int32,
        me: fx.Int32,
        s_cap: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            topk_ids,
            route_idx,
            plan,
            send_route,
            send_info,
            down_ids,
            expert_ids,
            sched,
            num_routes,
            me,
            s_cap,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =============================================================================
# completion barrier
# =============================================================================


def mega_finish_kernel(*, E_LOCAL, NPES):
    """After the mega kernel (same stream): tell every rank this rank's remote adds are
    done, wait for all ranks, then advance the epoch (status[STATUS_EPOCH]).
    ``E_LOCAL``: experts per rank, ``NPES``: ranks (the workspace layout)."""
    _, _, FFLAG, _ = ws_layout(E_LOCAL, NPES)

    @flyc.kernel(
        name=f"a8w4_megamoe_finish_e{E_LOCAL}" + ("" if NPES == 8 else f"_n{NPES}"),
        known_block_size=[64, 1, 1],
    )
    def kernel(ws: fx.Int64, ws_tab: fx.Int64, status: fx.Int64, me: fx.Int32):
        tid = fx.Int32(gpu.thread_id("x"))
        status_p = global_typed_ptr(status, T.i32)
        epoch = status_p[fx.Int32(STATUS_EPOCH)] + fx.Int32(1)
        if tid < fx.Int32(NPES):
            tab = global_typed_ptr(ws_tab, T.i64, align=8)
            _sys_store_at(
                fx.Int64(tab[tid]) + fx.Int64((fx.Int32(FFLAG) + me) * fx.Int32(4)),
                epoch,
            )
            _flag_wait(
                fx.Int64(ws) + fx.Int64((fx.Int32(FFLAG) + tid) * fx.Int32(4)), epoch
            )
        if tid == fx.Int32(0):  # one wave: every lane's wait is over
            _store(status_p, fx.Int32(STATUS_EPOCH), epoch)

    @flyc.jit
    def launch(
        ws: fx.Int64,
        ws_tab: fx.Int64,
        status: fx.Int64,
        me: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(ws, ws_tab, status, me).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch
