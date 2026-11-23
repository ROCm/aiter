# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 EP8 MoE mega kernel: dispatch, GateUp and Down (with the combine) in one launch.

ep_prep and ep_plan (comm.py) run first and build the plan.
Workgroups take task tickets from per-XCD queues (persistent: one workgroup per CU taking tickets until the queues
are empty). A workgroup takes a ticket only when it is running and its current task (if any) waits for nothing more,
and a task only waits for lower tickets of its queue or for data that sources push without waiting, so the grid
needs no co-residency. Each queue holds:
  [0, producers)  P2P-write this rank's routes in the order the destinations' queues need them (ep_prep), then add
                  each route to arrived[its sorted block] on its destination.
  then            units of NG GateUp tiles of block u and ND Down tiles of block u - lag: Down, and the combine
                  traffic its epilogue sends, runs under the GateUp of later blocks instead of after all of it.
GateUp(b) waits until arrived[b] reaches the routes block b receives, writes a2 and counts done[b];
Down(b) waits until done[b] == NG and adds its rows into the source ranks' out with BF16 atomics.
Only GateUp(b, 0) waits for arrived[b]: it copies block b's A scales into the [K/128, rows] layout and sets
scaled[b], which the block's other GateUp tiles wait for.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ...tensor_shim import _run_compiled, current_stream_arg, ptr_arg
from ..host import launcher
from .comm import ARRIVED, E_LOCAL, NPES, ROW_BITS, SCHED_EXPECTED
from .down import down_task_emitter
from .gateup import (
    OOB_OFFSET,
    SORT_BLOCK_M,
    gateup_task_emitter,
)
from .util import (
    _load_i32,
    _rsrc_of,
    _uniform,
    atomic_add_i32,
    buffer_rsrc_at,
    load_i32_atomic,
    spin_wait_ge,
)

# Routes per producer wave between two signals.
CHUNK = 8
# sc0 nt sc1: written through to the destination, so a vmcnt wait covers visibility before the signal.
P2P_AUX = 19
# Task queues: one per XCD.
QUEUES = 8


def compile_ep_mega(
    hidden_dim, inter_dim, topk, token_shift, producers, lag, swiglu_limit=0.0, wq="fp8"
):
    """The mega kernel's launcher. lag counts units (block pairs) of one queue.

    One task queue per XCD (producers split evenly), so the tiles of a block share that XCD's L2; a workgroup whose
    queue is empty takes tasks from the next queues. Queue q holds an eighth of the blocks, block pairs' tasks
    alternating tile by tile so both read each B tile together. Workgroups loop over tickets (launch about one per
    CU) and take the next ticket during the current task's epilogue.
    GateUp writes a2 through to memory (sc1) and both waits invalidate only the L1: all data a task waits for is in
    local memory, whose L2 lines are kept coherent by probes, and nothing reads it before it is written within a
    launch. GateUp tile 0 of a block copies the block's A scales from the token-major recv_scale [rows, K/128] into
    pRecvScaleT [K/128, rows], so that a K block's scale load of 16 rows touches one cache line instead of 16.
    GateUp and Down skip the MFMAs of a block's 64-row packets past its valid rows (see the emitters).
    swiglu_limit > 0: the GateUp activation clamps gate and up (see gateup_task_emitter).
    wq: "fp8" weights, or "fp4": a8w4's prepare_weights (fd2) output, decoded to FP8 in the loaders.
    topk need not be a power of two: the producers then divide a route index by it instead of shifting.
    """
    H, I = hidden_dim, inter_dim
    # A closure value: FlyDSL's cache key holds closure scalars but not the module globals that nested functions read,
    # so with ROW_BITS itself a changed width would reuse a cached kernel built with the old one.
    row_bits = ROW_BITS
    KB = H // 128
    gateup = gateup_task_emitter(I, H, topk, E_LOCAL, float(swiglu_limit), wq)
    down = down_task_emitter(H, I, topk, E_LOCAL, token_shift, wq)
    # Closures must hold the emitters themselves: FlyDSL's cache key only recurses into callable closure values.
    emit_gateup, emit_down, SharedTiles = gateup.emit, down.emit, gateup.SharedTiles
    NG, ND = gateup.n_tiles, down.n_tiles
    unit = NG + ND
    topk_shift = topk.bit_length() - 1
    topk_pow2 = 1 << topk_shift == topk
    assert 0 < topk <= 8 and H % 1024 == 0 and H // 128 * 4 <= 16 * 64
    # The transpose: each thread copies half a row's scales with dwordx4 loads.
    assert KB % 8 == 0
    scale_lanes = H // 128 * 4 // 16
    queues = QUEUES
    assert producers % queues == 0
    queue_producers = producers // queues
    # Blocks of a queue go in pairs.
    pair = 2

    @flyc.kernel(known_block_size=[512, 1, 1])
    def mega_kernel(
        pRecvA: fx.Pointer,
        pW1: fx.Pointer,
        pA2: fx.Pointer,
        pA2Scale: fx.Pointer,
        pRecvScale: fx.Pointer,
        pW1Scale: fx.Pointer,
        pGuIds: fx.Pointer,
        pW2: fx.Pointer,
        pW2Scale: fx.Pointer,
        pDownIds: fx.Pointer,
        pRecvW: fx.Pointer,
        pOutTable: fx.Pointer,
        pOutScaleTable: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pSched: fx.Pointer,
        pWs: fx.Pointer,
        pX: fx.Pointer,
        pXScale: fx.Pointer,
        pTopkW: fx.Pointer,
        pSendRoute: fx.Pointer,
        pSendInfo: fx.Pointer,
        pPeerA: fx.Pointer,
        pPeerScale: fx.Pointer,
        pPeerIds: fx.Pointer,
        pPeerW: fx.Pointer,
        pPeerWs: fx.Pointer,
        pTrace: fx.Pointer,
        pRecvScaleT: fx.Pointer,
        num_routes: fx.Int32,
        num_tokens: fx.Int32,
        me: fx.Int32,
        num_blocks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = _uniform(tid // 64)
        zero = fx.Int32(0).ir_value()
        aux = ir.IntegerAttr.get(T.i32, P2P_AUX)
        lds = fx.SharedAllocator().allocate(SharedTiles).peek()
        sched = fx.Int64(fx.ptrtoint(pSched))
        rows_cap = num_blocks * SORT_BLOCK_M

        def hwreg(name):
            return fx.Int32(
                llvm.inline_asm(
                    T.i32,
                    [],
                    f"s_getreg_b32 $0, hwreg({name})",
                    "=s",
                    has_side_effects=True,
                )
            )

        def ticket_slot(index):
            return fx.make_view(lds.ab.ptr + index, fx.make_layout(1, 1))

        valid_rows = _uniform(_load_i32(_rsrc_of(pNumValid, 4), 0))
        blocks = (valid_rows + (SORT_BLOCK_M - 1)) // SORT_BLOCK_M
        blocks = (blocks < num_blocks).select(blocks, num_blocks)

        def queue_layout(queue):
            """(first block, block stride, blocks) of a queue."""
            first = blocks * queue // queues
            return first, 1, blocks * (queue + 1) // queues - first

        def queue_tasks(queue):
            return queue_producers + unit * pair * (
                (queue_layout(queue)[2] + (pair - 1)) // pair
            )

        def lane0_add(addr, value):
            """Atomic add by lane 0 only; the old value in lane 0. The address goes through an opaque VGPR zero: at a
            uniform address LLVM's atomic optimizer sums the lanes in a 64-step loop and reads the result back at
            once (to broadcast it), so a prefetched ticket could not be left in flight.
            """
            taken = fx.Int32(0)
            if lane == 0:
                lane_zero = fx.Int32(
                    llvm.inline_asm(
                        T.i32, [], "v_mov_b32 $0, 0", "=v", has_side_effects=True
                    )
                )
                taken = atomic_add_i32(fx.Int64(addr) + fx.Int64(lane_zero), value)
            return taken

        def take_ticket(after_task=False, prefetched=None):
            """(queue, ticket) of this workgroup's next task; ticket -1 when every queue is empty.

            prefetched: wave 0's ticket of the home queue taken during the previous task (possibly past its end).
            """
            if const_expr(after_task):
                # The previous task's last LDS reads are done before the slots are overwritten.
                rocdl.s_barrier()
            # Wave-uniform branches only around barriers (a lane-divergent one lets LLVM move the barrier).
            if wave == 0:
                # This XCD's queue first (its blocks then share the XCD's L2), the others once it is empty. Plain
                # loads find the queues that may still hold tasks: once all are handed out, the leftover workgroups
                # of the (over-sized) grid exit after one round of loads instead of contending atomics (an atomic
                # per leftover workgroup, even on the home queue only, slows the tasks still running).
                home = hwreg("HW_REG_XCC_ID") & (queues - 1)
                cand = home
                got = fx.Int32(-1)
                if const_expr(prefetched is not None):
                    early = _uniform(prefetched)
                    got = (early < queue_tasks(home)).select(early, fx.Int32(-1))
                if got < 0:
                    open_mask = fx.Int32(0)
                    for q in range_constexpr(queues):
                        seen = _uniform(
                            load_i32_atomic(sched + fx.Int64(q * 4), "agent-one-as")
                        )
                        open_mask = open_mask | (
                            seen < queue_tasks(fx.Int32(q))
                        ).select(fx.Int32(1 << q), fx.Int32(0))
                    tries = fx.Int32(0)
                    while (got < 0) & (tries < queues):
                        cand = (home + tries) & (queues - 1)
                        if ((open_mask >> cand) & 1) != 0:
                            taken = _uniform(lane0_add(sched + fx.Int64(cand * 4), 1))
                            got = (taken < queue_tasks(cand)).select(
                                taken, fx.Int32(-1)
                            )
                        tries = tries + 1
                ticket_slot(0).store(Vec.from_elements([cand], fx.Int32))
                ticket_slot(1).store(Vec.from_elements([got], fx.Int32))
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            queue = _uniform(Vec(ticket_slot(0).load())[0])
            ticket = _uniform(Vec(ticket_slot(1).load())[0])
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()
            return queue, ticket

        def produce(index):
            x_rsrc = _rsrc_of(pX, num_tokens * H)
            xs_rsrc = _rsrc_of(pXScale, num_tokens * (scale_lanes * 16))
            w_rsrc = _rsrc_of(pTopkW, num_routes * 4)
            route_rsrc = _rsrc_of(pSendRoute, num_routes * 4)
            info_rsrc = _rsrc_of(pSendInfo, num_routes * 4)
            oob = fx.Int32(OOB_OFFSET)
            scale_lane = lane < scale_lanes
            # Lane r < NPES holds the (lo, hi) halves of rank r's buffer bases.
            peer_lane = (lane & (NPES - 1)) * 8
            bases = []
            for table in (pPeerA, pPeerScale, pPeerIds, pPeerW, pPeerWs):
                table_rsrc = _rsrc_of(table, NPES * 8)
                bases.append(
                    (
                        _load_i32(table_rsrc, peer_lane),
                        _load_i32(table_rsrc, peer_lane + 4),
                    )
                )

            def peer_rsrc(which, rank, num_bytes):
                lo, hi = [
                    fx.Uint32(rocdl.readlane(T.i32, half.ir_value(), rank.ir_value()))
                    for half in bases[which]
                ]
                return buffer_rsrc_at((fx.Uint64(hi) << 32) | fx.Uint64(lo), num_bytes)

            def lane_value(values, j):
                return fx.Int32(
                    rocdl.readlane(T.i32, values.ir_value(), fx.Int32(j).ir_value())
                )

            def token_slot(route):
                """(token, slot) of a route index token * topk + slot (route >= 0)."""
                if const_expr(topk_pow2):
                    return route >> topk_shift, route & (topk - 1)
                token = fx.Int32(fx.Uint32(route) // topk)
                return token, route - token * topk

            # The send list is sorted by when the destinations' queues need the routes. Producer workgroup `index`
            # sends its chunks index, index + producers, ... of CHUNK routes, its waves taking the next one from a
            # counter in LDS word 2 (after the ticket slots): all producers send the earliest blocks first, and a
            # workgroup's waves finish together. (One global counter for all 128 producer waves measured the same, with
            # its atomics on the ticket counters' cache line. Sending faster, the previous chunk's stores retiring
            # under this chunk's loads, was measured slower overall: the GateUp tasks of the first blocks then run
            # against more memory and power contention.)
            chunks = (num_routes + (CHUNK - 1)) // CHUNK
            counter = llvm.inttoptr(
                ir.Type.parse("!llvm.ptr<3>"),
                fx.Int32(fx.ptrtoint(lds.ab.ptr + 2)).ir_value(),
            )
            if wave == 0:
                ticket_slot(2).store(Vec.from_elements([fx.Int32(0)], fx.Int32))
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_barrier()

            def claim():
                taken = fx.Int32(0)
                if lane == 0:
                    taken = fx.Int32(
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.add,
                            counter,
                            fx.Int32(1).ir_value(),
                            llvm.AtomicOrdering.monotonic,
                            syncscope="workgroup",
                            alignment=4,
                        ).result
                    )
                return taken

            chunk = index + producers * _uniform(claim())
            while chunk < chunks:
                position = chunk * CHUNK + lane
                claimed = claim()
                mine = (lane < CHUNK) & (position < num_routes)
                offset = mine.select(position * 4, oob)
                routes = _load_i32(route_rsrc, offset)
                infos = _load_i32(info_rsrc, offset)
                send = mine & (infos >= 0)
                sends = send.select(fx.Int32(1), fx.Int32(0))
                # All loads of the chunk before any store: a store may alias a later load for all LLVM knows.
                pending = []
                for j in range_constexpr(CHUNK):
                    route = lane_value(routes, j)
                    ok = lane_value(sends, j) != 0
                    token, slot = token_slot(route)
                    data = [
                        rocdl.raw_ptr_buffer_load(
                            T.i32x4,
                            x_rsrc,
                            ok.select(
                                lane * 16 + part * 1024 + token * H, oob
                            ).ir_value(),
                            zero,
                            zero,
                        )
                        for part in range_constexpr(H // 1024)
                    ]
                    scales = rocdl.raw_ptr_buffer_load(
                        T.i32x4,
                        xs_rsrc,
                        (ok & scale_lane)
                        .select(lane * 16 + token * (scale_lanes * 16), oob)
                        .ir_value(),
                        zero,
                        zero,
                    )
                    weight = _load_i32(w_rsrc, ok.select(route * 4, oob))
                    pending.append(
                        (
                            route,
                            lane_value(infos, j),
                            ok,
                            token,
                            slot,
                            data,
                            scales,
                            weight,
                        )
                    )
                for route, info, ok, token, slot, data, scales, weight in pending:
                    row = info & ((1 << row_bits) - 1)
                    dest = ok.select(info >> (row_bits + 5), fx.Int32(0))
                    a_dst = peer_rsrc(0, dest, rows_cap * H)
                    for part in range_constexpr(H // 1024):
                        rocdl.raw_ptr_buffer_store(
                            data[part],
                            a_dst,
                            ok.select(
                                lane * 16 + part * 1024 + row * H, oob
                            ).ir_value(),
                            zero,
                            aux=aux,
                        )
                    rocdl.raw_ptr_buffer_store(
                        scales,
                        peer_rsrc(1, dest, rows_cap * (scale_lanes * 16)),
                        (ok & scale_lane)
                        .select(lane * 16 + row * (scale_lanes * 16), oob)
                        .ir_value(),
                        zero,
                        aux=aux,
                    )
                    row_off = (ok & (lane == 0)).select(row * 4, oob).ir_value()
                    down_id = (me << token_shift | token) | (slot << 24)
                    rocdl.raw_ptr_buffer_store(
                        down_id.ir_value(),
                        peer_rsrc(2, dest, rows_cap * 4),
                        row_off,
                        zero,
                        aux=aux,
                    )
                    rocdl.raw_ptr_buffer_store(
                        weight.ir_value(),
                        peer_rsrc(3, dest, rows_cap * 4),
                        row_off,
                        zero,
                        aux=aux,
                    )
                rocdl.s_waitcnt(vmcnt=0)
                expert = infos >> row_bits
                source = ((expert >> 5) & (NPES - 1)) * 4
                ws_lo, ws_hi = [
                    fx.Uint32(
                        rocdl.ds_bpermute(T.i32, source.ir_value(), half.ir_value())
                    )
                    for half in bases[4]
                ]
                block_of_row = (infos & ((1 << row_bits) - 1)) // SORT_BLOCK_M
                arrived = fx.Int64(
                    (fx.Uint64(ws_hi) << 32) | fx.Uint64(ws_lo)
                ) + fx.Int64((ARRIVED + block_of_row) * 4)
                if send:
                    atomic_add_i32(arrived, 1, "one-as")
                chunk = index + producers * _uniform(claimed)

        def prefetch_hook():
            """Epilog hook taking the next ticket of the home queue while the epilogue runs: the task has no waits
            left, so holding a ticket cannot delay one it depends on. Returns (hook, holder of wave 0's ticket).
            """
            holder = []

            def hook():
                queue_addr = sched + fx.Int64(
                    (hwreg("HW_REG_XCC_ID") & (queues - 1)) * 4
                )
                taken = fx.Int32(-1)
                if wave == 0:
                    # Read back only at the next take_ticket: the epilogue's own loads retire it on the way.
                    taken = lane0_add(queue_addr, 1)
                holder.append(taken)

            return hook, holder

        def transpose_block_scales(block, expected, flag):
            """GateUp tile 0: block's A scales from recv_scale [rows, KB] into pRecvScaleT [KB, rows] (padding rows 0),
            then flag += 1. Thread t copies row t % 256, K blocks [KB/2 (t // 256), +KB/2).
            """
            # An opaque copy of rows_cap: with rows_cap itself, LLVM merges the products here and the GateUp's equal
            # ones into a VGPR phi after this branch, and wraps the GateUp's row-id and K-loop scale loads (their
            # descriptor and soffset) in waterfall loops.
            rows = fx.Int32(
                llvm.inline_asm(
                    T.i32,
                    [rows_cap.ir_value()],
                    "s_mov_b32 $0, $1",
                    "=s,s",
                    has_side_effects=True,
                )
            )
            src = _rsrc_of(pRecvScale, rows * (KB * 4))
            dst = _rsrc_of(pRecvScaleT, rows * (KB * 4))
            row = tid % SORT_BLOCK_M
            half = _uniform(tid // SORT_BLOCK_M)
            sorted_row = block * SORT_BLOCK_M + row
            src_offset = (row < expected).select(
                sorted_row * (KB * 4) + half * (KB * 2), fx.Int32(OOB_OFFSET)
            )
            parts = [
                Vec(
                    rocdl.raw_ptr_buffer_load(
                        T.i32x4, src, (src_offset + i * 16).ir_value(), zero, zero
                    )
                )
                for i in range_constexpr(KB // 8)
            ]
            for i in range_constexpr(KB // 2):
                rocdl.raw_ptr_buffer_store(
                    parts[i // 4][i % 4].ir_value(),
                    dst,
                    (sorted_row * 4).ir_value(),
                    ((half * (KB // 2) + i) * rows * 4).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 16),
                )
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            if wave == 0:
                lane0_add(flag, 1)

        def gateup_task(block, tile, expert):
            # Only this block's routes: about two sources' worth, not the whole expert.
            expected = _uniform(
                _load_i32(
                    _rsrc_of(pSched, (SCHED_EXPECTED + 2 * num_blocks) * 4),
                    (SCHED_EXPECTED + block) * 4,
                )
            )
            arrived = fx.Int64(fx.ptrtoint(pWs)) + fx.Int64((ARRIVED + block) * 4)
            scaled = sched + fx.Int64((SCHED_EXPECTED + 2 * num_blocks + block) * 4)
            if wave == 0:
                # scaled[b] is set after arrived[b] was complete.
                if tile == 0:
                    spin_wait_ge(arrived, expected, "one-as", True)
                else:
                    spin_wait_ge(scaled, 1, "agent-one-as", True)
            rocdl.s_barrier()
            if tile == 0:
                transpose_block_scales(block, expected, scaled)
            hook, holder = prefetch_hook()
            emit_gateup(
                lds,
                pRecvA,
                pW1,
                pA2,
                pA2Scale,
                pRecvScaleT,
                pW1Scale,
                pGuIds,
                rows_cap,
                num_blocks,
                True,
                lambda: (block, tile, expert),
                tid=tid,
                epilog_hook=hook,
                rows=expected,
            )
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_barrier()
            if wave == 0:
                lane0_add(
                    sched + fx.Int64((SCHED_EXPECTED + num_blocks + block) * 4), 1
                )
            return holder[0]

        def down_task(block, tile, expert):
            # issued before the wait, which hides its latency
            rows = _uniform(
                _load_i32(
                    _rsrc_of(pSched, (SCHED_EXPECTED + 2 * num_blocks) * 4),
                    (SCHED_EXPECTED + block) * 4,
                )
            )
            if wave == 0:
                spin_wait_ge(
                    sched + fx.Int64((SCHED_EXPECTED + num_blocks + block) * 4),
                    NG,
                    "agent-one-as",
                    True,
                )
            rocdl.s_barrier()
            hook, holder = prefetch_hook()
            emit_down(
                lds,
                pA2,
                pW2,
                pOutTable,
                pA2Scale,
                pW2Scale,
                pDownIds,
                pRecvW,
                NPES << token_shift,
                num_blocks,
                True,
                lambda: (block, tile, expert),
                tid=tid,
                epilog_hook=hook,
                rows=rows,
            )
            return holder[0]

        def run_task(queue, ticket):
            """Runs a task; returns the ticket prefetched during it (wave 0; -1 if none)."""
            next_ticket = fx.Int32(-1)
            if ticket < queue_producers:
                produce(queue * queue_producers + ticket)
            else:
                index = ticket - queue_producers
                first, stride, own = queue_layout(queue)
                # Groups of `pair` blocks; their GateUp (Down) tasks alternate between the blocks tile by tile, so
                # neighbouring blocks of one expert read each B tile at about the same time.
                groups = (own + (pair - 1)) // pair
                gate_tasks, down_tasks = pair * NG, pair * ND
                group_tasks = gate_tasks + down_tasks
                lead = (groups < lag).select(groups, fx.Int32(lag))
                # GateUp of the first `lead` groups; then units of GateUp(group u) and Down(group u - lead); then the
                # Down of the last `lead` groups.
                in_head = index < gate_tasks * lead
                mid = index - gate_tasks * lead
                unit_of = mid // group_tasks
                slot = mid - unit_of * group_tasks
                in_mid = mid < group_tasks * (groups - lead)
                tail = mid - group_tasks * (groups - lead)
                is_gateup = in_head | (in_mid & (slot < gate_tasks))
                group = in_head.select(
                    index // gate_tasks,
                    in_mid.select(
                        lead + unit_of - (slot >= gate_tasks).select(lead, fx.Int32(0)),
                        groups - lead + tail // down_tasks,
                    ),
                )
                sub = in_head.select(
                    index % gate_tasks,
                    in_mid.select(
                        (slot < gate_tasks).select(slot, slot - gate_tasks),
                        tail % down_tasks,
                    ),
                )
                local = group * pair + sub % pair
                tile = sub // pair
                block = first + local * stride
                # An odd last block leaves its pair partner's tasks empty.
                if local < own:
                    expert = _uniform(
                        _load_i32(_rsrc_of(pExpertIds, num_blocks * 4), block * 4)
                    )
                    if is_gateup:
                        next_ticket = gateup_task(block, tile, expert)
                    else:
                        next_ticket = down_task(block, tile, expert)
            return next_ticket

        kernel_pointers = (
            pRecvA,
            pW1,
            pA2,
            pA2Scale,
            pRecvScale,
            pW1Scale,
            pGuIds,
            pW2,
            pW2Scale,
            pDownIds,
            pRecvW,
            pOutTable,
            pOutScaleTable,
            pExpertIds,
            pNumValid,
            pSched,
            pWs,
            pX,
            pXScale,
            pTopkW,
            pSendRoute,
            pSendInfo,
            pPeerA,
            pPeerScale,
            pPeerIds,
            pPeerW,
            pPeerWs,
            pTrace,
            pRecvScaleT,
        )

        def reload_args():
            """Kernel arguments and thread id read again through values LLVM cannot see through."""
            ptr4 = ir.Type.parse("!llvm.ptr<4>")
            segment = llvm.call_intrinsic(
                ptr4, "llvm.amdgcn.kernarg.segment.ptr", [], [], []
            )
            base = fx.Int64(
                llvm.inline_asm(
                    T.i64,
                    [llvm.ptrtoint(T.i64, segment)],
                    "s_mov_b64 $0, $1",
                    "=s,s",
                    has_side_effects=True,
                )
            )

            def load(offset, ty, align):
                return llvm.LoadOp(
                    ty, llvm.inttoptr(ptr4, (base + offset).ir_value()), alignment=align
                ).result

            # Pointers are 8-byte kernel arguments, the i32 scalars follow them.
            values = [
                fx.inttoptr(p.type, fx.Int64(load(8 * i, T.i64, 8)))
                for i, p in enumerate(kernel_pointers)
            ]
            values += [
                fx.Int32(load(8 * len(kernel_pointers) + 4 * j, T.i32, 4))
                for j in range(4)
            ]
            thread = fx.Int32(fx.thread_idx.x).ir_value()
            values.append(
                fx.Int32(
                    llvm.inline_asm(
                        T.i32,
                        [thread],
                        "v_mov_b32 $0, $1",
                        "=v,v",
                        has_side_effects=True,
                    )
                )
            )
            return values

        def rebind(values):
            nonlocal pRecvA, pW1, pA2, pA2Scale, pRecvScale, pW1Scale, pGuIds, pW2, pW2Scale, pDownIds, pRecvW
            nonlocal pOutTable, pOutScaleTable, pExpertIds, pNumValid, pSched, pWs, pX, pXScale, pTopkW, pSendRoute
            nonlocal pSendInfo, pPeerA, pPeerScale, pPeerIds, pPeerW, pPeerWs, pTrace, pRecvScaleT, num_routes, num_tokens
            nonlocal me, num_blocks, tid, lane, wave, sched, rows_cap
            (
                pRecvA,
                pW1,
                pA2,
                pA2Scale,
                pRecvScale,
                pW1Scale,
                pGuIds,
                pW2,
                pW2Scale,
                pDownIds,
                pRecvW,
                pOutTable,
                pOutScaleTable,
                pExpertIds,
                pNumValid,
                pSched,
                pWs,
                pX,
                pXScale,
                pTopkW,
                pSendRoute,
                pSendInfo,
                pPeerA,
                pPeerScale,
                pPeerIds,
                pPeerW,
                pPeerWs,
                pTrace,
                pRecvScaleT,
                num_routes,
                num_tokens,
                me,
                num_blocks,
                tid,
            ) = values
            lane = tid % 64
            wave = _uniform(tid // 64)
            sched = fx.Int64(fx.ptrtoint(pSched))
            rows_cap = num_blocks * SORT_BLOCK_M

        # A workgroup takes its next ticket in the epilogue of its task, when that task waits for nothing more: a
        # task still only waits for tickets that running workgroups hold (or for producers, which never wait), and
        # the ticket's atomic latency hides under the epilogue.
        queue, ticket = take_ticket()
        while ticket >= 0:
            # Every task re-reads the arguments: with loop-invariant ones LLVM hoists all task kinds' address math
            # out of the loop and runs out of registers.
            rebind(reload_args())
            next_ticket = run_task(queue, ticket)
            queue, ticket = take_ticket(True, next_ticket)

    @flyc.jit
    def launch(
        pRecvA: fx.Pointer,
        pW1: fx.Pointer,
        pA2: fx.Pointer,
        pA2Scale: fx.Pointer,
        pRecvScale: fx.Pointer,
        pW1Scale: fx.Pointer,
        pGuIds: fx.Pointer,
        pW2: fx.Pointer,
        pW2Scale: fx.Pointer,
        pDownIds: fx.Pointer,
        pRecvW: fx.Pointer,
        pOutTable: fx.Pointer,
        pOutScaleTable: fx.Pointer,
        pExpertIds: fx.Pointer,
        pNumValid: fx.Pointer,
        pSched: fx.Pointer,
        pWs: fx.Pointer,
        pX: fx.Pointer,
        pXScale: fx.Pointer,
        pTopkW: fx.Pointer,
        pSendRoute: fx.Pointer,
        pSendInfo: fx.Pointer,
        pPeerA: fx.Pointer,
        pPeerScale: fx.Pointer,
        pPeerIds: fx.Pointer,
        pPeerW: fx.Pointer,
        pPeerWs: fx.Pointer,
        pTrace: fx.Pointer,
        pRecvScaleT: fx.Pointer,
        num_routes: fx.Int32,
        num_tokens: fx.Int32,
        me: fx.Int32,
        num_blocks: fx.Int32,
        num_workgroups: fx.Int32,
        stream: fx.Stream,
    ):
        mega_kernel(
            pRecvA,
            pW1,
            pA2,
            pA2Scale,
            pRecvScale,
            pW1Scale,
            pGuIds,
            pW2,
            pW2Scale,
            pDownIds,
            pRecvW,
            pOutTable,
            pOutScaleTable,
            pExpertIds,
            pNumValid,
            pSched,
            pWs,
            pX,
            pXScale,
            pTopkW,
            pSendRoute,
            pSendInfo,
            pPeerA,
            pPeerScale,
            pPeerIds,
            pPeerW,
            pPeerWs,
            pTrace,
            pRecvScaleT,
            num_routes,
            num_tokens,
            me,
            num_blocks,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(grid=(num_workgroups, 1, 1), block=(512, 1, 1), stream=stream)

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch


def mega_grid(op, producers):
    """Workgroups covering every task (each queue may add one empty pair partner)."""
    return producers + (op.hidden // 256 + op.inter // 128) * (
        op.s_cap // SORT_BLOCK_M + 8
    )


def ep_mega(op, x_fp8, x_scale, topk_w, producers, lag):
    """Run the mega kernel for an LdsMegaMoE `op` whose ep_prep and ep_plan ran for this call."""
    launch = launcher(
        compile_ep_mega,
        x_fp8.device.index,
        hidden_dim=op.hidden,
        inter_dim=op.inter,
        topk=op.topk,
        token_shift=op.token_shift,
        producers=producers,
        lag=lag,
        swiglu_limit=op.swiglu_limit,
        wq=op.wq,
    )
    # Persistent: about one workgroup per CU.
    grid = min(
        mega_grid(op, producers),
        torch.cuda.get_device_properties(x_fp8.device).multi_processor_count,
    )
    t = op.tables

    def ptr(tensor):
        return ptr_arg(tensor, fx.Int32)

    _run_compiled(
        launch,
        ptr(op.recv_a),
        ptr(op.w1),
        ptr(op.a2),
        ptr(op.a2_scale_t),
        ptr(op.recv_scale),
        ptr(op.w1_scale),
        ptr(op.gu_ids),
        ptr(op.w2),
        ptr(op.w2_scale),
        ptr(op.down_ids),
        ptr(op.recv_w),
        ptr(t["out"]),
        ptr(t["out"]),
        ptr(op.expert_ids),
        ptr(op.num_valid),
        ptr(op.sched),
        ptr(op.ws),
        ptr(x_fp8),
        ptr(x_scale),
        ptr(topk_w),
        ptr(op.send_route),
        ptr(op.send_info),
        ptr(t["recv_a"]),
        ptr(t["recv_scale"]),
        ptr(t["down_ids"]),
        ptr(t["recv_w"]),
        ptr(t["ws"]),
        ptr(op.sched),
        ptr(op.recv_scale_t),
        x_fp8.shape[0] * op.topk,
        x_fp8.shape[0],
        op.rank,
        op.s_cap // SORT_BLOCK_M,
        grid,
        current_stream_arg(),
    )
