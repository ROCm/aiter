# SPDX-License-Identifier: MIT
"""Device emitter for gfx1250 MegaMoE Stage1 TDM dispatch.

This is deliberately an emitter, not a kernel: the metadata producer and the
payload TDM execute inside the grouped GEMM1 kernel that calls it.

Payload TDM must walk the same warp partition as COUNT. The standalone
``ep_dispatch_tdm`` kernel documents why: ``tok_map`` is only guaranteed visible
to the warp that wrote it (workgroup barrier, not grid), and a grid-strided
reread of another block's slots sees the host -1 fill. Decoding -1 as
``dest_tok = -1`` issues a TDM store one token before the peer recv buffer and
faults the fabric.

Leader election is the standalone COUNT ``ballot`` + ``mbcnt_lo`` match-any.
``max_recv`` is the arena slot bound (``world_size * max_tokens_per_rank``),
not ``2 * cur_tokens``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.rocdl import ballot, mbcnt_lo, readfirstlane, readlane
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.communication_ops_utils import traced
from . import tdm_prims as TDM


# int64 descriptor slots, shared with mega_moe.py.
TOK_MAP = 0
DEST_CTR = 1
DISP_BARRIER = 2
TOTAL_RECV = 3
DISPATCH_STATE = 4
ARENA_HANDLE = 5
RANK = 6
OFF_TOK_OFF = 7
OFF_RECV_NUM = 8
OFF_TIS = 9
OFF_OUT_IDX = 10
OFF_OUT_WTS = 11
OFF_OUT_TOK = 12
OFF_PAYLOAD_READY = 13


@traced
def emit_stage1_dispatch_tdm(
    *,
    tok_map,
    dest_ctr,
    disp_barrier,
    total_recv,
    arena_handle,
    rank: int,
    off_tok_off: int,
    off_recv_num: int,
    off_tis: int,
    off_out_idx: int,
    off_out_wts: int,
    off_out_tok: int,
    off_payload_ready: int,
    send_wire,
    input_idx,
    input_wts,
    dispatch_id,
    is_dispatch,
    generation,
    expected,
    parity,
    metadata_gate_addr,
    payload_lds,
    control_lds,
    cur_tokens: int,
    dispatch_blocks: int,
    async_payload: bool,
    num_waves: int,
    topk: int,
    experts_per_rank: int,
    max_tokens: int,
    max_recv: int,
    wire_stride: int,
):
    """Emit metadata dispatch plus a tile-visible TDM payload producer."""
    tid = fx.Int32(fx.thread_idx.x)
    wave = readfirstlane(T.i32, tid // fx.Int32(32))
    lane = tid % fx.Int32(32)
    sentinel = fx.Int32(2 * max_recv)
    zero_i32 = arith.constant(0, type=T.i32)
    tok_map = fx.Int64(tok_map)
    dest_ctr = fx.Int64(dest_ctr)
    disp_barrier = fx.Int64(disp_barrier)
    total_recv = fx.Int64(total_recv)
    rank = fx.Int32(rank)
    win = cco.Window(fx.Int64(arena_handle))

    idx_rsrc = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(input_idx)))
    wt_rsrc = create_buffer_resource_from_addr(fx.Int64(fx.ptrtoint(input_wts)))
    send_wire_addr = fx.Int64(fx.ptrtoint(send_wire))
    map_rsrc = create_buffer_resource_from_addr(tok_map)
    peer_idx_rsrc = [
        create_buffer_resource_from_addr(
            fx.Int64(win.lsa_ptr(fx.Int32(p), off_out_idx))
        )
        for p in range(2)
    ]
    peer_wts_rsrc = [
        create_buffer_resource_from_addr(
            fx.Int64(win.lsa_ptr(fx.Int32(p), off_out_wts))
        )
        for p in range(2)
    ]
    peer_tis_rsrc = [
        create_buffer_resource_from_addr(
            fx.Int64(win.lsa_ptr(fx.Int32(p), off_tis))
        )
        for p in range(2)
    ]
    ctl = fx.Int64(control_lds)

    def s_n(peer):
        return ctl + fx.Int64(peer) * fx.Int64(4)

    def s_base(peer):
        return ctl + (fx.Int64(peer) + fx.Int64(2)) * fx.Int64(4)

    def s_run(peer):
        return ctl + (fx.Int64(peer) + fx.Int64(4)) * fx.Int64(4)

    global_warp = dispatch_id * fx.Int32(num_waves) + wave
    total_warps = fx.Int32(dispatch_blocks * num_waves)
    tile_pitch = (wire_stride + 127) // 128 * 128
    tile_i32 = fx.Int32(payload_lds) + wave * fx.Int32(tile_pitch)
    group = TDM.tdm_group1(wire_stride, 1, 1)

    if is_dispatch:
        if tid < fx.Int32(2):
            comm_ops.store_i32_lds(s_n(tid), fx.Int32(0))
            comm_ops.store_i32_lds(s_run(tid), fx.Int32(0))
        fx.barrier()

        # COUNT: one recv slot per (token, destination peer), deduplicating
        # multiple experts on the same peer exactly like ep_dispatch_tdm.
        for tok in range(global_warp, fx.Int32(cur_tokens), total_warps):
            slot = tok * fx.Int32(topk) + lane
            active = lane < fx.Int32(topk)
            safe_slot = arith.select(active, slot, tok * fx.Int32(topk))
            expert = buffer_load(idx_rsrc, safe_slot, vec_width=1, dtype=T.i32)
            weight = buffer_load(wt_rsrc, safe_slot, vec_width=1, dtype=T.f32)
            peer = expert // fx.Int32(experts_per_rank)
            valid = active & (expert >= fx.Int32(0)) & (peer < fx.Int32(2))
            # Same type as ``valid``, always false, then the per-peer match-any.
            keep = valid & (lane < fx.Int32(0))
            for p in range_constexpr(2):
                pred = valid & (peer == fx.Int32(p))
                mask = ballot(T.i32, pred)
                below = mbcnt_lo(T.i32, mask, zero_i32)
                keep = keep | (pred & (below == zero_i32))
            if active:
                # Scratch the resolved expert in tok_map. FINALIZE walks this
                # exact warp partition, so it can reuse the value instead of
                # issuing a second input-index load before overwriting the slot
                # with its final destination encoding.
                buffer_store(expert, map_rsrc, slot)
            if keep:
                comm_ops.atomic_add_lds(s_n(peer), fx.Int32(1))
        comm_ops.waitcnt_stores()
        fx.barrier()

        # RESERVE: one fabric atomic per (workgroup, peer). Besides reducing
        # traffic, this keeps lsa_ptr's peer operand in the same proven shape as
        # the standalone TDM dispatch instead of feeding it a divergent route.
        if tid < fx.Int32(2):
            count = comm_ops.load_i32_lds(s_n(tid))
            base = fx.Int32(0)
            if count > fx.Int32(0):
                base = comm_ops.atomic_add_system(
                    fx.Int64(win.lsa_ptr(tid, off_tok_off)), count
                )
                comm_ops.atomic_add_system(
                    dest_ctr + fx.Int64(tid) * fx.Int64(4), count
                )
            comm_ops.store_i32_lds(s_base(tid), base)
        fx.barrier()

        # FINALIZE: hand each leader a slot from its block-local reserved run,
        # broadcast that destination to all routes for the same token/peer, and
        # publish metadata plus tok_map.
        for tok in range(global_warp, fx.Int32(cur_tokens), total_warps):
            slot = tok * fx.Int32(topk) + lane
            active = lane < fx.Int32(topk)
            safe_slot = arith.select(active, slot, tok * fx.Int32(topk))
            expert = buffer_load(map_rsrc, safe_slot, vec_width=1, dtype=T.i32)
            weight = buffer_load(wt_rsrc, safe_slot, vec_width=1, dtype=T.f32)
            peer = expert // fx.Int32(experts_per_rank)
            valid = active & (expert >= fx.Int32(0)) & (peer < fx.Int32(2))
            keep = valid & (lane < fx.Int32(0))
            for p in range_constexpr(2):
                pred = valid & (peer == fx.Int32(p))
                mask = ballot(T.i32, pred)
                below = mbcnt_lo(T.i32, mask, zero_i32)
                keep = keep | (pred & (below == zero_i32))
            local_slot = fx.Int32(0)
            if keep:
                local_slot = comm_ops.atomic_add_lds(s_run(peer), fx.Int32(1))
            dest_tok = fx.Int32(-1)
            if keep:
                dest_tok = comm_ops.load_i32_lds(s_base(peer)) + local_slot
            pub = keep & (dest_tok >= fx.Int32(0)) & (
                dest_tok < fx.Int32(max_recv)
            )
            flat = arith.select(
                pub, peer * fx.Int32(max_recv) + dest_tok, sentinel
            )
            if active:
                buffer_store(flat, map_rsrc, slot)

            for p in range_constexpr(2):
                found_p = valid & (lane < fx.Int32(0))
                dst_p = fx.Int32(0)
                for l in range_constexpr(topk):
                    keep_l = readlane(
                        T.i32,
                        arith.select(keep, fx.Int32(1), fx.Int32(0)),
                        l,
                    )
                    peer_l = readlane(T.i32, peer, l)
                    dst_l = readlane(T.i32, dest_tok, l)
                    hit = (keep_l != fx.Int32(0)) & (peer_l == fx.Int32(p))
                    dst_p = arith.select(hit, dst_l, dst_p)
                    found_p = found_p | hit
                meta_pub = active & found_p & (dst_p >= fx.Int32(0)) & (
                    dst_p < fx.Int32(max_recv)
                )
                if meta_pub:
                    buffer_store(
                        expert,
                        peer_idx_rsrc[p],
                        dst_p * fx.Int32(topk) + lane,
                    )
                    buffer_store(
                        arith.bitcast(T.i32, weight),
                        peer_wts_rsrc[p],
                        dst_p * fx.Int32(topk) + lane,
                    )
                if meta_pub & (lane == fx.Int32(0)):
                    buffer_store(
                        rank * fx.Int32(max_tokens) + tok,
                        peer_tis_rsrc[p],
                        dst_p,
                    )

        # Match standalone SIGNAL ordering: total_recv must be reset before the
        # grid rendezvous.  Resetting it in the signalling wave after the
        # rendezvous races the per-source atomic adds below and can erase the
        # count the planner consumes.
        if (dispatch_id == fx.Int32(0)) & (tid == fx.Int32(0)):
            buffer_store(
                fx.Int32(0),
                create_buffer_resource_from_addr(total_recv),
                0,
            )
        comm_ops.waitcnt_stores()
        comm_ops.fence_system_release()
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.atomic_add_system(disp_barrier, fx.Int32(1))

        if (dispatch_id == fx.Int32(0)) & (wave == fx.Int32(0)):
            if lane == fx.Int32(0):
                if const_expr(async_payload):
                    comm_ops.spin_until_eq_i32(
                        disp_barrier, fx.Int32(dispatch_blocks)
                    )
                    comm_ops.store_i32_system(
                        disp_barrier, fx.Int32(0), fx.Int32(0)
                    )
                else:
                    # Payload producers may advance the counter immediately;
                    # use a threshold rather than an exact phase value.
                    comm_ops.spin_until_gt_i32(
                        disp_barrier, fx.Int32(dispatch_blocks - 1)
                    )
            for peer in range(lane, fx.Int32(2), fx.Int32(32)):
                remote_recv = fx.Int64(win.lsa_ptr(peer, off_recv_num)) + fx.Int64(
                    rank
                ) * fx.Int64(4)
                comm_ops.spin_until_eq_i32(remote_recv, fx.Int32(0))
                count = buffer_load(
                    create_buffer_resource_from_addr(dest_ctr),
                    peer,
                    vec_width=1,
                    dtype=T.i32,
                )
                comm_ops.store_i32_system(
                    remote_recv, fx.Int32(0), count + fx.Int32(1)
                )
            for peer in range(lane, fx.Int32(2), fx.Int32(32)):
                local_recv = fx.Int64(win.lsa_ptr(rank, off_recv_num)) + fx.Int64(
                    peer
                ) * fx.Int64(4)
                count = comm_ops.spin_until_gt_i32(local_recv, fx.Int32(0)) - fx.Int32(
                    1
                )
                comm_ops.store_i32_system(local_recv, fx.Int32(0), fx.Int32(0))
                comm_ops.atomic_add_system(total_recv, count)
                buffer_store(
                    fx.Int32(0),
                    create_buffer_resource_from_addr(dest_ctr),
                    peer,
                )
            if lane == fx.Int32(0):
                comm_ops.store_i32_system(
                    fx.Int64(win.lsa_ptr(rank, off_tok_off)), fx.Int32(0), fx.Int32(0)
                )
                if const_expr(async_payload):
                    comm_ops.fence_system_release()
                    comm_ops.store_i32_system(
                        metadata_gate_addr, fx.Int32(0), fx.Int32(expected)
                    )

        # Same warp partition as COUNT: this warp only TDM-sends tokens it routed.
        for tok in range(global_warp, fx.Int32(cur_tokens), total_warps):
            probe_lane = arith.select(lane < fx.Int32(topk), lane, fx.Int32(0))
            flat = buffer_load(
                map_rsrc,
                tok * fx.Int32(topk) + probe_lane,
                vec_width=1,
                dtype=T.i32,
            )
            live = (
                (lane < fx.Int32(topk)) & (flat >= fx.Int32(0)) & (flat < sentinel)
            )
            if ballot(T.i32, live) != fx.Int32(0):
                TDM.tdm_load(
                    TDM.tdm_group0(
                        tile_i32,
                        send_wire_addr + fx.Int64(tok) * fx.Int64(wire_stride),
                    ),
                    group,
                )
                TDM.tdm_wait(0)
                live_i = arith.select(live, fx.Int32(1), fx.Int32(0))
                for l in range_constexpr(topk):
                    live_l = readlane(T.i32, live_i, l)
                    flat_l = readlane(T.i32, flat, l)
                    if live_l != fx.Int32(0):
                        peer = flat_l // fx.Int32(max_recv)
                        dst = flat_l - peer * fx.Int32(max_recv)
                        in_bounds = (
                            (peer >= fx.Int32(0))
                            & (peer < fx.Int32(2))
                            & (dst >= fx.Int32(0))
                            & (dst < fx.Int32(max_recv))
                        )
                        if in_bounds:
                            TDM.tdm_store(
                                TDM.tdm_group0(
                                    tile_i32,
                                    fx.Int64(win.lsa_ptr(peer, off_out_tok))
                                    + fx.Int64(dst) * fx.Int64(wire_stride),
                                ),
                                group,
                            )
                TDM.tdm_wait(0)
                comm_ops.fence_system_release()
                if lane < fx.Int32(topk):
                    if live:
                        peer = flat // fx.Int32(max_recv)
                        dst = flat - peer * fx.Int32(max_recv)
                        in_bounds = (
                            (peer >= fx.Int32(0))
                            & (peer < fx.Int32(2))
                            & (dst >= fx.Int32(0))
                            & (dst < fx.Int32(max_recv))
                        )
                        if in_bounds:
                            comm_ops.store_i32_system(
                                fx.Int64(win.lsa_ptr(peer, off_payload_ready))
                                + fx.Int64(parity * fx.Int32(max_recv) + dst)
                                * fx.Int64(4),
                                fx.Int32(0),
                                fx.Int32(expected),
                            )

        if const_expr(not async_payload):
            comm_ops.waitcnt_stores()
            comm_ops.fence_system_release()
            fx.barrier()
            if tid == fx.Int32(0):
                comm_ops.atomic_add_system(disp_barrier, fx.Int32(1))
            if (dispatch_id == fx.Int32(0)) & (wave == fx.Int32(0)):
                if lane == fx.Int32(0):
                    comm_ops.spin_until_gt_i32(
                        disp_barrier, fx.Int32(dispatch_blocks * 2 - 1)
                    )
                    comm_ops.store_i32_system(
                        disp_barrier, fx.Int32(0), fx.Int32(0)
                    )
                    comm_ops.fence_system_release()
                    comm_ops.store_i32_system(
                        metadata_gate_addr, fx.Int32(0), fx.Int32(expected)
                    )

    return (
        fx.Int64(win.lsa_ptr(rank, off_out_tok)),
        fx.Int64(win.lsa_ptr(rank, off_payload_ready)),
        fx.Int64(win.lsa_ptr(rank, off_out_idx)),
        fx.Int64(win.lsa_ptr(rank, off_out_wts)),
        rank,
    )


def make_stage1_dispatch_tdm_kernel(
    *,
    dispatch_blocks: int,
    num_waves: int,
    topk: int,
    experts_per_rank: int,
    max_tokens: int,
    max_recv: int,
    wire_stride: int,
):
    """Compile the Stage1 emitter as a standalone overlap producer."""
    block_threads = int(num_waves) * 32
    tile_pitch = (int(wire_stride) + 127) // 128 * 128

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def stage1_dispatch(
        descriptor: Int64,
        send_wire: Int64,
        input_idx: Int64,
        input_wts: Int64,
        cur_tokens: Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        smem = fx.SharedAllocator(static=False)
        ticket_lds = fx.Int64(fx.ptrtoint(smem.allocate(8, 8)._ptr))
        payload_lds = smem.allocate(num_waves * tile_pitch, 128)._ptr
        control_lds = smem.allocate(128, 128)._ptr

        desc = create_buffer_resource_from_addr(descriptor)

        def load_ptr(slot):
            return fx.Int64(
                buffer_load(desc, fx.Int32(slot), vec_width=1, dtype=T.i64)
            )

        state = load_ptr(DISPATCH_STATE)
        if tid == fx.Int32(0):
            ticket = fx.Int64(comm_ops.atomic_add_agent(state, fx.Int64(1)))
            generation = ticket // fx.Int64(dispatch_blocks)
            dispatch_id = fx.Int32(
                ticket - generation * fx.Int64(dispatch_blocks)
            )
            comm_ops.store_i32_lds(ticket_lds, dispatch_id)
            comm_ops.store_i32_lds(ticket_lds + fx.Int64(4), fx.Int32(generation))
        fx.barrier()
        dispatch_id = readfirstlane(T.i32, comm_ops.load_i32_lds(ticket_lds))
        generation = readfirstlane(
            T.i32, comm_ops.load_i32_lds(ticket_lds + fx.Int64(4))
        )
        parity = generation & fx.Int32(1)
        expected = (generation // fx.Int32(2)) + fx.Int32(1)
        if (dispatch_id == fx.Int32(0)) & (tid == fx.Int32(0)):
            comm_ops.store_i32_system(
                state + fx.Int64(8) + fx.Int64(parity) * fx.Int64(4),
                fx.Int32(0),
                expected,
            )

        emit_stage1_dispatch_tdm(
            tok_map=load_ptr(TOK_MAP),
            dest_ctr=load_ptr(DEST_CTR),
            disp_barrier=load_ptr(DISP_BARRIER),
            total_recv=load_ptr(TOTAL_RECV),
            arena_handle=load_ptr(ARENA_HANDLE),
            rank=fx.Int32(load_ptr(RANK)),
            off_tok_off=fx.Int32(load_ptr(OFF_TOK_OFF)),
            off_recv_num=fx.Int32(load_ptr(OFF_RECV_NUM)),
            off_tis=fx.Int32(load_ptr(OFF_TIS)),
            off_out_idx=fx.Int32(load_ptr(OFF_OUT_IDX)),
            off_out_wts=fx.Int32(load_ptr(OFF_OUT_WTS)),
            off_out_tok=fx.Int32(load_ptr(OFF_OUT_TOK)),
            off_payload_ready=fx.Int32(load_ptr(OFF_PAYLOAD_READY)),
            send_wire=send_wire,
            input_idx=input_idx,
            input_wts=input_wts,
            dispatch_id=dispatch_id,
            is_dispatch=True,
            generation=generation,
            expected=expected,
            parity=parity,
            metadata_gate_addr=state
            + fx.Int64(24)
            + fx.Int64(parity) * fx.Int64(4),
            payload_lds=arith.index_cast(
                T.i32, fx.index_cast(T.index, fx.ptrtoint(payload_lds))
            ),
            control_lds=fx.Int64(fx.ptrtoint(control_lds)),
            cur_tokens=cur_tokens,
            dispatch_blocks=dispatch_blocks,
            async_payload=True,
            num_waves=num_waves,
            topk=topk,
            experts_per_rank=experts_per_rank,
            max_tokens=max_tokens,
            max_recv=max_recv,
            wire_stride=wire_stride,
        )

    @flyc.jit
    def run(
        descriptor: Int64,
        send_wire: Int64,
        input_idx: Int64,
        input_wts: Int64,
        cur_tokens: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        stage1_dispatch(
            descriptor, send_wire, input_idx, input_wts, cur_tokens
        ).launch(
            grid=(dispatch_blocks, 1, 1),
            block=[block_threads, 1, 1],
            stream=stream,
        )

    return run
