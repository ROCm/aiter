# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Send-side compact route plan for gfx1250 TDM dispatch.

A low-LDS persistent-enough grid counts every local route into
``(dest_rank, local_expert)`` buckets, allgathers the histogram, then writes
the destination compact row into ``tok_map``. Dispatch copies payload straight
onto that row, so the receiver never runs ``moe_route_g2l_lds``.

This is the gfx1250 counterpart of ``mega_moe_prepare`` + ``emit_dispatch_group``:
the plan is a separate launch whose LDS is a few kilobytes of counters, not the
320 KiB TDM payload tile, so a software grid barrier here does not stall
max-LDS dispatch CTAs.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.rocdl import readlane
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from .config import _LANE_MASK as LANE_MASK
from .config import _LOG2_WAVE_SIZE as LOG2_WAVE
from .config import _WAVE_SIZE as WAVE
from . import tdm_prims as TDM

PLAN_BLOCKS = 1
PLAN_WAVES = 32
PLAN_THREADS = PLAN_WAVES * WAVE


@flyc.jit
def _wave32_inclusive_scan_i32(value, lane):
    """Inclusive sum within one gfx1250 wave32."""
    value_raw = value.ir_value()
    zero_raw = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.rocdl.update_dpp(
            T.i32, zero_raw, value_raw, dpp, 0xF, 0xF, True
        )
        value = (lane >= fx.Int32(shift)).select(value + fx.Int32(remote), value)
        value_raw = value.ir_value()
    source16 = (lane & fx.Int32(0x10)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(
        T.i32, source16 * fx.Int32(4), value
    )
    return (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)


def compact_row_capacity(
    *,
    max_recv: int,
    topk: int,
    experts_per_rank: int,
    tile_m: int,
) -> int:
    """Static CUDAGraph-safe bound matching grouped_moe contiguous_m."""
    tile_m = int(tile_m)
    ub = (
        int(max_recv) * int(topk)
        + int(experts_per_rank) * tile_m
        - int(topk)
    )
    aligned = ((ub + tile_m - 1) // tile_m) * tile_m
    return max(tile_m, aligned)


@functools.cache
def compile_tdm_compact_plan(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    topk: int,
    tile_m: int,
    compact_cap: int,
    off_hist: int,
    off_done: int,
    hist_stride: int,
):
    """Compile the compact-plan kernel. ``hist_stride`` is ``npes * segs`` (one parity)."""
    if WAVE != 32:
        raise ValueError("compact plan requires gfx1250 wave32")
    epr = int(experts_per_rank)
    segs = int(npes) * epr
    if segs > 1024:
        raise ValueError(f"compact plan LDS hist supports at most 1024 segments, got {segs}")
    tile_m = int(tile_m)
    compact_cap = int(compact_cap)
    peer_bits = max(1, (int(npes) - 1).bit_length())
    peer_mask = (1 << peer_bits) - 1
    if compact_cap >= (1 << (31 - peer_bits)):
        raise ValueError(
            "compact row encoding exceeds positive i32: "
            f"cap={compact_cap} peers={npes}"
        )
    hist_stride = int(hist_stride)
    dropped = -1

    @flyc.kernel(name="tdm_compact_plan", known_block_size=[PLAN_THREADS, 1, 1])
    def kernel(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        window = cco.Window(arena)

        rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_bhist = create_buffer_resource_from_addr(addr_block_hist)
        rsrc_base = create_buffer_resource_from_addr(addr_send_base)
        rsrc_mm = create_buffer_resource_from_addr(addr_masked_m)
        rsrc_psum = create_buffer_resource_from_addr(addr_psum)
        rsrc_bar = create_buffer_resource_from_addr(addr_barrier)

        smem = fx.SharedAllocator(static=False)
        hist_ptr = smem.allocate(segs * 4, 128)._ptr
        total_ptr = smem.allocate(segs * 4, 16)._ptr
        pref_ptr = smem.allocate(segs * 4, 16)._ptr
        matrix_ptr = smem.allocate(npes * segs * 4, 128)._ptr
        lds_hist = fx.Int64(fx.ptrtoint(hist_ptr))
        lds_total = fx.Int64(fx.ptrtoint(total_ptr))
        lds_pref = fx.Int64(fx.ptrtoint(pref_ptr))
        lds_matrix = fx.Int64(fx.ptrtoint(matrix_ptr))

        for s in range(tid, segs, PLAN_THREADS):
            comm_ops.store_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4), arith.constant(0))
        fx.barrier()

        n_routes = inp_cur_tok * fx.Int32(topk)
        for route in range(bid * PLAN_THREADS + tid, n_routes, PLAN_BLOCKS * PLAN_THREADS):
            expert = buffer_load(rsrc_idx, route, vec_width=1, dtype=T.i32)
            dest_pe = expert // epr
            valid = (expert >= 0) & (dest_pe >= 0) & (dest_pe < npes)
            local_e = expert - dest_pe * fx.Int32(epr)
            segment = dest_pe * fx.Int32(epr) + local_e
            intra = arith.constant(0)
            if valid:
                intra = comm_ops.atomic_add_lds(
                    lds_hist + fx.Int64(segment) * fx.Int64(4), arith.constant(1)
                )
            # Pack (segment, intra) into tok_map for the fill pass. Dropped
            # routes keep the sentinel so dispatch never names them.
            packed = arith.select(
                valid,
                segment | (intra << arith.constant(16)),
                arith.constant(dropped),
            )
            buffer_store(packed, rsrc_map, route)

        fx.barrier()
        if const_expr(PLAN_BLOCKS > 1):
            for s in range(tid, segs, PLAN_THREADS):
                cnt = comm_ops.load_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4))
                buffer_store(cnt, rsrc_bhist, bid * segs + s)
            comm_ops.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
                next_gen = gen + arith.constant(1)
                arrive = comm_ops.atomic_add_system(addr_barrier, arith.constant(1))
                if arrive != PLAN_BLOCKS - 1:
                    comm_ops.spin_until_eq_i32(addr_barrier + fx.Int64(4), next_gen)
                    comm_ops.fence_agent_acquire()
                else:
                    comm_ops.fence_system_release()
                    buffer_store(arith.constant(0), rsrc_bar, 0)
                    buffer_store(next_gen, rsrc_bar, 2)
                    comm_ops.fence_agent_release()
                    buffer_store(next_gen, rsrc_bar, 1)
            fx.barrier()

        if bid == 0:
            if const_expr(PLAN_BLOCKS > 1):
                for s in range(tid, segs, PLAN_THREADS):
                    total = arith.constant(0)
                    for blk in range_constexpr(PLAN_BLOCKS):
                        cnt = buffer_load(
                            rsrc_bhist, blk * segs + s, vec_width=1, dtype=T.i32
                        )
                        buffer_store(total, rsrc_bhist, blk * segs + s)
                        total = total + cnt
                    buffer_store(total, rsrc_base, s)
                comm_ops.waitcnt_stores()
                fx.barrier()

            if tid == 0:
                gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32) + arith.constant(
                    1
                )
                buffer_store(gen, rsrc_bar, 2)
            fx.barrier()
            gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
            parity = gen & arith.constant(1)
            hist_off = off_hist + parity * hist_stride * 4
            done_off = off_done + parity * npes * 4
            # One wave publishes the dense source histogram to each peer.
            # Typical EP4 is 384 dwords = 12 whole TDM rows.
            if const_expr(PLAN_BLOCKS == 1 and segs % 32 == 0):
                if warp < npes:
                    peer_hist = fx.Int64(
                        window.lsa_ptr(warp, hist_off)
                    ) + fx.Int64(rank * segs * 4)
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_hist)), peer_hist
                        ),
                        TDM.tdm_group1(32, segs // 32, 4),
                    )
                fx.barrier()
                TDM.tdm_wait(0)
            else:
                hist_vec = 2 if segs % 2 == 0 else 1
                for peer in range_constexpr(npes):
                    peer_hist = fx.Int64(window.lsa_ptr(peer, hist_off)) + fx.Int64(
                        rank * segs * 4
                    )
                    peer_hist_rsrc = create_buffer_resource_from_addr(peer_hist)
                    for s in range(
                        tid * hist_vec,
                        segs,
                        PLAN_THREADS * hist_vec,
                    ):
                        vals = [
                            comm_ops.load_i32_lds(
                                lds_hist + fx.Int64(s + i) * fx.Int64(4)
                            )
                            for i in range_constexpr(hist_vec)
                        ]
                        buffer_store(
                            fx.Vector.from_elements(vals, dtype=fx.Int32),
                            peer_hist_rsrc,
                            s,
                        )
                comm_ops.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                comm_ops.fence_system_release()
            fx.barrier()
            if tid < npes:
                comm_ops.store_i32_system(
                    fx.Int64(window.lsa_ptr(tid, done_off)),
                    fx.Int32(rank),
                    gen,
                )
            comm_ops.waitcnt_stores()
            fx.barrier()
            if tid < npes:
                comm_ops.wait_i32_until_equals(
                    fx.Int64(window.lsa_ptr(rank, done_off))
                    + fx.Int64(tid) * fx.Int64(4),
                    gen,
                )
            if tid == 0:
                comm_ops.fence_system_acquire()
            fx.barrier()

            matrix_n = npes * segs
            if const_expr(matrix_n % 32 == 0):
                TDM.tdm_load(
                    TDM.tdm_group0(
                        arith.trunci(T.i32, arith.unwrap(lds_matrix)),
                        fx.Int64(window.lsa_ptr(my_lsa_rank, hist_off)),
                    ),
                    TDM.tdm_group1(32, matrix_n // 32, 4),
                )
                TDM.tdm_wait(0)
            else:
                local_hist_rsrc = create_buffer_resource_from_addr(
                    fx.Int64(window.lsa_ptr(my_lsa_rank, hist_off))
                )
                for s in range(tid, matrix_n, PLAN_THREADS):
                    comm_ops.store_i32_lds(
                        lds_matrix + fx.Int64(s) * fx.Int64(4),
                        buffer_load(local_hist_rsrc, s, vec_width=1, dtype=T.i32),
                    )
            fx.barrier()
            for idx in range(tid, segs, PLAN_THREADS):
                dest = idx // fx.Int32(epr)
                e = idx - dest * fx.Int32(epr)
                total = arith.constant(0)
                my_prefix = arith.constant(0)
                for src in range_constexpr(npes):
                    cnt = comm_ops.load_i32_lds(
                        lds_matrix
                        + fx.Int64(src * segs + dest * fx.Int32(epr) + e)
                        * fx.Int64(4)
                    )
                    if src == rank:
                        my_prefix = total
                    total = total + cnt
                comm_ops.store_i32_lds(lds_total + fx.Int64(idx) * fx.Int64(4), total)
                comm_ops.store_i32_lds(lds_pref + fx.Int64(idx) * fx.Int64(4), my_prefix)
                if dest == rank:
                    buffer_store(total, rsrc_mm, e)
            fx.barrier()
            if warp < npes:
                dest = warp
                carry = arith.constant(0)
                for chunk in range(0, epr, WAVE):
                    e = fx.Int32(chunk) + lane
                    in_expert = e < fx.Int32(epr)
                    safe_e = in_expert.select(e, fx.Int32(0))
                    idx = dest * fx.Int32(epr) + safe_e
                    total = arith.constant(0)
                    my_prefix = arith.constant(0)
                    if in_expert:
                        total = comm_ops.load_i32_lds(
                            lds_total + fx.Int64(idx) * fx.Int64(4)
                        )
                        my_prefix = comm_ops.load_i32_lds(
                            lds_pref + fx.Int64(idx) * fx.Int64(4)
                        )
                    aligned = (total + fx.Int32(tile_m - 1)) // fx.Int32(tile_m)
                    aligned = aligned * fx.Int32(tile_m)
                    inclusive = _wave32_inclusive_scan_i32(aligned, lane)
                    expert_start = carry + inclusive - aligned
                    if in_expert:
                        send = expert_start + my_prefix
                        if dest == rank:
                            buffer_store(expert_start + total, rsrc_psum, e)
                        buffer_store(send, rsrc_base, idx)
                        comm_ops.store_i32_lds(
                            lds_hist + fx.Int64(idx) * fx.Int64(4), send
                        )
                    carry = carry + readlane(T.i32, inclusive, WAVE - 1)
            comm_ops.waitcnt_stores()
            fx.barrier()

        if const_expr(PLAN_BLOCKS > 1):
            if tid == 0:
                gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32)
                next_gen = gen + arith.constant(1)
                arrive = comm_ops.atomic_add_system(addr_barrier, arith.constant(1))
                if arrive != PLAN_BLOCKS - 1:
                    comm_ops.spin_until_eq_i32(addr_barrier + fx.Int64(4), next_gen)
                    comm_ops.fence_agent_acquire()
                else:
                    comm_ops.fence_system_release()
                    buffer_store(arith.constant(0), rsrc_bar, 0)
                    buffer_store(next_gen, rsrc_bar, 2)
                    comm_ops.fence_agent_release()
                    buffer_store(next_gen, rsrc_bar, 1)
            fx.barrier()

        # Fill dest_row = send_base[segment] + intra. PLAN_BLOCKS is one, so
        # intra is already the full source-local exclusive offset.
        for route in range(bid * PLAN_THREADS + tid, n_routes, PLAN_THREADS):
            packed = buffer_load(rsrc_map, route, vec_width=1, dtype=T.i32)
            valid = packed >= 0
            segment = packed & arith.constant(0xFFFF)
            intra = packed >> arith.constant(16)
            dest_pe = segment // fx.Int32(epr)
            send_base = comm_ops.load_i32_lds(
                lds_hist + fx.Int64(segment) * fx.Int64(4)
            )
            dest_row = send_base + intra
            in_cap = dest_row < compact_cap
            # Dispatch consumes this directly: low bits select the peer and
            # the remaining bits are the final expert row. Keeping it positive
            # preserves -1 as the dropped-route sentinel and removes the
            # per-route division by compact_cap in dispatch.
            flat = (dest_row << fx.Int32(peer_bits)) | (
                dest_pe & fx.Int32(peer_mask)
            )
            buffer_store(
                arith.select(valid & in_cap, flat, arith.constant(dropped)),
                rsrc_map,
                route,
            )

    @flyc.jit
    def launch(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        kernel(
            arena,
            addr_inp_idx,
            addr_tok_map,
            addr_block_hist,
            addr_send_base,
            addr_masked_m,
            addr_psum,
            addr_barrier,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(PLAN_BLOCKS, 1, 1),
            block=[PLAN_THREADS, 1, 1],
            stream=stream,
        )

    return launch
