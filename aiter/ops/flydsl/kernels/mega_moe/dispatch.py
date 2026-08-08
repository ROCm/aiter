# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""Compact dispatch path for MegaMoE v2 stage1."""

from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops


class DispatchSlot(IntEnum):
    PAIR_BASE = 0
    P2P_TOKEN = 1
    P2P_SCALE = 2
    P2P_WEIGHT = 3
    P2P_SRCMAP = 4
    SORTED_EXPERT = 5
    TILE_ROW_BASE = 6
    NUM_VALID = 7
    SRCMAP = 8
    LOCAL_HIST = 9
    COUNT_MATRIX = 10
    P2P_COUNT_MATRIX = 11
    COUNT_DONE = 12
    P2P_COUNT_DONE = 13
    TASK_ROW_BASE = 14
    LOCAL_CURSOR = 15
    P2P_PAYLOAD_READY = 16
    PAIR_ORDER = 17
    P2P_TASK_ROW_BASE = 18
    P2P_PLAN_READY = 19
    PLAN_READY = 20
    PAIR_READY = 21
    ENTRY_COUNT = 22
    EPOCH_GATE = 23
    PAIR_ORDER_READY = 24
    WORK_HEAD = 25
    WORK_TAIL = 26
    EXPERT_TILE_END = 27
    GROUP_DONE = 28
    RUNNING = 29
    P2P_RUNNING = 30
    LAUNCH_READY = 31
    P2P_LAUNCH_READY = 32
    MAX_EXPERT_TILES = 33
    PAYLOAD_CHUNK_DONE = 34
    TILE_READY = 35
    P2P_TILE_READY = 36
    TILE_EXPECTED = 37
    ACTIVE_PAYLOAD_BLOCKS = 38
    PAYLOAD_READY_ROWS = 39
    P2P_PAYLOAD_READY_ROWS = 40
    PAYLOAD_BLOCKS_PER_DESTINATION = 41
    PAYLOAD_CHUNKS_PER_DESTINATION = 42


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


@flyc.jit
def _wave_inclusive_scan_i32(value, lane):
    value_raw = value.ir_value()
    zero_raw = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.rocdl.update_dpp(T.i32, zero_raw, value_raw, dpp, 0xF, 0xF, True)
        value = (lane >= fx.Int32(shift)).select(value + fx.Int32(remote), value)
        value_raw = value.ir_value()
    source16 = (lane & fx.Int32(0x30)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(T.i32, source16 * fx.Int32(4), value)
    value = (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)
    source32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
    remote32 = fx.rocdl.ds_bpermute(T.i32, source32 * fx.Int32(4), value)
    return (lane >= fx.Int32(32)).select(value + fx.Int32(remote32), value)


@flyc.jit
def _wave_reduce_max_i32(value, lane):
    for distance in (1, 2, 4, 8, 16, 32):
        peer = fx.Int32(
            fx.rocdl.ds_bpermute(
                T.i32, (lane ^ fx.Int32(distance)) * fx.Int32(4), value
            )
        )
        value = (peer > value).select(peer, value)
    return value


# fmt: off
@flyc.jit
def emit_direct_fixed_slot_payload(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_cap, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    i32_cur_tok, dispatch_blocks, producer_slot, parity, expected,
):
# fmt: on
    """Allocate and publish routes directly into destination fixed slots."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    p_running = dp(DispatchSlot.P2P_RUNNING)
    p_source_done = dp(DispatchSlot.P2P_COUNT_DONE)
    a_producer_done = dp(DispatchSlot.GROUP_DONE)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    destination_groups = 2
    assert dispatch_blocks % destination_groups == 0, "direct fixed-slot dispatch needs even producer groups"
    producers_per_group = dispatch_blocks // destination_groups
    producer_group = producer_slot % fx.Int32(destination_groups)
    group_slot = producer_slot // fx.Int32(destination_groups)
    route = group_slot * fx.Int32(num_waves) + warp
    route_stride = fx.Int32(producers_per_group * num_waves)
    route_limit = i32_cur_tok * fx.Int32(fz_k)
    r_idx = crfa(addr_in_idx)
    r_wts = crfa(addr_in_wts)
    r_scales = crfa(addr_in_sc)

    for wk in range(route, route_limit, route_stride):
        source_token = wk // fx.Int32(fz_k)
        topk_slot = wk - source_token * fx.Int32(fz_k)
        global_expert_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            global_expert_lane = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
        global_expert = fx.Int32(fx.rocdl.readfirstlane(T.i32, global_expert_lane))
        valid_expert = (global_expert >= fx.Int32(0)) & (global_expert < fx.Int32(fz_total_experts))
        safe_expert = valid_expert.select(global_expert, fx.Int32(0))
        destination = safe_expert // fx.Int32(fz_epr)
        local_expert = safe_expert - destination * fx.Int32(fz_epr)
        offset_lane = fx.Int32(0)
        assigned = valid_expert & (destination % fx.Int32(destination_groups) == producer_group)
        if lane == fx.Int32(0):
            if assigned:
                remote_running = buffer_ops.buffer_load(
                    crfa(p_running), destination, vec_width=1, dtype=fx.Int64
                )
                offset_lane = fx.Int32(
                    comm_ops.atomic_add_system(
                        remote_running + fx.Int64(local_expert) * fx.Int64(4), fx.Int32(1)
                    )
                )
        expert_offset = fx.Int32(fx.rocdl.readlane(T.i32, offset_lane, 0))
        publish = assigned & (expert_offset < fx.Int32(fz_cap))
        payload_row = local_expert * fx.Int32(fz_cap) + expert_offset

        if publish:
            remote_token = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
            destination_rsrc = crfa(remote_token + fx.Int64(payload_row) * fx.Int64(fz_nbytes))
            source_rsrc = crfa(addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes))
            for column in range(lane * fx.Int32(4), fz_n_i32, 256):
                value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                buffer_ops.buffer_store(value, destination_rsrc, column)

            if const_expr(fz_enable_scales):
                if lane < fx.Int32(fz_scale_n_i32):
                    scale = buffer_ops.buffer_load(
                        r_scales, source_token * fx.Int32(fz_scale_n_i32) + lane,
                        vec_width=1, dtype=fx.Int32,
                    )
                    remote_scale = buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(scale, crfa(remote_scale), payload_row * fx.Int32(fz_scale_n_i32) + lane)

            if lane == fx.Int32(0):
                weight = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                remote_weights = buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64)
                remote_srcmap = buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64)
                buffer_ops.buffer_store(weight_bits, crfa(remote_weights), payload_row)
                buffer_ops.buffer_store(source_encoding, crfa(remote_srcmap), payload_row)

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_system_release()
        done = fx.Int32(
            comm_ops.atomic_add_agent(
                a_producer_done + fx.Int64(producer_group) * fx.Int64(4), fx.Int32(1)
            )
        )
        if done == fx.Int32(producers_per_group - 1):
            comm_ops.fence_agent_acquire()
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            for destination in range_constexpr(fz_npes):
                if producer_group == fx.Int32(destination % destination_groups):
                    remote_done = buffer_ops.buffer_load(
                        crfa(p_source_done), fx.Int32(destination), vec_width=1, dtype=fx.Int64
                    )
                    comm_ops.store_i32_system(remote_done, done_index, expected)


@flyc.jit
def emit_direct_fixed_slot_finalize(
    *, fz_npes, fz_epr, fz_cap, fz_mtpr, fz_rank, fz_tile_m, n_tiles, addr_disp, parity, expected
):
    """Finalize local fixed slots as soon as every source publishes this destination."""
    assert 0 < fz_epr <= 64, "direct fixed-slot finalize requires 1..64 experts per rank"
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_running = dp(DispatchSlot.RUNNING)
    a_source_done = dp(DispatchSlot.COUNT_DONE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_work_tail = dp(DispatchSlot.WORK_TAIL)
    a_expert_tile_end = dp(DispatchSlot.EXPERT_TILE_END)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    if warp == fx.Int32(0):
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            mori_shmem.int32_wait_until_equals(a_source_done + fx.Int64(done_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()

        valid_expert = lane < fx.Int32(fz_epr)
        safe_expert = valid_expert.select(lane, fx.Int32(0))
        count = buffer_ops.buffer_load(crfa(a_running), safe_expert, vec_width=1, dtype=fx.Int32)
        count = valid_expert.select(count, fx.Int32(0))
        overflow_flag = (count > fx.Int32(fz_cap)).select(fx.Int32(1), fx.Int32(0))
        overflow_prefix = _wave_inclusive_scan_i32(overflow_flag, lane)
        overflow_count = fx.Int32(fx.rocdl.readlane(T.i32, overflow_prefix, fz_epr - 1))
        no_overflow = overflow_count == fx.Int32(0)
        safe_count = (count <= fx.Int32(fz_cap)).select(count, fx.Int32(0))
        num_expert_tiles = (safe_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
        max_expert_tiles = _wave_reduce_max_i32(num_expert_tiles, lane)
        inclusive_tiles = _wave_inclusive_scan_i32(num_expert_tiles, lane)
        metadata_base = inclusive_tiles - num_expert_tiles
        total_tiles = fx.Int32(fx.rocdl.readlane(T.i32, inclusive_tiles, fz_epr - 1))

        if valid_expert:
            if no_overflow:
                global_expert = fx.Int32(fz_rank * fz_epr) + safe_expert
                payload_base = safe_expert * fx.Int32(fz_cap)
                for tile in range(fx.Int32(0), num_expert_tiles, 1):
                    metadata_index = metadata_base + tile
                    buffer_ops.buffer_store(global_expert, crfa(a_se), metadata_index)
                    buffer_ops.buffer_store(payload_base + tile * fx.Int32(fz_tile_m), crfa(a_trb), metadata_index)
                padded_rows = num_expert_tiles * fx.Int32(fz_tile_m)
                for pad in range(fx.Int32(0), padded_rows - safe_count, 1):
                    buffer_ops.buffer_store(fx.Int32(fz_npes * fz_mtpr), crfa(a_sm), payload_base + safe_count + pad)
                buffer_ops.buffer_store(metadata_base + num_expert_tiles, crfa(a_expert_tile_end), safe_expert)
            else:
                buffer_ops.buffer_store(fx.Int32(0), crfa(a_expert_tile_end), safe_expert)
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_running), safe_expert)

        if lane == fx.Int32(0):
            num_valid = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            ready_work = no_overflow.select(total_tiles * fx.Int32(n_tiles), fx.Int32(0))
            buffer_ops.buffer_store(num_valid, crfa(a_nv), fx.Int32(0))
            # num_valid[1] is a device-visible overflow status.
            buffer_ops.buffer_store(overflow_count, crfa(a_nv), fx.Int32(1))
            buffer_ops.buffer_store(ready_work, crfa(a_work_tail), fx.Int32(0))
            buffer_ops.buffer_store(max_expert_tiles, crfa(a_max_expert_tiles), fx.Int32(0))

        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = buffer_ops.buffer_load(crfa(p_plan_ready), source, vec_width=1, dtype=fx.Int64)
            ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.store_i32_system(remote_ready, ready_index, expected)
    fx.barrier()


# fmt: off
@flyc.jit
def emit_dispatch_plan(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_mtpr, fz_rank, fz_tile_m, fz_total_experts, addr_disp,
    i32_cur_tok, addr_in_idx, parity, expected, external_grouping, external_counting,
    dispatch_blocks, payload_chunk_rows=0, payload_tile_ready=False,
):
# fmt: on
    """Build a destination-owned compact plan in one producer-only CTA."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_bc = dp(DispatchSlot.COUNT_MATRIX)
    p_bc = dp(DispatchSlot.P2P_COUNT_MATRIX)
    a_cd = dp(DispatchSlot.COUNT_DONE)
    p_cd = dp(DispatchSlot.P2P_COUNT_DONE)
    a_lc = dp(DispatchSlot.LOCAL_CURSOR)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    p_mb = dp(DispatchSlot.P2P_TASK_ROW_BASE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_expert_tile_end = dp(DispatchSlot.EXPERT_TILE_END)
    a_group_done = dp(DispatchSlot.GROUP_DONE)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)
    a_tile_ready = dp(DispatchSlot.TILE_READY)
    a_tile_expected = dp(DispatchSlot.TILE_EXPECTED)
    a_active_payload_blocks = dp(DispatchSlot.ACTIVE_PAYLOAD_BLOCKS)
    a_payload_blocks_per_destination = dp(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
    a_payload_chunks_per_destination = dp(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    block_threads = num_waves * 64

    gtid = tid
    gnt = fx.Int32(block_threads)
    wl = i32_cur_tok * fx.Int32(fz_k)
    r_idx = crfa(addr_in_idx)
    r_lh = crfa(a_lh)
    r_bc = crfa(a_bc)
    r_pair_base = crfa(a_pair_base)
    r_pair = crfa(a_pair_order)
    r_lc = crfa(a_lc)
    if const_expr(external_counting):
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(a_group_done, fx.Int32(dispatch_blocks))
            comm_ops.fence_agent_acquire()
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_group_done), fx.Int32(0))
            fx.rocdl.s_waitcnt(0)
            comm_ops.fence_agent_release()
    else:
        if const_expr(num_waves >= 8):
            for wk0 in range(gtid, wl, gnt * fx.Int32(2)):
                wk1 = wk0 + gnt
                valid_wk1 = wk1 < wl
                safe_wk1 = valid_wk1.select(wk1, fx.Int32(0))
                expert0 = buffer_ops.buffer_load(r_idx, wk0, vec_width=1, dtype=fx.Int32)
                expert1 = buffer_ops.buffer_load(r_idx, safe_wk1, vec_width=1, dtype=fx.Int32)
                valid0 = (expert0 >= fx.Int32(0)) & (expert0 < fx.Int32(fz_total_experts))
                valid1 = valid_wk1 & (expert1 >= fx.Int32(0)) & (expert1 < fx.Int32(fz_total_experts))
                if valid0:
                    comm_ops.atomic_add_agent(a_lh + fx.Int64(expert0) * fx.Int64(4), fx.Int32(1))
                if valid1:
                    comm_ops.atomic_add_agent(a_lh + fx.Int64(expert1) * fx.Int64(4), fx.Int32(1))
        else:
            for wk in range(gtid, wl, gnt):
                expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    comm_ops.atomic_add_agent(a_lh + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    comm_ops.fence_agent_acquire()

    if const_expr(payload_tile_ready and dispatch_blocks > 32):
        if warp == fx.Int32(0):
            active_payload_blocks = fx.Int32(0)
            for destination in range_constexpr(fz_npes):
                max_source_count = fx.Int32(0)
                for local_expert in range(lane, fz_epr, 64):
                    ge = fx.Int32(destination * fz_epr) + local_expert
                    source_count = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
                    max_source_count = (source_count > max_source_count).select(
                        source_count, max_source_count
                    )
                max_source_count = _wave_reduce_max_i32(max_source_count, lane)
                if lane == fx.Int32(0):
                    chunks_per_destination = (
                        max_source_count + fx.Int32(payload_chunk_rows - 1)
                    ) // fx.Int32(payload_chunk_rows)
                    chunks_per_destination = (chunks_per_destination > fx.Int32(0)).select(
                        chunks_per_destination, fx.Int32(1)
                    )
                    chunks_per_destination = (chunks_per_destination > fx.Int32(4)).select(
                        chunks_per_destination, fx.Int32(4)
                    )
                    buffer_ops.buffer_store(
                        chunks_per_destination, crfa(a_payload_chunks_per_destination),
                        fx.Int32(destination),
                    )
                    active_per_destination = chunks_per_destination
                    active_per_destination = (active_per_destination > fx.Int32(4)).select(
                        active_per_destination, fx.Int32(4)
                    )
                    max_per_destination = fx.Int32(dispatch_blocks // fz_npes)
                    active_per_destination = (active_per_destination < max_per_destination).select(
                        active_per_destination, max_per_destination
                    )
                    buffer_ops.buffer_store(
                        active_per_destination, crfa(a_payload_blocks_per_destination),
                        fx.Int32(destination),
                    )
                    active_payload_blocks = active_payload_blocks + active_per_destination
            if lane == fx.Int32(0):
                buffer_ops.buffer_store(
                    active_payload_blocks, crfa(a_active_payload_blocks), fx.Int32(0)
                )
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        comm_ops.fence_agent_release()

    # Transpose the source histogram into each destination's count matrix.
    for ge in range(gtid, fz_total_experts, gnt):
        destination = ge // fx.Int32(fz_epr)
        local_expert = ge - destination * fx.Int32(fz_epr)
        remote_bigcnt = buffer_ops.buffer_load(crfa(p_bc), destination, vec_width=1, dtype=fx.Int64)
        count = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
        buffer_ops.buffer_store(count, crfa(remote_bigcnt), fx.Int32(fz_rank * fz_epr) + local_expert)
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    # Warp 0 plans local experts after all source matrices arrive.
    if warp == fx.Int32(0):
        comm_ops.fence_system_release()
        for peer in range(lane, fz_npes, 64):
            remote_done = buffer_ops.buffer_load(crfa(p_cd), peer, vec_width=1, dtype=fx.Int64)
            done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.store_i32_system(remote_done, done_index, expected)
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            mori_shmem.int32_wait_until_equals(a_cd + fx.Int64(done_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()

        r_se = crfa(a_se)
        r_trb = crfa(a_trb)
        r_nv = crfa(a_nv)
        r_sm = crfa(a_sm)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        for expert_chunk in range_constexpr((fz_epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(fz_epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            ge = fx.Int32(fz_rank * fz_epr + local_expert)
            source_counts = []
            total_count = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                source_count = buffer_ops.buffer_load(
                    r_bc, fx.Int32(source * fz_epr) + safe_expert, vec_width=1, dtype=fx.Int32
                )
                source_count = valid_expert.select(source_count, fx.Int32(0))
                source_counts.append(source_count)
                total_count = total_count + source_count
            num_tiles = (total_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
            chunk_max = _wave_reduce_max_i32(num_tiles, lane)
            max_expert_tiles = (chunk_max > max_expert_tiles).select(
                chunk_max, max_expert_tiles
            )
            padded_rows = num_tiles * fx.Int32(fz_tile_m)
            inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
            local_row_base = row_carry + inclusive_rows - padded_rows

            sender_prefix = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                if valid_expert:
                    remote_my_base = buffer_ops.buffer_load(crfa(p_mb), fx.Int32(source), vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(local_row_base + sender_prefix, crfa(remote_my_base), ge)
                sender_prefix = sender_prefix + source_counts[source]

            if valid_expert:
                if const_expr(payload_tile_ready):
                    base_tile = local_row_base // fx.Int32(fz_tile_m)
                    for tile in range(fx.Int32(0), num_tiles, 1):
                        tile_index = base_tile + tile
                        buffer_ops.buffer_store(fx.Int32(0), crfa(a_tile_ready), tile_index)
                        buffer_ops.buffer_store(fx.Int32(1), crfa(a_tile_expected), tile_index)
                    sender_prefix = fx.Int32(0)
                    for source in range_constexpr(fz_npes):
                        source_count = source_counts[source]
                        source_active = source_count > fx.Int32(0)
                        source_boundary = source_active & (sender_prefix > fx.Int32(0))
                        source_boundary = source_boundary & (
                            sender_prefix % fx.Int32(fz_tile_m) != fx.Int32(0)
                        )
                        if source_boundary:
                            tile_index = base_tile + sender_prefix // fx.Int32(fz_tile_m)
                            value = buffer_ops.buffer_load(
                                crfa(a_tile_expected), tile_index, vec_width=1, dtype=fx.Int32
                            )
                            buffer_ops.buffer_store(value + fx.Int32(1), crfa(a_tile_expected), tile_index)
                        for chunk_offset in range(
                            fx.Int32(payload_chunk_rows), source_count, payload_chunk_rows
                        ):
                            boundary = sender_prefix + chunk_offset
                            boundary_unaligned = boundary % fx.Int32(fz_tile_m) != fx.Int32(0)
                            if boundary_unaligned:
                                tile_index = base_tile + boundary // fx.Int32(fz_tile_m)
                                value = buffer_ops.buffer_load(
                                    crfa(a_tile_expected), tile_index, vec_width=1, dtype=fx.Int32
                                )
                                buffer_ops.buffer_store(value + fx.Int32(1), crfa(a_tile_expected), tile_index)
                        sender_prefix = sender_prefix + source_count
                buffer_ops.buffer_store(
                    (local_row_base + padded_rows) // fx.Int32(fz_tile_m), crfa(a_expert_tile_end), local_expert
                )
                for tile in range(fx.Int32(0), num_tiles, 1):
                    metadata_index = (local_row_base // fx.Int32(fz_tile_m)) + tile
                    buffer_ops.buffer_store(ge, r_se, metadata_index)
                    buffer_ops.buffer_store(local_row_base + tile * fx.Int32(fz_tile_m), r_trb, metadata_index)
                padding = padded_rows - total_count
                for pad in range(fx.Int32(0), padding, 1):
                    buffer_ops.buffer_store(fx.Int32(fz_npes * fz_mtpr), r_sm, local_row_base + total_count + pad)

            last_lane = min(63, fz_epr - expert_chunk * 64 - 1)
            row_carry = row_carry + fx.Int32(fx.rocdl.readlane(T.i32, inclusive_rows, last_lane))

        if lane == fx.Int32(0):
            buffer_ops.buffer_store(row_carry, r_nv, fx.Int32(0))
            buffer_ops.buffer_store(max_expert_tiles, crfa(a_max_expert_tiles), fx.Int32(0))
        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = buffer_ops.buffer_load(crfa(p_plan_ready), source, vec_width=1, dtype=fx.Int64)
            ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
            comm_ops.store_i32_system(remote_ready, ready_index, expected)
        fx.rocdl.s_waitcnt(0)
    elif warp == fx.Int32(1):
        # Build the global-expert exclusive prefix cooperatively.
        pairs_per_lane = (fz_total_experts + 63) // 64
        lane_base = lane * fx.Int32(pairs_per_lane)
        lane_total = fx.Int32(0)
        lane_counts = []
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            safe_ge = valid_ge.select(ge, fx.Int32(0))
            source_count = buffer_ops.buffer_load(r_lh, safe_ge, vec_width=1, dtype=fx.Int32)
            source_count = valid_ge.select(source_count, fx.Int32(0))
            lane_counts.append(source_count)
            lane_total = lane_total + source_count
        lane_prefix = _wave_inclusive_scan_i32(lane_total, lane) - lane_total
        source_prefix = lane_prefix
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(fz_total_experts)
            if valid_ge:
                buffer_ops.buffer_store(source_prefix, r_pair_base, ge)
                buffer_ops.buffer_store(source_prefix, r_lc, ge)
            source_prefix = source_prefix + lane_counts[item]
        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_agent_release()
        if lane == fx.Int32(0):
            comm_ops.store_i32_system(a_pair_ready, parity, expected)

    if const_expr(not external_grouping):
        # Warp 1 groups immediately; later waves wait for its prefix.
        if warp > fx.Int32(0):
            if warp > fx.Int32(1):
                if lane == fx.Int32(0):
                    mori_shmem.int32_wait_until_equals(a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected)
                    comm_ops.fence_agent_acquire()
            group_tid = (warp - fx.Int32(1)) * fx.Int32(64) + lane
            group_threads = fx.Int32((num_waves - 1) * 64)
            for wk in range(group_tid, wl, group_threads):
                expert = buffer_ops.buffer_load(r_idx, wk, vec_width=1, dtype=fx.Int32)
                valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
                if valid:
                    position = fx.Int32(
                        comm_ops.atomic_add_agent(a_lc + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
                    )
                    buffer_ops.buffer_store(wk, r_pair, position)

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        if const_expr(external_grouping):
            active_group_blocks = fx.Int32(dispatch_blocks)
            if const_expr(payload_tile_ready and dispatch_blocks > 32):
                active_group_blocks = buffer_ops.buffer_load(
                    crfa(a_active_payload_blocks), fx.Int32(0), vec_width=1, dtype=fx.Int32
                )
            mori_shmem.int32_wait_until_equals(a_group_done, active_group_blocks)
            comm_ops.fence_agent_acquire()
        comm_ops.fence_agent_release()
        comm_ops.store_i32_system(a_pair_order_ready, parity, expected)


# fmt: off
@flyc.jit
def emit_dispatch_group(
    *, num_waves, fz_k, fz_total_experts, addr_disp, i32_cur_tok, addr_in_idx, dispatch_blocks,
    producer_slot, parity, expected, external_counting, adaptive_grouping=False,
):
# fmt: on
    """Count and group disjoint route spans across payload producer CTAs."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_local_hist = dp(DispatchSlot.LOCAL_HIST)
    a_local_cursor = dp(DispatchSlot.LOCAL_CURSOR)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_group_done = dp(DispatchSlot.GROUP_DONE)
    a_active_payload_blocks = dp(DispatchSlot.ACTIVE_PAYLOAD_BLOCKS)
    r_idx = crfa(addr_in_idx)
    r_pair = crfa(a_pair_order)
    tid = fx.thread_idx.x
    block_threads = fx.Int32(num_waves * 64)
    group_tid = producer_slot * block_threads + tid
    group_threads = fx.Int32(dispatch_blocks) * block_threads
    route_limit = i32_cur_tok * fx.Int32(fz_k)

    if const_expr(external_counting):
        for route in range(group_tid, route_limit, group_threads):
            expert = buffer_ops.buffer_load(r_idx, route, vec_width=1, dtype=fx.Int32)
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
            if valid:
                comm_ops.atomic_add_agent(a_local_hist + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
    if tid == fx.Int32(0):
        mori_shmem.int32_wait_until_equals(a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected)
        comm_ops.fence_agent_acquire()
    fx.barrier()

    active_group_blocks = fx.Int32(dispatch_blocks)
    if const_expr(adaptive_grouping and dispatch_blocks > 32):
        active_group_blocks = buffer_ops.buffer_load(
            crfa(a_active_payload_blocks), fx.Int32(0), vec_width=1, dtype=fx.Int32
        )
    group_active = producer_slot < active_group_blocks
    if group_active:
        active_group_tid = producer_slot * block_threads + tid
        active_group_threads = active_group_blocks * block_threads
        for route in range(active_group_tid, route_limit, active_group_threads):
            expert = buffer_ops.buffer_load(r_idx, route, vec_width=1, dtype=fx.Int32)
            valid = (expert >= fx.Int32(0)) & (expert < fx.Int32(fz_total_experts))
            if valid:
                position = fx.Int32(
                    comm_ops.atomic_add_agent(a_local_cursor + fx.Int64(expert) * fx.Int64(4), fx.Int32(1))
                )
                buffer_ops.buffer_store(route, r_pair, position)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
            mori_shmem.int32_wait_until_equals(a_pair_order_ready + fx.Int64(parity) * fx.Int64(4), expected)
            comm_ops.fence_agent_acquire()
    fx.barrier()


# fmt: off
@flyc.jit
def emit_dispatch_payload(
    *, num_waves, fz_epr, fz_k, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32, fz_safe_end_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_wts, addr_in_sc, dispatch_blocks,
    producer_slot, parity, expected, producers_per_destination, chunks_per_destination,
    payload_chunk_rows=0,
    payload_tile_ready=False,
):
# fmt: on
    """Produce independently publishable expert payloads from a compact plan."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_mb = dp(DispatchSlot.TASK_ROW_BASE)
    p_payload_ready = dp(DispatchSlot.P2P_PAYLOAD_READY)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_chunk_done = dp(DispatchSlot.PAYLOAD_CHUNK_DONE)
    p_tile_ready = dp(DispatchSlot.P2P_TILE_READY)
    p_payload_ready_rows = dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    r_pair_base = crfa(a_pair_base)
    r_lh = crfa(a_lh)
    r_mb = crfa(a_mb)
    r_pair = crfa(a_pair_order)
    r_wts = crfa(addr_in_wts)
    r_chunk_done = crfa(a_chunk_done)
    row0 = warp
    row_stride = fx.Int32(num_waves)

    def _publish_task(destination, local_expert, ge):
        comm_ops.fence_system_release()
        ready_remote = buffer_ops.buffer_load(crfa(p_payload_ready), destination, vec_width=1, dtype=fx.Int64)
        ready_index = parity * fx.Int32(fz_epr) + local_expert
        comm_ops.atomic_add_system(ready_remote + fx.Int64(ready_index) * fx.Int64(4), fx.Int32(1))
        buffer_ops.buffer_store(fx.Int32(0), r_lh, ge)

    def _finish_task(destination, local_expert, ge, num_chunks):
        if const_expr(payload_chunk_rows > 0):
            comm_ops.fence_system_release()
            completed = fx.Int32(
                comm_ops.atomic_add_agent(a_chunk_done + fx.Int64(ge) * fx.Int64(4), fx.Int32(1))
            )
            if completed == num_chunks - fx.Int32(1):
                comm_ops.fence_agent_acquire()
                buffer_ops.buffer_store(fx.Int32(0), r_chunk_done, ge)
                _publish_task(destination, local_expert, ge)
        else:
            _publish_task(destination, local_expert, ge)

    num_destinations = fz_total_experts // fz_epr
    if const_expr(payload_chunk_rows > 0):
        assert dispatch_blocks % num_destinations == 0
        task_limit = fx.Int32(fz_epr) * chunks_per_destination
        task0 = producer_slot // fx.Int32(num_destinations)
        task_stride = fx.Int32(producers_per_destination)
    else:
        task_limit = fx.Int32(fz_total_experts)
        task0 = producer_slot
        task_stride = fx.Int32(dispatch_blocks)
    hoist_remote_resources = fz_mtpr >= 1024
    producer_destination = producer_slot % fx.Int32(num_destinations)
    ready_index = parity * fx.Int32(num_destinations) + producer_destination
    if tid == fx.Int32(0):
        mori_shmem.int32_wait_until_equals(a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()
    destination_ready_rows = fx.Int32(0)
    if const_expr(payload_tile_ready):
        if tid == fx.Int32(0):
            remote_ready_rows = buffer_ops.buffer_load(
                crfa(p_payload_ready_rows), producer_destination, vec_width=1, dtype=fx.Int64
            )
            destination_ready_rows = buffer_ops.buffer_load(
                crfa(remote_ready_rows), fx.Int32(0), vec_width=1, dtype=fx.Int32
            )
    fx.barrier()
    for task_index in range(task0, task_limit, task_stride):
        if const_expr(payload_chunk_rows > 0):
            chunk_id = task_index // fx.Int32(fz_epr)
            rotated_expert = task_index - chunk_id * fx.Int32(fz_epr)
            rotation = (chunk_id * fx.Int32(17)) % fx.Int32(fz_epr)
            local_expert = (rotated_expert + fx.Int32(fz_epr) - rotation) % fx.Int32(fz_epr)
        else:
            chunk_id = fx.Int32(0)
            local_expert = task_index // fx.Int32(num_destinations)
        destination = producer_destination
        ge = destination * fx.Int32(fz_epr) + local_expert
        source_count_lane = fx.Int32(0)
        source_base_lane = fx.Int32(0)
        destination_base_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            source_count_lane = buffer_ops.buffer_load(r_lh, ge, vec_width=1, dtype=fx.Int32)
            source_base_lane = buffer_ops.buffer_load(r_pair_base, ge, vec_width=1, dtype=fx.Int32)
            destination_base_lane = buffer_ops.buffer_load(r_mb, ge, vec_width=1, dtype=fx.Int32)
        source_count = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_count_lane))
        source_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_base_lane))
        destination_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, destination_base_lane))
        if const_expr(payload_chunk_rows > 0):
            num_chunks = (source_count + fx.Int32(payload_chunk_rows - 1)) // fx.Int32(
                payload_chunk_rows
            )
            num_chunks = (num_chunks > fx.Int32(0)).select(num_chunks, fx.Int32(1))
            chunk_active = chunk_id < num_chunks
            chunk_begin = chunk_id * fx.Int32(payload_chunk_rows)
            chunk_limit = chunk_begin + fx.Int32(payload_chunk_rows)
            chunk_end = (source_count < chunk_limit).select(source_count, chunk_limit)
            row_begin = chunk_active.select(chunk_begin, fx.Int32(0))
            row_end = chunk_active.select(chunk_end, fx.Int32(0))
        else:
            num_chunks = fx.Int32(1)
            chunk_active = fx.Int32(0) == fx.Int32(0)
            row_begin = fx.Int32(0)
            row_end = source_count
        if const_expr(hoist_remote_resources):
            wts_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64))
            srcmap_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64))
            token_remote = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
            if const_expr(fz_enable_scales):
                scale_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64))
        for row in range(row_begin + row0, row_end, row_stride):
            wk_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                wk_lane = buffer_ops.buffer_load(r_pair, source_base + row, vec_width=1, dtype=fx.Int32)
            wk = fx.Int32(fx.rocdl.readfirstlane(T.i32, wk_lane))
            source_token = wk // fx.Int32(fz_k)
            topk_slot = wk % fx.Int32(fz_k)
            destination_row = destination_base + row

            def _copy_route_header():
                weight = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                if const_expr(hoist_remote_resources):
                    buffer_ops.buffer_store(weight_bits, wts_remote_rsrc, destination_row)
                    buffer_ops.buffer_store(source_encoding, srcmap_remote_rsrc, destination_row)
                else:
                    wts_remote = buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(weight_bits, crfa(wts_remote), destination_row)
                    srcmap_remote = buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(source_encoding, crfa(srcmap_remote), destination_row)

            if lane == fx.Int32(0):
                _copy_route_header()

            if const_expr(fz_enable_scales):
                scale_lane = lane
                if const_expr(fz_scale_n_i32 % 4 == 0):
                    scale_offset = scale_lane * fx.Int32(4)
                    if scale_offset < fx.Int32(fz_scale_n_i32):
                        scale = buffer_ops.buffer_load(
                            crfa(addr_in_sc), source_token * fx.Int32(fz_scale_n_i32) + scale_offset,
                            vec_width=4, dtype=fx.Int32,
                        )
                        if const_expr(hoist_remote_resources):
                            buffer_ops.buffer_store(
                                scale, scale_remote_rsrc, destination_row * fx.Int32(fz_scale_n_i32) + scale_offset
                            )
                        else:
                            row_scale_remote = crfa(
                                buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                            )
                            buffer_ops.buffer_store(
                                scale, row_scale_remote, destination_row * fx.Int32(fz_scale_n_i32) + scale_offset
                            )
                elif scale_lane < fx.Int32(fz_scale_n_i32):
                    scale = buffer_ops.buffer_load(
                        crfa(addr_in_sc), source_token * fx.Int32(fz_scale_n_i32) + scale_lane,
                        vec_width=1, dtype=fx.Int32,
                    )
                    if const_expr(hoist_remote_resources):
                        buffer_ops.buffer_store(
                            scale, scale_remote_rsrc, destination_row * fx.Int32(fz_scale_n_i32) + scale_lane
                        )
                    else:
                        row_scale_remote = crfa(
                            buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                        )
                        buffer_ops.buffer_store(
                            scale, row_scale_remote, destination_row * fx.Int32(fz_scale_n_i32) + scale_lane
                        )
            source_rsrc = crfa(addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes))
            if const_expr(hoist_remote_resources):
                destination_rsrc = crfa(token_remote + fx.Int64(destination_row) * fx.Int64(fz_nbytes))
            else:
                row_token_remote = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
                destination_rsrc = crfa(row_token_remote + fx.Int64(destination_row) * fx.Int64(fz_nbytes))
            lane_offset = lane * fx.Int32(4)
            if const_expr(fz_safe_end_i32 > 0):
                for column in range(lane_offset, fz_safe_end_i32, 512):
                    value0 = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    value1 = buffer_ops.buffer_load(
                        source_rsrc, column + fx.Int32(256), vec_width=4, dtype=fx.Int32
                    )
                    buffer_ops.buffer_store(value0, destination_rsrc, column)
                    buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
            if const_expr(fz_safe_end_i32 < fz_n_i32):
                for column in range(lane_offset + fz_safe_end_i32, fz_n_i32, 256):
                    value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    buffer_ops.buffer_store(value, destination_rsrc, column)

        if chunk_active:
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
            if tid == fx.Int32(0):
                if const_expr(payload_tile_ready):
                    if row_end > row_begin:
                        comm_ops.fence_system_release()
                        remote_tile_ready = buffer_ops.buffer_load(
                            crfa(p_tile_ready), destination, vec_width=1, dtype=fx.Int64
                        )
                        first_tile = (destination_base + row_begin) // destination_ready_rows
                        last_tile = (destination_base + row_end - fx.Int32(1)) // destination_ready_rows
                        for tile in range(first_tile, last_tile + fx.Int32(1), 1):
                            comm_ops.atomic_add_system(
                                remote_tile_ready + fx.Int64(tile) * fx.Int64(4), fx.Int32(1)
                            )
                _finish_task(destination, local_expert, ge, num_chunks)
            fx.barrier()
