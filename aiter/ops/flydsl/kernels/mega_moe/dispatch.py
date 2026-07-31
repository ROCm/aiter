# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Compact dispatch path for MegaMoE v2 stage1."""

from enum import IntEnum

import mori.ir.flydsl as mori_shmem

import flydsl.compiler as flyc
import flydsl.expr as fx
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
    ACTIVE_EXPERTS = 28
    ACTIVE_COUNT = 29
    RUNNING = 30
    P2P_RUNNING = 31


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
    a_producer_done = dp(DispatchSlot.ACTIVE_COUNT)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    destination_groups = 2
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
    i32_cur_tok, addr_in_idx, parity, expected, active_expert_producer, external_grouping, external_counting,
    dispatch_blocks,
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
    a_active_experts = dp(DispatchSlot.ACTIVE_EXPERTS)
    a_active_count = dp(DispatchSlot.ACTIVE_COUNT)
    p_payload_ready = dp(DispatchSlot.P2P_PAYLOAD_READY)

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
    if const_expr(active_expert_producer):
        if tid == fx.Int32(0):
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_active_count), fx.Int32(0))
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
    if const_expr(external_counting):
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(a_active_count, fx.Int32(dispatch_blocks))
            comm_ops.fence_agent_acquire()
            buffer_ops.buffer_store(fx.Int32(0), crfa(a_active_count), fx.Int32(0))
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
        for expert_chunk in range_constexpr((fz_epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(fz_epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            ge = fx.Int32(fz_rank * fz_epr + local_expert)
            source_counts = []
            total_count = fx.Int32(0)
            zero_sources = fx.Int32(0)
            for source in range_constexpr(fz_npes):
                source_count = buffer_ops.buffer_load(
                    r_bc, fx.Int32(source * fz_epr) + safe_expert, vec_width=1, dtype=fx.Int32
                )
                source_count = valid_expert.select(source_count, fx.Int32(0))
                source_counts.append(source_count)
                total_count = total_count + source_count
                zero_sources = zero_sources + (source_count == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
            if const_expr(active_expert_producer):
                if valid_expert & (zero_sources > fx.Int32(0)):
                    local_payload_ready = buffer_ops.buffer_load(
                        crfa(p_payload_ready), fx.Int32(fz_rank), vec_width=1, dtype=fx.Int64
                    )
                    ready_index = parity * fx.Int32(fz_epr) + local_expert
                    comm_ops.store_i32_system(
                        local_payload_ready, ready_index, expected - fx.Int32(fz_npes) + zero_sources
                    )
            num_tiles = (total_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
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
            if const_expr(active_expert_producer):
                if valid_ge & (source_count > fx.Int32(0)):
                    active_slot = fx.Int32(comm_ops.atomic_add_agent(a_active_count, fx.Int32(1)))
                    buffer_ops.buffer_store(
                        (ge % fx.Int32(fz_epr)) * fx.Int32(fz_npes) + ge // fx.Int32(fz_epr),
                        crfa(a_active_experts), active_slot,
                    )
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
            mori_shmem.int32_wait_until_equals(a_active_count, fx.Int32(dispatch_blocks))
            comm_ops.fence_agent_acquire()
        comm_ops.fence_agent_release()
        buffer_ops.buffer_store(expected, crfa(a_pair_order_ready), parity)


# fmt: off
@flyc.jit
def emit_dispatch_group(
    *, num_waves, fz_k, fz_total_experts, addr_disp, i32_cur_tok, addr_in_idx, dispatch_blocks,
    producer_slot, parity, expected, external_counting,
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
    a_group_done = dp(DispatchSlot.ACTIVE_COUNT)
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

    for route in range(group_tid, route_limit, group_threads):
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
    producer_slot, parity, expected, active_expert_producer, cooperative_payload_copy,
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
    a_active_experts = dp(DispatchSlot.ACTIVE_EXPERTS)
    a_active_count = dp(DispatchSlot.ACTIVE_COUNT)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    r_pair_base = crfa(a_pair_base)
    r_lh = crfa(a_lh)
    r_mb = crfa(a_mb)
    r_pair = crfa(a_pair_order)
    r_wts = crfa(addr_in_wts)
    if const_expr(active_expert_producer):
        task_limit = buffer_ops.buffer_load(crfa(a_active_count), fx.Int32(0), vec_width=1, dtype=fx.Int32)
    else:
        task_limit = fx.Int32(fz_total_experts)
    task0 = producer_slot
    task_stride = fx.Int32(dispatch_blocks)
    if const_expr(cooperative_payload_copy):
        row0 = fx.Int32(0)
        row_stride = fx.Int32(1)
    else:
        row0 = warp
        row_stride = fx.Int32(num_waves)

    def _publish_task(destination, local_expert, ge):
        comm_ops.fence_system_release()
        ready_remote = buffer_ops.buffer_load(crfa(p_payload_ready), destination, vec_width=1, dtype=fx.Int64)
        ready_index = parity * fx.Int32(fz_epr) + local_expert
        comm_ops.atomic_add_system(ready_remote + fx.Int64(ready_index) * fx.Int64(4), fx.Int32(1))
        buffer_ops.buffer_store(fx.Int32(0), r_lh, ge)

    num_destinations = fz_total_experts // fz_epr
    hoist_remote_resources = fz_mtpr >= 1024
    if const_expr(not active_expert_producer):
        producer_destination = producer_slot % fx.Int32(num_destinations)
        ready_index = parity * fx.Int32(num_destinations) + producer_destination
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected)
            comm_ops.fence_system_acquire()
        fx.barrier()
    for task_index in range(task0, task_limit, task_stride):
        if const_expr(active_expert_producer):
            task = buffer_ops.buffer_load(crfa(a_active_experts), task_index, vec_width=1, dtype=fx.Int32)
            local_expert = task // fx.Int32(num_destinations)
            destination = task % fx.Int32(num_destinations)
            ready_index = parity * fx.Int32(num_destinations) + destination
            if tid == fx.Int32(0):
                mori_shmem.int32_wait_until_equals(a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected)
                comm_ops.fence_system_acquire()
            fx.barrier()
        else:
            task = task_index
            local_expert = task // fx.Int32(num_destinations)
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
        if const_expr(hoist_remote_resources):
            wts_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64))
            srcmap_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64))
            token_remote = buffer_ops.buffer_load(crfa(p_rx), destination, vec_width=1, dtype=fx.Int64)
            if const_expr(fz_enable_scales):
                scale_remote_rsrc = crfa(buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64))
        for row in range(row0, source_count, row_stride):
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

            if const_expr(not cooperative_payload_copy):
                if lane == fx.Int32(0):
                    _copy_route_header()
            else:
                if tid == fx.Int32(0):
                    _copy_route_header()

            if const_expr(fz_enable_scales):
                if const_expr(not cooperative_payload_copy):
                    scale_lane = lane
                else:
                    scale_lane = tid
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
            if const_expr(not cooperative_payload_copy):
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
            else:
                thread_offset = tid * fx.Int32(4)
                thread_stride = fx.Int32(num_waves * 64 * 4)
                for column in range(thread_offset, fz_n_i32, thread_stride):
                    value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    buffer_ops.buffer_store(value, destination_rsrc, column)

        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            _publish_task(destination, local_expert, ge)
        fx.barrier()
