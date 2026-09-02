# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""Compact dispatch path for MegaMoE v2 stage1."""

from enum import IntEnum

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
    PAYLOAD_READY_ROWS = 39
    P2P_PAYLOAD_READY_ROWS = 40
    PAYLOAD_BLOCKS_PER_DESTINATION = 41
    PAYLOAD_CHUNKS_PER_DESTINATION = 42
    TILE_INPUT_BASE = 48
    GROUP_TASK_BASE = 49
    P2P_GROUP_TASK_BASE = 50
    ROUTE_SEGMENT = 51
    PREP_ENTRY_COUNT = 52
    PREP_EPOCH_GATE = 53
    READY_TILE_QUEUE = 54
    P2P_READY_TILE_QUEUE = 55
    READY_TILE_EPOCH = 56
    P2P_READY_TILE_EPOCH = 57
    READY_TILE_TAIL = 58
    P2P_READY_TILE_TAIL = 59
    P2P_TILE_EXPECTED = 60
    FANOUT_PAIR_CONFIG = 61
    BLOCK_HIST = 62


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


@flyc.jit
def _load_fanout_pair(
    addr_pair_config,
    destination,
    parity,
    *,
    npes,
    fanout_masks,
    runtime_fanout,
):
    """Return the selected mask and canonical expert for one destination.

    The runtime table is double-buffered by the Stage1 parity.  Each entry is
    ``lo | hi << 8 | enabled << 16``.  Keeping expert identities out of the
    compile key lets one AOT artifact serve every routing distribution.
    """
    selected_mask = fx.Int64(0)
    canonical = fx.Int32(0)
    if const_expr(runtime_fanout):
        packed = comm_ops.load_i32_system(
            addr_pair_config,
            parity * fx.Int32(npes) + destination,
        )
        # ``packed`` controls route classification, destination offsets, and
        # the producer task mapping.  Keep the runtime-table VMEM dependency
        # explicit before those values cross lane/control-flow boundaries.
        fx.rocdl.s_waitcnt(0)
        enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
        pair_a = packed & fx.Int32(0xFF)
        pair_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
        pair_mask = (fx.Int64(1) << fx.Int64(pair_a)) | (
            fx.Int64(1) << fx.Int64(pair_b)
        )
        selected_mask = enabled.select(pair_mask, fx.Int64(0))
        canonical = enabled.select(pair_a, fx.Int32(0))
    else:
        for peer in range_constexpr(len(fanout_masks)):
            mask = int(fanout_masks[peer])
            selected_mask = (destination == fx.Int32(peer)).select(
                fx.Int64(mask), selected_mask
            )
            if mask:
                canonical_expert = (mask & -mask).bit_length() - 1
                canonical = (destination == fx.Int32(peer)).select(
                    fx.Int32(canonical_expert), canonical
                )
    return selected_mask, canonical


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


@flyc.jit
def _increment_i32(rsrc, index):
    value = buffer_ops.buffer_load(rsrc, index, vec_width=1, dtype=fx.Int32)
    buffer_ops.buffer_store(value + fx.Int32(1), rsrc, index)


@flyc.jit
def _classify_fanout_wave_route(
    expert,
    lane,
    addr_pair_config,
    parity,
    *,
    fz_k,
    fz_epr,
    fz_total_experts,
    fz_npes,
    fanout_masks,
    runtime_fanout,
):
    """Classify one route after a wave has loaded eight-token groups.

    Lanes are split into groups of eight.  The first ``topk`` lanes load one
    expert each, so every expert id is fetched from HBM exactly once.  The
    lanes exchange those ids with ds_bpermute instead of reloading the whole
    top-k list for every logical route.
    """
    destination = expert // fx.Int32(fz_epr)
    local_expert = expert - destination * fx.Int32(fz_epr)
    selected_mask, canonical = _load_fanout_pair(
        addr_pair_config,
        destination,
        parity,
        npes=fz_npes,
        fanout_masks=fanout_masks,
        runtime_fanout=runtime_fanout,
    )

    group_lane = lane & fx.Int32(~7)
    token_mask = fx.Int64(0)
    member_slots = fx.Int32(0)
    for slot in range_constexpr(fz_k):
        source_lane = group_lane + fx.Int32(slot)
        peer_expert = fx.Int32(
            fx.rocdl.ds_bpermute(T.i32, source_lane * fx.Int32(4), expert)
        )
        valid = (peer_expert >= fx.Int32(0)) & (
            peer_expert < fx.Int32(fz_total_experts)
        )
        same_destination = peer_expert // fx.Int32(fz_epr) == destination
        peer_local = peer_expert - destination * fx.Int32(fz_epr)
        safe_local = (valid & same_destination).select(peer_local, fx.Int32(0))
        peer_bit = fx.Int64(1) << fx.Int64(safe_local)
        token_mask = (valid & same_destination).select(
            token_mask | peer_bit, token_mask
        )
        selected_member = valid & same_destination
        selected_member = selected_member & (
            ((selected_mask >> fx.Int64(safe_local)) & fx.Int64(1)) != fx.Int64(0)
        )
        member_slots = selected_member.select(
            member_slots | fx.Int32(1 << slot), member_slots
        )

    selected = selected_mask != fx.Int64(0)
    matched = selected & ((token_mask & selected_mask) == selected_mask)
    member_bit = (selected_mask >> fx.Int64(local_expert)) & fx.Int64(1)
    shared_member = matched & (member_bit != fx.Int64(0))
    emit = (~shared_member) | (local_expert == canonical)
    shared_segment = fx.Int32(fz_total_experts) + destination
    segment = shared_member.select(shared_segment, expert)
    return segment, emit, member_slots


@flyc.jit
def _configure_payload_geometry(
    addr_local_hist,
    addr_chunk_counts,
    addr_block_counts,
    lane,
    *,
    fz_npes,
    fz_epr,
    fz_total_experts,
    fanout_enabled,
    payload_chunk_rows,
    dispatch_blocks,
):
    crfa = buffer_ops.create_buffer_resource_from_addr
    local_hist = crfa(addr_local_hist)
    chunk_counts = crfa(addr_chunk_counts)
    block_counts = crfa(addr_block_counts)
    max_blocks = fx.Int32(dispatch_blocks // fz_npes)
    for destination in range_constexpr(fz_npes):
        max_source_count = fx.Int32(0)
        for local_expert in range(lane, fz_epr, 64):
            ge = fx.Int32(destination * fz_epr) + local_expert
            source_count = buffer_ops.buffer_load(
                local_hist, ge, vec_width=1, dtype=fx.Int32
            )
            max_source_count = (source_count > max_source_count).select(
                source_count, max_source_count
            )
        max_source_count = _wave_reduce_max_i32(max_source_count, lane)
        if const_expr(fanout_enabled):
            group_count = fx.Int32(0)
            if lane == fx.Int32(0):
                group_count = buffer_ops.buffer_load(
                    local_hist,
                    fx.Int32(fz_total_experts + destination),
                    vec_width=1,
                    dtype=fx.Int32,
                )
            group_count = fx.Int32(fx.rocdl.readfirstlane(T.i32, group_count))
            max_source_count = (group_count > max_source_count).select(
                group_count, max_source_count
            )
        if lane == fx.Int32(0):
            chunks = (max_source_count + fx.Int32(payload_chunk_rows - 1)) // fx.Int32(
                payload_chunk_rows
            )
            chunks = (chunks > fx.Int32(0)).select(chunks, fx.Int32(1))
            buffer_ops.buffer_store(chunks, chunk_counts, fx.Int32(destination))
            # Tasks are flattened as chunk x expert. Even one chunk contains
            # enough independent expert tasks to keep every producer useful.
            buffer_ops.buffer_store(max_blocks, block_counts, fx.Int32(destination))


@flyc.jit
def _store_expert_metadata(
    addr_sorted_expert,
    addr_tile_row_base,
    addr_tile_input_base,
    addr_srcmap,
    ge,
    local_row_base,
    input_row_base,
    total_count,
    num_tiles,
    padded_rows,
    *,
    fz_tile_m,
    invalid_source,
):
    crfa = buffer_ops.create_buffer_resource_from_addr
    sorted_expert = crfa(addr_sorted_expert)
    tile_row_base = crfa(addr_tile_row_base)
    tile_input_base = crfa(addr_tile_input_base)
    srcmap = crfa(addr_srcmap)
    base_tile = local_row_base // fx.Int32(fz_tile_m)
    for tile in range(fx.Int32(0), num_tiles, 1):
        metadata_index = base_tile + tile
        buffer_ops.buffer_store(ge, sorted_expert, metadata_index)
        buffer_ops.buffer_store(
            local_row_base + tile * fx.Int32(fz_tile_m), tile_row_base, metadata_index
        )
        buffer_ops.buffer_store(
            input_row_base + tile * fx.Int32(fz_tile_m),
            tile_input_base,
            metadata_index,
        )
    padding = padded_rows - total_count
    for pad in range(fx.Int32(0), padding, 1):
        buffer_ops.buffer_store(
            fx.Int32(invalid_source), srcmap, local_row_base + total_count + pad
        )


@flyc.jit
def _initialize_section_ready(
    addr_tile_ready,
    addr_tile_expected,
    count_rsrc,
    count_column,
    row_base,
    num_tiles,
    *,
    fz_npes,
    count_stride,
    payload_chunk_rows,
    fz_tile_m,
):
    tile_expected = buffer_ops.create_buffer_resource_from_addr(addr_tile_expected)
    base_tile = row_base // fx.Int32(fz_tile_m)
    for tile in range(fx.Int32(0), num_tiles, 1):
        tile_index = base_tile + tile
        # Producers update TILE_READY with system-scope atomics and consumers
        # observe it with a system-scope wait.  Reset it in the same coherence
        # domain; an ordinary VMEM store can leave the atomic path observing
        # the previous layout when a runtime fanout pair changes tile bounds.
        comm_ops.store_i32_system(addr_tile_ready, tile_index, fx.Int32(0))
        buffer_ops.buffer_store(fx.Int32(1), tile_expected, tile_index)

    sender_prefix = fx.Int32(0)
    for source in range_constexpr(fz_npes):
        source_count = buffer_ops.buffer_load(
            count_rsrc,
            fx.Int32(source * count_stride) + count_column,
            vec_width=1,
            dtype=fx.Int32,
            cache_modifier=2,
        )
        source_active = source_count > fx.Int32(0)
        source_boundary = source_active & (sender_prefix > fx.Int32(0))
        source_boundary = source_boundary & (
            sender_prefix % fx.Int32(fz_tile_m) != fx.Int32(0)
        )
        if source_boundary:
            tile_index = base_tile + sender_prefix // fx.Int32(fz_tile_m)
            _increment_i32(tile_expected, tile_index)
        for chunk_offset in range(
            fx.Int32(payload_chunk_rows), source_count, payload_chunk_rows
        ):
            boundary = sender_prefix + chunk_offset
            if boundary % fx.Int32(fz_tile_m) != fx.Int32(0):
                tile_index = base_tile + boundary // fx.Int32(fz_tile_m)
                _increment_i32(tile_expected, tile_index)
        sender_prefix = sender_prefix + source_count


@flyc.jit
def _copy_token_row(source_rsrc, destination_rsrc, lane, *, fz_safe_end_i32, fz_n_i32):
    lane_offset = lane * fx.Int32(4)
    if const_expr(fz_safe_end_i32 > 0):
        for column in range(lane_offset, fz_safe_end_i32, 512):
            value0 = buffer_ops.buffer_load(
                source_rsrc, column, vec_width=4, dtype=fx.Int32
            )
            value1 = buffer_ops.buffer_load(
                source_rsrc, column + fx.Int32(256), vec_width=4, dtype=fx.Int32
            )
            buffer_ops.buffer_store(value0, destination_rsrc, column)
            buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
    if const_expr(fz_safe_end_i32 < fz_n_i32):
        for column in range(lane_offset + fz_safe_end_i32, fz_n_i32, 256):
            value = buffer_ops.buffer_load(
                source_rsrc, column, vec_width=4, dtype=fx.Int32
            )
            buffer_ops.buffer_store(value, destination_rsrc, column)


@flyc.jit
def _publish_tile_range(
    p_tile_ready,
    p_tile_expected,
    p_ready_tile_queue,
    p_ready_tile_epoch,
    p_ready_tile_tail,
    destination,
    destination_base,
    row_begin,
    row_end,
    rows_per_tile,
    payload_epoch,
    parity,
    *,
    ready_tile_queue,
    tile_state_stride,
):
    if row_end > row_begin:
        crfa = buffer_ops.create_buffer_resource_from_addr
        comm_ops.fence_system_release()
        remote_tile_ready = buffer_ops.buffer_load(
            crfa(p_tile_ready), destination, vec_width=1, dtype=fx.Int64
        )
        state_byte_offset = fx.Int64(parity) * fx.Int64(tile_state_stride) * fx.Int64(4)
        remote_tile_ready = remote_tile_ready + state_byte_offset
        if const_expr(ready_tile_queue):
            remote_tile_expected = buffer_ops.buffer_load(
                crfa(p_tile_expected), destination, vec_width=1, dtype=fx.Int64
            )
            remote_queue = buffer_ops.buffer_load(
                crfa(p_ready_tile_queue), destination, vec_width=1, dtype=fx.Int64
            )
            remote_queue_epoch = buffer_ops.buffer_load(
                crfa(p_ready_tile_epoch), destination, vec_width=1, dtype=fx.Int64
            )
            remote_queue_tail = buffer_ops.buffer_load(
                crfa(p_ready_tile_tail), destination, vec_width=1, dtype=fx.Int64
            )
            remote_tile_expected = remote_tile_expected + state_byte_offset
            remote_queue = remote_queue + state_byte_offset
            remote_queue_epoch = remote_queue_epoch + state_byte_offset
            remote_queue_tail = remote_queue_tail + fx.Int64(parity) * fx.Int64(4)
        first_tile = (destination_base + row_begin) // rows_per_tile
        last_tile = (destination_base + row_end - fx.Int32(1)) // rows_per_tile
        for tile in range(first_tile, last_tile + fx.Int32(1), 1):
            previous = fx.Int32(
                comm_ops.atomic_add_system(
                    remote_tile_ready + fx.Int64(tile) * fx.Int64(4), fx.Int32(1)
                )
            )
            if const_expr(ready_tile_queue):
                expected = buffer_ops.buffer_load(
                    crfa(remote_tile_expected), tile, vec_width=1, dtype=fx.Int32
                )
                if previous + fx.Int32(1) == expected:
                    # The final RMW observes the release sequence from all
                    # payload publishers for this tile.  Acquire it before
                    # publishing the completion-order queue entry.
                    comm_ops.fence_system_acquire()
                    ready_slot = fx.Int32(
                        comm_ops.atomic_add_system(remote_queue_tail, fx.Int32(1))
                    )
                    buffer_ops.buffer_store(tile, crfa(remote_queue), ready_slot)
                    fx.rocdl.s_waitcnt(0)
                    comm_ops.fence_system_release()
                    comm_ops.store_i32_system(
                        remote_queue_epoch, ready_slot, payload_epoch
                    )


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
    a_tib = dp(DispatchSlot.TILE_INPUT_BASE)
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
            comm_ops.wait_i32_until_equals(a_source_done + fx.Int64(done_index) * fx.Int64(4), expected)
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
                    buffer_ops.buffer_store(
                        payload_base + tile * fx.Int32(fz_tile_m),
                        crfa(a_tib),
                        metadata_index,
                    )
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


@flyc.jit
def _derive_allgather_offsets(
    addr_disp,
    parity,
    *,
    rank,
    npes,
    epr,
    total_experts,
    total_segments,
    fanout_masks,
    runtime_fanout,
):
    """Derive this source's remote row offsets from the gathered histogram."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(slot):
        return buffer_ops.buffer_load(
            rdisp, fx.Int32(int(slot)), vec_width=1, dtype=fx.Int64
        )

    r_count = crfa(dp(DispatchSlot.COUNT_MATRIX))
    r_task_base = crfa(dp(DispatchSlot.TASK_ROW_BASE))
    r_group_base = crfa(dp(DispatchSlot.GROUP_TASK_BASE))
    r_ready_rows_table = crfa(dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS))
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    fanout_enabled = bool(fanout_masks) or runtime_fanout
    tid = fx.thread_idx.x
    warp = tid >> fx.Int32(6)
    lane = tid & fx.Int32(63)
    if warp < fx.Int32(npes):
        destination = warp
        destination_tile_m_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            destination_ready_rows = buffer_ops.buffer_load(
                r_ready_rows_table,
                destination,
                vec_width=1,
                dtype=fx.Int64,
            )
            destination_tile_m_lane = buffer_ops.buffer_load(
                crfa(destination_ready_rows),
                fx.Int32(0),
                vec_width=1,
                dtype=fx.Int32,
            )
        fx.rocdl.s_waitcnt(0)
        destination_tile_m = fx.Int32(
            fx.rocdl.readfirstlane(T.i32, destination_tile_m_lane)
        )
        selected_mask, _ = _load_fanout_pair(
            addr_pair_config,
            destination,
            parity,
            npes=npes,
            fanout_masks=fanout_masks,
            runtime_fanout=runtime_fanout,
        )
        destination_group_count = fx.Int32(0)
        destination_group_source_prefix = fx.Int32(0)
        if const_expr(fanout_enabled):
            destination_group_counts = []
            for source in range_constexpr(npes):
                destination_group_counts.append(
                    buffer_ops.buffer_load(
                        r_count,
                        fx.Int32(source * total_segments + total_experts)
                        + destination,
                        vec_width=1,
                        dtype=fx.Int32,
                        mask=lane == fx.Int32(0),
                        cache_modifier=2,
                    )
                )
            fx.rocdl.s_waitcnt(0)
            group_count_lane = fx.Int32(0)
            group_source_prefix_lane = fx.Int32(0)
            for source in range_constexpr(npes):
                source_group_count = destination_group_counts[source]
                if const_expr(source < rank):
                    group_source_prefix_lane = (
                        group_source_prefix_lane + source_group_count
                    )
                group_count_lane = group_count_lane + source_group_count
            destination_group_count = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, group_count_lane)
            )
            destination_group_source_prefix = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, group_source_prefix_lane)
            )
        row_carry = fx.Int32(0)
        for expert_chunk in range_constexpr((epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            ge = destination * fx.Int32(epr) + safe_expert
            group_member = valid_expert & (
                ((selected_mask >> fx.Int64(safe_expert)) & fx.Int64(1))
                != fx.Int64(0)
            )
            normal_source_counts = []
            for source in range_constexpr(npes):
                normal_source_counts.append(
                    buffer_ops.buffer_load(
                        r_count,
                        fx.Int32(source * total_segments) + ge,
                        vec_width=1,
                        dtype=fx.Int32,
                        cache_modifier=2,
                    )
                )
            fx.rocdl.s_waitcnt(0)
            normal_count = fx.Int32(0)
            normal_source_prefix = fx.Int32(0)
            group_count = group_member.select(
                destination_group_count, fx.Int32(0)
            )
            group_source_prefix = group_member.select(
                destination_group_source_prefix, fx.Int32(0)
            )
            for source in range_constexpr(npes):
                source_count = normal_source_counts[source]
                source_count = valid_expert.select(source_count, fx.Int32(0))
                if const_expr(source < rank):
                    normal_source_prefix = normal_source_prefix + source_count
                normal_count = normal_count + source_count
            group_rows = (
                (group_count + destination_tile_m - fx.Int32(1))
                // destination_tile_m
            ) * destination_tile_m
            normal_rows = (
                (normal_count + destination_tile_m - fx.Int32(1))
                // destination_tile_m
            ) * destination_tile_m
            padded_rows = group_rows + normal_rows
            inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
            local_row_base = row_carry + inclusive_rows - padded_rows
            if valid_expert:
                buffer_ops.buffer_store(
                    local_row_base + group_rows + normal_source_prefix,
                    r_task_base,
                    ge,
                )
                if group_member:
                    buffer_ops.buffer_store(
                        local_row_base + group_source_prefix,
                        r_group_base,
                        ge,
                    )
            last_lane = min(63, epr - expert_chunk * 64 - 1)
            row_carry = row_carry + fx.Int32(
                fx.rocdl.readlane(T.i32, inclusive_rows, last_lane)
            )


@flyc.jit
def _derive_next_fanout_pairs(
    addr_count_matrix,
    addr_pair_config,
    parity,
    lane,
    *,
    npes,
    epr,
    total_experts,
    total_segments,
):
    """Select next-launch pairs from the already-gathered expert histogram.

    Classified common rows are restored into both selected experts before the
    top-two reduction, so the reconstructed marginal histogram is exact.
    Every rank owns the same gathered matrix and therefore makes the same
    deterministic choice without another communication phase.
    """
    counts = buffer_ops.create_buffer_resource_from_addr(addr_count_matrix)
    next_parity = parity ^ fx.Int32(1)
    score_stride = fx.Int32(epr + 1)
    for destination in range_constexpr(npes):
        current_mask, _ = _load_fanout_pair(
            addr_pair_config,
            fx.Int32(destination),
            parity,
            npes=npes,
            fanout_masks=(),
            runtime_fanout=True,
        )
        valid_expert = lane < fx.Int32(epr)
        safe_expert = valid_expert.select(lane, fx.Int32(0))
        ge = fx.Int32(destination * epr) + safe_expert
        normal_count = fx.Int32(0)
        for source in range_constexpr(npes):
            source_count = buffer_ops.buffer_load(
                counts,
                fx.Int32(source * total_segments) + ge,
                vec_width=1,
                dtype=fx.Int32,
                cache_modifier=2,
            )
            normal_count = normal_count + valid_expert.select(
                source_count, fx.Int32(0)
            )
        group_count_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            for source in range_constexpr(npes):
                group_count_lane = group_count_lane + buffer_ops.buffer_load(
                    counts,
                    fx.Int32(source * total_segments + total_experts + destination),
                    vec_width=1,
                    dtype=fx.Int32,
                    cache_modifier=2,
                )
        group_count = fx.Int32(
            fx.rocdl.readfirstlane(T.i32, group_count_lane)
        )
        current_member = (
            (current_mask >> fx.Int64(safe_expert)) & fx.Int64(1)
        ) != fx.Int64(0)
        total_count = normal_count + (valid_expert & current_member).select(
            group_count, fx.Int32(0)
        )
        score = valid_expert.select(
            total_count * score_stride + fx.Int32(epr) - safe_expert,
            fx.Int32(-1),
        )
        best_score = _wave_reduce_max_i32(score, lane)
        best_expert = fx.Int32(epr) - (best_score % score_stride)
        second_score = (safe_expert != best_expert).select(
            score, fx.Int32(-1)
        )
        second_score = _wave_reduce_max_i32(second_score, lane)
        second_expert = fx.Int32(epr) - (second_score % score_stride)
        second_count = second_score // score_stride
        pair_a = (best_expert < second_expert).select(best_expert, second_expert)
        pair_b = (best_expert < second_expert).select(second_expert, best_expert)
        enabled = (
            (second_count > fx.Int32(0))
            & (pair_a >= fx.Int32(0))
            & (pair_b < fx.Int32(epr))
            & (pair_a != pair_b)
        )
        packed = pair_a | (pair_b << fx.Int32(8)) | enabled.select(
            fx.Int32(1 << 16), fx.Int32(0)
        )
        if lane == fx.Int32(0):
            comm_ops.store_i32_system(
                addr_pair_config,
                next_parity * fx.Int32(npes) + fx.Int32(destination),
                packed,
            )
    fx.rocdl.s_waitcnt(0)
    comm_ops.fence_agent_release()


# fmt: off
@flyc.jit
def emit_dispatch_plan(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_mtpr, fz_rank, fz_tile_m, fz_total_experts, addr_disp,
    i32_cur_tok, addr_in_idx, parity, expected,
    dispatch_blocks, group_blocks, group_done_slot, group_phase_base, payload_chunk_rows=0, payload_tile_ready=False, fanout_masks=(),
    tile_state_stride=0,
    runtime_fanout=False,
    dynamic_fanout=False,
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
    a_tib = dp(DispatchSlot.TILE_INPUT_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_block_hist = dp(DispatchSlot.BLOCK_HIST)
    a_bc = dp(DispatchSlot.COUNT_MATRIX)
    p_bc = dp(DispatchSlot.P2P_COUNT_MATRIX)
    a_cd = dp(DispatchSlot.COUNT_DONE)
    p_cd = dp(DispatchSlot.P2P_COUNT_DONE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_pair_order_ready = dp(DispatchSlot.PAIR_ORDER_READY)
    a_expert_tile_end = dp(DispatchSlot.EXPERT_TILE_END)
    a_group_done = dp(DispatchSlot.GROUP_DONE) + fx.Int64(group_done_slot * 4)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)
    a_tile_ready = dp(DispatchSlot.TILE_READY)
    a_tile_expected = dp(DispatchSlot.TILE_EXPECTED)
    if const_expr(payload_tile_ready):
        assert tile_state_stride > 0
        tile_state_byte_offset = (
            fx.Int64(parity)
            * fx.Int64(tile_state_stride)
            * fx.Int64(4)
        )
        a_tile_ready = a_tile_ready + tile_state_byte_offset
        a_tile_expected = a_tile_expected + tile_state_byte_offset
    a_payload_blocks_per_destination = dp(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
    a_payload_chunks_per_destination = dp(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    block_threads = num_waves * 64

    gtid = tid
    gnt = fx.Int32(block_threads)
    fanout_enabled = bool(fanout_masks) or runtime_fanout
    total_segments = fz_total_experts + (fz_npes if fanout_enabled else 0)
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    r_lh = crfa(a_lh)
    r_block_hist = crfa(a_block_hist)
    r_bc = crfa(a_bc)
    r_pair_base = crfa(a_pair_base)
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(
            a_group_done,
            group_phase_base + fx.Int32(group_blocks),
        )
        comm_ops.fence_agent_acquire()
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    comm_ops.fence_agent_acquire()

    # Every prepare worker publishes a complete per-segment histogram.  The
    # owner reduces those rows with ordinary loads/stores, so counting and the
    # later pair-order fill need no global per-route atomics.
    for segment in range(tid, total_segments, block_threads):
        segment_count = fx.Int32(0)
        for group_block in range_constexpr(group_blocks):
            segment_count = segment_count + buffer_ops.buffer_load(
                r_block_hist,
                fx.Int32(group_block * total_segments) + segment,
                vec_width=1,
                dtype=fx.Int32,
            )
        buffer_ops.buffer_store(segment_count, r_lh, segment)
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    if const_expr(payload_tile_ready):
        if warp == fx.Int32(0):
            _configure_payload_geometry(
                a_lh,
                a_payload_chunks_per_destination,
                a_payload_blocks_per_destination,
                lane,
                fz_npes=fz_npes,
                fz_epr=fz_epr,
                fz_total_experts=fz_total_experts,
                fanout_enabled=fanout_enabled,
                payload_chunk_rows=payload_chunk_rows,
                dispatch_blocks=dispatch_blocks,
            )
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        comm_ops.fence_agent_release()

    # Exchange route counts once.  The full-histogram path makes the offset
    # calculation deterministic on every source and removes remote base
    # push-back stores from the destination planner.
    count_stride = total_segments
    for destination in range_constexpr(fz_npes):
        remote_bigcnt = buffer_ops.buffer_load(
            crfa(p_bc), destination, vec_width=1, dtype=fx.Int64
        )
        for segment in range(gtid, total_segments, gnt):
            count = buffer_ops.buffer_load(
                r_lh, segment, vec_width=1, dtype=fx.Int32
            )
            buffer_ops.buffer_store(
                count,
                crfa(remote_bigcnt),
                fx.Int32(fz_rank * total_segments) + segment,
            )
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    # Warp 0 plans local experts after all source matrices arrive.
    if warp == fx.Int32(0):
        comm_ops.fence_system_release()
        for peer in range(lane, fz_npes, 64):
            remote_done = buffer_ops.buffer_load(crfa(p_cd), peer, vec_width=1, dtype=fx.Int64)
            comm_ops.atomic_add_system(
                remote_done + fx.Int64(parity) * fx.Int64(4),
                fx.Int32(1),
            )
        if lane == fx.Int32(0):
            comm_ops.wait_i32_until_equals(
                a_cd + fx.Int64(parity) * fx.Int64(4), expected
            )
        comm_ops.fence_system_acquire()

        if const_expr(runtime_fanout and dynamic_fanout):
            _derive_next_fanout_pairs(
                a_bc,
                addr_pair_config,
                parity,
                lane,
                npes=fz_npes,
                epr=fz_epr,
                total_experts=fz_total_experts,
                total_segments=total_segments,
            )

        r_nv = crfa(a_nv)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        local_fanout_mask, canonical_expert = _load_fanout_pair(
            addr_pair_config,
            fx.Int32(fz_rank),
            parity,
            npes=fz_npes,
            fanout_masks=fanout_masks,
            runtime_fanout=runtime_fanout,
        )
        for expert_chunk in range_constexpr((fz_epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(fz_epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            ge = fx.Int32(fz_rank * fz_epr + local_expert)
            safe_ge = fx.Int32(fz_rank * fz_epr) + safe_expert
            normal_source_counts = []
            normal_count = fx.Int32(0)
            group_source_counts = []
            group_count = fx.Int32(0)
            group_member = fx.Int32(0) == fx.Int32(1)
            if const_expr(fanout_enabled):
                group_member = valid_expert & (
                    (
                        (local_fanout_mask >> fx.Int64(safe_expert))
                        & fx.Int64(1)
                    )
                    != fx.Int64(0)
                )
            for source in range_constexpr(fz_npes):
                source_count = buffer_ops.buffer_load(
                    r_bc,
                    fx.Int32(source * count_stride) + safe_ge,
                    vec_width=1,
                    dtype=fx.Int32,
                    cache_modifier=2,
                )
                source_count = valid_expert.select(source_count, fx.Int32(0))
                normal_source_counts.append(source_count)
                normal_count = normal_count + source_count
                if const_expr(fanout_enabled):
                    source_group_count = buffer_ops.buffer_load(
                        r_bc,
                        fx.Int32(source * count_stride + fz_total_experts + fz_rank),
                        vec_width=1,
                        dtype=fx.Int32,
                        cache_modifier=2,
                    )
                    source_group_count = group_member.select(
                        source_group_count, fx.Int32(0)
                    )
                else:
                    source_group_count = fx.Int32(0)
                group_source_counts.append(source_group_count)
                group_count = group_count + source_group_count

            group_num_tiles = (
                group_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            normal_num_tiles = (
                normal_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            num_tiles = group_num_tiles + normal_num_tiles
            chunk_max = _wave_reduce_max_i32(num_tiles, lane)
            max_expert_tiles = (chunk_max > max_expert_tiles).select(
                chunk_max, max_expert_tiles
            )
            group_padded_rows = group_num_tiles * fx.Int32(fz_tile_m)
            normal_padded_rows = normal_num_tiles * fx.Int32(fz_tile_m)
            padded_rows = group_padded_rows + normal_padded_rows
            inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
            local_row_base = row_carry + inclusive_rows - padded_rows
            group_row_base = local_row_base
            normal_row_base = local_row_base + group_padded_rows
            group_input_base = fx.Int32(
                fx.rocdl.readlane(T.i32, group_row_base, canonical_expert)
            )

            if valid_expert:
                if const_expr(payload_tile_ready):
                    if group_member:
                        _initialize_section_ready(
                            a_tile_ready,
                            a_tile_expected,
                            r_bc,
                            fx.Int32(fz_total_experts + fz_rank),
                            group_row_base,
                            group_num_tiles,
                            fz_npes=fz_npes,
                            count_stride=count_stride,
                            payload_chunk_rows=payload_chunk_rows,
                            fz_tile_m=fz_tile_m,
                        )
                    _initialize_section_ready(
                        a_tile_ready,
                        a_tile_expected,
                        r_bc,
                        safe_ge,
                        normal_row_base,
                        normal_num_tiles,
                        fz_npes=fz_npes,
                        count_stride=count_stride,
                        payload_chunk_rows=payload_chunk_rows,
                        fz_tile_m=fz_tile_m,
                    )
                buffer_ops.buffer_store(
                    (local_row_base + padded_rows) // fx.Int32(fz_tile_m),
                    crfa(a_expert_tile_end),
                    local_expert,
                )
                if group_member:
                    _store_expert_metadata(
                        a_se,
                        a_trb,
                        a_tib,
                        a_sm,
                        ge,
                        group_row_base,
                        group_input_base,
                        group_count,
                        group_num_tiles,
                        group_padded_rows,
                        fz_tile_m=fz_tile_m,
                        invalid_source=fz_npes * fz_mtpr,
                    )
                _store_expert_metadata(
                    a_se,
                    a_trb,
                    a_tib,
                    a_sm,
                    ge,
                    normal_row_base,
                    normal_row_base,
                    normal_count,
                    normal_num_tiles,
                    normal_padded_rows,
                    fz_tile_m=fz_tile_m,
                    invalid_source=fz_npes * fz_mtpr,
                )

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
        pairs_per_lane = (total_segments + 63) // 64
        lane_base = lane * fx.Int32(pairs_per_lane)
        lane_total = fx.Int32(0)
        lane_counts = []
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(total_segments)
            safe_ge = valid_ge.select(ge, fx.Int32(0))
            source_count = buffer_ops.buffer_load(r_lh, safe_ge, vec_width=1, dtype=fx.Int32)
            source_count = valid_ge.select(source_count, fx.Int32(0))
            lane_counts.append(source_count)
            lane_total = lane_total + source_count
        lane_prefix = _wave_inclusive_scan_i32(lane_total, lane) - lane_total
        source_prefix = lane_prefix
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(total_segments)
            if valid_ge:
                buffer_ops.buffer_store(source_prefix, r_pair_base, ge)
                block_prefix = source_prefix
                for group_block in range_constexpr(group_blocks):
                    block_index = (
                        fx.Int32(group_block * total_segments) + ge
                    )
                    block_count = buffer_ops.buffer_load(
                        r_block_hist,
                        block_index,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                    buffer_ops.buffer_store(
                        block_prefix, r_block_hist, block_index
                    )
                    block_prefix = block_prefix + block_count
            source_prefix = source_prefix + lane_counts[item]
        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_agent_release()

    # All compact offsets are derived locally from the exchanged histogram.
    # The prepare kernel owns this single synchronization edge; MegaStage1
    # never recounts or regroups routes.
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    # Warp 0 observes COUNT_DONE and performs the system acquire above, but
    # every warp participates in the deterministic offset derivation below.
    # Give those warps their own system acquire after the CTA barrier so their
    # COUNT_MATRIX loads cannot reuse lines fetched before the remote peers'
    # histogram stores became visible.
    comm_ops.fence_system_acquire()
    _derive_allgather_offsets(
        addr_disp,
        parity,
        rank=fz_rank,
        npes=fz_npes,
        epr=fz_epr,
        total_experts=fz_total_experts,
        total_segments=total_segments,
        fanout_masks=fanout_masks,
        runtime_fanout=runtime_fanout,
    )
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_system_release()
        comm_ops.store_i32_system(a_pair_ready, parity, expected)

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(
            a_group_done,
            group_phase_base + fx.Int32(group_blocks * 2),
        )
        comm_ops.fence_agent_acquire()
        comm_ops.store_i32_system(a_pair_order_ready, parity, expected)


# fmt: off
@flyc.jit
def emit_dispatch_group(
    *, num_waves, fz_npes, fz_k, fz_epr, fz_total_experts, addr_disp, i32_cur_tok, addr_in_idx,
    dispatch_blocks, group_done_slot, producer_slot, parity, expected, fanout_masks=(),
    runtime_fanout=False,
    count_scratch=None,
):
# fmt: on
    """Count and group disjoint route spans across payload producer CTAs."""
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(int(i)), vec_width=1, dtype=fx.Int64)

    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_block_hist = dp(DispatchSlot.BLOCK_HIST)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_group_done = dp(DispatchSlot.GROUP_DONE) + fx.Int64(group_done_slot * 4)
    a_route_segment = dp(DispatchSlot.ROUTE_SEGMENT)
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    r_idx = crfa(addr_in_idx)
    r_block_hist = crfa(a_block_hist)
    r_pair = crfa(a_pair_order)
    r_route_segment = crfa(a_route_segment)
    tid = fx.thread_idx.x
    block_threads = fx.Int32(num_waves * 64)
    group_tid = producer_slot * block_threads + tid
    group_threads = fx.Int32(dispatch_blocks) * block_threads
    route_limit = i32_cur_tok * fx.Int32(fz_k)
    fanout_enabled = bool(fanout_masks) or runtime_fanout
    aggregate_counting = 3 if fanout_enabled else 1
    count_segments = fz_total_experts + (fz_npes if fanout_enabled else 0)
    assert count_scratch is not None
    def record_count(segment):
        scratch_addr = fx.Int64(fx.ptrtoint(count_scratch))
        return fx.Int32(
            comm_ops.atomic_add_workgroup(
                scratch_addr + fx.Int64(segment) * fx.Int64(4),
                fx.Int32(1),
            )
        )

    for segment in range(tid, count_segments, block_threads):
        fx.ptr_store(fx.Int32(0), count_scratch + fx.Int64(segment))
    fx.barrier()

    if const_expr(fanout_enabled):
        assert fz_k == 6, "wave-grouped fanout classification requires topk=6"
        lane = tid & fx.Int32(63)
        warp = tid >> fx.Int32(6)
        wave_id = producer_slot * fx.Int32(num_waves) + warp
        token_batch0 = wave_id * fx.Int32(8)
        token_stride = fx.Int32(dispatch_blocks * num_waves * 8)
        topk_slot = lane & fx.Int32(7)
        active_slot = topk_slot < fx.Int32(fz_k)
        for token_batch in range(token_batch0, i32_cur_tok, token_stride):
            token = token_batch + (lane >> fx.Int32(3))
            active_route = active_slot & (token < i32_cur_tok)
            safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
            safe_slot = active_slot.select(topk_slot, fx.Int32(0))
            route = safe_token * fx.Int32(fz_k) + safe_slot
            expert = buffer_ops.buffer_load(
                r_idx, route, vec_width=1, dtype=fx.Int32
            )
            valid = active_route & (expert >= fx.Int32(0))
            valid = valid & (expert < fx.Int32(fz_total_experts))
            safe_expert = valid.select(expert, fx.Int32(0))
            segment, emit, member_slots = _classify_fanout_wave_route(
                safe_expert,
                lane,
                addr_pair_config,
                parity,
                fz_k=fz_k,
                fz_epr=fz_epr,
                fz_total_experts=fz_total_experts,
                fz_npes=fz_npes,
                fanout_masks=fanout_masks,
                runtime_fanout=runtime_fanout,
            )
            cached_segment = segment | (member_slots << fx.Int32(16))
            cached_segment = (valid & emit).select(
                cached_segment, fx.Int32(-1)
            )
            if valid & emit:
                intra_rank = record_count(segment)
                if const_expr(aggregate_counting == 3):
                    cached_segment = (
                        segment
                        | (member_slots << fx.Int32(9))
                        | (intra_rank << fx.Int32(15))
                    )
            if active_route:
                buffer_ops.buffer_store(cached_segment, r_route_segment, route)
    else:
        for route in range(group_tid, route_limit, group_threads):
            expert = buffer_ops.buffer_load(
                r_idx, route, vec_width=1, dtype=fx.Int32
            )
            valid = (expert >= fx.Int32(0)) & (
                expert < fx.Int32(fz_total_experts)
            )
            safe_expert = valid.select(expert, fx.Int32(0))
            intra_block = fx.Int32(0)
            if valid:
                intra_block = record_count(safe_expert)
            cached_segment = valid.select(
                safe_expert | (intra_block << fx.Int32(9)),
                fx.Int32(-1),
            )
            buffer_ops.buffer_store(cached_segment, r_route_segment, route)

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    for segment in range(tid, count_segments, block_threads):
        block_count = fx.ptr_load(count_scratch + fx.Int64(segment))
        block_index = producer_slot * fx.Int32(count_segments) + segment
        buffer_ops.buffer_store(block_count, r_block_hist, block_index)
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
        comm_ops.wait_i32_until_equals(
            a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected
        )
        comm_ops.fence_agent_acquire()
    fx.barrier()

    fill_lane = tid & fx.Int32(63)
    fill_warp = tid >> fx.Int32(6)
    fill_wave_id = producer_slot * fx.Int32(num_waves) + fill_warp
    fill_token_batch0 = fill_wave_id * fx.Int32(8)
    fill_token_stride = fx.Int32(dispatch_blocks * num_waves * 8)
    fill_topk_slot = fill_lane & fx.Int32(7)
    fill_active_slot = fill_topk_slot < fx.Int32(fz_k)
    for segment in range(tid, count_segments, block_threads):
        block_index = producer_slot * fx.Int32(count_segments) + segment
        block_base = buffer_ops.buffer_load(
            r_block_hist,
            block_index,
            vec_width=1,
            dtype=fx.Int32,
        )
        fx.ptr_store(block_base, count_scratch + fx.Int64(segment))
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if const_expr(aggregate_counting == 3):
        assert fanout_enabled
        for token_batch in range(
            fill_token_batch0, i32_cur_tok, fill_token_stride
        ):
            token = token_batch + (fill_lane >> fx.Int32(3))
            active_route = fill_active_slot & (token < i32_cur_tok)
            safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
            safe_slot = fill_active_slot.select(fill_topk_slot, fx.Int32(0))
            route = safe_token * fx.Int32(fz_k) + safe_slot
            packed_segment = buffer_ops.buffer_load(
                r_route_segment, route, vec_width=1, dtype=fx.Int32
            )
            emit = active_route & (packed_segment >= fx.Int32(0))
            segment = packed_segment & fx.Int32(0x1FF)
            member_slots = (packed_segment >> fx.Int32(9)) & fx.Int32(0x3F)
            intra_rank = (packed_segment >> fx.Int32(15)) & fx.Int32(0xFFFF)
            if emit:
                block_base = fx.ptr_load(count_scratch + fx.Int64(segment))
                position = block_base + intra_rank
                shared_group = segment >= fx.Int32(fz_total_experts)
                group_entry = token | (member_slots << fx.Int32(24))
                pair_entry = shared_group.select(group_entry, route)
                buffer_ops.buffer_store(pair_entry, r_pair, position)
    else:
        for route in range(group_tid, route_limit, group_threads):
            packed_segment = buffer_ops.buffer_load(
                r_route_segment, route, vec_width=1, dtype=fx.Int32
            )
            emit = packed_segment >= fx.Int32(0)
            segment = packed_segment & fx.Int32(0x1FF)
            intra_block = packed_segment >> fx.Int32(9)
            if emit:
                block_base = fx.ptr_load(
                    count_scratch + fx.Int64(segment)
                )
                buffer_ops.buffer_store(
                    route,
                    r_pair,
                    block_base + intra_block,
                )

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    if tid == fx.Int32(0):
        comm_ops.fence_agent_release()
        comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
    fx.barrier()


# fmt: off
@flyc.jit
def emit_dispatch_payload(
    *, num_waves, fz_epr, fz_k, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32, fz_safe_end_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    dispatch_blocks,
    producer_slot, parity, expected, producers_per_destination, chunks_per_destination,
    payload_chunk_rows=0,
    payload_tile_ready=False,
    ready_tile_queue=False,
    tile_state_stride=0,
    indexed_payload=False,
    fanout_masks=(),
    runtime_fanout=False,
):
# fmt: on
    """Produce independently publishable expert payloads from a compact plan."""
    if const_expr(payload_tile_ready):
        assert tile_state_stride > 0
    crfa = buffer_ops.create_buffer_resource_from_addr
    rdisp = crfa(addr_disp)

    def dp(i):
        return buffer_ops.buffer_load(rdisp, fx.Int32(i), vec_width=1, dtype=fx.Int64)

    fanout_enabled = bool(fanout_masks) or runtime_fanout
    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_mb = dp(DispatchSlot.TASK_ROW_BASE)
    a_gb = fx.Int64(0)
    if const_expr(fanout_enabled):
        a_gb = dp(DispatchSlot.GROUP_TASK_BASE)
    p_payload_ready = dp(DispatchSlot.P2P_PAYLOAD_READY)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_chunk_done = dp(DispatchSlot.PAYLOAD_CHUNK_DONE)
    p_tile_ready = fx.Int64(0)
    p_payload_ready_rows = fx.Int64(0)
    if const_expr(payload_tile_ready):
        p_tile_ready = dp(DispatchSlot.P2P_TILE_READY)
        p_payload_ready_rows = dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS)
    p_tile_expected = fx.Int64(0)
    p_ready_tile_queue = fx.Int64(0)
    p_ready_tile_epoch = fx.Int64(0)
    p_ready_tile_tail = fx.Int64(0)
    if const_expr(ready_tile_queue):
        p_tile_expected = dp(DispatchSlot.P2P_TILE_EXPECTED)
        p_ready_tile_queue = dp(DispatchSlot.P2P_READY_TILE_QUEUE)
        p_ready_tile_epoch = dp(DispatchSlot.P2P_READY_TILE_EPOCH)
        p_ready_tile_tail = dp(DispatchSlot.P2P_READY_TILE_TAIL)
    addr_pair_config = fx.Int64(0)
    if const_expr(fanout_enabled):
        addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    r_pair_base = crfa(a_pair_base)
    r_lh = crfa(a_lh)
    r_mb = crfa(a_mb)
    r_gb = crfa(a_gb)
    r_pair = crfa(a_pair_order)
    r_idx = crfa(fx.Int64(0))
    if const_expr(fanout_enabled):
        r_idx = crfa(addr_in_idx)
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
    segments_per_destination = fz_epr + (1 if fanout_enabled else 0)
    payload_epoch = fx.Int32(0)
    if const_expr(ready_tile_queue):
        payload_epoch = (expected // fx.Int32(num_destinations)) * fx.Int32(2) - parity
    if const_expr(payload_chunk_rows > 0):
        assert dispatch_blocks % num_destinations == 0
        task_limit = fx.Int32(segments_per_destination) * chunks_per_destination
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
        comm_ops.wait_i32_until_equals(a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected)
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
            chunk_id = task_index // fx.Int32(segments_per_destination)
            rotated_segment = task_index - chunk_id * fx.Int32(
                segments_per_destination
            )
            rotation = (chunk_id * fx.Int32(17)) % fx.Int32(
                segments_per_destination
            )
            local_segment = (
                rotated_segment
                + fx.Int32(segments_per_destination)
                - rotation
            ) % fx.Int32(segments_per_destination)
        else:
            chunk_id = fx.Int32(0)
            local_segment = task_index // fx.Int32(num_destinations)
        destination = producer_destination
        selected_mask = fx.Int64(0)
        group_task = fx.Int32(0) != fx.Int32(0)
        local_expert = local_segment
        if const_expr(fanout_enabled):
            selected_mask, canonical_expert = _load_fanout_pair(
                addr_pair_config,
                destination,
                parity,
                npes=num_destinations,
                fanout_masks=fanout_masks,
                runtime_fanout=runtime_fanout,
            )
            group_task = local_segment == fx.Int32(fz_epr)
            local_expert = group_task.select(canonical_expert, local_segment)
        ge = destination * fx.Int32(fz_epr) + local_expert
        segment = ge
        if const_expr(fanout_enabled):
            segment = group_task.select(
                fx.Int32(fz_total_experts) + destination, ge
            )
        source_count_lane = fx.Int32(0)
        source_base_lane = fx.Int32(0)
        destination_base_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            source_count_lane = buffer_ops.buffer_load(
                r_lh, segment, vec_width=1, dtype=fx.Int32
            )
            source_base_lane = buffer_ops.buffer_load(
                r_pair_base, segment, vec_width=1, dtype=fx.Int32
            )
            destination_base_lane = buffer_ops.buffer_load(
                r_mb, ge, vec_width=1, dtype=fx.Int32
            )
            if const_expr(fanout_enabled):
                destination_base_lane = group_task.select(
                    buffer_ops.buffer_load(r_gb, ge, vec_width=1, dtype=fx.Int32),
                    destination_base_lane,
                )
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
                scale_remote_rsrc = crfa(
                    buffer_ops.buffer_load(crfa(p_sc), destination, vec_width=1, dtype=fx.Int64)
                )
        for row in range(row_begin + row0, row_end, row_stride):
            wk_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                wk_lane = buffer_ops.buffer_load(r_pair, source_base + row, vec_width=1, dtype=fx.Int32)
            wk = fx.Int32(fx.rocdl.readfirstlane(T.i32, wk_lane))
            source_token = wk // fx.Int32(fz_k)
            topk_slot = wk % fx.Int32(fz_k)
            group_member_slots = fx.Int32(0)
            if const_expr(fanout_enabled):
                group_member_slots = (wk >> fx.Int32(24)) & fx.Int32(0xFF)
                source_token = group_task.select(
                    wk & fx.Int32(0xFFFFFF), source_token
                )
                topk_slot = group_task.select(fx.Int32(0), topk_slot)
            destination_row = destination_base + row
            source_key = fx.Int32(fz_rank * fz_mtpr) + source_token

            def _copy_route_header(route_index, route_row):
                weight = buffer_ops.buffer_load(
                    r_wts, route_index, vec_width=1, dtype=fx.Float32
                )
                route_slot = topk_slot
                if const_expr(fanout_enabled):
                    route_slot = route_index % fx.Int32(fz_k)
                source_encoding = source_key | (route_slot << fx.Int32(24))
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                if const_expr(hoist_remote_resources):
                    buffer_ops.buffer_store(weight_bits, wts_remote_rsrc, route_row)
                    buffer_ops.buffer_store(source_encoding, srcmap_remote_rsrc, route_row)
                else:
                    wts_remote = buffer_ops.buffer_load(crfa(p_wts), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(weight_bits, crfa(wts_remote), route_row)
                    srcmap_remote = buffer_ops.buffer_load(crfa(p_sm), destination, vec_width=1, dtype=fx.Int64)
                    buffer_ops.buffer_store(source_encoding, crfa(srcmap_remote), route_row)

            if lane == fx.Int32(0):
                if const_expr(fanout_enabled):
                    if group_task:
                        for slot in range_constexpr(fz_k):
                            member_active = (
                                group_member_slots >> fx.Int32(slot)
                            ) & fx.Int32(1)
                            if member_active != fx.Int32(0):
                                member_route = (
                                    source_token * fx.Int32(fz_k) + fx.Int32(slot)
                                )
                                member_ge = buffer_ops.buffer_load(
                                    r_idx,
                                    member_route,
                                    vec_width=1,
                                    dtype=fx.Int32,
                                )
                                member_base = buffer_ops.buffer_load(
                                    r_gb, member_ge, vec_width=1, dtype=fx.Int32
                                )
                                _copy_route_header(member_route, member_base + row)
                    else:
                        _copy_route_header(wk, destination_row)
                else:
                    _copy_route_header(wk, destination_row)

            payload_row = destination_row
            if const_expr(indexed_payload):
                payload_row = source_key
            if const_expr(fz_enable_scales):
                scale_lane = lane
                if const_expr(fz_scale_n_i32 % 4 == 0):
                    scale_offset = scale_lane * fx.Int32(4)
                    if scale_offset < fx.Int32(fz_scale_n_i32):
                        scale = buffer_ops.buffer_load(
                            crfa(addr_in_sc),
                            source_token * fx.Int32(fz_scale_n_i32) + scale_offset,
                            vec_width=4,
                            dtype=fx.Int32,
                        )
                        if const_expr(hoist_remote_resources):
                            buffer_ops.buffer_store(
                                scale,
                                scale_remote_rsrc,
                                payload_row * fx.Int32(fz_scale_n_i32) + scale_offset,
                            )
                        else:
                            row_scale_remote = crfa(
                                buffer_ops.buffer_load(
                                    crfa(p_sc), destination, vec_width=1, dtype=fx.Int64
                                )
                            )
                            buffer_ops.buffer_store(
                                scale,
                                row_scale_remote,
                                payload_row * fx.Int32(fz_scale_n_i32) + scale_offset,
                            )
                elif scale_lane < fx.Int32(fz_scale_n_i32):
                    scale = buffer_ops.buffer_load(
                        crfa(addr_in_sc),
                        source_token * fx.Int32(fz_scale_n_i32) + scale_lane,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                    if const_expr(hoist_remote_resources):
                        buffer_ops.buffer_store(
                            scale,
                            scale_remote_rsrc,
                            payload_row * fx.Int32(fz_scale_n_i32) + scale_lane,
                        )
                    else:
                        row_scale_remote = crfa(
                            buffer_ops.buffer_load(
                                crfa(p_sc), destination, vec_width=1, dtype=fx.Int64
                            )
                        )
                        buffer_ops.buffer_store(
                            scale,
                            row_scale_remote,
                            payload_row * fx.Int32(fz_scale_n_i32) + scale_lane,
                        )
            source_rsrc = crfa(addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes))
            if const_expr(hoist_remote_resources):
                destination_rsrc = crfa(
                    token_remote + fx.Int64(payload_row) * fx.Int64(fz_nbytes)
                )
            else:
                row_token_remote = buffer_ops.buffer_load(
                    crfa(p_rx), destination, vec_width=1, dtype=fx.Int64
                )
                destination_rsrc = crfa(
                    row_token_remote + fx.Int64(payload_row) * fx.Int64(fz_nbytes)
                )
            _copy_token_row(
                source_rsrc,
                destination_rsrc,
                lane,
                fz_safe_end_i32=fz_safe_end_i32,
                fz_n_i32=fz_n_i32,
            )

        if chunk_active:
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
            if tid == fx.Int32(0):
                if const_expr(payload_tile_ready):
                    if const_expr(fanout_enabled):
                        if group_task:
                            for member in range_constexpr(fz_epr):
                                member_bit = (
                                    selected_mask >> fx.Int64(member)
                                ) & fx.Int64(1)
                                if member_bit != fx.Int64(0):
                                    member_ge = destination * fx.Int32(
                                        fz_epr
                                    ) + fx.Int32(member)
                                    member_base = buffer_ops.buffer_load(
                                        r_gb,
                                        member_ge,
                                        vec_width=1,
                                        dtype=fx.Int32,
                                    )
                                    _publish_tile_range(
                                        p_tile_ready,
                                        p_tile_expected,
                                        p_ready_tile_queue,
                                        p_ready_tile_epoch,
                                        p_ready_tile_tail,
                                        destination,
                                        member_base,
                                        row_begin,
                                        row_end,
                                        destination_ready_rows,
                                        payload_epoch,
                                        parity,
                                        ready_tile_queue=ready_tile_queue,
                                        tile_state_stride=tile_state_stride,
                                    )
                        else:
                            _publish_tile_range(
                                p_tile_ready,
                                p_tile_expected,
                                p_ready_tile_queue,
                                p_ready_tile_epoch,
                                p_ready_tile_tail,
                                destination,
                                destination_base,
                                row_begin,
                                row_end,
                                destination_ready_rows,
                                payload_epoch,
                                parity,
                                ready_tile_queue=ready_tile_queue,
                                tile_state_stride=tile_state_stride,
                            )
                    else:
                        _publish_tile_range(
                            p_tile_ready,
                            p_tile_expected,
                            p_ready_tile_queue,
                            p_ready_tile_epoch,
                            p_ready_tile_tail,
                            destination,
                            destination_base,
                            row_begin,
                            row_end,
                            destination_ready_rows,
                            payload_epoch,
                            parity,
                            ready_tile_queue=ready_tile_queue,
                            tile_state_stride=tile_state_stride,
                        )
                _finish_task(destination, local_expert, segment, num_chunks)
            fx.barrier()
