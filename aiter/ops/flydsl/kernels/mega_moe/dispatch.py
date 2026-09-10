# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""Compact dispatch path for MegaMoE v2 stage1."""

from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor


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
    PAIR_ORDER = 15
    P2P_PLAN_READY = 16
    PLAN_READY = 17
    PAIR_READY = 18
    ENTRY_COUNT = 19
    EPOCH_GATE = 20
    PAIR_ORDER_READY = 21
    WORK_HEAD = 22
    WORK_TAIL = 23
    EXPERT_TILE_END = 24
    GROUP_DONE = 25
    RUNNING = 26
    P2P_RUNNING = 27
    LAUNCH_READY = 28
    P2P_LAUNCH_READY = 29
    MAX_EXPERT_TILES = 30
    PAYLOAD_CHUNK_DONE = 31
    TILE_READY = 32
    P2P_TILE_READY = 33
    TILE_EXPECTED = 34
    PAYLOAD_READY_ROWS = 35
    P2P_PAYLOAD_READY_ROWS = 36
    PAYLOAD_BLOCKS_PER_DESTINATION = 37
    PAYLOAD_CHUNKS_PER_DESTINATION = 38
    TILE_INPUT_BASE = 39
    GROUP_TASK_BASE = 40
    ROUTE_SEGMENT = 41
    PREP_ENTRY_COUNT = 42
    PREP_EPOCH_GATE = 43
    READY_TILE_QUEUE = 44
    P2P_READY_TILE_QUEUE = 45
    READY_TILE_EPOCH = 46
    P2P_READY_TILE_EPOCH = 47
    READY_TILE_TAIL = 48
    P2P_READY_TILE_TAIL = 49
    P2P_TILE_EXPECTED = 50
    FANOUT_PAIR_CONFIG = 51
    BLOCK_HIST = 52


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


@flyc.jit
def _load_fanout_pair(
    addr_pair_config,
    destination,
    parity,
    *,
    npes,
):
    """Return the selected expert pair for one destination.

    The runtime table is double-buffered by the Stage1 parity.  Each entry is
    ``lo | hi << 8 | enabled << 16``.  Keeping expert identities out of the
    compile key lets one AOT artifact serve every routing distribution.  Keep
    the pair as ids instead of an i64 bitmap so a destination may own more than
    64 experts (Kimi-K3 EP8 owns 112).
    """
    packed = comm_ops.load_i32_system(
        addr_pair_config,
        parity * fx.Int32(npes) + destination,
    )
    # ``packed`` controls route classification, destination offsets, and the
    # producer task mapping. Keep the VMEM dependency explicit before those
    # values cross lane/control-flow boundaries.
    fx.rocdl.s_waitcnt(0)
    enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
    pair_a = packed & fx.Int32(0xFF)
    pair_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
    pair_mask = (fx.Int64(1) << fx.Int64(pair_a)) | (fx.Int64(1) << fx.Int64(pair_b))
    selected_mask = enabled.select(pair_mask, fx.Int64(0))
    canonical = enabled.select(pair_a, fx.Int32(0))
    return selected_mask, pair_a, pair_b, enabled, canonical


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
def _increment_i32(buffer, index):
    buffer[index] = buffer[index] + fx.Int32(1)


@flyc.jit
def _classify_fanout_wave_route(
    expert,
    route_expert,
    lane,
    addr_pair_config,
    parity,
    *,
    fz_k,
    fz_epr,
    fz_total_experts,
    fz_npes,
):
    """Classify one route after a wave has loaded grouped top-k routes.

    Top-k up to eight uses the historical eight-lane group.  Top-k 9..16 uses
    sixteen lanes, keeping every route load coalesced and compile-time
    specialized.  The selected fanout pair is represented by expert ids, not
    a 64-bit mask, so local expert ids above 63 remain valid.
    """
    destination = expert // fx.Int32(fz_epr)
    local_expert = expert - destination * fx.Int32(fz_epr)
    selected_mask, pair_a, pair_b, pair_enabled, canonical = _load_fanout_pair(
        addr_pair_config,
        destination,
        parity,
        npes=fz_npes,
    )

    assert 0 < fz_k <= 16, "wave-grouped fanout classification supports topk <= 16"
    member_slots = fx.Int32(0)
    if const_expr(fz_epr <= 64 and fz_k <= 8):
        # Preserve the established DSV4-Pro instructions: its Stage1/Stage2
        # overlap is sensitive even to small changes in compact prepare.
        group_lane = lane & fx.Int32(~7)
        token_mask = fx.Int64(0)
        for slot in range_constexpr(fz_k):
            source_lane = group_lane + fx.Int32(slot)
            peer_expert = fx.Int32(
                fx.rocdl.ds_bpermute(T.i32, source_lane * fx.Int32(4), route_expert)
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
    else:
        route_group_width = 8 if fz_k <= 8 else 16
        group_lane = lane & fx.Int32(~(route_group_width - 1))
        slot_a = fx.Int32(0)
        slot_b = fx.Int32(0)
        has_a = fx.Int32(0) != fx.Int32(0)
        has_b = fx.Int32(0) != fx.Int32(0)
        for slot in range_constexpr(fz_k):
            source_lane = group_lane + fx.Int32(slot)
            peer_expert = fx.Int32(
                fx.rocdl.ds_bpermute(T.i32, source_lane * fx.Int32(4), route_expert)
            )
            valid = (peer_expert >= fx.Int32(0)) & (
                peer_expert < fx.Int32(fz_total_experts)
            )
            same_destination = peer_expert // fx.Int32(fz_epr) == destination
            peer_local = peer_expert - destination * fx.Int32(fz_epr)
            safe_local = (valid & same_destination).select(peer_local, fx.Int32(0))
            member_a = valid & same_destination & pair_enabled & (safe_local == pair_a)
            member_b = valid & same_destination & pair_enabled & (safe_local == pair_b)
            has_a = has_a | member_a
            has_b = has_b | member_b
            if const_expr(fz_k <= 8):
                selected_member = member_a | member_b
                member_slots = selected_member.select(
                    member_slots | fx.Int32(1 << slot), member_slots
                )
            slot_a = member_a.select(fx.Int32(slot), slot_a)
            slot_b = member_b.select(fx.Int32(slot), slot_b)
        matched = pair_enabled & has_a & has_b
        shared_member = matched & ((local_expert == pair_a) | (local_expert == pair_b))
        if const_expr(fz_k > 8):
            # Two four-bit slots fit topk <= 16 in the existing 32-bit entry.
            member_slots = slot_a | (slot_b << fx.Int32(4))
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
    payload_chunk_rows,
    dispatch_blocks,
):
    local_hist = ptr_buf_tensor(addr_local_hist, fx.Int32)
    chunk_counts = ptr_buf_tensor(addr_chunk_counts, fx.Int32)
    block_counts = ptr_buf_tensor(addr_block_counts, fx.Int32)
    max_blocks = fx.Int32(dispatch_blocks // fz_npes)
    for destination in range_constexpr(fz_npes):
        max_source_count = fx.Int32(0)
        for local_expert in range(lane, fz_epr, 64):
            ge = fx.Int32(destination * fz_epr) + local_expert
            source_count = local_hist[ge]
            max_source_count = (source_count > max_source_count).select(
                source_count, max_source_count
            )
        max_source_count = _wave_reduce_max_i32(max_source_count, lane)
        group_count = fx.Int32(0)
        if lane == fx.Int32(0):
            group_count = local_hist[fx.Int32(fz_total_experts + destination)]
        group_count = fx.Int32(fx.rocdl.readfirstlane(T.i32, group_count))
        max_source_count = (group_count > max_source_count).select(
            group_count, max_source_count
        )
        if lane == fx.Int32(0):
            chunks = (max_source_count + fx.Int32(payload_chunk_rows - 1)) // fx.Int32(
                payload_chunk_rows
            )
            chunks = (chunks > fx.Int32(0)).select(chunks, fx.Int32(1))
            chunk_counts[fx.Int32(destination)] = chunks
            # Tasks are flattened as chunk x expert. Even one chunk contains
            # enough independent expert tasks to keep every producer useful.
            block_counts[fx.Int32(destination)] = max_blocks


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
    sorted_expert = ptr_buf_tensor(addr_sorted_expert, fx.Int32)
    tile_row_base = ptr_buf_tensor(addr_tile_row_base, fx.Int32)
    tile_input_base = ptr_buf_tensor(addr_tile_input_base, fx.Int32)
    srcmap = ptr_buf_tensor(addr_srcmap, fx.Int32)
    base_tile = local_row_base // fx.Int32(fz_tile_m)
    for tile in range(fx.Int32(0), num_tiles, 1):
        metadata_index = base_tile + tile
        sorted_expert[metadata_index] = ge
        tile_row_base[metadata_index] = local_row_base + tile * fx.Int32(fz_tile_m)
        tile_input_base[metadata_index] = input_row_base + tile * fx.Int32(fz_tile_m)
    padding = padded_rows - total_count
    for pad in range(fx.Int32(0), padding, 1):
        srcmap[local_row_base + total_count + pad] = fx.Int32(invalid_source)


@flyc.jit
def _initialize_section_ready(
    addr_tile_ready,
    addr_tile_expected,
    count_buffer,
    count_column,
    row_base,
    num_tiles,
    *,
    fz_npes,
    count_stride,
    payload_chunk_rows,
    fz_tile_m,
):
    tile_expected = ptr_buf_tensor(addr_tile_expected, fx.Int32)
    base_tile = row_base // fx.Int32(fz_tile_m)
    for tile in range(fx.Int32(0), num_tiles, 1):
        tile_index = base_tile + tile
        # Producers update TILE_READY with system-scope atomics and consumers
        # observe it with a system-scope wait.  Reset it in the same coherence
        # domain; an ordinary VMEM store can leave the atomic path observing
        # the previous layout when a runtime fanout pair changes tile bounds.
        comm_ops.store_i32_system(addr_tile_ready, tile_index, fx.Int32(0))
        tile_expected[tile_index] = fx.Int32(1)

    sender_prefix = fx.Int32(0)
    for source in range_constexpr(fz_npes):
        source_count = buf_copy_load(
            count_buffer,
            fx.Int32(source * count_stride) + count_column,
            fx.Int32,
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
def _copy_token_row(
    source_buffer, destination_buffer, lane, *, fz_safe_end_i32, fz_n_i32
):
    safe_end_vec = fz_safe_end_i32 // 4
    n_vec = fz_n_i32 // 4
    if const_expr(fz_safe_end_i32 > 0):
        for unit in range(lane, safe_end_vec, 128):
            value0 = buf_copy_load(source_buffer, unit, fx.Int32, unit_elems=4)
            value1 = buf_copy_load(
                source_buffer, unit + fx.Int32(64), fx.Int32, unit_elems=4
            )
            buf_copy_store(destination_buffer, unit, value0, fx.Int32, unit_elems=4)
            buf_copy_store(
                destination_buffer,
                unit + fx.Int32(64),
                value1,
                fx.Int32,
                unit_elems=4,
            )
    if const_expr(fz_safe_end_i32 < fz_n_i32):
        for unit in range(lane + safe_end_vec, n_vec, 64):
            value = buf_copy_load(source_buffer, unit, fx.Int32, unit_elems=4)
            buf_copy_store(destination_buffer, unit, value, fx.Int32, unit_elems=4)


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
    tile_state_stride,
):
    if row_end > row_begin:
        comm_ops.fence_system_release()
        remote_tile_ready = ptr_buf_tensor(p_tile_ready, fx.Int64)[destination]
        state_byte_offset = fx.Int64(parity) * fx.Int64(tile_state_stride) * fx.Int64(4)
        remote_tile_ready = remote_tile_ready + state_byte_offset
        remote_tile_expected = ptr_buf_tensor(p_tile_expected, fx.Int64)[destination]
        remote_queue = ptr_buf_tensor(p_ready_tile_queue, fx.Int64)[destination]
        remote_queue_epoch = ptr_buf_tensor(p_ready_tile_epoch, fx.Int64)[destination]
        remote_queue_tail = ptr_buf_tensor(p_ready_tile_tail, fx.Int64)[destination]
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
            expected = ptr_buf_tensor(remote_tile_expected, fx.Int32)[tile]
            if previous + fx.Int32(1) == expected:
                # The final RMW observes the release sequence from all payload
                # publishers before publishing the completion-order entry.
                comm_ops.fence_system_acquire()
                ready_slot = fx.Int32(
                    comm_ops.atomic_add_system(remote_queue_tail, fx.Int32(1))
                )
                ptr_buf_tensor(remote_queue, fx.Int32)[ready_slot] = tile
                fx.rocdl.s_waitcnt(0)
                comm_ops.fence_system_release()
                comm_ops.store_i32_system(remote_queue_epoch, ready_slot, payload_epoch)


# fmt: off
@flyc.jit
def emit_direct_fixed_slot_payload(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_cap, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    i32_cur_tok, dispatch_blocks, producer_slot, parity, expected,
):
# fmt: on
    """Allocate and publish routes directly into destination fixed slots."""
    dispatch_table = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(i):
        return dispatch_table[fx.Int32(int(i))]

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
    idx_buffer = ptr_buf_tensor(addr_in_idx, fx.Int32)
    weight_buffer = ptr_buf_tensor(addr_in_wts, fx.Float32)
    scale_buffer = ptr_buf_tensor(addr_in_sc, fx.Int32)
    token_table = ptr_buf_tensor(p_rx, fx.Int64)
    scale_table = ptr_buf_tensor(p_sc, fx.Int64)
    weight_table = ptr_buf_tensor(p_wts, fx.Int64)
    srcmap_table = ptr_buf_tensor(p_sm, fx.Int64)
    running_table = ptr_buf_tensor(p_running, fx.Int64)
    source_done_table = ptr_buf_tensor(p_source_done, fx.Int64)

    for wk in range(route, route_limit, route_stride):
        source_token = wk // fx.Int32(fz_k)
        topk_slot = wk - source_token * fx.Int32(fz_k)
        global_expert_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            global_expert_lane = idx_buffer[wk]
        global_expert = fx.Int32(fx.rocdl.readfirstlane(T.i32, global_expert_lane))
        valid_expert = (global_expert >= fx.Int32(0)) & (global_expert < fx.Int32(fz_total_experts))
        safe_expert = valid_expert.select(global_expert, fx.Int32(0))
        destination = safe_expert // fx.Int32(fz_epr)
        local_expert = safe_expert - destination * fx.Int32(fz_epr)
        offset_lane = fx.Int32(0)
        assigned = valid_expert & (destination % fx.Int32(destination_groups) == producer_group)
        if lane == fx.Int32(0):
            if assigned:
                remote_running = running_table[destination]
                offset_lane = fx.Int32(
                    comm_ops.atomic_add_system(
                        remote_running + fx.Int64(local_expert) * fx.Int64(4), fx.Int32(1)
                    )
                )
        expert_offset = fx.Int32(fx.rocdl.readlane(T.i32, offset_lane, 0))
        publish = assigned & (expert_offset < fx.Int32(fz_cap))
        payload_row = local_expert * fx.Int32(fz_cap) + expert_offset

        if publish:
            remote_token = token_table[destination]
            destination_buffer = ptr_buf_tensor(
                remote_token + fx.Int64(payload_row) * fx.Int64(fz_nbytes),
                fx.Int32,
                unit_elems=4,
            )
            source_buffer = ptr_buf_tensor(
                addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes),
                fx.Int32,
                unit_elems=4,
            )
            for unit in range(lane, fz_n_i32 // 4, 64):
                value = buf_copy_load(
                    source_buffer, unit, fx.Int32, unit_elems=4
                )
                buf_copy_store(
                    destination_buffer, unit, value, fx.Int32, unit_elems=4
                )

            if const_expr(fz_enable_scales):
                if lane < fx.Int32(fz_scale_n_i32):
                    scale = scale_buffer[
                        source_token * fx.Int32(fz_scale_n_i32) + lane
                    ]
                    remote_scale = scale_table[destination]
                    ptr_buf_tensor(remote_scale, fx.Int32)[
                        payload_row * fx.Int32(fz_scale_n_i32) + lane
                    ] = scale

            if lane == fx.Int32(0):
                weight = weight_buffer[wk]
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                remote_weights = weight_table[destination]
                remote_srcmap = srcmap_table[destination]
                ptr_buf_tensor(remote_weights, fx.Int32)[payload_row] = weight_bits
                ptr_buf_tensor(remote_srcmap, fx.Int32)[payload_row] = source_encoding

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
                    remote_done = source_done_table[fx.Int32(destination)]
                    comm_ops.store_i32_system(remote_done, done_index, expected)


@flyc.jit
def emit_direct_fixed_slot_finalize(
    *, fz_npes, fz_epr, fz_cap, fz_mtpr, fz_rank, fz_tile_m, n_tiles, addr_disp, parity, expected
):
    """Finalize local fixed slots as soon as every source publishes this destination."""
    assert 0 < fz_epr <= 64, "direct fixed-slot finalize requires 1..64 experts per rank"
    dispatch_table = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(i):
        return dispatch_table[fx.Int32(int(i))]

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
    sorted_expert = ptr_buf_tensor(a_se, fx.Int32)
    tile_row_base = ptr_buf_tensor(a_trb, fx.Int32)
    tile_input_base = ptr_buf_tensor(a_tib, fx.Int32)
    num_valid_buffer = ptr_buf_tensor(a_nv, fx.Int32)
    srcmap = ptr_buf_tensor(a_sm, fx.Int32)
    running = ptr_buf_tensor(a_running, fx.Int32)
    plan_ready_table = ptr_buf_tensor(p_plan_ready, fx.Int64)
    work_tail = ptr_buf_tensor(a_work_tail, fx.Int32)
    expert_tile_end = ptr_buf_tensor(a_expert_tile_end, fx.Int32)
    max_expert_tiles_buffer = ptr_buf_tensor(a_max_expert_tiles, fx.Int32)

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
        count = running[safe_expert]
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
                    sorted_expert[metadata_index] = global_expert
                    tile_row_base[metadata_index] = (
                        payload_base + tile * fx.Int32(fz_tile_m)
                    )
                    tile_input_base[metadata_index] = (
                        payload_base + tile * fx.Int32(fz_tile_m)
                    )
                padded_rows = num_expert_tiles * fx.Int32(fz_tile_m)
                for pad in range(fx.Int32(0), padded_rows - safe_count, 1):
                    srcmap[payload_base + safe_count + pad] = fx.Int32(
                        fz_npes * fz_mtpr
                    )
                expert_tile_end[safe_expert] = metadata_base + num_expert_tiles
            else:
                expert_tile_end[safe_expert] = fx.Int32(0)
            running[safe_expert] = fx.Int32(0)

        if lane == fx.Int32(0):
            num_valid = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            ready_work = no_overflow.select(total_tiles * fx.Int32(n_tiles), fx.Int32(0))
            num_valid_buffer[fx.Int32(0)] = num_valid
            # num_valid[1] is a device-visible overflow status.
            num_valid_buffer[fx.Int32(1)] = overflow_count
            work_tail[fx.Int32(0)] = ready_work
            max_expert_tiles_buffer[fx.Int32(0)] = max_expert_tiles

        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = plan_ready_table[source]
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
):
    """Derive this source's remote row offsets from the gathered histogram."""
    dispatch_table = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(slot):
        return dispatch_table[fx.Int32(int(slot))]

    count_buffer = ptr_buf_tensor(dp(DispatchSlot.COUNT_MATRIX), fx.Int32)
    task_base = ptr_buf_tensor(dp(DispatchSlot.TASK_ROW_BASE), fx.Int32)
    group_base = ptr_buf_tensor(dp(DispatchSlot.GROUP_TASK_BASE), fx.Int32)
    ready_rows_table = ptr_buf_tensor(
        dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS), fx.Int64
    )
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    tid = fx.thread_idx.x
    warp = tid >> fx.Int32(6)
    lane = tid & fx.Int32(63)
    if warp < fx.Int32(npes):
        destination = warp
        destination_tile_m_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            destination_ready_rows = ready_rows_table[destination]
            destination_tile_m_lane = ptr_buf_tensor(
                destination_ready_rows, fx.Int32
            )[fx.Int32(0)]
        fx.rocdl.s_waitcnt(0)
        destination_tile_m = fx.Int32(
            fx.rocdl.readfirstlane(T.i32, destination_tile_m_lane)
        )
        selected_mask, pair_a, pair_b, pair_enabled, _ = _load_fanout_pair(
            addr_pair_config,
            destination,
            parity,
            npes=npes,
        )
        destination_group_count = fx.Int32(0)
        destination_group_source_prefix = fx.Int32(0)
        destination_group_counts = []
        for source in range_constexpr(npes):
            destination_group_counts.append(
                buf_copy_load(
                    count_buffer,
                    fx.Int32(source * total_segments + total_experts)
                    + destination,
                    fx.Int32,
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
            if const_expr(epr <= 64):
                group_member = valid_expert & (
                    ((selected_mask >> fx.Int64(safe_expert)) & fx.Int64(1))
                    != fx.Int64(0)
                )
            else:
                group_member = valid_expert & pair_enabled & (
                    (safe_expert == pair_a) | (safe_expert == pair_b)
                )
            normal_source_counts = []
            for source in range_constexpr(npes):
                normal_source_counts.append(
                    buf_copy_load(
                        count_buffer,
                        fx.Int32(source * total_segments) + ge,
                        fx.Int32,
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
                task_base[ge] = local_row_base + group_rows + normal_source_prefix
                if group_member:
                    group_base[ge] = local_row_base + group_source_prefix
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
    counts = ptr_buf_tensor(addr_count_matrix, fx.Int32)
    next_parity = parity ^ fx.Int32(1)
    score_stride = fx.Int32(epr + 1)
    for destination in range_constexpr(npes):
        (
            current_mask,
            current_pair_a,
            current_pair_b,
            current_pair_enabled,
            _,
        ) = _load_fanout_pair(
            addr_pair_config,
            fx.Int32(destination),
            parity,
            npes=npes,
        )
        if const_expr(epr <= 64):
            valid_expert = lane < fx.Int32(epr)
            safe_expert = valid_expert.select(lane, fx.Int32(0))
            ge = fx.Int32(destination * epr) + safe_expert
            normal_count = fx.Int32(0)
            for source in range_constexpr(npes):
                source_count = buf_copy_load(
                    counts,
                    fx.Int32(source * total_segments) + ge,
                    fx.Int32,
                    cache_modifier=2,
                )
                normal_count = normal_count + valid_expert.select(
                    source_count, fx.Int32(0)
                )
            group_count_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                for source in range_constexpr(npes):
                    group_count_lane = group_count_lane + buf_copy_load(
                        counts,
                        fx.Int32(
                            source * total_segments + total_experts + destination
                        ),
                        fx.Int32,
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
        else:
            group_count_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                for source in range_constexpr(npes):
                    group_count_lane = group_count_lane + buf_copy_load(
                        counts,
                        fx.Int32(
                            source * total_segments + total_experts + destination
                        ),
                        fx.Int32,
                        cache_modifier=2,
                    )
            group_count = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, group_count_lane)
            )
            lane_best_score = fx.Int32(-1)
            lane_second_score = fx.Int32(-1)
            for expert_chunk in range_constexpr((epr + 63) // 64):
                local_expert = fx.Int32(expert_chunk * 64) + lane
                valid_expert = local_expert < fx.Int32(epr)
                safe_expert = valid_expert.select(local_expert, fx.Int32(0))
                ge = fx.Int32(destination * epr) + safe_expert
                normal_count = fx.Int32(0)
                for source in range_constexpr(npes):
                    source_count = buf_copy_load(
                        counts,
                        fx.Int32(source * total_segments) + ge,
                        fx.Int32,
                        cache_modifier=2,
                    )
                    normal_count = normal_count + valid_expert.select(
                        source_count, fx.Int32(0)
                    )
                current_member = current_pair_enabled & (
                    (safe_expert == current_pair_a)
                    | (safe_expert == current_pair_b)
                )
                total_count = normal_count + (valid_expert & current_member).select(
                    group_count, fx.Int32(0)
                )
                score = valid_expert.select(
                    total_count * score_stride + fx.Int32(epr) - safe_expert,
                    fx.Int32(-1),
                )
                new_best = score > lane_best_score
                lane_second_score = new_best.select(
                    lane_best_score,
                    (score > lane_second_score).select(score, lane_second_score),
                )
                lane_best_score = new_best.select(score, lane_best_score)

            best_score = _wave_reduce_max_i32(lane_best_score, lane)
            best_expert = fx.Int32(epr) - (best_score % score_stride)
            lane_second_candidate = (lane_best_score == best_score).select(
                lane_second_score, lane_best_score
            )
            second_score = _wave_reduce_max_i32(lane_second_candidate, lane)
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
    dispatch_blocks, group_blocks, group_done_slot, group_phase_base, payload_chunk_rows, tile_state_stride,
):
# fmt: on
    """Build a destination-owned compact plan in one producer-only CTA."""
    dispatch_table = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(i):
        return dispatch_table[fx.Int32(i)]

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
    assert payload_chunk_rows > 0 and tile_state_stride > 0
    tile_state_byte_offset = (
        fx.Int64(parity) * fx.Int64(tile_state_stride) * fx.Int64(4)
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
    total_segments = fz_total_experts + fz_npes
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    local_hist = ptr_buf_tensor(a_lh, fx.Int32)
    block_hist = ptr_buf_tensor(a_block_hist, fx.Int32)
    count_matrix = ptr_buf_tensor(a_bc, fx.Int32)
    pair_base = ptr_buf_tensor(a_pair_base, fx.Int32)
    count_matrix_table = ptr_buf_tensor(p_bc, fx.Int64)
    count_done_table = ptr_buf_tensor(p_cd, fx.Int64)
    plan_ready_table = ptr_buf_tensor(p_plan_ready, fx.Int64)
    expert_tile_end = ptr_buf_tensor(a_expert_tile_end, fx.Int32)
    max_expert_tiles_buffer = ptr_buf_tensor(a_max_expert_tiles, fx.Int32)
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
            segment_count = segment_count + block_hist[
                fx.Int32(group_block * total_segments) + segment
            ]
        local_hist[segment] = segment_count
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    if warp == fx.Int32(0):
        _configure_payload_geometry(
            a_lh,
            a_payload_chunks_per_destination,
            a_payload_blocks_per_destination,
            lane,
            fz_npes=fz_npes,
            fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
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
        remote_bigcnt = count_matrix_table[destination]
        remote_count_matrix = ptr_buf_tensor(remote_bigcnt, fx.Int32)
        for segment in range(gtid, total_segments, gnt):
            count = local_hist[segment]
            remote_count_matrix[fx.Int32(fz_rank * total_segments) + segment] = (
                count
            )
    fx.rocdl.s_waitcnt(0)
    fx.barrier()

    # Warp 0 plans local experts after all source matrices arrive.
    if warp == fx.Int32(0):
        comm_ops.fence_system_release()
        for peer in range(lane, fz_npes, 64):
            remote_done = count_done_table[peer]
            comm_ops.atomic_add_system(
                remote_done + fx.Int64(parity) * fx.Int64(4),
                fx.Int32(1),
            )
        if lane == fx.Int32(0):
            comm_ops.wait_i32_until_equals(
                a_cd + fx.Int64(parity) * fx.Int64(4), expected
            )
        comm_ops.fence_system_acquire()

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

        num_valid_buffer = ptr_buf_tensor(a_nv, fx.Int32)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        (
            local_fanout_mask,
            pair_a,
            pair_b,
            pair_enabled,
            canonical_expert,
        ) = _load_fanout_pair(
            addr_pair_config,
            fx.Int32(fz_rank),
            parity,
            npes=fz_npes,
        )
        pair_a_group_base = fx.Int32(0)
        pair_b_group_base = fx.Int32(0)
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
            if const_expr(fz_epr <= 64):
                group_member = valid_expert & (
                    (
                        (local_fanout_mask >> fx.Int64(safe_expert))
                        & fx.Int64(1)
                    )
                    != fx.Int64(0)
                )
            else:
                group_member = valid_expert & pair_enabled & (
                    (safe_expert == pair_a) | (safe_expert == pair_b)
                )
            for source in range_constexpr(fz_npes):
                source_count = buf_copy_load(
                    count_matrix,
                    fx.Int32(source * count_stride) + safe_ge,
                    fx.Int32,
                    cache_modifier=2,
                )
                source_count = valid_expert.select(source_count, fx.Int32(0))
                normal_source_counts.append(source_count)
                normal_count = normal_count + source_count
                source_group_count = buf_copy_load(
                    count_matrix,
                    fx.Int32(source * count_stride + fz_total_experts + fz_rank),
                    fx.Int32,
                    cache_modifier=2,
                )
                source_group_count = group_member.select(
                    source_group_count, fx.Int32(0)
                )
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
            if const_expr(fz_epr <= 64):
                group_input_base = fx.Int32(
                    fx.rocdl.readlane(T.i32, group_row_base, canonical_expert)
                )
            else:
                group_input_base = fx.Int32(0)
                pair_a_base_lane = (valid_expert & (local_expert == pair_a)).select(
                    group_row_base, fx.Int32(0)
                )
                pair_b_base_lane = (valid_expert & (local_expert == pair_b)).select(
                    group_row_base, fx.Int32(0)
                )
                pair_a_in_chunk = (pair_a >= fx.Int32(expert_chunk * 64)) & (
                    pair_a < fx.Int32(min(fz_epr, (expert_chunk + 1) * 64))
                )
                pair_b_in_chunk = (pair_b >= fx.Int32(expert_chunk * 64)) & (
                    pair_b < fx.Int32(min(fz_epr, (expert_chunk + 1) * 64))
                )
                pair_a_group_base = pair_a_in_chunk.select(
                    _wave_reduce_max_i32(pair_a_base_lane, lane), pair_a_group_base
                )
                pair_b_group_base = pair_b_in_chunk.select(
                    _wave_reduce_max_i32(pair_b_base_lane, lane), pair_b_group_base
                )

            if valid_expert:
                if group_member:
                    _initialize_section_ready(
                        a_tile_ready,
                        a_tile_expected,
                        count_matrix,
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
                    count_matrix,
                    safe_ge,
                    normal_row_base,
                    normal_num_tiles,
                    fz_npes=fz_npes,
                    count_stride=count_stride,
                    payload_chunk_rows=payload_chunk_rows,
                    fz_tile_m=fz_tile_m,
                )
                expert_tile_end[local_expert] = (
                    local_row_base + padded_rows
                ) // fx.Int32(fz_tile_m)
                if const_expr(fz_epr <= 64):
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

        if const_expr(fz_epr > 64):
            group_count_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                for source in range_constexpr(fz_npes):
                    group_count_lane = group_count_lane + buf_copy_load(
                        count_matrix,
                        fx.Int32(source * count_stride + fz_total_experts + fz_rank),
                        fx.Int32,
                        cache_modifier=2,
                    )
            group_count = fx.Int32(
                fx.rocdl.readfirstlane(T.i32, group_count_lane)
            )
            group_num_tiles = (
                group_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            group_padded_rows = group_num_tiles * fx.Int32(fz_tile_m)
            if (lane == fx.Int32(0)) & pair_enabled:
                pair_a_ge = fx.Int32(fz_rank * fz_epr) + pair_a
                pair_b_ge = fx.Int32(fz_rank * fz_epr) + pair_b
                _store_expert_metadata(
                    a_se,
                    a_trb,
                    a_tib,
                    a_sm,
                    pair_a_ge,
                    pair_a_group_base,
                    pair_a_group_base,
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
                    pair_b_ge,
                    pair_b_group_base,
                    pair_a_group_base,
                    group_count,
                    group_num_tiles,
                    group_padded_rows,
                    fz_tile_m=fz_tile_m,
                    invalid_source=fz_npes * fz_mtpr,
                )

        if lane == fx.Int32(0):
            num_valid_buffer[fx.Int32(0)] = row_carry
            max_expert_tiles_buffer[fx.Int32(0)] = max_expert_tiles
        fx.rocdl.s_waitcnt(0)
        comm_ops.fence_system_release()
        for source in range(lane, fz_npes, 64):
            remote_ready = plan_ready_table[source]
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
            source_count = local_hist[safe_ge]
            source_count = valid_ge.select(source_count, fx.Int32(0))
            lane_counts.append(source_count)
            lane_total = lane_total + source_count
        lane_prefix = _wave_inclusive_scan_i32(lane_total, lane) - lane_total
        source_prefix = lane_prefix
        for item in range_constexpr(pairs_per_lane):
            ge = lane_base + fx.Int32(item)
            valid_ge = ge < fx.Int32(total_segments)
            if valid_ge:
                pair_base[ge] = source_prefix
                block_prefix = source_prefix
                for group_block in range_constexpr(group_blocks):
                    block_index = (
                        fx.Int32(group_block * total_segments) + ge
                    )
                    block_count = block_hist[block_index]
                    block_hist[block_index] = block_prefix
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
    dispatch_blocks, group_done_slot, producer_slot, parity, expected,
    count_scratch=None,
):
# fmt: on
    """Count and group disjoint route spans across payload producer CTAs."""
    dispatch_table = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(i):
        return dispatch_table[fx.Int32(int(i))]

    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_block_hist = dp(DispatchSlot.BLOCK_HIST)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_group_done = dp(DispatchSlot.GROUP_DONE) + fx.Int64(group_done_slot * 4)
    a_route_segment = dp(DispatchSlot.ROUTE_SEGMENT)
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    idx_buffer = ptr_buf_tensor(addr_in_idx, fx.Int32)
    block_hist = ptr_buf_tensor(a_block_hist, fx.Int32)
    pair_order = ptr_buf_tensor(a_pair_order, fx.Int32)
    route_segment = ptr_buf_tensor(a_route_segment, fx.Int32)
    tid = fx.thread_idx.x
    block_threads = fx.Int32(num_waves * 64)
    count_segments = fz_total_experts + fz_npes
    assert count_segments <= 1024, "route metadata supports at most 1024 segments"
    packed_topk6_metadata = fz_k <= 6 and count_segments <= 512
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

    assert 0 < fz_k <= 16, "wave-grouped fanout requires topk <= 16"
    route_group_width = 8 if fz_k <= 8 else 16
    route_group_shift = 3 if route_group_width == 8 else 4
    tokens_per_wave = 64 // route_group_width
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    wave_id = producer_slot * fx.Int32(num_waves) + warp
    token_batch0 = wave_id * fx.Int32(tokens_per_wave)
    token_stride = fx.Int32(dispatch_blocks * num_waves * tokens_per_wave)
    topk_slot = lane & fx.Int32(route_group_width - 1)
    active_slot = topk_slot < fx.Int32(fz_k)
    for token_batch in range(token_batch0, i32_cur_tok, token_stride):
        token = token_batch + (lane >> fx.Int32(route_group_shift))
        active_route = active_slot & (token < i32_cur_tok)
        safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
        safe_slot = active_slot.select(topk_slot, fx.Int32(0))
        route = safe_token * fx.Int32(fz_k) + safe_slot
        expert = idx_buffer[route]
        valid = active_route & (expert >= fx.Int32(0))
        valid = valid & (expert < fx.Int32(fz_total_experts))
        safe_expert = valid.select(expert, fx.Int32(0))
        segment, emit, member_slots = _classify_fanout_wave_route(
            safe_expert,
            expert,
            lane,
            addr_pair_config,
            parity,
            fz_k=fz_k,
            fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            fz_npes=fz_npes,
        )
        cached_segment = fx.Int32(-1)
        if valid & emit:
            intra_rank = record_count(segment)
            if const_expr(packed_topk6_metadata):
                cached_segment = (
                    segment
                    | (member_slots << fx.Int32(9))
                    | (intra_rank << fx.Int32(15))
                )
            else:
                cached_segment = (
                    segment
                    | (member_slots << fx.Int32(10))
                    | (intra_rank << fx.Int32(18))
                )
        if active_route:
            route_segment[route] = cached_segment

    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    for segment in range(tid, count_segments, block_threads):
        block_count = fx.ptr_load(count_scratch + fx.Int64(segment))
        block_index = producer_slot * fx.Int32(count_segments) + segment
        block_hist[block_index] = block_count
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

    for segment in range(tid, count_segments, block_threads):
        block_index = producer_slot * fx.Int32(count_segments) + segment
        block_base = block_hist[block_index]
        fx.ptr_store(block_base, count_scratch + fx.Int64(segment))
    fx.rocdl.s_waitcnt(0)
    fx.barrier()
    for token_batch in range(
        token_batch0, i32_cur_tok, token_stride
    ):
        token = token_batch + (lane >> fx.Int32(route_group_shift))
        active_route = active_slot & (token < i32_cur_tok)
        safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
        safe_slot = active_slot.select(topk_slot, fx.Int32(0))
        route = safe_token * fx.Int32(fz_k) + safe_slot
        packed_segment = route_segment[route]
        if const_expr(packed_topk6_metadata):
            emit = active_route & (packed_segment >= fx.Int32(0))
            segment = packed_segment & fx.Int32(0x1FF)
            member_slots = (packed_segment >> fx.Int32(9)) & fx.Int32(0x3F)
            intra_rank = (packed_segment >> fx.Int32(15)) & fx.Int32(0xFFFF)
        else:
            emit = active_route & (packed_segment != fx.Int32(-1))
            segment = packed_segment & fx.Int32(0x3FF)
            member_slots = (packed_segment >> fx.Int32(10)) & fx.Int32(0xFF)
            intra_rank = (packed_segment >> fx.Int32(18)) & fx.Int32(0x3FFF)
        if emit:
            block_base = fx.ptr_load(count_scratch + fx.Int64(segment))
            position = block_base + intra_rank
            shared_group = segment >= fx.Int32(fz_total_experts)
            group_entry = token | (member_slots << fx.Int32(24))
            pair_entry = shared_group.select(group_entry, route)
            pair_order[position] = pair_entry

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
    payload_chunk_rows,
    tile_state_stride,
    indexed_payload=False,
):
# fmt: on
    """Produce independently publishable expert payloads from a compact plan."""
    assert payload_chunk_rows > 0 and tile_state_stride > 0
    dispatch_table = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(i):
        return dispatch_table[fx.Int32(i)]

    p_rx = dp(DispatchSlot.P2P_TOKEN)
    p_sc = dp(DispatchSlot.P2P_SCALE)
    p_wts = dp(DispatchSlot.P2P_WEIGHT)
    p_sm = dp(DispatchSlot.P2P_SRCMAP)
    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_mb = dp(DispatchSlot.TASK_ROW_BASE)
    a_gb = dp(DispatchSlot.GROUP_TASK_BASE)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_plan_ready = dp(DispatchSlot.PLAN_READY)
    a_chunk_done = dp(DispatchSlot.PAYLOAD_CHUNK_DONE)
    p_tile_ready = dp(DispatchSlot.P2P_TILE_READY)
    p_payload_ready_rows = dp(DispatchSlot.P2P_PAYLOAD_READY_ROWS)
    p_tile_expected = dp(DispatchSlot.P2P_TILE_EXPECTED)
    p_ready_tile_queue = dp(DispatchSlot.P2P_READY_TILE_QUEUE)
    p_ready_tile_epoch = dp(DispatchSlot.P2P_READY_TILE_EPOCH)
    p_ready_tile_tail = dp(DispatchSlot.P2P_READY_TILE_TAIL)
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    pair_base = ptr_buf_tensor(a_pair_base, fx.Int32)
    local_hist = ptr_buf_tensor(a_lh, fx.Int32)
    task_base = ptr_buf_tensor(a_mb, fx.Int32)
    group_base = ptr_buf_tensor(a_gb, fx.Int32)
    pair_order = ptr_buf_tensor(a_pair_order, fx.Int32)
    idx_buffer = ptr_buf_tensor(addr_in_idx, fx.Int32)
    weight_buffer = ptr_buf_tensor(addr_in_wts, fx.Float32)
    chunk_done = ptr_buf_tensor(a_chunk_done, fx.Int32)
    token_table = ptr_buf_tensor(p_rx, fx.Int64)
    scale_table = ptr_buf_tensor(p_sc, fx.Int64)
    weight_table = ptr_buf_tensor(p_wts, fx.Int64)
    srcmap_table = ptr_buf_tensor(p_sm, fx.Int64)
    payload_ready_rows_table = ptr_buf_tensor(p_payload_ready_rows, fx.Int64)
    source_scale_buffer = ptr_buf_tensor(addr_in_sc, fx.Int32)
    source_scale_vec_buffer = ptr_buf_tensor(
        addr_in_sc, fx.Int32, unit_elems=4
    )
    row0 = warp
    row_stride = fx.Int32(num_waves)

    def _finish_task(chunk_done_buffer, ge, num_chunks):
        comm_ops.fence_system_release()
        completed = fx.Int32(
            comm_ops.atomic_add_agent(
                a_chunk_done + fx.Int64(ge) * fx.Int64(4), fx.Int32(1)
            )
        )
        if completed == num_chunks - fx.Int32(1):
            comm_ops.fence_agent_acquire()
            chunk_done_buffer[ge] = fx.Int32(0)

    num_destinations = fz_total_experts // fz_epr
    segments_per_destination = fz_epr + 1
    payload_epoch = (expected // fx.Int32(num_destinations)) * fx.Int32(2) - parity
    assert dispatch_blocks % num_destinations == 0
    task_limit = fx.Int32(segments_per_destination) * chunks_per_destination
    task0 = producer_slot // fx.Int32(num_destinations)
    task_stride = fx.Int32(producers_per_destination)
    hoist_remote_resources = fz_mtpr >= 1024
    producer_destination = producer_slot % fx.Int32(num_destinations)
    ready_index = parity * fx.Int32(num_destinations) + producer_destination
    if tid == fx.Int32(0):
        comm_ops.wait_i32_until_equals(a_plan_ready + fx.Int64(ready_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()
    destination_ready_rows = fx.Int32(0)
    if tid == fx.Int32(0):
        remote_ready_rows = payload_ready_rows_table[producer_destination]
        destination_ready_rows = ptr_buf_tensor(remote_ready_rows, fx.Int32)[
            fx.Int32(0)
        ]
    fx.barrier()
    for task_index in range(task0, task_limit, task_stride):
        chunk_id = task_index // fx.Int32(segments_per_destination)
        rotated_segment = task_index - chunk_id * fx.Int32(
            segments_per_destination
        )
        rotation = (chunk_id * fx.Int32(17)) % fx.Int32(
            segments_per_destination
        )
        local_segment = (
            rotated_segment + fx.Int32(segments_per_destination) - rotation
        ) % fx.Int32(segments_per_destination)
        destination = producer_destination
        (
            selected_mask,
            pair_a,
            pair_b,
            pair_enabled,
            canonical_expert,
        ) = _load_fanout_pair(
            addr_pair_config,
            destination,
            parity,
            npes=num_destinations,
        )
        group_task = local_segment == fx.Int32(fz_epr)
        local_expert = group_task.select(canonical_expert, local_segment)
        ge = destination * fx.Int32(fz_epr) + local_expert
        segment = group_task.select(
            fx.Int32(fz_total_experts) + destination, ge
        )
        source_count_lane = fx.Int32(0)
        source_base_lane = fx.Int32(0)
        destination_base_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            source_count_lane = local_hist[segment]
            source_base_lane = pair_base[segment]
            destination_base_lane = task_base[ge]
            destination_base_lane = group_task.select(
                group_base[ge],
                destination_base_lane,
            )
        source_count = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_count_lane))
        source_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, source_base_lane))
        destination_base = fx.Int32(fx.rocdl.readfirstlane(T.i32, destination_base_lane))
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
        if const_expr(hoist_remote_resources):
            remote_weight_buffer = ptr_buf_tensor(
                weight_table[destination], fx.Int32
            )
            remote_srcmap_buffer = ptr_buf_tensor(
                srcmap_table[destination], fx.Int32
            )
            token_remote = token_table[destination]
            if const_expr(fz_enable_scales):
                remote_scale_buffer = ptr_buf_tensor(
                    scale_table[destination], fx.Int32
                )
                remote_scale_vec_buffer = ptr_buf_tensor(
                    scale_table[destination], fx.Int32, unit_elems=4
                )
        for row in range(row_begin + row0, row_end, row_stride):
            wk_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                wk_lane = pair_order[source_base + row]
            wk = fx.Int32(fx.rocdl.readfirstlane(T.i32, wk_lane))
            source_token = wk // fx.Int32(fz_k)
            topk_slot = wk % fx.Int32(fz_k)
            group_member_slots = (wk >> fx.Int32(24)) & fx.Int32(0xFF)
            source_token = group_task.select(
                wk & fx.Int32(0xFFFFFF), source_token
            )
            topk_slot = group_task.select(fx.Int32(0), topk_slot)
            destination_row = destination_base + row
            source_key = fx.Int32(fz_rank * fz_mtpr) + source_token

            def _copy_route_header(route_index, route_row):
                weight = weight_buffer[route_index]
                route_slot = route_index % fx.Int32(fz_k)
                source_encoding = source_key | (route_slot << fx.Int32(24))
                weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                if const_expr(hoist_remote_resources):
                    remote_weight_buffer[route_row] = weight_bits
                    remote_srcmap_buffer[route_row] = source_encoding
                else:
                    wts_remote = weight_table[destination]
                    ptr_buf_tensor(wts_remote, fx.Int32)[route_row] = weight_bits
                    srcmap_remote = srcmap_table[destination]
                    ptr_buf_tensor(srcmap_remote, fx.Int32)[route_row] = (
                        source_encoding
                    )

            if lane == fx.Int32(0):
                if group_task:
                    if const_expr(fz_k <= 8):
                        for slot in range_constexpr(fz_k):
                            member_active = (
                                group_member_slots >> fx.Int32(slot)
                            ) & fx.Int32(1)
                            if member_active != fx.Int32(0):
                                member_route = (
                                    source_token * fx.Int32(fz_k)
                                    + fx.Int32(slot)
                                )
                                member_ge = idx_buffer[member_route]
                                member_base = group_base[member_ge]
                                _copy_route_header(
                                    member_route, member_base + row
                                )
                    else:
                        slot_a = group_member_slots & fx.Int32(0xF)
                        slot_b = (group_member_slots >> fx.Int32(4)) & fx.Int32(
                            0xF
                        )
                        for member_slot in (slot_a, slot_b):
                            member_route = (
                                source_token * fx.Int32(fz_k) + member_slot
                            )
                            member_ge = idx_buffer[member_route]
                            member_base = group_base[member_ge]
                            _copy_route_header(member_route, member_base + row)
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
                        scale_unit = (
                            source_token * fx.Int32(fz_scale_n_i32) + scale_offset
                        ) // fx.Int32(4)
                        scale = buf_copy_load(
                            source_scale_vec_buffer,
                            scale_unit,
                            fx.Int32,
                            unit_elems=4,
                        )
                        if const_expr(hoist_remote_resources):
                            buf_copy_store(
                                remote_scale_vec_buffer,
                                (
                                    payload_row * fx.Int32(fz_scale_n_i32)
                                    + scale_offset
                                )
                                // fx.Int32(4),
                                scale,
                                fx.Int32,
                                unit_elems=4,
                            )
                        else:
                            row_scale_remote = ptr_buf_tensor(
                                scale_table[destination], fx.Int32, unit_elems=4
                            )
                            buf_copy_store(
                                row_scale_remote,
                                (
                                    payload_row * fx.Int32(fz_scale_n_i32)
                                    + scale_offset
                                )
                                // fx.Int32(4),
                                scale,
                                fx.Int32,
                                unit_elems=4,
                            )
                elif scale_lane < fx.Int32(fz_scale_n_i32):
                    scale = source_scale_buffer[
                        source_token * fx.Int32(fz_scale_n_i32) + scale_lane
                    ]
                    if const_expr(hoist_remote_resources):
                        remote_scale_buffer[
                            payload_row * fx.Int32(fz_scale_n_i32) + scale_lane
                        ] = scale
                    else:
                        row_scale_remote = ptr_buf_tensor(
                            scale_table[destination], fx.Int32
                        )
                        row_scale_remote[
                            payload_row * fx.Int32(fz_scale_n_i32) + scale_lane
                        ] = scale
            source_buffer = ptr_buf_tensor(
                addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes),
                fx.Int32,
                unit_elems=4,
            )
            if const_expr(hoist_remote_resources):
                destination_buffer = ptr_buf_tensor(
                    token_remote + fx.Int64(payload_row) * fx.Int64(fz_nbytes),
                    fx.Int32,
                    unit_elems=4,
                )
            else:
                row_token_remote = token_table[destination]
                destination_buffer = ptr_buf_tensor(
                    row_token_remote
                    + fx.Int64(payload_row) * fx.Int64(fz_nbytes),
                    fx.Int32,
                    unit_elems=4,
                )
            _copy_token_row(
                source_buffer,
                destination_buffer,
                lane,
                fz_safe_end_i32=fz_safe_end_i32,
                fz_n_i32=fz_n_i32,
            )

        if chunk_active:
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
            if tid == fx.Int32(0):
                if group_task:
                    if const_expr(fz_epr <= 64):
                        for member in range_constexpr(fz_epr):
                            member_bit = (
                                selected_mask >> fx.Int64(member)
                            ) & fx.Int64(1)
                            if member_bit != fx.Int64(0):
                                member_ge = destination * fx.Int32(fz_epr) + member
                                member_base = group_base[member_ge]
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
                                    tile_state_stride=tile_state_stride,
                                )
                    elif pair_enabled:
                        for member in (pair_a, pair_b):
                            member_ge = destination * fx.Int32(fz_epr) + member
                            member_base = group_base[member_ge]
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
                        tile_state_stride=tile_state_stride,
                    )
                _finish_task(chunk_done, segment, num_chunks)
            fx.barrier()
