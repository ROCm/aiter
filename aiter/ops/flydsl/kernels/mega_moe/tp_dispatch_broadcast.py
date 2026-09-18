# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""Naive TP copy of EP fixed-slot dispatch: send each published row to all ranks.

Slot allocation stays on the expert-owner rank (so finalize still unblocks).
The token/scale/header payload is then written to every peer at the same
``payload_row``. Compute is wrong: those rows collide with other ranks' local
experts. This is only a send-volume / fused-kernel occupancy experiment.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from .dispatch import DispatchSlot


# fmt: off
@flyc.jit
def emit_tp_broadcast_fixed_slot_payload(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_cap, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    i32_cur_tok, dispatch_blocks, producer_slot, parity, expected,
):
# fmt: on
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
    assert dispatch_blocks % destination_groups == 0
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
            source_rsrc = crfa(addr_in_tok + fx.Int64(source_token) * fx.Int64(fz_nbytes))
            for dest_i in range_constexpr(fz_npes):
                dest = fx.Int32(dest_i)
                remote_token = buffer_ops.buffer_load(crfa(p_rx), dest, vec_width=1, dtype=fx.Int64)
                destination_rsrc = crfa(remote_token + fx.Int64(payload_row) * fx.Int64(fz_nbytes))
                for column in range(lane * fx.Int32(4), fz_n_i32, 256):
                    value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
                    buffer_ops.buffer_store(value, destination_rsrc, column)
                if const_expr(fz_enable_scales):
                    if lane < fx.Int32(fz_scale_n_i32):
                        scale = buffer_ops.buffer_load(
                            r_scales, source_token * fx.Int32(fz_scale_n_i32) + lane,
                            vec_width=1, dtype=fx.Int32,
                        )
                        remote_scale = buffer_ops.buffer_load(crfa(p_sc), dest, vec_width=1, dtype=fx.Int64)
                        buffer_ops.buffer_store(
                            scale, crfa(remote_scale), payload_row * fx.Int32(fz_scale_n_i32) + lane
                        )
                if lane == fx.Int32(0):
                    weight = buffer_ops.buffer_load(r_wts, wk, vec_width=1, dtype=fx.Float32)
                    weight_bits = fx.Vector.from_elements([weight], fx.Float32).bitcast(fx.Int32)[0]
                    source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                    remote_weights = buffer_ops.buffer_load(crfa(p_wts), dest, vec_width=1, dtype=fx.Int64)
                    remote_srcmap = buffer_ops.buffer_load(crfa(p_sm), dest, vec_width=1, dtype=fx.Int64)
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
