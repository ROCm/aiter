# SPDX-License-Identifier: Apache-2.0
"""Single-launch Stage2 producer/consumer kernel.

Producer workgroups run the natural-grid route GEMM. Completion is tracked per
N tile; the final ``service_groups`` producers become communication consumers
and execute the selected direct, reduce-broadcast, or reduce-scatter/all-gather
path. This keeps the native GEMM tiling and avoids an artificial window.
"""

import functools
import hashlib

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T, as_ir_value

from .... import communication_ops_utils as comm_ops
from ....mxfp4_gemm_common import global_typed_ptr, lds_typed_ptr
from .collectives import (
    buffer_tensor_from_addr,
    decode_scaled_fp8_f32,
    e8m0_scale,
    load_bf16,
    load_buffer,
    load_e8m0_scale,
    load_fp8_words,
    pack_fp8_words,
    peer_base,
    store_bf16,
    store_buffer,
    store_fp8_words,
)
from .config import PRODUCER_COUNTER_STRIDE, SLOTS, MegakernelConfig
from .producer import compile_megakernel_producer

CPOL_COHERENT = 0x1 | 0x10


def _decode_scaled_fp8_bf16(words, scale):
    """Decode packed FP8 directly to the BF16 final-output representation."""

    values = []
    for word in range_constexpr(len(words)):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16),
                    as_ir_value(words[word]),
                    as_ir_value(scale),
                    bool(half),
                )
            )
            values.extend((pair[0], pair[1]))
    return values


def _mxfp8_scale(values, lane, vector_width):
    local_max = fx.Float32(1e-10).maximumf(fmath.absf(values).reduce(ReductionOp.MAX))
    max_bits = local_max.bitcast(fx.Int32)
    remote_bits = fx.rocdl.ds_bpermute(
        T.i32, (lane ^ fx.Int32(1)) * fx.Int32(4), max_bits
    )
    local_max = local_max.maximumf(fx.Int32(remote_bits).bitcast(fx.Float32))
    if const_expr(vector_width == 8):
        max_bits = local_max.bitcast(fx.Int32)
        remote_bits = fx.rocdl.ds_bpermute(
            T.i32, (lane ^ fx.Int32(2)) * fx.Int32(4), max_bits
        )
        local_max = local_max.maximumf(fx.Int32(remote_bits).bitcast(fx.Float32))
    return e8m0_scale(local_max)


@flyc.jit
def _store_mxfp8_scale(
    payload_buffer,
    scale_buffer,
    payload_bytes,
    offset,
    vector_width,
    e8m0,
    wide_scale=False,
    cache_modifier=0,
):
    if const_expr(wide_scale):
        if offset % fx.Int32(32) == fx.Int32(0):
            store_buffer(
                scale_buffer,
                offset // fx.Int32(32),
                e8m0 << fx.Int32(23),
                fx.Int32,
                cache_modifier=cache_modifier,
            )
    elif const_expr(vector_width == 8):
        if offset % fx.Int32(32) == fx.Int32(0):
            store_buffer(
                scale_buffer,
                offset // fx.Int32(32),
                e8m0.to(fx.Int8),
                fx.Int8,
                cache_modifier=cache_modifier,
            )
    else:
        store_buffer(
            payload_buffer,
            fx.Int32(payload_bytes // 4) + offset // fx.Int32(vector_width),
            e8m0 << fx.Int32(23),
            fx.Int32,
            cache_modifier=cache_modifier,
        )


def _load_mxfp8_scale(
    payload_buffer,
    scale_buffer,
    payload_bytes,
    offset,
    vector_width,
    cache_modifier,
    wide_scale=False,
):
    if const_expr(wide_scale):
        return fx.Int32(
            load_buffer(
                scale_buffer,
                offset // fx.Int32(32),
                fx.Int32,
                cache_modifier=cache_modifier,
            )
        ).bitcast(fx.Float32)
    if const_expr(vector_width == 8):
        return load_e8m0_scale(
            scale_buffer,
            offset // fx.Int32(32),
            cache_modifier,
        )
    return fx.Int32(
        load_buffer(
            payload_buffer,
            fx.Int32(payload_bytes // 4) + offset // fx.Int32(vector_width),
            fx.Int32,
            cache_modifier=cache_modifier,
        )
    ).bitcast(fx.Float32)


@flyc.jit
def emit_service_tile(
    config,
    workspace,
    workspace_flat_base,
    shared_partial,
    shared_partial_flat_base,
    specialized_rank,
    n_tile,
    tid,
    service_group,
    service_marker_ptr,
):
    hidden_dim = config.shape.model_dim
    topk = config.shape.topk
    tp_size = config.shape.tp_size
    rank = fx.Int32(specialized_rank)
    payload_bytes = config.payload_bytes
    partial_payload_bytes = config.partial_payload_bytes
    partial_bytes = config.partial_bytes
    partial_scale_sideband = not config.shared_bf16_partials and (
        config.vector_width == 8 or config.wide_partial_scales
    )
    reduce_items = config.m * config.tile_n // config.vector_width
    optimized_m8_direct = config.collective == "direct" and config.m == 8
    partial_store_cache_modifier = CPOL_COHERENT if optimized_m8_direct else 0
    local_workspace_base = fx.Int64(ptrtoint(workspace))
    state_n_tile = (n_tile // fx.Int32(config.service_tile_group)) * fx.Int32(
        config.service_tile_group
    )
    tile_byte_offset = fx.Int64(state_n_tile) * fx.Int64(8)
    epoch_address = (
        local_workspace_base + fx.Int64(config.epoch_offset) + tile_byte_offset
    )
    expected = fx.Int64(comm_ops.load_i64_global(epoch_address)) + fx.Int64(1)
    expected_i32 = fx.Int32(expected)
    slot = expected & fx.Int64(1)
    output_resource = buffer_tensor_from_addr(
        local_workspace_base + fx.Int64(config.output_offset),
        fx.BFloat16,
        payload_bytes,
    )
    shared_resource = buffer_tensor_from_addr(
        fx.Int64(ptrtoint(shared_partial)),
        fx.BFloat16,
        payload_bytes,
    )
    route_resource = None
    accumulator_resource = None
    clear_accumulator_resource = None
    if const_expr(config.producer_mode == "atomic_shared"):
        accumulator_resource = buffer_tensor_from_addr(
            local_workspace_base
            + fx.Int64(config.route_offset)
            + slot * fx.Int64(payload_bytes),
            fx.BFloat16,
            payload_bytes,
        )
        clear_accumulator_resource = buffer_tensor_from_addr(
            local_workspace_base
            + fx.Int64(config.route_offset)
            + (slot ^ fx.Int64(1)) * fx.Int64(payload_bytes),
            fx.BFloat16,
            payload_bytes,
        )
    else:
        route_resource = buffer_tensor_from_addr(
            local_workspace_base + fx.Int64(config.route_offset),
            fx.BFloat16,
            config.route_bytes,
        )
    partial_resource = buffer_tensor_from_addr(
        local_workspace_base + slot * fx.Int64(partial_bytes),
        fx.Int32,
        (partial_payload_bytes if partial_scale_sideband else partial_bytes),
    )
    partial_scale_resource = None
    if const_expr(partial_scale_sideband):
        partial_scale_resource = buffer_tensor_from_addr(
            local_workspace_base
            + slot * fx.Int64(partial_bytes)
            + fx.Int64(partial_payload_bytes),
            fx.Int32 if config.wide_partial_scales else fx.Int8,
            config.partial_scale_bytes,
        )
    if const_expr(config.collective == "rsag"):
        reduced_resource = buffer_tensor_from_addr(
            local_workspace_base
            + fx.Int64(config.reduced_offset)
            + slot * fx.Int64(config.reduced_shard_bytes),
            fx.Int32,
            (
                config.reduced_payload_bytes
                if const_expr(config.vector_width == 8)
                else config.reduced_shard_bytes
            ),
        )
        reduced_scale_resource = None
        if const_expr(config.vector_width == 8):
            reduced_scale_resource = buffer_tensor_from_addr(
                local_workspace_base
                + fx.Int64(config.reduced_offset)
                + slot * fx.Int64(config.reduced_shard_bytes)
                + fx.Int64(config.reduced_payload_bytes),
                fx.Int8,
                config.reduced_scale_bytes,
            )
    service_stride = config.block_threads * config.service_groups
    service_start = tid + service_group * fx.Int32(config.block_threads)
    retain_underfilled_local_partial = (
        config.collective == "direct"
        and not config.shared_bf16_partials
        and reduce_items < service_stride
    )
    retain_full_local_partials = (
        config.collective == "direct"
        and not config.shared_bf16_partials
        and reduce_items % service_stride == 0
        and reduce_items <= 4 * service_stride
    )
    retain_local_partials = (
        retain_underfilled_local_partial or retain_full_local_partials
    )
    if const_expr(config.uses_rsag):
        if tid < fx.Int32(tp_size):
            gather_slot = state_n_tile * fx.Int32(tp_size) + tid
            comm_ops.spin_until_ge_i32_system(
                local_workspace_base
                + fx.Int64(config.gather_done_offset)
                + fx.Int64(gather_slot) * fx.Int64(4),
                expected_i32 - fx.Int32(SLOTS),
            )
        gpu.barrier()

    def emit_local_reduce_item(item):
        token = item // fx.Int32(config.tile_n // config.vector_width)
        tile_item = item - token * fx.Int32(config.tile_n // config.vector_width)
        output_offset = (
            token * fx.Int32(hidden_dim)
            + n_tile * fx.Int32(config.tile_n)
            + tile_item * fx.Int32(config.vector_width)
        )
        if const_expr(config.producer_mode == "atomic_shared"):
            shared_values = load_bf16(
                shared_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).to(fx.Float32)
            reduced_f32 = shared_values + load_bf16(
                accumulator_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).to(fx.Float32)
        else:
            shared_values = load_bf16(
                shared_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).to(fx.Float32)

            def load_bf16_route(route_slot):
                route_offset = (
                    (token * fx.Int32(topk) + fx.Int32(route_slot))
                    * fx.Int32(hidden_dim)
                    + n_tile * fx.Int32(config.tile_n)
                    + tile_item * fx.Int32(config.vector_width)
                )
                return load_bf16(
                    route_resource,
                    route_offset,
                    config.vector_width,
                    config.local_load_cache_modifier,
                ).to(fx.Float32)

            local_even = shared_values + load_bf16_route(0)
            if const_expr(topk == 1):
                reduced_f32 = local_even
            else:
                local_odd = load_bf16_route(1)
                for route_slot in range_constexpr(2, topk):
                    if const_expr(route_slot == topk - 1 or route_slot % 2 == 0):
                        local_odd = local_odd + load_bf16_route(route_slot)
                    else:
                        local_even = local_even + load_bf16_route(route_slot)
                reduced_f32 = local_even + local_odd

        if const_expr(config.shared_bf16_partials):
            if const_expr(config.producer_mode == "atomic_shared"):
                store_bf16(
                    output_resource,
                    output_offset,
                    reduced_f32.to(fx.BFloat16),
                    config.vector_width,
                )
                zero_bf16 = fx.Float32(0.0).to(fx.BFloat16)
                store_bf16(
                    clear_accumulator_resource,
                    output_offset,
                    fx.Vector.from_elements(
                        [zero_bf16 for _ in range_constexpr(config.vector_width)],
                        fx.BFloat16,
                    ),
                    config.vector_width,
                    cache_modifier=2,
                )
            else:
                store_bf16(
                    shared_resource,
                    output_offset,
                    reduced_f32.to(fx.BFloat16),
                    config.vector_width,
                )
            return None

        partial_e8m0, quant_scale = _mxfp8_scale(
            reduced_f32,
            tid & fx.Int32(63),
            config.vector_width,
        )
        packed_words = config.vector_width // 4
        packed = pack_fp8_words(reduced_f32, quant_scale, packed_words)
        store_fp8_words(
            partial_resource,
            output_offset,
            packed,
            packed_words,
            cache_modifier=partial_store_cache_modifier,
        )
        _store_mxfp8_scale(
            partial_resource,
            partial_scale_resource,
            partial_payload_bytes,
            output_offset,
            config.vector_width,
            partial_e8m0,
            config.wide_partial_scales,
            cache_modifier=partial_store_cache_modifier,
        )
        return packed, partial_e8m0

    retained_local_partials = []

    def emit_local_reduce_items():
        if const_expr(retain_underfilled_local_partial):
            initial = [fx.Int32(0) for _ in range_constexpr(config.vector_width // 4)]
            initial.append(fx.Int32(0))
            for item, _ in range(
                service_start,
                fx.Int32(reduce_items),
                fx.Int32(service_stride),
                init=initial,
            ):
                packed, partial_e8m0 = emit_local_reduce_item(fx.Int32(item))
                retained = yield [*packed, partial_e8m0]
            retained_local_partials.append((retained[:-1], retained[-1]))
        elif const_expr(retain_full_local_partials):
            for iteration in range_constexpr(reduce_items // service_stride):
                item = service_start + fx.Int32(iteration * service_stride)
                retained_local_partials.append(emit_local_reduce_item(item))
        else:
            for item in range(
                service_start,
                fx.Int32(reduce_items),
                fx.Int32(service_stride),
            ):
                emit_local_reduce_item(item)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        gpu.barrier()

    emit_local_reduce_items()

    def load_reduced_partials(
        offset,
        retained_local_partial=None,
        source_rotation=None,
        vector_width=None,
    ):
        load_vector_width = vector_width or config.vector_width

        def load_peer(peer, cache_modifier=None, use_retained=False):
            if const_expr(config.shared_bf16_partials):
                if const_expr(config.producer_mode == "atomic_shared"):
                    peer_resource = buffer_tensor_from_addr(
                        peer_base(workspace_flat_base, peer)
                        + fx.Int64(config.output_offset),
                        fx.BFloat16,
                        payload_bytes,
                    )
                else:
                    peer_resource = buffer_tensor_from_addr(
                        peer_base(shared_partial_flat_base, peer),
                        fx.BFloat16,
                        payload_bytes,
                    )
                return load_bf16(
                    peer_resource,
                    offset,
                    load_vector_width,
                    (
                        cache_modifier
                        if const_expr(cache_modifier is not None)
                        else (
                            config.remote_load_cache_modifier
                            if const_expr(peer != specialized_rank)
                            else config.local_load_cache_modifier
                        )
                    ),
                ).to(fx.Float32)
            if const_expr(use_retained):
                retained_packed, retained_e8m0 = retained_local_partial
                scale = (fx.Uint32(retained_e8m0) << fx.Uint32(23)).bitcast(fx.Float32)
                return fx.Vector.from_elements(
                    decode_scaled_fp8_f32(retained_packed, scale),
                    fx.Float32,
                )
            load_offset = offset
            peer_resource = buffer_tensor_from_addr(
                peer_base(workspace_flat_base, peer) + slot * fx.Int64(partial_bytes),
                fx.Int32,
                (partial_payload_bytes if partial_scale_sideband else partial_bytes),
            )
            if const_expr(cache_modifier is None):
                cache_modifier = (
                    config.remote_load_cache_modifier
                    if const_expr(peer != specialized_rank)
                    else config.local_load_cache_modifier
                )
            peer_scale_resource = None
            if const_expr(partial_scale_sideband):
                peer_scale_resource = buffer_tensor_from_addr(
                    peer_base(workspace_flat_base, peer)
                    + slot * fx.Int64(partial_bytes)
                    + fx.Int64(partial_payload_bytes),
                    fx.Int32 if config.wide_partial_scales else fx.Int8,
                    config.partial_scale_bytes,
                )
            if const_expr(config.wide_partial_scales):
                scale = _load_mxfp8_scale(
                    peer_resource,
                    peer_scale_resource,
                    partial_payload_bytes,
                    load_offset,
                    config.vector_width,
                    cache_modifier,
                    config.wide_partial_scales,
                )
            words = load_fp8_words(
                peer_resource,
                load_offset // fx.Int32(4),
                word_count=load_vector_width // 4,
                load_width=load_vector_width // 4,
                cache_modifier=cache_modifier,
            )
            if const_expr(not config.wide_partial_scales):
                scale = _load_mxfp8_scale(
                    peer_resource,
                    peer_scale_resource,
                    partial_payload_bytes,
                    load_offset,
                    config.vector_width,
                    cache_modifier,
                    config.wide_partial_scales,
                )
            return fx.Vector.from_elements(
                decode_scaled_fp8_f32(words, scale),
                fx.Float32,
            )

        if const_expr(config.wide_partial_scales):
            reduced_even = load_peer(
                specialized_rank,
                use_retained=retain_local_partials,
            )
            source_rotation = n_tile
            first_delta = source_rotation % fx.Int32(tp_size - 1) + fx.Int32(1)
            first_peer = (rank + first_delta) & fx.Int32(tp_size - 1)
            reduced_odd = load_peer(first_peer, config.remote_load_cache_modifier)
            for remote_step in range_constexpr(1, tp_size - 1):
                remote_delta = (
                    (source_rotation + fx.Int32(remote_step)) % fx.Int32(tp_size - 1)
                ) + fx.Int32(1)
                peer = (rank + remote_delta) & fx.Int32(tp_size - 1)
                if const_expr(remote_step % 2 == 1):
                    reduced_even = reduced_even + load_peer(
                        peer, config.remote_load_cache_modifier
                    )
                else:
                    reduced_odd = reduced_odd + load_peer(
                        peer, config.remote_load_cache_modifier
                    )
            reduced = reduced_even + reduced_odd
        elif const_expr(source_rotation is not None):
            first_peer = (rank + source_rotation + fx.Int32(1)) & fx.Int32(tp_size - 1)
            second_peer = (rank + source_rotation + fx.Int32(2)) & fx.Int32(tp_size - 1)
            reduced_even = load_peer(first_peer, config.remote_load_cache_modifier)
            reduced_odd = load_peer(second_peer, config.remote_load_cache_modifier)
            for peer_step in range_constexpr(3, tp_size + 1):
                peer = (rank + source_rotation + fx.Int32(peer_step)) & fx.Int32(
                    tp_size - 1
                )
                if const_expr(peer_step % 2 == 1):
                    reduced_even = reduced_even + load_peer(
                        peer, config.remote_load_cache_modifier
                    )
                else:
                    reduced_odd = reduced_odd + load_peer(
                        peer, config.remote_load_cache_modifier
                    )
            reduced = reduced_even + reduced_odd
        else:
            reduced_even = load_peer(
                specialized_rank,
                use_retained=retain_local_partials,
            )
            reduced_odd = load_peer((specialized_rank + 1) % tp_size)
            for peer_step in range_constexpr(2, tp_size):
                peer = (specialized_rank + peer_step) % tp_size
                if const_expr(peer_step % 2 == 0):
                    reduced_even = reduced_even + load_peer(peer)
                else:
                    reduced_odd = reduced_odd + load_peer(peer)
            reduced = reduced_even + reduced_odd
        return reduced

    def emit_direct_reduce():
        def emit_allreduce_item(item, retained_local_partial=None):
            token = item // fx.Int32(config.tile_n // config.vector_width)
            tile_item = item - token * fx.Int32(config.tile_n // config.vector_width)
            offset = (
                token * fx.Int32(hidden_dim)
                + n_tile * fx.Int32(config.tile_n)
                + tile_item * fx.Int32(config.vector_width)
            )
            reduced = load_reduced_partials(
                offset,
                retained_local_partial,
            )
            store_bf16(
                output_resource,
                offset,
                reduced.to(fx.BFloat16),
                config.vector_width,
            )

        if const_expr(retain_underfilled_local_partial):
            for item in range(
                service_start,
                fx.Int32(reduce_items),
                fx.Int32(service_stride),
            ):
                emit_allreduce_item(item, retained_local_partials[0])
        elif const_expr(retain_full_local_partials):
            for iteration in range_constexpr(reduce_items // service_stride):
                item = service_start + fx.Int32(iteration * service_stride)
                emit_allreduce_item(
                    item,
                    retained_local_partials[iteration],
                )
        else:
            for item in range(
                service_start,
                fx.Int32(reduce_items),
                fx.Int32(service_stride),
            ):
                emit_allreduce_item(item)

    def reset_tile_state_values():
        for tile_delta in range_constexpr(config.service_tile_group):
            counter_slot = fx.Int64(0)
            if const_expr(config.producer_mode == "atomic_shared"):
                counter_slot = slot ^ fx.Int64(1)
            counter_address = (
                local_workspace_base
                + fx.Int64(config.producer_done_offset)
                + (
                    fx.Int64(state_n_tile + fx.Int32(tile_delta))
                    * fx.Int64(config.producer_counter_slots)
                    + counter_slot
                )
                * fx.Int64(PRODUCER_COUNTER_STRIDE)
            )
            fx.ptr_store(
                fx.Int32(0),
                global_typed_ptr(counter_address, T.i32),
            )
        if const_expr(config.service_groups > 1):
            fx.ptr_store(
                fx.Int32(0),
                global_typed_ptr(
                    local_workspace_base
                    + fx.Int64(config.service_done_offset)
                    + tile_byte_offset,
                    T.i32,
                ),
            )
            fx.ptr_store(
                fx.Int32(0),
                global_typed_ptr(
                    local_workspace_base
                    + fx.Int64(config.reduce_done_offset)
                    + tile_byte_offset,
                    T.i32,
                ),
            )
            fx.ptr_store(
                fx.Int32(0),
                global_typed_ptr(
                    local_workspace_base
                    + fx.Int64(config.gather_service_done_offset)
                    + tile_byte_offset,
                    T.i32,
                ),
            )
        fx.ptr_store(expected, global_typed_ptr(epoch_address, T.i64, align=8))

    def emit_rsag_reduce():
        collective_vector_width = config.vector_width

        def emit_gather_ack(barrier=True):
            if tid < fx.Int32(tp_size):
                remote_slot = state_n_tile * fx.Int32(tp_size) + rank
                comm_ops.store_i32_global_system_monotonic(
                    peer_base(workspace_flat_base, tid)
                    + fx.Int64(config.gather_done_offset)
                    + fx.Int64(remote_slot) * fx.Int64(4),
                    expected_i32,
                )
            if const_expr(barrier):
                gpu.barrier()

        def wait_for_gather_acks():
            if const_expr(
                config.collective == "rs_broadcast"
                and config.producer_mode == "atomic_shared"
            ):
                if tid < fx.Int32(tp_size):
                    local_slot = state_n_tile * fx.Int32(tp_size) + tid
                    comm_ops.spin_until_ge_i32_system(
                        local_workspace_base
                        + fx.Int64(config.gather_done_offset)
                        + fx.Int64(local_slot) * fx.Int64(4),
                        expected_i32,
                    )
                gpu.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_system_acquire()
                gpu.barrier()

        def publish_gather_completion():
            if const_expr(config.service_groups == 1):
                if const_expr(config.collective == "rs_broadcast"):
                    emit_gather_ack()
                    wait_for_gather_acks()
                else:
                    if tid == fx.Int32(0):
                        comm_ops.fence_system_release()
                    gpu.barrier()
                    emit_gather_ack()
            else:
                gather_done_address = (
                    local_workspace_base
                    + fx.Int64(config.gather_service_done_offset)
                    + tile_byte_offset
                )
                if tid == fx.Int32(0):
                    comm_ops.fence_agent_release()
                    arrival = fx.Int32(
                        comm_ops.atomic_add_agent_one_as(
                            gather_done_address, fx.Int32(1)
                        )
                    )
                    fx.ptr_store(arrival, service_marker_ptr)
                gpu.barrier()
                gather_arrival = fx.Int32(fx.ptr_load(service_marker_ptr))
                if gather_arrival == fx.Int32(
                    config.service_groups * config.service_tile_group - 1
                ):
                    if tid == fx.Int32(0):
                        comm_ops.fence_agent_acquire()
                        comm_ops.fence_system_release()
                    gpu.barrier()
                    emit_gather_ack(barrier=False)
                    wait_for_gather_acks()
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    if tid == fx.Int32(0):
                        reset_tile_state_values()

        def emit_reduce_scatter_items():
            vectors_per_token = config.tile_n // collective_vector_width
            shard_tokens = config.m // tp_size
            service_item = tid + service_group * fx.Int32(config.block_threads)
            first_shard_token = service_item // fx.Int32(vectors_per_token)
            vector_lane = service_item - first_shard_token * fx.Int32(vectors_per_token)

            for shard_token in range(
                first_shard_token,
                fx.Int32(shard_tokens),
                fx.Int32(
                    config.block_threads * config.service_groups // vectors_per_token,
                ),
            ):
                token = rank * fx.Int32(shard_tokens) + shard_token
                offset = (
                    token * fx.Int32(hidden_dim)
                    + n_tile * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(collective_vector_width)
                )
                reduced = load_reduced_partials(
                    offset,
                    source_rotation=(
                        shard_token
                        if const_expr(
                            config.collective == "rsag" and config.service_groups > 1
                        )
                        else None
                    ),
                    vector_width=collective_vector_width,
                )
                reduced_bf16 = reduced.to(fx.BFloat16)
                if const_expr(config.collective == "rsag"):
                    # Publish the local RS/AG shard directly.
                    store_bf16(
                        output_resource,
                        offset,
                        reduced_bf16,
                        collective_vector_width,
                        config.remote_store_cache_modifier,
                    )
                reduced_offset = (
                    n_tile * fx.Int32(config.m * config.tile_n // tp_size)
                    + shard_token * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(collective_vector_width)
                )
                if const_expr(config.collective == "rs_broadcast"):
                    for peer_step in range_constexpr(tp_size):
                        peer = (specialized_rank + peer_step + 1) % tp_size
                        if const_expr(peer == specialized_rank):
                            peer_output_resource = output_resource
                        else:
                            peer_output_resource = buffer_tensor_from_addr(
                                peer_base(workspace_flat_base, peer)
                                + fx.Int64(config.output_offset),
                                fx.BFloat16,
                                payload_bytes,
                            )
                        store_bf16(
                            peer_output_resource,
                            offset,
                            reduced_bf16,
                            collective_vector_width,
                            config.remote_store_cache_modifier,
                        )
                else:
                    reduced_e8m0, reduced_quant_scale = _mxfp8_scale(
                        reduced,
                        tid & fx.Int32(63),
                        collective_vector_width,
                    )
                    store_fp8_words(
                        reduced_resource,
                        reduced_offset,
                        pack_fp8_words(
                            reduced,
                            reduced_quant_scale,
                            collective_vector_width // 4,
                        ),
                        collective_vector_width // 4,
                    )
                    _store_mxfp8_scale(
                        reduced_resource,
                        reduced_scale_resource,
                        config.reduced_payload_bytes,
                        reduced_offset,
                        collective_vector_width,
                        reduced_e8m0,
                    )
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()

        emit_reduce_scatter_items()

        if const_expr(config.collective == "rs_broadcast"):
            publish_gather_completion()
            return

        def emit_reduced_exchange(propagate_acquire):
            if tid < fx.Int32(tp_size):
                remote_slot = state_n_tile * fx.Int32(tp_size) + rank
                comm_ops.store_i32_global_system_monotonic(
                    peer_base(workspace_flat_base, tid)
                    + fx.Int64(config.owner_ready_offset)
                    + fx.Int64(remote_slot) * fx.Int64(4),
                    expected_i32,
                )
                local_slot = state_n_tile * fx.Int32(tp_size) + tid
                comm_ops.spin_until_ge_i32_system(
                    local_workspace_base
                    + fx.Int64(config.owner_ready_offset)
                    + fx.Int64(local_slot) * fx.Int64(4),
                    expected_i32,
                )
            gpu.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_system_acquire()
            if const_expr(propagate_acquire):
                gpu.barrier()

        if const_expr(config.service_groups == 1):
            if tid == fx.Int32(0):
                comm_ops.fence_system_release()
            gpu.barrier()
            emit_reduced_exchange(True)
        else:
            reduce_done_address = (
                local_workspace_base
                + fx.Int64(config.reduce_done_offset)
                + tile_byte_offset
            )
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                arrival = fx.Int32(
                    comm_ops.atomic_add_agent_one_as(reduce_done_address, fx.Int32(1))
                )
                fx.ptr_store(arrival, service_marker_ptr)
            gpu.barrier()
            reduce_arrival = fx.Int32(fx.ptr_load(service_marker_ptr))
            if reduce_arrival == fx.Int32(
                config.service_groups * config.service_tile_group - 1
            ):
                if tid == fx.Int32(0):
                    comm_ops.fence_agent_acquire()
                    comm_ops.fence_system_release()
                gpu.barrier()
                emit_reduced_exchange(False)
                if tid == fx.Int32(0):
                    comm_ops.store_i32_global_agent_release(
                        local_workspace_base
                        + fx.Int64(config.reduced_collective_ready_offset)
                        + fx.Int64(state_n_tile) * fx.Int64(4),
                        expected_i32,
                    )

            if tid == fx.Int32(0):
                comm_ops.spin_until_ge_i32_agent(
                    local_workspace_base
                    + fx.Int64(config.reduced_collective_ready_offset)
                    + fx.Int64(state_n_tile) * fx.Int64(4),
                    expected_i32,
                )
                comm_ops.fence_agent_acquire()
            gpu.barrier()

        def emit_gather_source(source, source_start, source_stride):
            gather_vector_width = config.vector_width
            gather_cache_modifier = (
                config.remote_load_cache_modifier
                if const_expr(config.gather_load_cache_modifier < 0)
                else config.gather_load_cache_modifier
            )
            vectors_per_token = config.tile_n // gather_vector_width
            shard_tokens = config.m // tp_size
            first_token = source_start // fx.Int32(vectors_per_token)
            vector_lane = source_start - first_token * fx.Int32(vectors_per_token)
            source_resource = buffer_tensor_from_addr(
                peer_base(workspace_flat_base, source)
                + fx.Int64(config.reduced_offset)
                + slot * fx.Int64(config.reduced_shard_bytes),
                fx.Int32,
                (
                    config.reduced_payload_bytes
                    if const_expr(config.vector_width == 8)
                    else config.reduced_shard_bytes
                ),
            )
            source_scale_resource = None
            if const_expr(config.vector_width == 8):
                source_scale_resource = buffer_tensor_from_addr(
                    peer_base(workspace_flat_base, source)
                    + fx.Int64(config.reduced_offset)
                    + slot * fx.Int64(config.reduced_shard_bytes)
                    + fx.Int64(config.reduced_payload_bytes),
                    fx.Int8,
                    config.reduced_scale_bytes,
                )
            for source_token in range(
                first_token,
                fx.Int32(shard_tokens),
                fx.Int32(source_stride // vectors_per_token),
            ):
                offset = (
                    (source * fx.Int32(shard_tokens) + source_token)
                    * fx.Int32(hidden_dim)
                    + n_tile * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(gather_vector_width)
                )
                reduced_offset = (
                    n_tile * fx.Int32(config.m * config.tile_n // tp_size)
                    + source_token * fx.Int32(config.tile_n)
                    + vector_lane * fx.Int32(gather_vector_width)
                )
                words = load_fp8_words(
                    source_resource,
                    reduced_offset // fx.Int32(4),
                    word_count=gather_vector_width // 4,
                    load_width=gather_vector_width // 4,
                    cache_modifier=gather_cache_modifier,
                )
                scale = _load_mxfp8_scale(
                    source_resource,
                    source_scale_resource,
                    config.reduced_payload_bytes,
                    reduced_offset,
                    gather_vector_width,
                    gather_cache_modifier,
                )
                values = fx.Vector.from_elements(
                    _decode_scaled_fp8_bf16(words, scale),
                    fx.BFloat16,
                )
                store_bf16(
                    output_resource,
                    offset,
                    values,
                    gather_vector_width,
                    config.remote_store_cache_modifier,
                )

        def emit_remote_gather_source(source, source_start, source_stride):
            if source != rank:
                emit_gather_source(source, source_start, source_stride)

        # Split SG4 waves across its two source shards.
        if const_expr(config.service_groups == 1):
            parallel_sources = min(4, tp_size)
            threads_per_source = config.block_threads // parallel_sources
            source_lane = tid // fx.Int32(threads_per_source)
            for source_iteration in range_constexpr(tp_size // parallel_sources):
                source = (
                    n_tile + source_lane + fx.Int32(source_iteration * parallel_sources)
                ) % fx.Int32(tp_size)
                emit_remote_gather_source(
                    source,
                    tid % fx.Int32(threads_per_source),
                    threads_per_source,
                )
        elif const_expr(config.service_groups == 4):
            sources_per_group = tp_size // config.service_groups
            threads_per_source = config.block_threads // sources_per_group
            source_lane = tid // fx.Int32(threads_per_source)
            source_phase = (n_tile + source_lane) % fx.Int32(sources_per_group)
            source = service_group + source_phase * fx.Int32(config.service_groups)
            emit_remote_gather_source(
                source,
                tid % fx.Int32(threads_per_source),
                threads_per_source,
            )
        else:
            # Assign each service workgroup to one source rank at a time.
            for source_iteration in range_constexpr(tp_size // config.service_groups):
                source_phase = (n_tile + fx.Int32(source_iteration)) % fx.Int32(
                    tp_size // config.service_groups
                )
                source = service_group + source_phase * fx.Int32(config.service_groups)
                emit_remote_gather_source(source, tid, config.block_threads)
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
        gpu.barrier()
        publish_gather_completion()

    if const_expr(config.service_groups == 1):
        if tid == fx.Int32(0):
            comm_ops.fence_system_release()
        if const_expr(config.single_pass_direct):
            gpu.barrier()

        if tid < fx.Int32(tp_size):
            remote_slot = state_n_tile * fx.Int32(tp_size) + rank
            comm_ops.store_i32_global_system_monotonic(
                peer_base(workspace_flat_base, tid)
                + fx.Int64(config.rank_ready_offset)
                + fx.Int64(remote_slot) * fx.Int64(4),
                expected_i32,
            )
            local_slot = state_n_tile * fx.Int32(tp_size) + tid
            ready_address = (
                local_workspace_base
                + fx.Int64(config.rank_ready_offset)
                + fx.Int64(local_slot) * fx.Int64(4)
            )
            if const_expr(config.single_pass_direct):
                comm_ops.spin_until_ge_i32_system(
                    ready_address,
                    expected_i32,
                    acquire=not optimized_m8_direct,
                    sleep=False,
                )
            else:
                comm_ops.spin_until_ge_i32_system(
                    ready_address,
                    expected_i32,
                )
        if const_expr(not config.single_pass_direct):  # noqa: SIM102
            if tid == fx.Int32(0):
                comm_ops.fence_system_acquire()
        gpu.barrier()
    else:
        service_done_address = (
            local_workspace_base
            + fx.Int64(config.service_done_offset)
            + tile_byte_offset
        )

        def publish_partials_and_exchange():
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                arrival = fx.Int32(
                    comm_ops.atomic_add_agent_one_as(service_done_address, fx.Int32(1))
                )
                fx.ptr_store(arrival, service_marker_ptr)
            gpu.barrier()
            service_arrival = fx.Int32(fx.ptr_load(service_marker_ptr))
            if service_arrival == fx.Int32(
                config.service_groups * config.service_tile_group - 1
            ):
                if tid == fx.Int32(0):
                    comm_ops.fence_agent_acquire()
                    local_ready_slot = state_n_tile * fx.Int32(tp_size) + rank
                    comm_ops.store_i32_global_system_release(
                        local_workspace_base
                        + fx.Int64(config.rank_ready_offset)
                        + fx.Int64(local_ready_slot) * fx.Int64(4),
                        expected_i32,
                    )
                gpu.barrier()

                if tid < fx.Int32(tp_size):
                    peer_ready_slot = state_n_tile * fx.Int32(tp_size) + tid
                    comm_ops.spin_until_ge_i32_system(
                        peer_base(workspace_flat_base, tid)
                        + fx.Int64(config.rank_ready_offset)
                        + fx.Int64(peer_ready_slot) * fx.Int64(4),
                        expected_i32,
                    )
                gpu.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_system_acquire()
                    comm_ops.store_i32_global_agent_release(
                        local_workspace_base
                        + fx.Int64(config.collective_ready_offset)
                        + fx.Int64(state_n_tile) * fx.Int64(4),
                        expected_i32,
                    )

        publish_partials_and_exchange()

        def wait_for_collective():
            if tid == fx.Int32(0):
                comm_ops.spin_until_ge_i32_agent(
                    local_workspace_base
                    + fx.Int64(config.collective_ready_offset)
                    + fx.Int64(state_n_tile) * fx.Int64(4),
                    expected_i32,
                )
                comm_ops.fence_agent_acquire()
            gpu.barrier()

    if const_expr(config.uses_rsag):
        if const_expr(config.service_groups > 1):
            wait_for_collective()
        emit_rsag_reduce()
    else:
        emit_direct_reduce()
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    if const_expr(not (config.uses_rsag and config.service_groups > 1)):  # noqa: SIM102
        if tid == fx.Int32(0):
            reset_tile_state_values()


@functools.cache
def compile_megakernel(
    config: MegakernelConfig,
    specialized_rank: int,
):
    """Compile the single MXMoE GEMM2 + TP kernel."""

    shape = config.shape
    if not 0 <= specialized_rank < shape.tp_size:
        raise ValueError(f"invalid TP rank {specialized_rank}")
    if config.collective == "direct" and config.producer_rows % config.compute_groups:
        raise ValueError("compute_groups must divide the producer row bound")

    dynamic_producer = config.collective != "direct"
    rows_per_group = (
        config.producer_rows // config.compute_groups if not dynamic_producer else None
    )
    launch_grid = (
        (config.compute_groups * config.n_tiles, 1, 1)
        if config.n_tile_cohort or config.flat_producer_grid
        else (config.n_tiles, config.compute_groups, 1)
    )
    legacy_config_repr = (
        repr(config)
        .replace("MegakernelConfig(", "Gemm2TPMegakernelConfig(", 1)
        .replace("shape=Shape(", "shape=Gemm2TPShape(", 1)
    )
    cache_config = hashlib.sha256(
        f"mxmoe_bf16_route_dynamic_scale_v5:{legacy_config_repr}".encode()
    ).hexdigest()[:16]

    def compose(*, module_name, emit_gemm2_tile, shared_storage):
        @flyc.kernel(
            name=(
                f"{module_name}_gemm2_tp_mega_r{specialized_rank}_m{config.m}"
                f"_cg{config.compute_groups}_v{config.vector_width}"
            ),
            known_block_size=[config.block_threads, 1, 1],
        )
        def kernel(
            workspace: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            shared_partial: fx.Pointer,
            shared_partial_flat_base: fx.Int64,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
        ):
            local_workspace_base = fx.Int64(ptrtoint(workspace))
            workspace_flat_base = fx.Int64(
                comm_ops.load_i64_global(
                    local_workspace_base + fx.Int64(config.flat_base_offset)
                )
            )
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
            if const_expr(config.flat_producer_grid):
                physical_block = fx.Int32(gpu.block_id("x"))
                n_tile = physical_block % fx.Int32(config.n_tiles)
                compute_group = physical_block // fx.Int32(config.n_tiles)
            elif const_expr(config.n_tile_cohort):
                physical_block = fx.Int32(gpu.block_id("x"))
                cohort_size = fx.Int32(config.n_tile_cohort)
                cohort_span = fx.Int32(config.n_tile_cohort * config.compute_groups)
                cohort_base = physical_block // cohort_span * cohort_size
                within_cohort = physical_block % cohort_span
                n_tile = cohort_base + within_cohort % cohort_size
                compute_group = within_cohort // cohort_size
            else:
                n_tile = fx.Int32(gpu.block_id("x"))
                compute_group = fx.Int32(gpu.block_id("y"))
            producer_output_address = local_workspace_base + fx.Int64(
                config.route_offset
            )
            producer_counter_slot = fx.Int64(0)
            if const_expr(config.producer_mode == "atomic_shared"):
                producer_epoch = fx.Int64(
                    comm_ops.load_i64_global(
                        local_workspace_base
                        + fx.Int64(config.epoch_offset)
                        + fx.Int64(n_tile) * fx.Int64(8)
                    )
                )
                producer_slot = (producer_epoch + fx.Int64(1)) & fx.Int64(1)
                producer_counter_slot = producer_slot
                producer_output_address = (
                    producer_output_address
                    + producer_slot * fx.Int64(config.payload_bytes)
                )
            producer_output = global_typed_ptr(
                producer_output_address,
                T.i8,
                align=1,
            )
            lds = fx.SharedAllocator().allocate(shared_storage).peek()
            service_marker_ptr = lds_typed_ptr(fx.Int32(ptrtoint(lds.buf.ptr)), T.i32)
            valid_rows = global_typed_ptr(fx.Int64(ptrtoint(num_valid_ids)), T.i32)[0]

            def emit_gemm(m_block):
                emit_gemm2_tile(
                    fx.Int64(ptrtoint(x)),
                    fx.Int64(ptrtoint(scale_x)),
                    fx.Int64(ptrtoint(w)),
                    fx.Int64(ptrtoint(scale_w)),
                    fx.Int64(ptrtoint(expert_ids)),
                    fx.Int64(ptrtoint(sorted_token_ids)),
                    fx.Int64(ptrtoint(sorted_weights)),
                    fx.Int64(ptrtoint(w)),
                    fx.Int64(ptrtoint(producer_output)),
                    m_block,
                    n_tile,
                    lane,
                    wave,
                    tokens,
                    size_expert_ids,
                    inter_dim,
                    model_dim,
                    lds,
                )

            if const_expr(dynamic_producer):
                total_m_tiles = (valid_rows + fx.Int32(config.tile_m - 1)) // fx.Int32(
                    config.tile_m
                )
                base_iterations = total_m_tiles // fx.Int32(config.compute_groups)
                remainder = total_m_tiles - base_iterations * fx.Int32(
                    config.compute_groups
                )
                has_extra = compute_group < remainder
                iteration_count = base_iterations + has_extra.select(
                    fx.Int32(1), fx.Int32(0)
                )
                start_tail = has_extra.select(compute_group, remainder)
                start_m_block = compute_group * base_iterations + start_tail
                for iteration in range(fx.Int32(0), iteration_count, fx.Int32(1)):
                    gpu.barrier()
                    emit_gemm(start_m_block + fx.Int32(iteration))
            else:
                for iteration in range_constexpr(rows_per_group):
                    if iteration:
                        gpu.barrier()
                    m_block = compute_group * fx.Int32(rows_per_group) + fx.Int32(
                        iteration
                    )
                    if m_block * fx.Int32(config.tile_m) < valid_rows:
                        emit_gemm(m_block)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
            gpu.barrier()
            if tid == fx.Int32(0):
                ticket = fx.Int32(
                    comm_ops.atomic_add_agent_one_as(
                        local_workspace_base
                        + fx.Int64(config.producer_done_offset)
                        + (
                            fx.Int64(n_tile) * fx.Int64(config.producer_counter_slots)
                            + producer_counter_slot
                        )
                        * fx.Int64(PRODUCER_COUNTER_STRIDE),
                        fx.Int32(1),
                    )
                )
                service_begin = fx.Int32(config.compute_groups - config.service_groups)
                fx.ptr_store(
                    (ticket >= service_begin).select(
                        ticket - service_begin + fx.Int32(1),
                        fx.Int32(0),
                    ),
                    service_marker_ptr,
                )
            gpu.barrier()

            service_marker = fx.Int32(fx.ptr_load(service_marker_ptr))
            if service_marker > fx.Int32(0):
                service_group = service_marker - fx.Int32(1)
                if config.service_groups > 1:
                    if tid == fx.Int32(0):
                        comm_ops.spin_until_ge_i32_agent(
                            local_workspace_base
                            + fx.Int64(config.producer_done_offset)
                            + (
                                fx.Int64(n_tile)
                                * fx.Int64(config.producer_counter_slots)
                                + producer_counter_slot
                            )
                            * fx.Int64(PRODUCER_COUNTER_STRIDE),
                            fx.Int32(config.compute_groups),
                        )
                        comm_ops.fence_agent_acquire()
                else:
                    if tid == fx.Int32(0):
                        comm_ops.fence_agent_acquire()
                gpu.barrier()
                emit_service_tile(
                    config,
                    workspace,
                    workspace_flat_base,
                    shared_partial,
                    shared_partial_flat_base,
                    specialized_rank,
                    n_tile,
                    tid,
                    service_group,
                    service_marker_ptr,
                )

        def launch(
            workspace,
            shared_partial,
            shared_partial_flat_base,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            stream,
        ):
            kernel(
                workspace,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                shared_partial,
                shared_partial_flat_base,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
                value_attrs=(
                    {"rocdl.waves_per_eu": config.waves_per_eu}
                    if config.waves_per_eu > 0
                    else None
                ),
            ).launch(
                grid=launch_grid,
                block=(config.block_threads, 1, 1),
                stream=stream,
            )

        launch.__name__ = (
            f"launch_gemm2_tp_mega_mxmoe_r{specialized_rank}_{cache_config}"
        )
        return flyc.jit(launch)

    return compile_megakernel_producer(config, compose)
