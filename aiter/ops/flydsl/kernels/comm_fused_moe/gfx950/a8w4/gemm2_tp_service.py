# SPDX-License-Identifier: Apache-2.0
"""TP communication service for the fused GEMM2 megakernel."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr

from .... import buffer_ops
from .... import communication_ops_utils as comm_ops
from .collectives import (
    decode_scaled_fp8_f32,
    load_fp8_words,
    pack_fp8_words,
    store_fp8_words,
)
from .sync import peer_base


def _load_bf16(resource, offset, vector_width, cache_modifier):
    values = []
    for chunk in range_constexpr(vector_width // 8):
        loaded = fx.Vector(
            buffer_ops.buffer_load(
                resource,
                offset + fx.Int32(chunk * 8),
                vec_width=8,
                dtype=T.bf16,
                cache_modifier=cache_modifier,
            )
        )
        values.extend(loaded[element] for element in range_constexpr(8))
    return fx.Vector.from_elements(values, fx.BFloat16)


def _decode_unscaled_fp8_f32(words):
    """Decode packed FP8 directly to FP32 when no E8M0 scale is needed."""

    values = []
    for word in range_constexpr(len(words)):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), words[word], bool(half))
            )
            values.extend((pair[0], pair[1]))
    return values


def _decode_scaled_fp8_bf16(words, scale):
    """Decode packed FP8 directly to the BF16 final-output representation."""

    values = []
    for word in range_constexpr(len(words)):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16),
                    arith.unwrap(words[word]),
                    arith.unwrap(scale),
                    bool(half),
                )
            )
            values.extend((pair[0], pair[1]))
    return values


def _atomic_add_i32_agent(addr, value):
    return llvm_d.AtomicRMWOp(
        llvm_d.AtomicBinOp.add,
        comm_ops._to_ptr_global(addr),
        arith.unwrap(value),
        llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    ).res


def _wait_i32_system_until_at_least(addr, expected):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(32),
            comm_ops._to_ptr_global(addr),
            alignment=4,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.OneAs,
        ).result

    loop = scf.WhileOp([T.i32], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        llvm_d.InlineAsmOp(None, [], "s_sleep 1", "", has_side_effects=True)
        scf.YieldOp([load()])
    return loop.results[0]


def _wait_i32_agent_until_at_least(addr, expected, *, sleep=True):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(32),
            comm_ops._to_ptr_global(addr),
            alignment=4,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        ).result

    loop = scf.WhileOp([T.i32], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        if sleep:
            llvm_d.InlineAsmOp(None, [], "s_sleep 1", "", has_side_effects=True)
        scf.YieldOp([load()])
    return loop.results[0]


def _store_i32_relaxed(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
    )


def _store_i32_agent_release(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def _store_i32_system_monotonic(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i32_system_release(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def _store_i64_relaxed(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=8,
    )


def _store_bf16(resource, offset, values, vector_width, cache_modifier=0):
    if vector_width == 8:
        buffer_ops.buffer_store(values, resource, offset, cache_modifier=cache_modifier)
        return
    for chunk in range_constexpr(vector_width // 8):
        chunk_values = fx.Vector.from_elements(
            [values[chunk * 8 + element] for element in range_constexpr(8)],
            fx.BFloat16,
        )
        buffer_ops.buffer_store(
            chunk_values,
            resource,
            offset + fx.Int32(chunk * 8),
            cache_modifier=cache_modifier,
        )


def emit_tile(
    config,
    workspace,
    workspace_flat_base,
    shared_partial,
    shared_partial_flat_base,
    specialized_rank,
    n_tile,
    tid,
    service_group,
    service_smem_base,
    *,
    hidden_dim,
    topk,
    tp_size,
    slots,
    producer_counter_stride,
):
    rank = fx.Int32(specialized_rank)
    payload_bytes = config.payload_bytes
    partial_bytes = config.partial_bytes
    reduce_items = config.m * config.tile_n // config.vector_width
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

    route_resource = buffer_ops.create_buffer_resource_from_addr(
        local_workspace_base + fx.Int64(config.route_offset),
        num_records_bytes=config.route_bytes,
    )
    output_resource = buffer_ops.create_buffer_resource_from_addr(
        local_workspace_base + fx.Int64(config.output_offset),
        num_records_bytes=payload_bytes,
    )
    shared_resource = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(shared_partial)),
        num_records_bytes=payload_bytes,
    )
    if config.producer_mode == "atomic_shared":
        producer_resource = shared_resource
    else:
        producer_resource = route_resource
    partial_resource = buffer_ops.create_buffer_resource_from_addr(
        local_workspace_base + slot * fx.Int64(partial_bytes),
        num_records_bytes=partial_bytes,
    )
    if config.collective == "rsag":
        reduced_resource = buffer_ops.create_buffer_resource_from_addr(
            local_workspace_base
            + fx.Int64(config.reduced_offset)
            + slot * fx.Int64(config.reduced_shard_bytes),
            num_records_bytes=config.reduced_shard_bytes,
        )
    service_stride = config.block_threads * config.service_groups
    service_start = tid + service_group * fx.Int32(config.block_threads)
    retain_local_partials = (
        config.collective == "direct"
        and reduce_items % service_stride == 0
        and reduce_items <= 4 * service_stride
    )
    if config.uses_rsag:
        reuse_waiter = scf.IfOp(arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(tp_size)))
        with ir.InsertionPoint(reuse_waiter.then_block):
            gather_slot = state_n_tile * fx.Int32(tp_size) + tid
            _wait_i32_system_until_at_least(
                local_workspace_base
                + fx.Int64(config.gather_done_offset)
                + fx.Int64(gather_slot) * fx.Int64(4),
                expected_i32 - fx.Int32(slots),
            )
            scf.YieldOp([])
        gpu.barrier()

    def emit_local_reduce_item(item):
        token = item // fx.Int32(config.tile_n // config.vector_width)
        tile_item = item - token * fx.Int32(config.tile_n // config.vector_width)
        output_offset = (
            token * fx.Int32(hidden_dim)
            + n_tile * fx.Int32(config.tile_n)
            + tile_item * fx.Int32(config.vector_width)
        )
        if config.producer_mode == "atomic_shared":
            reduced_f32 = _load_bf16(
                producer_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).extf(T.vec(config.vector_width, T.f32))
        else:
            shared_values = _load_bf16(
                shared_resource,
                output_offset,
                config.vector_width,
                config.local_load_cache_modifier,
            ).extf(T.vec(config.vector_width, T.f32))

            def load_bf16_route(route_slot):
                route_offset = (
                    (token * fx.Int32(topk) + fx.Int32(route_slot))
                    * fx.Int32(hidden_dim)
                    + n_tile * fx.Int32(config.tile_n)
                    + tile_item * fx.Int32(config.vector_width)
                )
                return _load_bf16(
                    producer_resource,
                    route_offset,
                    config.vector_width,
                    config.local_load_cache_modifier,
                ).extf(T.vec(config.vector_width, T.f32))

            def load_fp8_route(route_slot):
                column = n_tile * fx.Int32(config.tile_n) + tile_item * fx.Int32(
                    config.vector_width
                )
                route_row_offset = (
                    token * fx.Int32(topk) + fx.Int32(route_slot)
                ) * fx.Int32(hidden_dim)
                values = []
                for chunk in range_constexpr(config.vector_width // 8):
                    chunk_column = column + fx.Int32(chunk * 8)
                    words = load_fp8_words(
                        producer_resource,
                        (route_row_offset + chunk_column) // fx.Int32(4),
                        word_count=2,
                        load_width=2,
                        cache_modifier=config.local_load_cache_modifier,
                    )
                    values.extend(_decode_unscaled_fp8_f32(words))
                return fx.Vector.from_elements(values, fx.Float32)

            def load_route(route_slot):
                if const_expr(config.producer_mode == "routes_fp8_fixed"):
                    return load_fp8_route(route_slot)
                return load_bf16_route(route_slot)

            local_even = shared_values + load_route(0)
            if const_expr(topk == 1):
                reduced_f32 = local_even
            else:
                local_odd = load_route(1)
                for route_slot in range_constexpr(2, topk):
                    if const_expr(route_slot == topk - 1 or route_slot % 2 == 0):
                        local_odd = local_odd + load_route(route_slot)
                    else:
                        local_even = local_even + load_route(route_slot)
                reduced_f32 = local_even + local_odd

        quant_scale = fx.Int32((254 - config.fp8_scale_exponent) << 23).bitcast(
            fx.Float32
        )
        packed_words = config.vector_width // 4
        packed = pack_fp8_words(reduced_f32, quant_scale, packed_words)
        store_fp8_words(partial_resource, output_offset, packed, packed_words)
        return packed

    retained_local_partials = []

    def emit_local_reduce_items():
        if retain_local_partials:
            for iteration in range_constexpr(reduce_items // service_stride):
                item = service_start + fx.Int32(iteration * service_stride)
                retained_local_partials.append(emit_local_reduce_item(item))
        else:
            local_reduce_loop = scf.ForOp(
                arith.index_cast(T.index, service_start),
                arith.constant(reduce_items, index=True),
                arith.constant(service_stride, index=True),
            )
            with ir.InsertionPoint(local_reduce_loop.body):
                emit_local_reduce_item(
                    arith.index_cast(T.i32, local_reduce_loop.induction_variable)
                )
                scf.YieldOp([])
        fx.rocdl.s_waitcnt(0)
        gpu.barrier()

    if not config.shared_bf16_partials:
        emit_local_reduce_items()

    def load_reduced_partials(
        offset,
        retained_local_partial=None,
        source_rotation=None,
        vector_width=None,
    ):
        load_vector_width = vector_width or config.vector_width

        def load_peer(peer, cache_modifier=None):
            if config.shared_bf16_partials:
                peer_resource = buffer_ops.create_buffer_resource_from_addr(
                    peer_base(shared_partial_flat_base, peer),
                    num_records_bytes=payload_bytes,
                )
                return _load_bf16(
                    peer_resource,
                    offset,
                    load_vector_width,
                    (
                        cache_modifier
                        if cache_modifier is not None
                        else (
                            config.remote_load_cache_modifier
                            if peer != specialized_rank
                            else config.local_load_cache_modifier
                        )
                    ),
                ).extf(T.vec(load_vector_width, T.f32))
            if const_expr(retain_local_partials and peer == specialized_rank):
                scale = fx.Int32(config.fp8_scale_exponent << 23).bitcast(fx.Float32)
                return fx.Vector.from_elements(
                    decode_scaled_fp8_f32(retained_local_partial, scale),
                    fx.Float32,
                )
            load_offset = offset
            peer_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base(workspace_flat_base, peer) + slot * fx.Int64(partial_bytes),
                num_records_bytes=partial_bytes,
            )
            if cache_modifier is None:
                cache_modifier = (
                    config.remote_load_cache_modifier
                    if peer != specialized_rank
                    else config.local_load_cache_modifier
                )
            words = load_fp8_words(
                peer_resource,
                load_offset // fx.Int32(4),
                word_count=load_vector_width // 4,
                load_width=load_vector_width // 4,
                cache_modifier=cache_modifier,
            )
            scale = fx.Int32(config.fp8_scale_exponent << 23).bitcast(fx.Float32)
            return fx.Vector.from_elements(
                decode_scaled_fp8_f32(words, scale),
                fx.Float32,
            )

        # Rotate source ranks to spread peer traffic.
        if source_rotation is not None:
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
            reduced_even = load_peer(specialized_rank)
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
            _store_bf16(
                output_resource,
                offset,
                reduced.truncf(T.vec(config.vector_width, T.bf16)),
                config.vector_width,
            )

        if retain_local_partials:
            for iteration in range_constexpr(reduce_items // service_stride):
                item = service_start + fx.Int32(iteration * service_stride)
                emit_allreduce_item(
                    item,
                    retained_local_partials[iteration],
                )
        else:
            allreduce_loop = scf.ForOp(
                arith.index_cast(T.index, service_start),
                arith.constant(reduce_items, index=True),
                arith.constant(service_stride, index=True),
            )
            with ir.InsertionPoint(allreduce_loop.body):
                emit_allreduce_item(
                    arith.index_cast(T.i32, allreduce_loop.induction_variable)
                )
                scf.YieldOp([])

    def reset_tile_state_values():
        for tile_delta in range_constexpr(config.service_tile_group):
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.producer_done_offset)
                + fx.Int64(state_n_tile + fx.Int32(tile_delta))
                * fx.Int64(producer_counter_stride),
                fx.Int32(0),
            )
        if config.service_groups > 1:
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.service_done_offset)
                + tile_byte_offset,
                fx.Int32(0),
            )
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.reduce_done_offset)
                + tile_byte_offset,
                fx.Int32(0),
            )
            _store_i32_relaxed(
                local_workspace_base
                + fx.Int64(config.gather_service_done_offset)
                + tile_byte_offset,
                fx.Int32(0),
            )
        _store_i64_relaxed(epoch_address, expected)

    def emit_rsag_reduce():
        collective_vector_width = config.vector_width

        def emit_gather_ack(barrier=True):
            gather_ack = scf.IfOp(arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(tp_size)))
            with ir.InsertionPoint(gather_ack.then_block):
                remote_slot = state_n_tile * fx.Int32(tp_size) + rank
                _store_i32_system_monotonic(
                    peer_base(workspace_flat_base, tid)
                    + fx.Int64(config.gather_done_offset)
                    + fx.Int64(remote_slot) * fx.Int64(4),
                    expected_i32,
                )
                scf.YieldOp([])
            if barrier:
                gpu.barrier()

        def publish_gather_completion():
            if config.service_groups == 1:
                if config.collective == "rs_broadcast":
                    emit_gather_ack()
                else:
                    gather_release = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(gather_release.then_block):
                        comm_ops.fence_system_release()
                        scf.YieldOp([])
                    gpu.barrier()
                    emit_gather_ack()
            else:
                gather_done_address = (
                    local_workspace_base
                    + fx.Int64(config.gather_service_done_offset)
                    + tile_byte_offset
                )
                gather_publisher = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(gather_publisher.then_block):
                    comm_ops.fence_agent_release()
                    arrival = fx.Int32(
                        _atomic_add_i32_agent(gather_done_address, fx.Int32(1))
                    )
                    SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).store(arrival)
                    scf.YieldOp([])
                gpu.barrier()
                gather_arrival = fx.Int32(
                    SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).load()
                )
                coordinator_condition = arith.cmpi(
                    CmpIPredicate.eq,
                    gather_arrival,
                    fx.Int32(config.service_groups * config.service_tile_group - 1),
                )
                coordinator = scf.IfOp(coordinator_condition)
                with ir.InsertionPoint(coordinator.then_block):
                    gather_acquirer = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(gather_acquirer.then_block):
                        comm_ops.fence_agent_acquire()
                        comm_ops.fence_system_release()
                        scf.YieldOp([])
                    gpu.barrier()
                    emit_gather_ack(barrier=False)
                    fx.rocdl.s_waitcnt(0)
                    gather_resetter = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(gather_resetter.then_block):
                        reset_tile_state_values()
                        scf.YieldOp([])
                    scf.YieldOp([])

        def emit_reduce_scatter_items():
            vectors_per_token = config.tile_n // collective_vector_width
            shard_tokens = config.m // tp_size
            service_item = tid + service_group * fx.Int32(config.block_threads)
            first_shard_token = service_item // fx.Int32(vectors_per_token)
            vector_lane = service_item - first_shard_token * fx.Int32(vectors_per_token)

            reduce_scatter_loop = scf.ForOp(
                arith.index_cast(T.index, first_shard_token),
                arith.constant(shard_tokens, index=True),
                arith.constant(
                    config.block_threads * config.service_groups // vectors_per_token,
                    index=True,
                ),
            )
            with ir.InsertionPoint(reduce_scatter_loop.body):
                shard_token = arith.index_cast(
                    T.i32, reduce_scatter_loop.induction_variable
                )
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
                        if config.collective == "rsag" and config.service_groups > 1
                        else None
                    ),
                    vector_width=collective_vector_width,
                )
                reduced_bf16 = reduced.truncf(T.vec(collective_vector_width, T.bf16))
                if config.collective == "rsag":
                    # Publish the local RS/AG shard directly.
                    _store_bf16(
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
                if config.collective == "rs_broadcast":
                    for peer_step in range_constexpr(tp_size):
                        peer = (specialized_rank + peer_step + 1) % tp_size
                        if peer == specialized_rank:
                            peer_output_resource = output_resource
                        else:
                            peer_output_resource = (
                                buffer_ops.create_buffer_resource_from_addr(
                                    peer_base(workspace_flat_base, peer)
                                    + fx.Int64(config.output_offset),
                                    num_records_bytes=payload_bytes,
                                )
                            )
                        _store_bf16(
                            peer_output_resource,
                            offset,
                            reduced_bf16,
                            collective_vector_width,
                            config.remote_store_cache_modifier,
                        )
                else:
                    reduced_quant_scale = fx.Int32(
                        (254 - config.fp8_scale_exponent) << 23
                    ).bitcast(fx.Float32)
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
                scf.YieldOp([])
            fx.rocdl.s_waitcnt(0)
            gpu.barrier()

        emit_reduce_scatter_items()

        if config.collective == "rs_broadcast":
            publish_gather_completion()
            return

        def emit_reduced_exchange(propagate_acquire):
            reduced_exchange = scf.IfOp(
                arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(tp_size))
            )
            with ir.InsertionPoint(reduced_exchange.then_block):
                remote_slot = state_n_tile * fx.Int32(tp_size) + rank
                _store_i32_system_monotonic(
                    peer_base(workspace_flat_base, tid)
                    + fx.Int64(config.owner_ready_offset)
                    + fx.Int64(remote_slot) * fx.Int64(4),
                    expected_i32,
                )
                local_slot = state_n_tile * fx.Int32(tp_size) + tid
                _wait_i32_system_until_at_least(
                    local_workspace_base
                    + fx.Int64(config.owner_ready_offset)
                    + fx.Int64(local_slot) * fx.Int64(4),
                    expected_i32,
                )
                scf.YieldOp([])
            gpu.barrier()
            reduced_acquire = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(reduced_acquire.then_block):
                comm_ops.fence_system_acquire()
                scf.YieldOp([])
            if propagate_acquire:
                gpu.barrier()

        if config.service_groups == 1:
            reduced_release = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(reduced_release.then_block):
                comm_ops.fence_system_release()
                scf.YieldOp([])
            gpu.barrier()
            emit_reduced_exchange(True)
        else:
            reduce_done_address = (
                local_workspace_base
                + fx.Int64(config.reduce_done_offset)
                + tile_byte_offset
            )
            reduce_publisher = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(reduce_publisher.then_block):
                comm_ops.fence_agent_release()
                arrival = fx.Int32(
                    _atomic_add_i32_agent(reduce_done_address, fx.Int32(1))
                )
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).store(arrival)
                scf.YieldOp([])
            gpu.barrier()
            reduce_arrival = fx.Int32(
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).load()
            )
            coordinator = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.eq,
                    reduce_arrival,
                    fx.Int32(config.service_groups * config.service_tile_group - 1),
                )
            )
            with ir.InsertionPoint(coordinator.then_block):
                reduce_acquirer = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(reduce_acquirer.then_block):
                    comm_ops.fence_agent_acquire()
                    comm_ops.fence_system_release()
                    scf.YieldOp([])
                gpu.barrier()
                emit_reduced_exchange(False)
                reduced_publisher = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(reduced_publisher.then_block):
                    _store_i32_agent_release(
                        local_workspace_base
                        + fx.Int64(config.reduced_collective_ready_offset)
                        + fx.Int64(state_n_tile) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                scf.YieldOp([])

            reduced_waiter = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(reduced_waiter.then_block):
                _wait_i32_agent_until_at_least(
                    local_workspace_base
                    + fx.Int64(config.reduced_collective_ready_offset)
                    + fx.Int64(state_n_tile) * fx.Int64(4),
                    expected_i32,
                )
                comm_ops.fence_agent_acquire()
                scf.YieldOp([])
            gpu.barrier()

        def emit_gather_source(source, source_start, source_stride):
            gather_vector_width = config.vector_width
            gather_cache_modifier = (
                config.remote_load_cache_modifier
                if config.gather_load_cache_modifier < 0
                else config.gather_load_cache_modifier
            )
            vectors_per_token = config.tile_n // gather_vector_width
            shard_tokens = config.m // tp_size
            first_token = source_start // fx.Int32(vectors_per_token)
            vector_lane = source_start - first_token * fx.Int32(vectors_per_token)
            source_resource = buffer_ops.create_buffer_resource_from_addr(
                peer_base(workspace_flat_base, source)
                + fx.Int64(config.reduced_offset)
                + slot * fx.Int64(config.reduced_shard_bytes),
                num_records_bytes=config.reduced_shard_bytes,
            )
            gather_loop = scf.ForOp(
                arith.index_cast(T.index, first_token),
                arith.constant(shard_tokens, index=True),
                arith.constant(
                    source_stride // vectors_per_token,
                    index=True,
                ),
            )
            with ir.InsertionPoint(gather_loop.body):
                source_token = arith.index_cast(T.i32, gather_loop.induction_variable)
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
                scale = fx.Int32(config.fp8_scale_exponent << 23).bitcast(fx.Float32)
                values = fx.Vector.from_elements(
                    _decode_scaled_fp8_bf16(words, scale),
                    fx.BFloat16,
                )
                _store_bf16(
                    output_resource,
                    offset,
                    values,
                    gather_vector_width,
                    config.remote_store_cache_modifier,
                )
                scf.YieldOp([])

        def emit_remote_gather_source(source, source_start, source_stride):
            remote_source = scf.IfOp(arith.cmpi(CmpIPredicate.ne, source, rank))
            with ir.InsertionPoint(remote_source.then_block):
                emit_gather_source(source, source_start, source_stride)
                scf.YieldOp([])

        # Split SG4 waves across its two source shards.
        if config.service_groups == 1:
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
        elif config.service_groups == 4:
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
        fx.rocdl.s_waitcnt(0)
        gpu.barrier()
        publish_gather_completion()

    if config.service_groups == 1:
        release_publisher = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
        with ir.InsertionPoint(release_publisher.then_block):
            comm_ops.fence_system_release()
            scf.YieldOp([])
        if config.single_pass_direct:
            gpu.barrier()

        rank_exchange = scf.IfOp(arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(tp_size)))
        with ir.InsertionPoint(rank_exchange.then_block):
            remote_slot = state_n_tile * fx.Int32(tp_size) + rank
            _store_i32_system_monotonic(
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
            if config.single_pass_direct:
                _wait_i32_agent_until_at_least(
                    ready_address,
                    expected_i32,
                    sleep=False,
                )
            else:
                _wait_i32_system_until_at_least(
                    ready_address,
                    expected_i32,
                )
            scf.YieldOp([])
        if not config.single_pass_direct:
            rank_acquire = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(rank_acquire.then_block):
                comm_ops.fence_system_acquire()
                scf.YieldOp([])
        gpu.barrier()
    else:
        service_done_address = (
            local_workspace_base
            + fx.Int64(config.service_done_offset)
            + tile_byte_offset
        )

        def publish_partials_and_exchange():
            service_publisher = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(service_publisher.then_block):
                comm_ops.fence_agent_release()
                arrival = fx.Int32(
                    _atomic_add_i32_agent(service_done_address, fx.Int32(1))
                )
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).store(arrival)
                scf.YieldOp([])
            gpu.barrier()
            service_arrival = fx.Int32(
                SmemPtr(service_smem_base, 0, T.i32, shape=(1,)).load()
            )
            coordinator = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.eq,
                    service_arrival,
                    fx.Int32(config.service_groups * config.service_tile_group - 1),
                )
            )
            with ir.InsertionPoint(coordinator.then_block):
                service_acquirer = scf.IfOp(
                    arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                )
                with ir.InsertionPoint(service_acquirer.then_block):
                    comm_ops.fence_agent_acquire()
                    local_ready_slot = state_n_tile * fx.Int32(tp_size) + rank
                    _store_i32_system_release(
                        local_workspace_base
                        + fx.Int64(config.rank_ready_offset)
                        + fx.Int64(local_ready_slot) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                gpu.barrier()

                rank_waiter = scf.IfOp(
                    arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(tp_size))
                )
                with ir.InsertionPoint(rank_waiter.then_block):
                    peer_ready_slot = state_n_tile * fx.Int32(tp_size) + tid
                    _wait_i32_system_until_at_least(
                        peer_base(workspace_flat_base, tid)
                        + fx.Int64(config.rank_ready_offset)
                        + fx.Int64(peer_ready_slot) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                gpu.barrier()
                rank_acquire = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
                with ir.InsertionPoint(rank_acquire.then_block):
                    comm_ops.fence_system_acquire()
                    _store_i32_agent_release(
                        local_workspace_base
                        + fx.Int64(config.collective_ready_offset)
                        + fx.Int64(state_n_tile) * fx.Int64(4),
                        expected_i32,
                    )
                    scf.YieldOp([])
                scf.YieldOp([])

        publish_partials_and_exchange()

        def wait_for_collective():
            collective_waiter = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
            with ir.InsertionPoint(collective_waiter.then_block):
                _wait_i32_agent_until_at_least(
                    local_workspace_base
                    + fx.Int64(config.collective_ready_offset)
                    + fx.Int64(state_n_tile) * fx.Int64(4),
                    expected_i32,
                )
                comm_ops.fence_agent_acquire()
                scf.YieldOp([])
            gpu.barrier()

    if config.uses_rsag:
        if config.service_groups > 1:
            wait_for_collective()
        emit_rsag_reduce()
    else:
        emit_direct_reduce()
    fx.rocdl.s_waitcnt(0)
    if not (config.uses_rsag and config.service_groups > 1):
        reset_condition = arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
        state_resetter = scf.IfOp(reset_condition)
        with ir.InsertionPoint(state_resetter.then_block):
            reset_tile_state_values()
            scf.YieldOp([])
