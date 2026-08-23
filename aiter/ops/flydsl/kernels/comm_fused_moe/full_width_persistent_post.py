# SPDX-License-Identifier: Apache-2.0
"""Persistent full-width TP communication.

This module contains only the best M=2048 post pipelines found by the isolated
probe. Stage2 GEMM remains a separate launch. The communication-only form is
used with the ordinary local reduction, while the two-kernel form performs:

    local route/shared reduce -> compressed reduce-scatter -> all-gather

The constants are intentionally fixed until a different shape is promoted by
measurement.  Keeping one implementation here makes it the clean baseline for
later producer/service overlap work.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from . import full_width
from .collectives import (
    decode_scaled_fp8_f32,
    e8m0_scale,
    emit_tp_all_gather,
    emit_tp_reduce_scatter,
    load_e8m0_scale,
    load_fp8_words,
    pack_fp8_words,
    store_fp8_words,
)
from .persistent_window import _store_agent
from .sync import peer_base


POST_GRID = 91
BLOCK = 256
LOCAL_POST_GRID = 256
LOCAL_BLOCK = 512
LOCAL_VECTOR_WIDTH = 8
ROUTE_CACHE_MODIFIER = 0
SHARED_CACHE_MODIFIER = 0
REDUCE_SCATTER_GRID = 76
ALL_GATHER_GRID = 91
PARTIAL_GATE_OFFSET = 0
REDUCED_GATE_OFFSET = 8
WORKER_EPOCH_OFFSET = 16
WORKER_EPOCH_BYTES = LOCAL_POST_GRID * 8
LOCAL_COUNTER_OFFSET = WORKER_EPOCH_OFFSET + WORKER_EPOCH_BYTES
STATE_BYTES = (LOCAL_COUNTER_OFFSET + 8 + 255) // 256 * 256


def _wait_agent(addr, expected):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(64),
            comm_ops._to_ptr_global(addr),
            alignment=8,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        ).result

    loop = scf.WhileOp([T.i64], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i64])
    after = ir.Block.create_at_start(loop.after, [T.i64])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        scf.YieldOp([load()])
    return loop.results[0]


def _emit_wave_local_tile(
    route_row,
    payload_row,
    scale_row,
    shared_row,
    column,
):
    route_row_bytes = full_width.H + full_width.H // 8
    lane = fx.Int32(gpu.thread_idx.x) & fx.Int32(63)
    acc = fx.Vector.filled(LOCAL_VECTOR_WIDTH, 0.0, fx.Float32)

    def load_source(route_slot):
        source_words = []
        source_scales = []
        for chunk in range_constexpr(LOCAL_VECTOR_WIDTH // 8):
            source_words.append(
                load_fp8_words(
                    route_row,
                    fx.Int32(route_slot * (route_row_bytes // 4))
                    + column // fx.Int32(4)
                    + fx.Int32(chunk * 2),
                    word_count=2,
                    load_width=2,
                    cache_modifier=ROUTE_CACHE_MODIFIER,
                )
            )
            source_scales.append(
                load_e8m0_scale(
                    route_row,
                    fx.Int32(route_slot * route_row_bytes + full_width.H)
                    + column // fx.Int32(8)
                    + fx.Int32(chunk),
                    ROUTE_CACHE_MODIFIER,
                )
            )
        return source_words, source_scales

    def accumulate(source_words, source_scales, current):
        values = []
        for chunk in range_constexpr(LOCAL_VECTOR_WIDTH // 8):
            values.extend(
                decode_scaled_fp8_f32(
                    source_words[chunk], source_scales[chunk]
                )
            )
        return current + fx.Vector.from_elements(values, fx.Float32)

    for route_slot in range_constexpr(full_width.TOPK):
        words, scales = load_source(route_slot)
        acc = accumulate(words, scales, acc)

    shared_values = []
    for chunk in range_constexpr(LOCAL_VECTOR_WIDTH // 8):
        values = fx.Vector(
            buffer_ops.buffer_load(
                shared_row,
                column + fx.Int32(chunk * 8),
                vec_width=8,
                dtype=T.bf16,
                cache_modifier=SHARED_CACHE_MODIFIER,
            )
        ).extf(T.vec(8, T.f32))
        shared_values.extend(values[i] for i in range_constexpr(8))
    acc = acc + fx.Vector.from_elements(shared_values, fx.Float32)

    local_max = fx.Float32(1e-10).maximumf(
        fmath.absf(acc).reduce(ReductionOp.MAX)
    )
    max_bits = local_max.bitcast(fx.Int32)
    for xor_lane in (1, 2):
        remote_bits = fx.rocdl.ds_bpermute(
            T.i32,
            (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
            max_bits,
        )
        local_max = local_max.maximumf(
            fx.Int32(remote_bits).bitcast(fx.Float32)
        )
        max_bits = local_max.bitcast(fx.Int32)
    e8m0, quant_scale = e8m0_scale(local_max)
    store_fp8_words(
        payload_row,
        column,
        pack_fp8_words(acc, quant_scale, LOCAL_VECTOR_WIDTH // 4),
        LOCAL_VECTOR_WIDTH // 4,
    )

    scale_leader = scf.IfOp(
        arith.cmpi(
            CmpIPredicate.eq,
            lane & fx.Int32(32 // LOCAL_VECTOR_WIDTH - 1),
            fx.Int32(0),
        )
    )
    with ir.InsertionPoint(scale_leader.then_block):
        buffer_ops.buffer_store(
            e8m0.to(fx.Int8),
            scale_row,
            column // fx.Int32(32),
            offset_is_bytes=True,
        )
        scf.YieldOp([])


def _emit_wave_local(config, route, partial, shared, worker):
    route_row_bytes = full_width.H + full_width.H // 8
    tid = fx.Int32(gpu.thread_idx.x)
    lane_column = tid * fx.Int32(LOCAL_VECTOR_WIDTH)
    tokens = scf.ForOp(
        arith.index_cast(T.index, worker),
        arith.constant(config.m, index=True),
        arith.constant(LOCAL_POST_GRID, index=True),
    )
    with ir.InsertionPoint(tokens.body):
        token = arith.index_cast(T.i32, tokens.induction_variable)
        route_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(route))
            + fx.Int64(token) * fx.Int64(full_width.TOPK * route_row_bytes),
            num_records_bytes=full_width.TOPK * route_row_bytes,
        )
        shared_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(shared))
            + fx.Int64(token) * fx.Int64(full_width.H * 2),
            num_records_bytes=full_width.H * 2,
        )
        payload_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(partial))
            + fx.Int64(token) * fx.Int64(full_width.H),
            num_records_bytes=full_width.H,
        )
        scale_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(partial))
            + fx.Int64(config.m * full_width.H)
            + fx.Int64(token) * fx.Int64(full_width.GROUPS_PER_ROW),
            num_records_bytes=full_width.GROUPS_PER_ROW,
        )
        for tile in range_constexpr(2):
            _emit_wave_local_tile(
                route_row,
                payload_row,
                scale_row,
                shared_row,
                lane_column
                + fx.Int32(tile * LOCAL_BLOCK * LOCAL_VECTOR_WIDTH),
            )
        scf.YieldOp([])


def _publish_local_ready_atomic(local_state, publish_addr, tid):
    """Publish local completion without a coordinator slot scan."""

    fx.rocdl.s_waitcnt(0)
    gpu.barrier()
    leader = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0)))
    with ir.InsertionPoint(leader.then_block):
        counter = local_state + fx.Int64(LOCAL_COUNTER_OFFSET)
        comm_ops.fence_agent_release()
        done = fx.Int64(comm_ops.atomic_add_agent(counter, fx.Int64(1)))
        generation = done // fx.Int64(LOCAL_POST_GRID) + fx.Int64(1)
        last = scf.IfOp(
            arith.cmpi(
                CmpIPredicate.eq,
                done,
                generation * fx.Int64(LOCAL_POST_GRID) - fx.Int64(1),
            )
        )
        with ir.InsertionPoint(last.then_block):
            comm_ops.fence_agent_acquire()
            comm_ops.store_i64_global_system(publish_addr, generation)
            scf.YieldOp([])
        scf.YieldOp([])


def _publish_reduced_ready(
    local_state,
    reduced_payload_base,
    reduced_ready,
    rank,
    worker,
    expected,
):
    tid = fx.Int32(gpu.thread_idx.x)
    worker_epoch_base = local_state + fx.Int64(WORKER_EPOCH_OFFSET)
    mark_done = scf.IfOp(
        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
    )
    with ir.InsertionPoint(mark_done.then_block):
        _store_agent(worker_epoch_base + fx.Int64(worker) * fx.Int64(8), expected)
        scf.YieldOp([])

    coordinator = scf.IfOp(
        arith.cmpi(CmpIPredicate.eq, worker, fx.Int32(0))
    )
    with ir.InsertionPoint(coordinator.then_block):
        scan = scf.IfOp(
            arith.cmpi(CmpIPredicate.ult, tid, fx.Int32(REDUCE_SCATTER_GRID))
        )
        with ir.InsertionPoint(scan.then_block):
            _wait_agent(
                worker_epoch_base + fx.Int64(tid) * fx.Int64(8), expected
            )
            comm_ops.fence_agent_acquire()
            scf.YieldOp([])
        gpu.barrier()
        publish = scf.IfOp(
            arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
        )
        with ir.InsertionPoint(publish.then_block):
            comm_ops.store_i64_global_system(
                peer_base(reduced_payload_base, rank) + fx.Int64(reduced_ready),
                expected,
            )
            scf.YieldOp([])
        scf.YieldOp([])


@functools.cache
def compile_stage2_post(
    config: full_width.Config,
    include_local: bool = False,
):
    """Compile the communication post kernel, optionally including local reduce."""

    post_grid = LOCAL_POST_GRID if include_local else POST_GRID
    block_size = LOCAL_BLOCK if include_local else BLOCK
    partial_ready = config.m * (full_width.H + full_width.H // 32)
    reduced_ready = config.shard_rows * full_width.H

    @flyc.kernel(
        name=(
            "flydsl_fused_moe_full_persistent_post"
            f"_sr{config.shard_rows}_g{post_grid}_lr{int(include_local)}"
            f"_b{block_size}_cb{BLOCK}_v{LOCAL_VECTOR_WIDTH}"
            f"_rc{ROUTE_CACHE_MODIFIER}_sc{SHARED_CACHE_MODIFIER}"
            f"_rs{REDUCE_SCATTER_GRID}_ag{ALL_GATHER_GRID}"
        ),
        known_block_size=[block_size, 1, 1],
    )
    def kernel(
        route: fx.Pointer,
        shared: fx.Pointer,
        partial: fx.Pointer,
        partial_base: fx.Int64,
        owner_output: fx.Pointer,
        output: fx.Pointer,
        reduced_payload: fx.Pointer,
        reduced_payload_base: fx.Int64,
        reduced_scale: fx.Pointer,
        reduced_scale_base: fx.Int64,
        state: fx.Pointer,
        rank: fx.Int32,
    ):
        worker = fx.Int32(gpu.block_idx.x)
        tid = fx.Int32(gpu.thread_idx.x)
        local_state = fx.Int64(ptrtoint(state))
        worker_epoch = (
            local_state
            + fx.Int64(WORKER_EPOCH_OFFSET)
            + fx.Int64(worker) * fx.Int64(8)
        )
        expected = fx.Int64(comm_ops.load_i64_global(worker_epoch)) + fx.Int64(1)

        if include_local:
            _emit_wave_local(config, route, partial, shared, worker)
            _publish_local_ready_atomic(
                local_state,
                peer_base(partial_base, rank) + fx.Int64(partial_ready),
                tid,
            )

        if worker >= fx.Int32(REDUCE_SCATTER_GRID):
            if worker < fx.Int32(ALL_GATHER_GRID):
                if tid == fx.Int32(0):
                    llvm_d.StoreOp(
                        arith.unwrap(expected),
                        comm_ops._to_ptr_global(worker_epoch),
                        alignment=8,
                        ordering=llvm_d.AtomicOrdering.monotonic,
                        syncscope=fx.rocdl.SyncScope.AgentOneAs,
                    )

        active = scf.IfOp(
            arith.cmpi(CmpIPredicate.ult, worker, fx.Int32(ALL_GATHER_GRID))
        )
        with ir.InsertionPoint(active.then_block):
            if worker == fx.Int32(0):
                if not include_local:
                    if tid == fx.Int32(0):
                        comm_ops.store_i64_global_system(
                            peer_base(partial_base, rank)
                            + fx.Int64(partial_ready),
                            expected,
                        )

            if worker == fx.Int32(0):
                if tid < fx.Int32(full_width.TP):
                    comm_ops.wait_i64_system_until_at_least(
                        peer_base(partial_base, tid)
                        + fx.Int64(partial_ready),
                        expected,
                    )
                    comm_ops.fence_system_acquire()
                gpu.barrier()
                if tid == fx.Int32(0):
                    _store_agent(
                        local_state + fx.Int64(PARTIAL_GATE_OFFSET), expected
                    )
            if tid == fx.Int32(0):
                _wait_agent(local_state + fx.Int64(PARTIAL_GATE_OFFSET), expected)
                comm_ops.fence_agent_acquire()
            gpu.barrier()

            active_rs = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.ult,
                    worker,
                    fx.Int32(REDUCE_SCATTER_GRID),
                )
            )
            with ir.InsertionPoint(active_rs.then_block):
                def emit_reduce_scatter():
                    emit_tp_reduce_scatter(
                        partial_base,
                        owner_output,
                        reduced_payload,
                        reduced_scale,
                        rank,
                        worker,
                        tokens=config.m,
                        output_width=full_width.H,
                        payload_width=full_width.H,
                        shard_rows=config.shard_rows,
                        tp=full_width.TP,
                        block=BLOCK,
                        reduce_scatter_grid=REDUCE_SCATTER_GRID,
                    )

                if not include_local:
                    emit_reduce_scatter()
                else:
                    collective_thread = scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.ult,
                            tid,
                            fx.Int32(BLOCK),
                        )
                    )
                    with ir.InsertionPoint(collective_thread.then_block):
                        emit_reduce_scatter()
                        scf.YieldOp([])
                gpu.barrier()
                _publish_reduced_ready(
                    local_state,
                    reduced_payload_base,
                    reduced_ready,
                    rank,
                    worker,
                    expected,
                )
                scf.YieldOp([])

            if worker == fx.Int32(0):
                if tid < fx.Int32(full_width.TP):
                    comm_ops.wait_i64_system_until_at_least(
                        peer_base(reduced_payload_base, tid)
                        + fx.Int64(reduced_ready),
                        expected,
                    )
                    comm_ops.fence_system_acquire()
                gpu.barrier()
                if tid == fx.Int32(0):
                    _store_agent(
                        local_state + fx.Int64(REDUCED_GATE_OFFSET), expected
                    )
            if tid == fx.Int32(0):
                _wait_agent(local_state + fx.Int64(REDUCED_GATE_OFFSET), expected)
                comm_ops.fence_agent_acquire()
            gpu.barrier()

            def emit_all_gather():
                emit_tp_all_gather(
                    reduced_payload_base,
                    reduced_scale_base,
                    output,
                    rank,
                    worker,
                    output_width=full_width.H,
                    payload_width=full_width.H,
                    shard_rows=config.shard_rows,
                    tp=full_width.TP,
                    block=BLOCK,
                    all_gather_grid=ALL_GATHER_GRID,
                )

            if not include_local:
                emit_all_gather()
            else:
                collective_thread = scf.IfOp(
                    arith.cmpi(
                        CmpIPredicate.ult,
                        tid,
                        fx.Int32(BLOCK),
                    )
                )
                with ir.InsertionPoint(collective_thread.then_block):
                    emit_all_gather()
                    scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch(
        route,
        shared,
        partial,
        partial_base,
        owner_output,
        output,
        reduced_payload,
        reduced_payload_base,
        reduced_scale,
        reduced_scale_base,
        state,
        rank,
        stream,
    ):
        kernel(
            route,
            shared,
            partial,
            partial_base,
            owner_output,
            output,
            reduced_payload,
            reduced_payload_base,
            reduced_scale,
            reduced_scale_base,
            state,
            rank,
        ).launch(
            grid=(post_grid, 1, 1),
            block=(block_size, 1, 1),
            stream=stream,
        )

    return launch
