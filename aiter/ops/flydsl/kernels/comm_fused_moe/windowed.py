# SPDX-License-Identifier: Apache-2.0
"""Windowed communication-fused Stage2 kernels."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import buffer_ops
from ..mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm2_common
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


M = 32768
H = 7168
I = 384
E = 384
TOPK = 6
TP = 8
TILE_M = 64
TILE_N = 256
TILE_K = 128
WINDOW = 1024
SLOTS = 2
TILES_PER_WINDOW = WINDOW // TILE_N
BLOCK = 256
LOCAL_WORKERS = 2048
REDUCE_SCATTER_GRID = 92
ALL_GATHER_GRID = 91
SHARD_ROWS = M // TP
GROUPS_PER_ROW = WINDOW // 32


def _compile_compute(window: int, compose=None):
    return compile_mixed_moe_gemm2_common(
        model_dim=H,
        inter_dim=I,
        experts=E,
        topk=TOPK,
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="fp8",
        accumulate=False,
        persist_m=1,
        _n_tile_range=(
            window * TILES_PER_WINDOW,
            (window + 1) * TILES_PER_WINDOW,
        ),
        _compose_entry=compose,
    )


@functools.cache
def compile_stage2_compute(window: int):
    """Compile one compact 1024-column Stage2 producer."""
    return _compile_compute(window)


def _emit_local(route, partial, shared, worker):
    work = scf.ForOp(
        arith.index_cast(T.index, worker),
        arith.constant(M, index=True),
        arith.constant(LOCAL_WORKERS, index=True),
    )
    with ir.InsertionPoint(work.body):
        token = arith.index_cast(T.i32, work.induction_variable)
        tid = fx.Int32(gpu.thread_idx.x)
        column = tid * fx.Int32(8)
        active = scf.IfOp(arith.cmpi(CmpIPredicate.ult, column, fx.Int32(WINDOW)))
        with ir.InsertionPoint(active.then_block):
            route_row_bytes = WINDOW + WINDOW // 8
            route_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(route))
                + fx.Int64(token) * fx.Int64(TOPK * route_row_bytes),
                num_records_bytes=TOPK * route_row_bytes,
            )
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            for slot in range_constexpr(TOPK):
                words = load_fp8_words(
                    route_row,
                    fx.Int32(slot * (route_row_bytes // 4))
                    + column // fx.Int32(4),
                    word_count=2,
                    load_width=2,
                    cache_modifier=2,
                )
                scale = load_e8m0_scale(
                    route_row,
                    fx.Int32(slot * route_row_bytes + WINDOW)
                    + column // fx.Int32(8),
                    2,
                )
                values = decode_scaled_fp8_f32(words, scale)
                acc = acc + fx.Vector.from_elements(values, fx.Float32)

            shared_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(shared)) + fx.Int64(token) * fx.Int64(H * 2),
                num_records_bytes=H * 2,
            )
            shared_values = fx.Vector(
                buffer_ops.buffer_load(
                    shared_row,
                    column,
                    vec_width=8,
                    dtype=T.bf16,
                    cache_modifier=2,
                )
            ).extf(T.vec(8, T.f32))
            acc = acc + shared_values

            lane = tid & fx.Int32(63)
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
            packed = pack_fp8_words(acc, quant_scale, 2)
            payload_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(WINDOW),
                num_records_bytes=WINDOW,
            )
            store_fp8_words(payload_row, column, packed, 2)
            scale_leader = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, lane & fx.Int32(3), fx.Int32(0))
            )
            with ir.InsertionPoint(scale_leader.then_block):
                scale_row = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(partial))
                    + fx.Int64(M * WINDOW)
                    + fx.Int64(token) * fx.Int64(GROUPS_PER_ROW),
                    num_records_bytes=GROUPS_PER_ROW,
                )
                buffer_ops.buffer_store(
                    e8m0.to(fx.Int8),
                    scale_row,
                    column // fx.Int32(32),
                    offset_is_bytes=True,
                )
                scf.YieldOp([])
            scf.YieldOp([])
        scf.YieldOp([])


def _compose_cycle(*, has_reduce_scatter: bool, has_all_gather: bool):
    def compose(
        *,
        module_name,
        emit_gemm2,
        allocator,
    ):
        @flyc.kernel(
            name=(
                f"{module_name}_comm_cycle_rs{int(has_reduce_scatter)}"
                f"ag{int(has_all_gather)}"
            ),
            known_block_size=[BLOCK, 1, 1],
        )
        def kernel(
            route_out: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            bias: fx.Pointer,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
            local_route: fx.Pointer,
            local_partial: fx.Pointer,
            local_shared: fx.Pointer,
            partial_flat_base: fx.Int64,
            reduced_shard: fx.Pointer,
            reduced_payload: fx.Pointer,
            reduced_scale: fx.Pointer,
            gather_payload_base: fx.Int64,
            gather_scale_base: fx.Int64,
            gathered_output: fx.Pointer,
            rank: fx.Int32,
        ):
            linear = fx.Int32(gpu.block_idx.x)

            def emit_compute(worker):
                emit_gemm2(
                    route_out,
                    x,
                    w,
                    scale_x,
                    scale_w,
                    w,
                    scale_w,
                    sorted_token_ids,
                    expert_ids,
                    sorted_weights,
                    num_valid_ids,
                    bias,
                    tokens,
                    model_dim,
                    inter_dim,
                    size_expert_ids,
                    block_id=arith.index_cast(T.index, worker),
                )

            if const_expr(has_reduce_scatter):
                paired = arith.cmpi(
                    CmpIPredicate.ult,
                    linear,
                    fx.Int32(REDUCE_SCATTER_GRID * 3),
                )
                slot = linear % fx.Int32(3)
                is_service = arith.andi(
                    paired, arith.cmpi(CmpIPredicate.eq, slot, fx.Int32(0))
                )
                is_compute = arith.ori(
                    arith.cmpi(
                        CmpIPredicate.uge,
                        linear,
                        fx.Int32(REDUCE_SCATTER_GRID * 3),
                    ),
                    arith.andi(
                        paired,
                        arith.cmpi(CmpIPredicate.ne, slot, fx.Int32(0)),
                    ),
                )
                raw_compute = arith.select(
                    paired,
                    (linear // fx.Int32(3)) * fx.Int32(2) + slot - fx.Int32(1),
                    linear - fx.Int32(REDUCE_SCATTER_GRID),
                )
                compute_worker = arith.select(is_compute, raw_compute, fx.Int32(0))
                service_worker = linear // fx.Int32(3)
            else:
                compute_worker = linear

            if const_expr(has_reduce_scatter):
                compute_if = scf.IfOp(is_compute)
                with ir.InsertionPoint(compute_if.then_block):
                    emit_compute(compute_worker)
                    scf.YieldOp([])
            else:
                emit_compute(compute_worker)

            local_active = arith.cmpi(
                CmpIPredicate.ult, compute_worker, fx.Int32(LOCAL_WORKERS)
            )
            if const_expr(has_reduce_scatter):
                local_active = arith.andi(is_compute, local_active)
            local_if = scf.IfOp(local_active)
            with ir.InsertionPoint(local_if.then_block):
                _emit_local(local_route, local_partial, local_shared, compute_worker)
                scf.YieldOp([])

            if const_expr(has_reduce_scatter):
                reduce_scatter_if = scf.IfOp(is_service)
                with ir.InsertionPoint(reduce_scatter_if.then_block):
                    emit_tp_reduce_scatter(
                        partial_flat_base,
                        reduced_shard,
                        reduced_payload,
                        reduced_scale,
                        rank,
                        service_worker,
                        tokens=M,
                        output_width=H,
                        payload_width=WINDOW,
                        shard_rows=SHARD_ROWS,
                        tp=TP,
                        block=BLOCK,
                        reduce_scatter_grid=REDUCE_SCATTER_GRID,
                    )
                    if const_expr(has_all_gather):
                        all_gather_if = scf.IfOp(
                            arith.cmpi(
                                CmpIPredicate.ult,
                                service_worker,
                                fx.Int32(ALL_GATHER_GRID),
                            )
                        )
                        with ir.InsertionPoint(all_gather_if.then_block):
                            emit_tp_all_gather(
                                gather_payload_base,
                                gather_scale_base,
                                gathered_output,
                                rank,
                                service_worker,
                                output_width=H,
                                payload_width=WINDOW,
                                shard_rows=SHARD_ROWS,
                                tp=TP,
                                block=BLOCK,
                                all_gather_grid=ALL_GATHER_GRID,
                            )
                            scf.YieldOp([])
                    scf.YieldOp([])

        @flyc.jit
        def launch(
            route_out,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            bias,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            local_route,
            local_partial,
            local_shared,
            partial_flat_base,
            reduced_shard,
            reduced_payload,
            reduced_scale,
            gather_payload_base,
            gather_scale_base,
            gathered_output,
            rank,
            stream,
        ):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            compute_workers = arith.index_cast(T.index, size_expert_ids) * arith.constant(
                TILES_PER_WINDOW, index=True
            )
            grid = compute_workers + arith.constant(
                REDUCE_SCATTER_GRID if has_reduce_scatter else 0,
                index=True,
            )
            kernel(
                route_out,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                bias,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
                local_route,
                local_partial,
                local_shared,
                partial_flat_base,
                reduced_shard,
                reduced_payload,
                reduced_scale,
                gather_payload_base,
                gather_scale_base,
                gathered_output,
                rank,
            ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

        return launch

    return compose


@functools.cache
def compile_stage2_cycle(
    window: int,
    has_reduce_scatter: bool,
    has_all_gather: bool,
):
    """Compile G/L with optional TP reduce-scatter/all-gather service CTAs."""
    return _compile_compute(
        window,
        _compose_cycle(
            has_reduce_scatter=has_reduce_scatter,
            has_all_gather=has_all_gather,
        ),
    )


@functools.cache
def compile_stage2_drain(
    has_local: bool,
    has_reduce_scatter: bool,
    has_all_gather: bool,
):
    """Compile the fixed pipeline tail without reserving GEMM LDS."""

    @flyc.kernel(
        name=(
            f"comm_fused_moe_window_drain_l{int(has_local)}"
            f"rs{int(has_reduce_scatter)}ag{int(has_all_gather)}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        route: fx.Pointer,
        partial: fx.Pointer,
        shared: fx.Pointer,
        partial_flat_base: fx.Int64,
        reduced_shard: fx.Pointer,
        reduced_payload: fx.Pointer,
        reduced_scale: fx.Pointer,
        gather_payload_base: fx.Int64,
        gather_scale_base: fx.Int64,
        gathered_output: fx.Pointer,
        rank: fx.Int32,
    ):
        worker = fx.Int32(gpu.block_idx.x)
        if const_expr(has_reduce_scatter):
            reduce_scatter_if = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.ult,
                    worker,
                    fx.Int32(REDUCE_SCATTER_GRID),
                )
            )
            with ir.InsertionPoint(reduce_scatter_if.then_block):
                emit_tp_reduce_scatter(
                    partial_flat_base,
                    reduced_shard,
                    reduced_payload,
                    reduced_scale,
                    rank,
                    worker,
                    tokens=M,
                    output_width=H,
                    payload_width=WINDOW,
                    shard_rows=SHARD_ROWS,
                    tp=TP,
                    block=BLOCK,
                    reduce_scatter_grid=REDUCE_SCATTER_GRID,
                )
                if const_expr(has_all_gather):
                    all_gather_if = scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.ult,
                            worker,
                            fx.Int32(ALL_GATHER_GRID),
                        )
                    )
                    with ir.InsertionPoint(all_gather_if.then_block):
                        emit_tp_all_gather(
                            gather_payload_base,
                            gather_scale_base,
                            gathered_output,
                            rank,
                            worker,
                            output_width=H,
                            payload_width=WINDOW,
                            shard_rows=SHARD_ROWS,
                            tp=TP,
                            block=BLOCK,
                            all_gather_grid=ALL_GATHER_GRID,
                        )
                        scf.YieldOp([])
                scf.YieldOp([])
        elif const_expr(has_all_gather):
            emit_tp_all_gather(
                gather_payload_base,
                gather_scale_base,
                gathered_output,
                rank,
                worker,
                output_width=H,
                payload_width=WINDOW,
                shard_rows=SHARD_ROWS,
                tp=TP,
                block=BLOCK,
                all_gather_grid=ALL_GATHER_GRID,
            )

        if const_expr(has_local):
            local_worker = worker - fx.Int32(
                REDUCE_SCATTER_GRID if has_reduce_scatter else 0
            )
            local_if = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.uge,
                    worker,
                    fx.Int32(
                        REDUCE_SCATTER_GRID if has_reduce_scatter else 0
                    ),
                )
            )
            with ir.InsertionPoint(local_if.then_block):
                _emit_local(route, partial, shared, local_worker)
                scf.YieldOp([])

    @flyc.jit
    def launch(
        route,
        partial,
        shared,
        partial_flat_base,
        reduced_shard,
        reduced_payload,
        reduced_scale,
        gather_payload_base,
        gather_scale_base,
        gathered_output,
        rank,
        stream,
    ):
        service_grid = (
            REDUCE_SCATTER_GRID
            if has_reduce_scatter
            else ALL_GATHER_GRID
            if has_all_gather
            else 0
        )
        kernel(
            route,
            partial,
            shared,
            partial_flat_base,
            reduced_shard,
            reduced_payload,
            reduced_scale,
            gather_payload_base,
            gather_scale_base,
            gathered_output,
            rank,
        ).launch(
            grid=(service_grid + (LOCAL_WORKERS if has_local else 0), 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
