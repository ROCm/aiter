# SPDX-License-Identifier: Apache-2.0
"""Full-width communication-fused Stage2 kernels."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
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


M = 2048
H = 7168
I = 384
E = 384
TOPK = 6
TP = 8
TILE_M = 32
TILE_N = 256
TILE_K = 128
SORT_BLOCK_M = 64
SHARD_ROWS = M // TP
BLOCK = 256
REDUCE_SCATTER_GRID = 128
ALL_GATHER_GRID = 126
VECTOR_WIDTH = 8
GROUPS_PER_ROW = H // 32
LOCAL_COLUMN_TILES = (H + BLOCK * VECTOR_WIDTH - 1) // (
    BLOCK * VECTOR_WIDTH
)
LOCAL_WORKERS = M * LOCAL_COLUMN_TILES


@functools.cache
def compile_stage2_compute():
    """Compile the full-width compact Stage2 producer."""
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
        sort_block_m=SORT_BLOCK_M,
    )


def _emit_local_reduce(route, partial, shared, token, column_base):
    """Reduce six routed rows plus the local shared partial into MXFP8."""
    route_row_bytes = H + H // 8
    tid = fx.Int32(gpu.thread_idx.x)
    column = tid * fx.Int32(VECTOR_WIDTH) + column_base
    active = scf.IfOp(arith.cmpi(CmpIPredicate.ult, column, fx.Int32(H)))
    with ir.InsertionPoint(active.then_block):
        route_addr = fx.Int64(ptrtoint(route)) + fx.Int64(token) * fx.Int64(
            TOPK * route_row_bytes
        )
        route_row = buffer_ops.create_buffer_resource_from_addr(
            route_addr, num_records_bytes=TOPK * route_row_bytes
        )
        acc = fx.Vector.filled(VECTOR_WIDTH, 0.0, fx.Float32)
        for route_slot in range_constexpr(TOPK):
            words = load_fp8_words(
                route_row,
                fx.Int32(route_slot * (route_row_bytes // 4))
                + column // fx.Int32(4),
                word_count=2,
                load_width=2,
                cache_modifier=2,
            )
            scale = load_e8m0_scale(
                route_row,
                fx.Int32(route_slot * route_row_bytes + H)
                + column // fx.Int32(8),
                2,
            )
            values = decode_scaled_fp8_f32(words, scale)
            acc = acc + fx.Vector.from_elements(values, fx.Float32)

        shared_addr = fx.Int64(ptrtoint(shared)) + fx.Int64(token) * fx.Int64(
            H * 2
        )
        shared_row = buffer_ops.create_buffer_resource_from_addr(
            shared_addr, num_records_bytes=H * 2
        )
        shared_values = fx.Vector(
            buffer_ops.buffer_load(
                shared_row,
                column,
                vec_width=VECTOR_WIDTH,
                dtype=T.bf16,
                cache_modifier=2,
            )
        ).extf(T.vec(VECTOR_WIDTH, T.f32))
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
        packed_words = pack_fp8_words(acc, quant_scale, 2)

        payload_addr = fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(
            H
        )
        payload_row = buffer_ops.create_buffer_resource_from_addr(
            payload_addr, num_records_bytes=H
        )
        store_fp8_words(payload_row, column, packed_words, 2)

        scale_leader = scf.IfOp(
            arith.cmpi(
                CmpIPredicate.eq,
                lane & fx.Int32(3),
                fx.Int32(0),
            )
        )
        with ir.InsertionPoint(scale_leader.then_block):
            scale_addr = (
                fx.Int64(ptrtoint(partial))
                + fx.Int64(M * H)
                + fx.Int64(token) * fx.Int64(GROUPS_PER_ROW)
            )
            scale_row = buffer_ops.create_buffer_resource_from_addr(
                scale_addr, num_records_bytes=GROUPS_PER_ROW
            )
            buffer_ops.buffer_store(
                e8m0.to(fx.Int8),
                scale_row,
                column // fx.Int32(32),
                offset_is_bytes=True,
            )
            scf.YieldOp([])
        scf.YieldOp([])


def _emit_local_worker(route, partial, shared, worker):
    work = scf.ForOp(
        arith.index_cast(T.index, worker),
        arith.constant(M * LOCAL_COLUMN_TILES, index=True),
        arith.constant(LOCAL_WORKERS, index=True),
    )
    with ir.InsertionPoint(work.body):
        item = arith.index_cast(T.i32, work.induction_variable)
        token = item // fx.Int32(LOCAL_COLUMN_TILES)
        tile = item - token * fx.Int32(LOCAL_COLUMN_TILES)
        _emit_local_reduce(
            route,
            partial,
            shared,
            token,
            tile * fx.Int32(BLOCK * VECTOR_WIDTH),
        )
        scf.YieldOp([])


@functools.cache
def compile_stage2_local_reduce():
    """Compile the local route/shared reduction."""

    @flyc.kernel(name="comm_fused_moe_local", known_block_size=[BLOCK, 1, 1])
    def kernel(
        route: fx.Pointer,
        partial: fx.Pointer,
        shared: fx.Pointer,
    ):
        _emit_local_worker(route, partial, shared, fx.Int32(gpu.block_idx.x))

    @flyc.jit
    def launch(route, partial, shared, stream):
        kernel(route, partial, shared).launch(
            grid=(LOCAL_WORKERS, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_stage2_tp_reduce_scatter():
    """Compile the TP reduce-scatter and reduced MXFP8 publication."""

    @flyc.kernel(
        name="comm_fused_moe_tp_reduce_scatter",
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        flat_base: fx.Int64,
        output: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        rank: fx.Int32,
    ):
        emit_tp_reduce_scatter(
            flat_base,
            output,
            payload,
            scales,
            rank,
            fx.Int32(gpu.block_idx.x),
            tokens=M,
            output_width=H,
            payload_width=H,
            shard_rows=SHARD_ROWS,
            tp=TP,
            block=BLOCK,
            reduce_scatter_grid=REDUCE_SCATTER_GRID,
        )

    @flyc.jit
    def launch(flat_base, output, payload, scales, rank, stream):
        kernel(flat_base, output, payload, scales, rank).launch(
            grid=(REDUCE_SCATTER_GRID, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_stage2_tp_all_gather():
    """Compile the TP all-gather of reduced shards."""

    @flyc.kernel(
        name="comm_fused_moe_tp_all_gather",
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        payload_flat_base: fx.Int64,
        scale_flat_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        emit_tp_all_gather(
            payload_flat_base,
            scale_flat_base,
            output,
            rank,
            fx.Int32(gpu.block_idx.x),
            output_width=H,
            payload_width=H,
            shard_rows=SHARD_ROWS,
            tp=TP,
            block=BLOCK,
            all_gather_grid=ALL_GATHER_GRID,
        )

    @flyc.jit
    def launch(payload_flat_base, scale_flat_base, output, rank, stream):
        kernel(payload_flat_base, scale_flat_base, output, rank).launch(
            grid=(ALL_GATHER_GRID, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
