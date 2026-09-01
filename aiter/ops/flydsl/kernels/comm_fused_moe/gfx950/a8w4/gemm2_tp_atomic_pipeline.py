# SPDX-License-Identifier: Apache-2.0
"""GEMM2/TP pipeline with atomic BF16 accumulation and MXFP8 communication."""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .... import buffer_ops
from .collectives import (
    decode_scaled_fp8_f32,
    e8m0_scale,
    load_e8m0_scale,
    load_fp8_words,
    pack_fp8_words,
    store_fp8_words,
)
from .shape import Gemm2TPShape
from .sync import peer_base

BLOCK = 256
VECTOR_WIDTH = 16
QUANT_BLOCK = 256


@dataclass(frozen=True)
class Gemm2TPAtomicPipelineConfig:
    shape: Gemm2TPShape
    m: int
    reduce_scatter_grid: int
    all_gather_grid: int

    def __post_init__(self):
        if self.m <= 0 or self.m % self.shape.tp_size:
            raise ValueError(
                f"m={self.m} must be a positive multiple of " f"TP={self.shape.tp_size}"
            )

    @property
    def shard_rows(self) -> int:
        return self.m // self.shape.tp_size


def _emit_quantize_item(config, local, partial, item):
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    column_tiles = (h + QUANT_BLOCK * VECTOR_WIDTH - 1) // (QUANT_BLOCK * VECTOR_WIDTH)
    token = item // fx.Int32(column_tiles)
    tile = item - token * fx.Int32(column_tiles)
    column = tile * fx.Int32(QUANT_BLOCK * VECTOR_WIDTH) + fx.Int32(
        gpu.thread_idx.x
    ) * fx.Int32(VECTOR_WIDTH)
    active = scf.IfOp(arith.cmpi(CmpIPredicate.ult, column, fx.Int32(h)))
    with ir.InsertionPoint(active.then_block):
        local_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(local)) + fx.Int64(token) * fx.Int64(h * 2),
            num_records_bytes=h * 2,
        )
        values = []
        for chunk in range(VECTOR_WIDTH // 8):
            loaded = fx.Vector(
                buffer_ops.buffer_load(
                    local_row,
                    column + fx.Int32(chunk * 8),
                    vec_width=8,
                    dtype=T.bf16,
                    cache_modifier=2,
                )
            ).extf(T.vec(8, T.f32))
            values.extend(loaded[element] for element in range(8))

        vector = fx.Vector.from_elements(values, fx.Float32)
        local_max = fx.Float32(1e-10).maximumf(
            fmath.absf(vector).reduce(ReductionOp.MAX)
        )
        lane = fx.Int32(gpu.thread_idx.x) & fx.Int32(63)
        remote_bits = fx.rocdl.ds_bpermute(
            T.i32,
            (lane ^ fx.Int32(1)) * fx.Int32(4),
            local_max.bitcast(fx.Int32),
        )
        local_max = local_max.maximumf(fx.Int32(remote_bits).bitcast(fx.Float32))
        e8m0, quant_scale = e8m0_scale(local_max)
        packed = pack_fp8_words(vector, quant_scale, VECTOR_WIDTH // 4)

        payload_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(h),
            num_records_bytes=h,
        )
        store_fp8_words(
            payload_row,
            column,
            packed,
            VECTOR_WIDTH // 4,
        )
        scale_leader = scf.IfOp(
            arith.cmpi(
                CmpIPredicate.eq,
                lane % fx.Int32(32 // VECTOR_WIDTH),
                fx.Int32(0),
            )
        )
        with ir.InsertionPoint(scale_leader.then_block):
            scale_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(partial))
                + fx.Int64(config.m * h)
                + fx.Int64(token) * fx.Int64(groups_per_row),
                num_records_bytes=groups_per_row,
            )
            buffer_ops.buffer_store(
                e8m0.to(fx.Int8),
                scale_row,
                column // fx.Int32(32),
                offset_is_bytes=True,
            )
            scf.YieldOp([])
        scf.YieldOp([])


@functools.cache
def compile_stage2_quantize(config: Gemm2TPAtomicPipelineConfig):
    """Quantize the BF16 atomic Stage2 result to MXFP8."""
    m = config.m
    shape = config.shape
    column_tiles = (shape.model_dim + QUANT_BLOCK * VECTOR_WIDTH - 1) // (
        QUANT_BLOCK * VECTOR_WIDTH
    )

    @flyc.kernel(
        name=(f"gemm2_tp_atomic_pipeline_{shape.tag}_m{m}_quant_v16_b256"),
        known_block_size=[QUANT_BLOCK, 1, 1],
    )
    def kernel(local: fx.Pointer, partial: fx.Pointer):
        item = fx.Int32(gpu.block_idx.x)
        _emit_quantize_item(config, local, partial, item)

    @flyc.jit
    def launch(local, partial, stream):
        kernel(local, partial).launch(
            grid=(m * column_tiles, 1, 1),
            block=(QUANT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch


def _store_bf16(resource, offset, values):
    for chunk in range_constexpr(VECTOR_WIDTH // 8):
        buffer_ops.buffer_store(
            fx.Vector.from_elements(
                [values[chunk * 8 + element] for element in range_constexpr(8)],
                fx.BFloat16,
            ),
            resource,
            offset + fx.Int32(chunk * 8),
        )


@functools.cache
def compile_stage2_tp_reduce_scatter(config: Gemm2TPAtomicPipelineConfig):
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    packs_per_group = 32 // VECTOR_WIDTH
    work_items = config.shard_rows * groups_per_row * packs_per_group

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_rs_g{config.reduce_scatter_grid}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        partial_base: fx.Int64,
        output: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        rank: fx.Int32,
    ):
        start = arith.index_cast(
            T.index,
            fx.Int32(gpu.block_idx.x) * fx.Int32(BLOCK) + fx.Int32(gpu.thread_idx.x),
        )
        loop = scf.ForOp(
            start,
            arith.constant(work_items, index=True),
            arith.constant(config.reduce_scatter_grid * BLOCK, index=True),
        )
        with ir.InsertionPoint(loop.body):
            item = arith.index_cast(T.i32, loop.induction_variable)
            group_item = item // fx.Int32(packs_per_group)
            pack_in_group = item - group_item * fx.Int32(packs_per_group)
            local_token = group_item // fx.Int32(groups_per_row)
            group = group_item - local_token * fx.Int32(groups_per_row)
            column = group * fx.Int32(32) + pack_in_group * fx.Int32(VECTOR_WIDTH)
            global_token = rank * fx.Int32(config.shard_rows) + local_token
            lane = fx.Int32(gpu.thread_idx.x) & fx.Int32(63)
            acc = fx.Vector.filled(VECTOR_WIDTH, 0.0, fx.Float32)
            for source_round in range_constexpr(shape.tp_size):
                source = (rank + local_token + fx.Int32(source_round)) % fx.Int32(
                    shape.tp_size
                )
                source_base = peer_base(partial_base, source)
                source_row = buffer_ops.create_buffer_resource_from_addr(
                    source_base + fx.Int64(global_token) * fx.Int64(h),
                    num_records_bytes=h,
                )
                words = load_fp8_words(
                    source_row,
                    column // fx.Int32(4),
                    word_count=VECTOR_WIDTH // 4,
                    load_width=4,
                    cache_modifier=2,
                )
                scale_row = buffer_ops.create_buffer_resource_from_addr(
                    source_base
                    + fx.Int64(config.m * h)
                    + fx.Int64(global_token) * fx.Int64(groups_per_row),
                    num_records_bytes=groups_per_row,
                )
                values = decode_scaled_fp8_f32(
                    words, load_e8m0_scale(scale_row, group, 2)
                )
                acc = acc + fx.Vector.from_elements(values, fx.Float32)

            local_max = fx.Float32(1e-10).maximumf(
                fmath.absf(acc).reduce(ReductionOp.MAX)
            )
            max_bits = local_max.bitcast(fx.Int32)
            for xor_lane in (1,):
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
            packed = pack_fp8_words(acc, quant_scale, VECTOR_WIDTH // 4)

            payload_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(payload)) + fx.Int64(local_token) * fx.Int64(h),
                num_records_bytes=h,
            )
            store_fp8_words(
                payload_row,
                column,
                packed,
                4,
            )
            if pack_in_group == fx.Int32(0):
                scale_row = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(scales))
                    + fx.Int64(local_token) * fx.Int64(groups_per_row),
                    num_records_bytes=groups_per_row,
                )
                buffer_ops.buffer_store(
                    e8m0.to(fx.Int8),
                    scale_row,
                    group,
                    offset_is_bytes=True,
                )

            decoded = decode_scaled_fp8_f32(
                packed,
                (fx.Uint32(e8m0) << fx.Uint32(23)).bitcast(fx.Float32),
            )
            output_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(output)) + fx.Int64(local_token) * fx.Int64(h * 2),
                num_records_bytes=h * 2,
            )
            _store_bf16(output_row, column, decoded)
            scf.YieldOp([])

    @flyc.jit
    def launch(partial_base, output, payload, scales, rank, stream):
        kernel(partial_base, output, payload, scales, rank).launch(
            grid=(config.reduce_scatter_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_stage2_tp_all_gather(config: Gemm2TPAtomicPipelineConfig):
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    packs_per_group = 32 // VECTOR_WIDTH
    work_items = config.shard_rows * groups_per_row * packs_per_group
    source_count = shape.tp_size - 1
    workers_per_source = config.all_gather_grid // source_count

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{config.m}"
            f"_ag_g{config.all_gather_grid}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        payload_base: fx.Int64,
        scale_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        worker = fx.Int32(gpu.block_idx.x)
        source_slot = worker % fx.Int32(source_count)
        source_block = worker // fx.Int32(source_count)
        source = (rank + source_slot + fx.Int32(1)) % fx.Int32(shape.tp_size)
        payload = buffer_ops.create_buffer_resource_from_addr(
            peer_base(payload_base, source), num_records_bytes=0xFFFFFFFF
        )
        scales = buffer_ops.create_buffer_resource_from_addr(
            peer_base(scale_base, source), num_records_bytes=0xFFFFFFFF
        )
        output_resource = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(output)), num_records_bytes=0xFFFFFFFF
        )
        start = arith.index_cast(
            T.index,
            source_block * fx.Int32(BLOCK) + fx.Int32(gpu.thread_idx.x),
        )
        loop = scf.ForOp(
            start,
            arith.constant(work_items, index=True),
            arith.constant(workers_per_source * BLOCK, index=True),
        )
        with ir.InsertionPoint(loop.body):
            item = arith.index_cast(T.i32, loop.induction_variable)
            group_item = item // fx.Int32(packs_per_group)
            pack_in_group = item - group_item * fx.Int32(packs_per_group)
            words = load_fp8_words(
                payload,
                item * fx.Int32(VECTOR_WIDTH // 4),
                word_count=VECTOR_WIDTH // 4,
                load_width=4,
                cache_modifier=1,
            )
            loaded = fx.Uint32(
                fx.Uint8(
                    buffer_ops.buffer_load(
                        scales,
                        group_item,
                        vec_width=1,
                        dtype=T.i8,
                        cache_modifier=1,
                    )
                )
            )
            scale = (loaded << fx.Uint32(23)).bitcast(fx.Float32)
            values = decode_scaled_fp8_f32(words, scale)
            shard_row = group_item // fx.Int32(groups_per_row)
            group = group_item - shard_row * fx.Int32(groups_per_row)
            output_column = (
                source * fx.Int32(config.shard_rows * h)
                + shard_row * fx.Int32(h)
                + group * fx.Int32(32)
                + pack_in_group * fx.Int32(VECTOR_WIDTH)
            )
            _store_bf16(output_resource, output_column, values)
            scf.YieldOp([])

    @flyc.jit
    def launch(payload_base, scale_base, output, rank, stream):
        kernel(payload_base, scale_base, output, rank).launch(
            grid=(config.all_gather_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
