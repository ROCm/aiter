# SPDX-License-Identifier: Apache-2.0
"""Multi-launch atomic BF16 and MXFP8 communication family."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

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
from .config import BLOCK, AtomicConfig

VECTOR_WIDTH = 16
QUANT_BLOCK = 256


@flyc.jit
def _emit_quantize_item(config, local, shared, partial, item, add_shared):
    shape = config.shape
    h = shape.model_dim
    groups_per_row = h // 32
    column_tiles = (h + QUANT_BLOCK * VECTOR_WIDTH - 1) // (QUANT_BLOCK * VECTOR_WIDTH)
    token = item // fx.Int32(column_tiles)
    tile = item - token * fx.Int32(column_tiles)
    column = tile * fx.Int32(QUANT_BLOCK * VECTOR_WIDTH) + fx.Int32(
        gpu.thread_id("x")
    ) * fx.Int32(VECTOR_WIDTH)
    if column < fx.Int32(h):
        local_row = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(local)) + fx.Int64(token) * fx.Int64(h * 2),
            fx.BFloat16,
            h * 2,
        )
        if const_expr(add_shared):
            shared_row = buffer_tensor_from_addr(
                fx.Int64(ptrtoint(shared)) + fx.Int64(token) * fx.Int64(h * 2),
                fx.BFloat16,
                h * 2,
            )
        values = []
        for chunk in range_constexpr(VECTOR_WIDTH // 8):
            chunk_offset = column + fx.Int32(chunk * 8)
            loaded = load_bf16(local_row, chunk_offset, 8, 2).to(fx.Float32)
            if const_expr(add_shared):
                shared_values = load_bf16(shared_row, chunk_offset, 8, 2).to(fx.Float32)
                loaded = (loaded + shared_values).to(fx.BFloat16).to(fx.Float32)
            values.extend(loaded[element] for element in range_constexpr(8))

        vector = fx.Vector.from_elements(values, fx.Float32)
        local_max = fx.Float32(1e-10).maximumf(
            fmath.absf(vector).reduce(ReductionOp.MAX)
        )
        lane = fx.Int32(gpu.thread_id("x")) & fx.Int32(63)
        remote_bits = fx.rocdl.ds_bpermute(
            T.i32,
            (lane ^ fx.Int32(1)) * fx.Int32(4),
            local_max.bitcast(fx.Int32),
        )
        local_max = local_max.maximumf(fx.Int32(remote_bits).bitcast(fx.Float32))
        e8m0, quant_scale = e8m0_scale(local_max)
        packed = pack_fp8_words(vector, quant_scale, VECTOR_WIDTH // 4)

        payload_row = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(h),
            fx.Int32,
            h,
        )
        store_fp8_words(
            payload_row,
            column,
            packed,
            VECTOR_WIDTH // 4,
        )
        if lane % fx.Int32(32 // VECTOR_WIDTH) == fx.Int32(0):
            scale_row = buffer_tensor_from_addr(
                fx.Int64(ptrtoint(partial))
                + fx.Int64(config.m * h)
                + fx.Int64(token) * fx.Int64(groups_per_row),
                fx.Int8,
                groups_per_row,
            )
            store_buffer(
                scale_row,
                column // fx.Int32(32),
                e8m0.to(fx.Int8),
                fx.Int8,
            )


@functools.cache
def compile_quantize(config: AtomicConfig, add_shared: bool):
    """Optionally add shared BF16 output, then quantize to MXFP8."""
    m = config.m
    shape = config.shape
    column_tiles = (shape.model_dim + QUANT_BLOCK * VECTOR_WIDTH - 1) // (
        QUANT_BLOCK * VECTOR_WIDTH
    )

    @flyc.kernel(
        name=(
            f"gemm2_tp_atomic_pipeline_{shape.tag}_m{m}_quant"
            f"{'_add_shared' if add_shared else ''}_v16_b256"
        ),
        known_block_size=[QUANT_BLOCK, 1, 1],
    )
    def kernel(local: fx.Pointer, shared: fx.Pointer, partial: fx.Pointer):
        item = fx.Int32(gpu.block_id("x"))
        _emit_quantize_item(config, local, shared, partial, item, add_shared)

    @flyc.jit
    def launch(local, shared, partial, stream):
        kernel(local, shared, partial).launch(
            grid=(m * column_tiles, 1, 1),
            block=(QUANT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_reduce_scatter(config: AtomicConfig):
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
        start = fx.Int32(gpu.block_id("x")) * fx.Int32(BLOCK) + fx.Int32(
            gpu.thread_id("x")
        )
        for item in range(
            start,
            fx.Int32(work_items),
            fx.Int32(config.reduce_scatter_grid * BLOCK),
        ):
            group_item = item // fx.Int32(packs_per_group)
            pack_in_group = item - group_item * fx.Int32(packs_per_group)
            local_token = group_item // fx.Int32(groups_per_row)
            group = group_item - local_token * fx.Int32(groups_per_row)
            column = group * fx.Int32(32) + pack_in_group * fx.Int32(VECTOR_WIDTH)
            global_token = rank * fx.Int32(config.shard_rows) + local_token
            lane = fx.Int32(gpu.thread_id("x")) & fx.Int32(63)
            acc = fx.Vector.filled(VECTOR_WIDTH, 0.0, fx.Float32)
            for source_round in range_constexpr(shape.tp_size):
                source = (rank + local_token + fx.Int32(source_round)) % fx.Int32(
                    shape.tp_size
                )
                source_base = peer_base(partial_base, source)
                source_row = buffer_tensor_from_addr(
                    source_base + fx.Int64(global_token) * fx.Int64(h),
                    fx.Int32,
                    h,
                )
                words = load_fp8_words(
                    source_row,
                    column // fx.Int32(4),
                    word_count=VECTOR_WIDTH // 4,
                    load_width=4,
                    cache_modifier=2,
                )
                scale_row = buffer_tensor_from_addr(
                    source_base
                    + fx.Int64(config.m * h)
                    + fx.Int64(global_token) * fx.Int64(groups_per_row),
                    fx.Int8,
                    groups_per_row,
                )
                values = decode_scaled_fp8_f32(
                    words, load_e8m0_scale(scale_row, group, 2)
                )
                acc = acc + fx.Vector.from_elements(values, fx.Float32)

            local_max = fx.Float32(1e-10).maximumf(
                fmath.absf(acc).reduce(ReductionOp.MAX)
            )
            max_bits = local_max.bitcast(fx.Int32)
            for xor_lane in range_constexpr(1, 2):
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

            payload_row = buffer_tensor_from_addr(
                fx.Int64(ptrtoint(payload)) + fx.Int64(local_token) * fx.Int64(h),
                fx.Int32,
                h,
            )
            store_fp8_words(
                payload_row,
                column,
                packed,
                4,
            )
            if pack_in_group == fx.Int32(0):
                scale_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(scales))
                    + fx.Int64(local_token) * fx.Int64(groups_per_row),
                    fx.Int8,
                    groups_per_row,
                )
                store_buffer(
                    scale_row,
                    group,
                    e8m0.to(fx.Int8),
                    fx.Int8,
                )

            decoded = decode_scaled_fp8_f32(
                packed,
                (fx.Uint32(e8m0) << fx.Uint32(23)).bitcast(fx.Float32),
            )
            output_row = buffer_tensor_from_addr(
                fx.Int64(ptrtoint(output)) + fx.Int64(local_token) * fx.Int64(h * 2),
                fx.BFloat16,
                h * 2,
            )
            store_bf16(output_row, column, decoded, VECTOR_WIDTH)

    @flyc.jit
    def launch(partial_base, output, payload, scales, rank, stream):
        kernel(partial_base, output, payload, scales, rank).launch(
            grid=(config.reduce_scatter_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_all_gather(config: AtomicConfig):
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
        worker = fx.Int32(gpu.block_id("x"))
        source_slot = worker % fx.Int32(source_count)
        source_block = worker // fx.Int32(source_count)
        source = (rank + source_slot + fx.Int32(1)) % fx.Int32(shape.tp_size)
        payload = buffer_tensor_from_addr(
            peer_base(payload_base, source),
            fx.Int32,
            config.shard_rows * h,
        )
        scales = buffer_tensor_from_addr(
            peer_base(scale_base, source),
            fx.Int8,
            config.shard_rows * groups_per_row,
        )
        output_resource = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(output)),
            fx.BFloat16,
            shape.tp_size * config.shard_rows * h * 2,
        )
        start = source_block * fx.Int32(BLOCK) + fx.Int32(gpu.thread_id("x"))
        for item in range(
            start,
            fx.Int32(work_items),
            fx.Int32(workers_per_source * BLOCK),
        ):
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
                fx.Uint8(load_buffer(scales, group_item, fx.Int8, cache_modifier=1))
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
            store_bf16(output_resource, output_column, values, VECTOR_WIDTH)

    @flyc.jit
    def launch(payload_base, scale_base, output, rank, stream):
        kernel(payload_base, scale_base, output, rank).launch(
            grid=(config.all_gather_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch
