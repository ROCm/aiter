# SPDX-License-Identifier: Apache-2.0
"""MXFP8 TP reduce-scatter and all-gather emitters for communication-fused MoE."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .... import buffer_ops
from .sync import peer_base


def e8m0_scale(local_max):
    working = (local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    exponent = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(exponent + fx.Int32(1), exponent)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    return e8m0, scale


def pack_fp8_words(values, scale, word_count):
    packed = []
    for word in range_constexpr(word_count):
        base = word * 4
        value = fx.rocdl.cvt_pk_fp8_f32(
            T.i32,
            values[base] * scale,
            values[base + 1] * scale,
            fx.Int32(0),
            0,
        )
        packed.append(
            fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                values[base + 2] * scale,
                values[base + 3] * scale,
                value,
                1,
            )
        )
    return packed


def quantize_group32(acc):
    local_max = fx.Float32(1e-10).maximumf(fmath.absf(acc).reduce(ReductionOp.MAX))
    e8m0, scale = e8m0_scale(local_max)
    return e8m0, pack_fp8_words(acc, scale, 8)


def decode_scaled_fp8_f32(words, scale):
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
            ).extf(T.vec(2, T.f32))
            values.extend((pair[0], pair[1]))
    return values


def decode_group32(e8m0, packed):
    scale = (fx.Uint32(e8m0) << fx.Uint32(23)).bitcast(fx.Float32)
    values = []
    for word in range_constexpr(8):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16),
                    arith.unwrap(packed[word]),
                    arith.unwrap(scale),
                    bool(half),
                )
            )
            values.extend((pair[0], pair[1]))
    return values


def load_fp8_words(
    resource,
    offset,
    *,
    word_count,
    load_width,
    cache_modifier,
):
    words = []
    for chunk in range_constexpr(word_count // load_width):
        raw = fx.Vector(
            buffer_ops.buffer_load(
                resource,
                offset + fx.Int32(chunk * load_width),
                vec_width=load_width,
                dtype=T.i32,
                cache_modifier=cache_modifier,
            )
        )
        words.extend(raw[word] for word in range_constexpr(load_width))
    return words


def load_e8m0_scale(resource, offset, cache_modifier):
    e8m0 = buffer_ops.buffer_load(
        resource,
        offset,
        vec_width=1,
        dtype=T.i8,
        cache_modifier=cache_modifier,
    )
    return (fx.Uint32(fx.Uint8(e8m0)) << fx.Uint32(23)).bitcast(fx.Float32)


def store_fp8_words(resource, offset, packed, store_width):
    for chunk in range_constexpr(len(packed) // store_width):
        begin = chunk * store_width
        buffer_ops.buffer_store(
            fx.Vector.from_elements(packed[begin : begin + store_width], fx.Int32),
            resource,
            offset + fx.Int32(begin * 4),
            offset_is_bytes=True,
        )


def _store_bf16_group32(resource, offset, values):
    for chunk in range_constexpr(4):
        buffer_ops.buffer_store(
            fx.Vector.from_elements(
                [values[chunk * 8 + element] for element in range_constexpr(8)],
                fx.BFloat16,
            ),
            resource,
            offset + fx.Int32(chunk * 8),
        )


def emit_tp_reduce_scatter(
    flat_base,
    output,
    payload,
    scales,
    rank,
    worker,
    *,
    tokens,
    output_width,
    payload_width,
    shard_rows,
    tp,
    block,
    reduce_scatter_grid,
):
    groups_per_row = payload_width // 32
    start = arith.index_cast(
        T.index, worker * fx.Int32(block) + fx.Int32(gpu.thread_idx.x)
    )
    loop = scf.ForOp(
        start,
        arith.constant(shard_rows * groups_per_row, index=True),
        arith.constant(reduce_scatter_grid * block, index=True),
    )
    with ir.InsertionPoint(loop.body):
        pack = arith.index_cast(T.i32, loop.induction_variable)
        local_token = pack // fx.Int32(groups_per_row)
        group = pack - local_token * fx.Int32(groups_per_row)
        column = group * fx.Int32(32)
        global_token = rank * fx.Int32(shard_rows) + local_token
        acc = fx.Vector.filled(32, 0.0, fx.Float32)

        for source_round in range_constexpr(tp):
            source = (rank + local_token + fx.Int32(source_round)) % fx.Int32(tp)
            base = peer_base(flat_base, source)
            source_row = buffer_ops.create_buffer_resource_from_addr(
                base + fx.Int64(global_token) * fx.Int64(payload_width),
                num_records_bytes=payload_width,
            )
            words = load_fp8_words(
                source_row,
                column // fx.Int32(4),
                word_count=8,
                load_width=4,
                cache_modifier=2,
            )
            scale_row = buffer_ops.create_buffer_resource_from_addr(
                base
                + fx.Int64(tokens * payload_width)
                + fx.Int64(global_token) * fx.Int64(groups_per_row),
                num_records_bytes=groups_per_row,
            )
            scale = load_e8m0_scale(scale_row, group, 2)
            values = decode_scaled_fp8_f32(words, scale)
            acc = acc + fx.Vector.from_elements(values, fx.Float32)

        e8m0, packed = quantize_group32(acc)
        payload_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(payload))
            + fx.Int64(local_token) * fx.Int64(payload_width),
            num_records_bytes=payload_width,
        )
        store_fp8_words(payload_row, column, packed, 4)
        scale_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(scales))
            + fx.Int64(local_token) * fx.Int64(groups_per_row),
            num_records_bytes=groups_per_row,
        )
        buffer_ops.buffer_store(
            e8m0.to(fx.Int8), scale_row, group, offset_is_bytes=True
        )

        decoded = decode_group32(e8m0, packed)
        output_row = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(output))
            + fx.Int64(local_token) * fx.Int64(output_width * 2),
            num_records_bytes=output_width * 2,
        )
        _store_bf16_group32(output_row, column, decoded)
        scf.YieldOp([])


def emit_tp_all_gather(
    payload_base,
    scale_base,
    output,
    rank,
    worker,
    *,
    output_width,
    payload_width,
    shard_rows,
    tp,
    block,
    all_gather_grid,
):
    groups_per_row = payload_width // 32
    source_slot = worker % fx.Int32(tp - 1)
    source_block = worker // fx.Int32(tp - 1)
    source = (rank + fx.Int32(1) + source_slot) % fx.Int32(tp)
    payload = buffer_ops.create_buffer_resource_from_addr(
        peer_base(payload_base, source), num_records_bytes=0xFFFFFFFF
    )
    scales = buffer_ops.create_buffer_resource_from_addr(
        peer_base(scale_base, source), num_records_bytes=0xFFFFFFFF
    )
    output_rsrc = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(output)), num_records_bytes=0xFFFFFFFF
    )
    start = arith.index_cast(
        T.index,
        source_block * fx.Int32(block) + fx.Int32(gpu.thread_idx.x),
    )
    loop = scf.ForOp(
        start,
        arith.constant(shard_rows * groups_per_row, index=True),
        arith.constant(all_gather_grid // (tp - 1) * block, index=True),
    )
    with ir.InsertionPoint(loop.body):
        group = arith.index_cast(T.i32, loop.induction_variable)
        words = load_fp8_words(
            payload,
            group * fx.Int32(8),
            word_count=8,
            load_width=4,
            cache_modifier=1,
        )
        scale_raw = buffer_ops.buffer_load(
            scales, group, vec_width=1, dtype=T.i8, cache_modifier=1
        )
        values = decode_group32(fx.Uint8(scale_raw), words)
        shard_row = group // fx.Int32(groups_per_row)
        group_in_row = group - shard_row * fx.Int32(groups_per_row)
        output_column = (
            source * fx.Int32(shard_rows * output_width)
            + shard_row * fx.Int32(output_width)
            + group_in_row * fx.Int32(32)
        )
        _store_bf16_group32(output_rsrc, output_column, values)
        scf.YieldOp([])
