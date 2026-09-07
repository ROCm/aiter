# SPDX-License-Identifier: Apache-2.0
"""Communication and synchronization primitives for comm-fused MoE."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T, as_ir_value

from .... import buffer_ops
from .... import communication_ops_utils as comm_ops

FLAT_VA_RANK_STRIDE = 1 << 32


def peer_base(flat_base, peer):
    return flat_base + fx.Int64(peer) * fx.Int64(FLAT_VA_RANK_STRIDE)


@functools.cache
def compile_epoch_barrier(tp_size: int):
    """Publish one symmetric workspace epoch and acquire all TP peers."""

    @flyc.kernel(
        name=f"comm_fused_moe_epoch_tp{tp_size}",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        workspace: fx.Pointer,
        flat_base: fx.Int64,
        ready_offset: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        local_base = fx.Int64(ptrtoint(workspace))
        expected = fx.Int64(
            comm_ops.load_i64_global(local_base + ready_offset)
        ) + fx.Int64(1)
        if tid == fx.Int32(0):
            comm_ops.store_i64_global_system(local_base + ready_offset, expected)
        gpu.barrier()
        if tid < fx.Int32(tp_size):
            peer_addr = peer_base(flat_base, tid)
            comm_ops.spin_until_ge_i64(peer_addr + ready_offset, expected)
            comm_ops.fence_system_acquire()

    def launch(workspace, flat_base, ready_offset, stream):
        kernel(workspace, flat_base, ready_offset).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.__name__ = f"launch_comm_fused_moe_epoch_tp{tp_size}"
    return flyc.jit(launch)


def atomic_add_i32_agent(addr, value):
    return llvm_d.AtomicRMWOp(
        llvm_d.AtomicBinOp.add,
        comm_ops._to_ptr_global(addr),
        as_ir_value(value),
        llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    ).res


def wait_i32_system_until_at_least(addr, expected, *, acquire=False, sleep=True):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(32),
            comm_ops._to_ptr_global(addr),
            alignment=4,
            volatile_=True,
            ordering=(
                llvm_d.AtomicOrdering.acquire
                if acquire
                else llvm_d.AtomicOrdering.monotonic
            ),
            syncscope=fx.rocdl.SyncScope.OneAs,
        ).result

    loop = scf.WhileOp([T.i32], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, as_ir_value(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        if sleep:
            llvm_d.InlineAsmOp(None, [], "s_sleep 1", "", has_side_effects=True)
        scf.YieldOp([load()])
    return loop.results[0]


def wait_i32_agent_until_at_least(addr, expected, *, sleep=True):
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
            arith.CmpIPredicate.slt, current, as_ir_value(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        if sleep:
            llvm_d.InlineAsmOp(None, [], "s_sleep 1", "", has_side_effects=True)
        scf.YieldOp([load()])
    return loop.results[0]


def store_i32_relaxed(addr, value):
    llvm_d.StoreOp(
        as_ir_value(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
    )


def store_i32_agent_release(addr, value):
    llvm_d.StoreOp(
        as_ir_value(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def store_i32_system_monotonic(addr, value):
    llvm_d.StoreOp(
        as_ir_value(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def store_i32_system_release(addr, value):
    llvm_d.StoreOp(
        as_ir_value(value),
        comm_ops._to_ptr_global(addr),
        alignment=4,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.OneAs,
    )


def store_i64_relaxed(addr, value):
    llvm_d.StoreOp(
        as_ir_value(value),
        comm_ops._to_ptr_global(addr),
        alignment=8,
    )


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
                    as_ir_value(words[word]),
                    as_ir_value(scale),
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
                    as_ir_value(packed[word]),
                    as_ir_value(scale),
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


def store_fp8_words(resource, offset, packed, store_width, cache_modifier=0):
    for chunk in range_constexpr(len(packed) // store_width):
        begin = chunk * store_width
        buffer_ops.buffer_store(
            fx.Vector.from_elements(packed[begin : begin + store_width], fx.Int32),
            resource,
            offset + fx.Int32(begin * 4),
            cache_modifier=cache_modifier,
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


@flyc.jit
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
    start = worker * fx.Int32(block) + fx.Int32(gpu.thread_id("x"))
    for pack in range(
        start,
        fx.Int32(shard_rows * groups_per_row),
        fx.Int32(reduce_scatter_grid * block),
    ):
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


@flyc.jit
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
    start = source_block * fx.Int32(block) + fx.Int32(gpu.thread_id("x"))
    for group in range(
        start,
        fx.Int32(shard_rows * groups_per_row),
        fx.Int32(all_gather_grid // (tp - 1) * block),
    ):
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
