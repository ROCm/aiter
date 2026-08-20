# SPDX-License-Identifier: Apache-2.0
"""Full-width TP8 communication-fused Stage2 kernels."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import buffer_ops, communication_ops_utils as comm_ops
from ..mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm2_common


M = 2048
H = 7168
I = 384
E = 384
TOPK = 6
TP = 8
FLAT_VA_RANK_STRIDE = 1 << 32
TILE_M = 32
TILE_N = 256
TILE_K = 128
SORT_BLOCK_M = 64
OWNER_ROWS = M // TP
BLOCK = 256
PEER_GRID = 128
FANOUT_GRID = 126
LOCAL_WORKERS = 640
VECTOR_WIDTH = 8
GROUPS_PER_ROW = H // 32
FANOUT_BLOCKS_PER_SOURCE = FANOUT_GRID // (TP - 1)
LOCAL_COLUMN_TILES = (H + BLOCK * VECTOR_WIDTH - 1) // (
    BLOCK * VECTOR_WIDTH
)


def _compile_compute():
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


@functools.cache
def compile_stage2_compute():
    """Compile the full-width compact Stage2 producer."""
    return _compile_compute()


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
            raw = fx.Vector(
                buffer_ops.buffer_load(
                    route_row,
                    fx.Int32(route_slot * (route_row_bytes // 4))
                    + column // fx.Int32(4),
                    vec_width=2,
                    dtype=T.i32,
                    cache_modifier=2,
                )
            )
            values = []
            for word in range_constexpr(2):
                for half in range_constexpr(2):
                    pair = fx.Vector(
                        fx.rocdl.cvt_pk_f32_fp8(T.f32x2, raw[word], bool(half))
                    )
                    values.extend((pair[0], pair[1]))
            scale_raw = buffer_ops.buffer_load(
                route_row,
                fx.Int32(route_slot * route_row_bytes + H)
                + column // fx.Int32(8),
                vec_width=1,
                dtype=T.i8,
                cache_modifier=2,
            )
            scale = (
                fx.Uint32(fx.Uint8(scale_raw)) << fx.Uint32(23)
            ).bitcast(fx.Float32)
            acc = acc + fx.Vector.from_elements(
                [value * scale for value in values], fx.Float32
            )

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
        working = (
            local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)
        ).bitcast(fx.Int32)
        mantissa = working & fx.Int32(0x7FFFFF)
        exponent = (working >> fx.Int32(23)) & fx.Int32(0xFF)
        e8m0 = (mantissa != fx.Int32(0)).select(exponent + fx.Int32(1), exponent)
        e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
        quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(
            fx.Float32
        )

        packed_words = []
        for word in range_constexpr(2):
            base = word * 4
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                acc[base] * quant_scale,
                acc[base + 1] * quant_scale,
                fx.Int32(0),
                0,
            )
            packed_words.append(
                fx.rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    acc[base + 2] * quant_scale,
                    acc[base + 3] * quant_scale,
                    packed,
                    1,
                )
            )

        payload_addr = fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(
            H
        )
        payload_row = buffer_ops.create_buffer_resource_from_addr(
            payload_addr, num_records_bytes=H
        )
        buffer_ops.buffer_store(
            fx.Vector.from_elements(packed_words, fx.Int32),
            payload_row,
            column,
            offset_is_bytes=True,
        )

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


def _quantize_group32(acc):
    local_max = fx.Float32(1e-10).maximumf(
        fmath.absf(acc).reduce(ReductionOp.MAX)
    )
    working = (
        local_max * fx.Int32(0x3B124925).bitcast(fx.Float32)
    ).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    exponent = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(exponent + fx.Int32(1), exponent)
    e8m0 = (e8m0 > fx.Int32(0xFF)).select(fx.Int32(0xFF), e8m0)
    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    packed_words = []
    for word in range_constexpr(8):
        base = word * 4
        packed = fx.rocdl.cvt_pk_fp8_f32(
            T.i32,
            acc[base] * quant_scale,
            acc[base + 1] * quant_scale,
            fx.Int32(0),
            0,
        )
        packed_words.append(
            fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                acc[base + 2] * quant_scale,
                acc[base + 3] * quant_scale,
                packed,
                1,
            )
        )
    return e8m0, packed_words


def _decode_group32(e8m0, packed_words):
    scale = (fx.Uint32(e8m0) << fx.Uint32(23)).bitcast(fx.Float32)
    values = []
    for word in range_constexpr(8):
        for half in range_constexpr(2):
            pair = fx.Vector(
                fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                    T.vec(2, T.bf16),
                    arith.unwrap(packed_words[word]),
                    arith.unwrap(scale),
                    bool(half),
                )
            )
            values.extend((pair[0], pair[1]))
    return values


def _peer_base(flat_base, peer):
    return flat_base + fx.Int64(peer) * fx.Int64(FLAT_VA_RANK_STRIDE)


def _reduce_pack(flat_base, output, payload, scales, rank, pack):
    local_token = pack // fx.Int32(GROUPS_PER_ROW)
    pack_in_row = pack - local_token * fx.Int32(GROUPS_PER_ROW)
    column = pack_in_row * fx.Int32(32)
    global_token = rank * fx.Int32(OWNER_ROWS) + local_token
    acc = fx.Vector.filled(32, 0.0, fx.Float32)

    for source_round in range_constexpr(TP):
        source = (rank + local_token + fx.Int32(source_round)) % fx.Int32(TP)
        peer_base = _peer_base(flat_base, source)
        source_row = buffer_ops.create_buffer_resource_from_addr(
            peer_base + fx.Int64(global_token) * fx.Int64(H),
            num_records_bytes=H,
        )
        raw_words = []
        for half in range_constexpr(2):
            raw = fx.Vector(
                buffer_ops.buffer_load(
                    source_row,
                    column // fx.Int32(4) + fx.Int32(half * 4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=2,
                )
            )
            for word in range_constexpr(4):
                raw_words.append(raw[word])
        scale_row = buffer_ops.create_buffer_resource_from_addr(
            peer_base
            + fx.Int64(M * H)
            + fx.Int64(global_token) * fx.Int64(GROUPS_PER_ROW),
            num_records_bytes=GROUPS_PER_ROW,
        )
        scale_raw = buffer_ops.buffer_load(
            scale_row,
            column // fx.Int32(32),
            vec_width=1,
            dtype=T.i8,
            cache_modifier=2,
        )
        scale = (
            fx.Uint32(fx.Uint8(scale_raw)) << fx.Uint32(23)
        ).bitcast(fx.Float32)
        values = []
        for word in range_constexpr(8):
            for half in range_constexpr(2):
                pair = fx.Vector(
                    fx.rocdl.cvt_pk_f32_fp8(
                        T.f32x2, raw_words[word], bool(half)
                    )
                )
                values.extend((pair[0] * scale, pair[1] * scale))
        acc = acc + fx.Vector.from_elements(values, fx.Float32)

    e8m0, packed_words = _quantize_group32(acc)
    payload_row = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(payload)) + fx.Int64(local_token) * fx.Int64(H),
        num_records_bytes=H,
    )
    for chunk in range_constexpr(2):
        buffer_ops.buffer_store(
            fx.Vector.from_elements(
                packed_words[chunk * 4 : chunk * 4 + 4], fx.Int32
            ),
            payload_row,
            column + fx.Int32(chunk * 16),
            offset_is_bytes=True,
        )
    scale_row = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(scales))
        + fx.Int64(local_token) * fx.Int64(GROUPS_PER_ROW),
        num_records_bytes=GROUPS_PER_ROW,
    )
    buffer_ops.buffer_store(
        e8m0.to(fx.Int8),
        scale_row,
        column // fx.Int32(32),
        offset_is_bytes=True,
    )

    output_values = _decode_group32(e8m0, packed_words)
    output_row = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(output)) + fx.Int64(local_token) * fx.Int64(H * 2),
        num_records_bytes=H * 2,
    )
    for chunk in range_constexpr(4):
        buffer_ops.buffer_store(
            fx.Vector.from_elements(
                [
                    output_values[chunk * 8 + element]
                    for element in range_constexpr(8)
                ],
                fx.BFloat16,
            ),
            output_row,
            column + fx.Int32(chunk * 8),
        )


def _emit_peer_reduce(flat_base, output, payload, scales, rank, worker):
    """Reduce owner packs from all TP ranks and publish MXFP8 owner data."""
    start = arith.index_cast(
        T.index, worker * fx.Int32(BLOCK) + fx.Int32(gpu.thread_idx.x)
    )
    loop = scf.ForOp(
        start,
        arith.constant(OWNER_ROWS * GROUPS_PER_ROW, index=True),
        arith.constant(PEER_GRID * BLOCK, index=True),
    )
    with ir.InsertionPoint(loop.body):
        _reduce_pack(
            flat_base,
            output,
            payload,
            scales,
            rank,
            arith.index_cast(T.i32, loop.induction_variable),
        )
        scf.YieldOp([])


def _emit_fanout(payload_flat_base, scale_flat_base, output, rank, worker):
    """Pull one source-owned token partition into the replicated output."""
    tid = fx.Int32(gpu.thread_idx.x)
    source_slot = worker % fx.Int32(TP - 1)
    source_block = worker // fx.Int32(TP - 1)
    source = (rank + fx.Int32(1) + source_slot) % fx.Int32(TP)
    payload = buffer_ops.create_buffer_resource_from_addr(
        _peer_base(payload_flat_base, source), num_records_bytes=0xFFFFFFFF
    )
    scales = buffer_ops.create_buffer_resource_from_addr(
        _peer_base(scale_flat_base, source), num_records_bytes=0xFFFFFFFF
    )
    output_rsrc = buffer_ops.create_buffer_resource_from_addr(
        fx.Int64(ptrtoint(output)), num_records_bytes=0xFFFFFFFF
    )

    start = arith.index_cast(T.index, source_block * fx.Int32(BLOCK) + tid)
    loop = scf.ForOp(
        start,
        arith.constant(OWNER_ROWS * GROUPS_PER_ROW, index=True),
        arith.constant(FANOUT_BLOCKS_PER_SOURCE * BLOCK, index=True),
    )
    with ir.InsertionPoint(loop.body):
        group = arith.index_cast(T.i32, loop.induction_variable)
        raw_words = []
        for half in range_constexpr(2):
            raw = fx.Vector(
                buffer_ops.buffer_load(
                    payload,
                    group * fx.Int32(8) + fx.Int32(half * 4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=1,
                )
            )
            for word in range_constexpr(4):
                raw_words.append(raw[word])
        scale_raw = buffer_ops.buffer_load(
            scales,
            group,
            vec_width=1,
            dtype=T.i8,
            cache_modifier=1,
        )
        scale = (
            fx.Uint32(fx.Uint8(scale_raw)) << fx.Uint32(23)
        ).bitcast(fx.Float32)
        values = []
        for word in range_constexpr(8):
            for half in range_constexpr(2):
                pair = fx.Vector(
                    fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                        T.vec(2, T.bf16),
                        raw_words[word].ir_value(),
                        scale.ir_value(),
                        bool(half),
                    )
                )
                values.extend((pair[0], pair[1]))
        owner_row = group // fx.Int32(GROUPS_PER_ROW)
        group_in_row = group - owner_row * fx.Int32(GROUPS_PER_ROW)
        output_column = (
            source * fx.Int32(OWNER_ROWS * H)
            + owner_row * fx.Int32(H)
            + group_in_row * fx.Int32(32)
        )
        for chunk in range_constexpr(4):
            buffer_ops.buffer_store(
                fx.Vector.from_elements(
                    [
                        values[chunk * 8 + element]
                        for element in range_constexpr(8)
                    ],
                    fx.BFloat16,
                ),
                output_rsrc,
                output_column + fx.Int32(chunk * 8),
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
def compile_stage2_peer_reduce():
    """Compile the TP peer reduction and owner MXFP8 publication."""

    @flyc.kernel(name="comm_fused_moe_peer", known_block_size=[BLOCK, 1, 1])
    def kernel(
        flat_base: fx.Int64,
        output: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        rank: fx.Int32,
    ):
        _emit_peer_reduce(
            flat_base,
            output,
            payload,
            scales,
            rank,
            fx.Int32(gpu.block_idx.x),
        )

    @flyc.jit
    def launch(flat_base, output, payload, scales, rank, stream):
        kernel(flat_base, output, payload, scales, rank).launch(
            grid=(PEER_GRID, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch


@functools.cache
def compile_stage2_fanout():
    """Compile the replicated owner fanout."""

    @flyc.kernel(name="comm_fused_moe_fanout", known_block_size=[BLOCK, 1, 1])
    def kernel(
        payload_flat_base: fx.Int64,
        scale_flat_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        _emit_fanout(
            payload_flat_base,
            scale_flat_base,
            output,
            rank,
            fx.Int32(gpu.block_idx.x),
        )

    @flyc.jit
    def launch(payload_flat_base, scale_flat_base, output, rank, stream):
        kernel(payload_flat_base, scale_flat_base, output, rank).launch(
            grid=(FANOUT_GRID, 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    return launch


@functools.cache
def compile_epoch_barrier():
    """Publish one symmetric workspace epoch and acquire all TP peers."""

    @flyc.kernel(name="comm_fused_moe_epoch_tp8", known_block_size=[64, 1, 1])
    def kernel(
        workspace: fx.Pointer,
        flat_base: fx.Int64,
        ready_offset: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        local_base = fx.Int64(ptrtoint(workspace))
        expected = fx.Int64(
            comm_ops.load_i64_global(local_base + ready_offset)
        ) + fx.Int64(1)
        if tid == fx.Int32(0):
            comm_ops.store_i64_global_system(local_base + ready_offset, expected)
        gpu.barrier()
        if tid < fx.Int32(TP):
            peer_base = _peer_base(flat_base, tid)
            comm_ops.wait_i64_system_until_at_least(
                peer_base + ready_offset, expected
            )
            comm_ops.fence_system_acquire()

    @flyc.jit
    def launch(workspace, flat_base, ready_offset, stream):
        kernel(workspace, flat_base, ready_offset).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


__all__ = [
    "FLAT_VA_RANK_STRIDE",
    "compile_epoch_barrier",
    "compile_stage2_compute",
    "compile_stage2_fanout",
    "compile_stage2_local_reduce",
    "compile_stage2_peer_reduce",
]
