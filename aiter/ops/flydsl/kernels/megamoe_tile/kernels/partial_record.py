# SPDX-License-Identifier: MIT
"""Pack/unpack the aligned BF16 node-partial return wire record."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class PartialRecordFormat:
    hidden_dim: int
    payload_bytes: int
    source_offset: int
    record_bytes: int
    padding_bytes: int


def partial_record_format(hidden_dim: int) -> PartialRecordFormat:
    """Return ``BF16[H] + u32 source + zero padding`` aligned to 256 B."""

    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0 or hidden_dim % 8:
        raise ValueError("hidden_dim must be positive and divisible by 8")
    payload = hidden_dim * 2
    record = _align_up(payload + 4, 256)
    return PartialRecordFormat(
        hidden_dim=hidden_dim,
        payload_bytes=payload,
        source_offset=payload,
        record_bytes=record,
        padding_bytes=record - payload - 4,
    )


@dataclass(frozen=True)
class PartialRecordModule:
    launch_pack: object
    launch_unpack: object
    format: PartialRecordFormat


def build_partial_record_module(hidden_dim: int) -> PartialRecordModule:
    """Build pack/unpack launchers for one compile-time hidden dimension."""

    fmt = partial_record_format(hidden_dim)
    if fmt.payload_bytes % 16 or fmt.padding_bytes % 4:
        raise ValueError("record geometry must support 16-byte payload and dword padding")

    payload_dwords = fmt.payload_bytes // 4
    record_dwords = fmt.record_bytes // 4
    source_dword = fmt.source_offset // 4
    padding_dwords = fmt.padding_bytes // 4
    tag = f"h{fmt.hidden_dim}_r{fmt.record_bytes}"

    @flyc.kernel(
        name=f"megamoe_partial_record_pack_{tag}",
        known_block_size=[256, 1, 1],
    )
    def pack_kernel(
        node_partial: fx.Int64,
        source_ids: fx.Int64,
        records: fx.Int64,
        record_count: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        record = fx.Int32(gpu.block_id("x"))
        if record < record_count:
            partial_rsrc = buffer_ops.create_buffer_resource_from_addr(node_partial)
            source_rsrc = buffer_ops.create_buffer_resource_from_addr(source_ids)
            record_rsrc = buffer_ops.create_buffer_resource_from_addr(records)
            source = buffer_ops.buffer_load(
                source_rsrc, record, vec_width=1, dtype=T.i32
            )
            invalid = (source < fx.Int32(0)) | (source >= max_source_tokens)
            valid = (source >= fx.Int32(0)) & (source < max_source_tokens)
            safe_source = valid.select(source, fx.Int32(0))
            src_base = safe_source * fx.Int32(payload_dwords)
            dst_base = record * fx.Int32(record_dwords)
            zero4 = fx.Vector.filled(4, 0, fx.Int32)

            for dword in range(tx * fx.Int32(4), payload_dwords, fx.Int32(1024)):
                value = zero4
                if valid:
                    value = buffer_ops.buffer_load(
                        partial_rsrc, src_base + dword, vec_width=4, dtype=T.i32
                    )
                buffer_ops.buffer_store(value, record_rsrc, dst_base + dword)

            if tx == fx.Int32(0):
                stored_source = valid.select(source, fx.Int32(-1))
                buffer_ops.buffer_store(
                    stored_source, record_rsrc, dst_base + fx.Int32(source_dword)
                )
                if invalid:
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

            if tx < fx.Int32(padding_dwords):
                buffer_ops.buffer_store(
                    fx.Int32(0),
                    record_rsrc,
                    dst_base + fx.Int32(source_dword + 1) + tx,
                )

        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tx == fx.Int32(0):
            comm_ops.fence_system_release()

    @flyc.kernel(
        name=f"megamoe_partial_record_unpack_{tag}",
        known_block_size=[256, 1, 1],
    )
    def unpack_kernel(
        records: fx.Int64,
        rows_out: fx.Int64,
        source_ids_out: fx.Int64,
        record_count: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        record = fx.Int32(gpu.block_id("x"))
        if record < record_count:
            record_rsrc = buffer_ops.create_buffer_resource_from_addr(records)
            rows_rsrc = buffer_ops.create_buffer_resource_from_addr(rows_out)
            source_out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                source_ids_out
            )
            src_base = record * fx.Int32(record_dwords)
            dst_base = record * fx.Int32(payload_dwords)
            source = buffer_ops.buffer_load(
                record_rsrc,
                src_base + fx.Int32(source_dword),
                vec_width=1,
                dtype=T.i32,
            )
            invalid = (source < fx.Int32(0)) | (source >= max_source_tokens)
            valid = (source >= fx.Int32(0)) & (source < max_source_tokens)
            zero4 = fx.Vector.filled(4, 0, fx.Int32)

            for dword in range(tx * fx.Int32(4), payload_dwords, fx.Int32(1024)):
                value = zero4
                if valid:
                    value = buffer_ops.buffer_load(
                        record_rsrc, src_base + dword, vec_width=4, dtype=T.i32
                    )
                buffer_ops.buffer_store(value, rows_rsrc, dst_base + dword)

            if tx == fx.Int32(0):
                buffer_ops.buffer_store(source, source_out_rsrc, record)
                if invalid:
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

            if tx < fx.Int32(padding_dwords):
                padding = buffer_ops.buffer_load(
                    record_rsrc,
                    src_base + fx.Int32(source_dword + 1) + tx,
                    vec_width=1,
                    dtype=T.i32,
                )
                if padding != fx.Int32(0):
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tx == fx.Int32(0):
            comm_ops.fence_system_release()

    @flyc.jit
    def launch_pack(
        node_partial: fx.Int64,
        source_ids: fx.Int64,
        records: fx.Int64,
        record_count: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
        stream: fx.Stream,
    ):
        pack_kernel(
            node_partial,
            source_ids,
            records,
            record_count,
            max_source_tokens,
            error_flag,
        ).launch(grid=(record_count, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_unpack(
        records: fx.Int64,
        rows_out: fx.Int64,
        source_ids_out: fx.Int64,
        record_count: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
        stream: fx.Stream,
    ):
        unpack_kernel(
            records,
            rows_out,
            source_ids_out,
            record_count,
            max_source_tokens,
            error_flag,
        ).launch(grid=(record_count, 1, 1), block=(256, 1, 1), stream=stream)

    launch_pack.record_bytes = fmt.record_bytes
    launch_pack.payload_bytes = fmt.payload_bytes
    launch_unpack.record_bytes = fmt.record_bytes
    launch_unpack.payload_bytes = fmt.payload_bytes
    return PartialRecordModule(launch_pack, launch_unpack, fmt)
