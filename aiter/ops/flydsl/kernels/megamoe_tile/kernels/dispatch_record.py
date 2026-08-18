# SPDX-License-Identifier: MIT
"""Device pack/unpack for the aligned dispatch wire record."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops

from ..wire import K3DispatchWireLayout


@dataclass(frozen=True)
class DispatchRecordModule:
    launch_pack: object
    launch_unpack: object
    layout: K3DispatchWireLayout


def build_dispatch_record_module(hidden_dim: int, topk: int) -> DispatchRecordModule:
    """Build token-row-selecting pack and contiguous unpack launchers."""

    hidden_dim = int(hidden_dim)
    topk = int(topk)
    if hidden_dim <= 0 or hidden_dim % 128:
        raise ValueError("hidden_dim must be positive and divisible by 128")
    if not 0 < topk < 64:
        raise ValueError("topk must be in [1, 63]")
    layout = K3DispatchWireLayout(
        hidden_bytes=hidden_dim // 2,
        scale_bytes=hidden_dim // 32,
        topk=topk,
    )
    route_end = layout.route_mask_offset + 8
    if route_end > layout.record_bytes:
        raise ValueError("dispatch fields exceed the aligned record stride")
    for value in (
        layout.hidden_bytes,
        layout.scale_bytes,
        layout.source_offset,
        layout.route_mask_offset,
        layout.record_bytes,
    ):
        if value % 4:
            raise ValueError("dispatch wire fields must be dword aligned")
    if layout.hidden_bytes % 16 or layout.scale_bytes % 16:
        raise ValueError("activation and scale rows must be 16-byte aligned")

    hidden_dw = layout.hidden_bytes // 4
    scale_dw = layout.scale_bytes // 4
    ids_dw = (layout.hidden_bytes + layout.scale_bytes) // 4
    weights_dw = ids_dw + topk
    source_dw = layout.source_offset // 4
    route_dw = layout.route_mask_offset // 4
    record_dw = layout.record_bytes // 4
    tail_dw = (layout.record_bytes - route_end) // 4
    allowed_mask = (1 << topk) - 1
    tag = f"h{hidden_dim}_k{topk}"

    @flyc.kernel(
        name=f"megamoe_dispatch_record_pack_{tag}",
        known_block_size=[256, 1, 1],
    )
    def pack_kernel(
        hidden_q: fx.Int64,
        scales: fx.Int64,
        topk_ids: fx.Int64,
        topk_weights: fx.Int64,
        source_tokens: fx.Int64,
        route_masks: fx.Int64,
        row_ids: fx.Int64,
        records: fx.Int64,
        record_count: fx.Int32,
        token_count: fx.Int32,
        num_experts: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        record = fx.Int32(gpu.block_id("x"))
        if record < record_count:
            hidden_rsrc = buffer_ops.create_buffer_resource_from_addr(hidden_q)
            scale_rsrc = buffer_ops.create_buffer_resource_from_addr(scales)
            ids_rsrc = buffer_ops.create_buffer_resource_from_addr(topk_ids)
            weights_rsrc = buffer_ops.create_buffer_resource_from_addr(topk_weights)
            source_rsrc = buffer_ops.create_buffer_resource_from_addr(source_tokens)
            mask_rsrc = buffer_ops.create_buffer_resource_from_addr(route_masks)
            row_rsrc = buffer_ops.create_buffer_resource_from_addr(row_ids)
            record_rsrc = buffer_ops.create_buffer_resource_from_addr(records)
            row = buffer_ops.buffer_load(row_rsrc, record, vec_width=1, dtype=T.i32)
            row_invalid = (row < fx.Int32(0)) | (row >= token_count)
            row_valid = (row >= fx.Int32(0)) & (row < token_count)
            safe_row = row_valid.select(row, fx.Int32(0))
            dst = record * fx.Int32(record_dw)
            zero4 = fx.Vector.filled(4, 0, fx.Int32)

            for dword in range(tx * fx.Int32(4), hidden_dw, fx.Int32(1024)):
                value = zero4
                if row_valid:
                    value = buffer_ops.buffer_load(
                        hidden_rsrc,
                        safe_row * fx.Int32(hidden_dw) + dword,
                        vec_width=4,
                        dtype=T.i32,
                    )
                buffer_ops.buffer_store(value, record_rsrc, dst + dword)

            for dword in range(tx * fx.Int32(4), scale_dw, fx.Int32(1024)):
                value = zero4
                if row_valid:
                    value = buffer_ops.buffer_load(
                        scale_rsrc,
                        safe_row * fx.Int32(scale_dw) + dword,
                        vec_width=4,
                        dtype=T.i32,
                    )
                buffer_ops.buffer_store(
                    value, record_rsrc, dst + fx.Int32(hidden_dw) + dword
                )

            if tx < fx.Int32(topk):
                index = safe_row * fx.Int32(topk) + tx
                expert = buffer_ops.buffer_load(
                    ids_rsrc, index, vec_width=1, dtype=T.i32
                )
                expert_invalid = row_invalid | (expert < fx.Int32(0)) | (
                    expert >= num_experts
                )
                expert_valid = row_valid & (expert >= fx.Int32(0)) & (
                    expert < num_experts
                )
                stored_expert = expert_valid.select(expert, fx.Int32(-1))
                buffer_ops.buffer_store(
                    stored_expert, record_rsrc, dst + fx.Int32(ids_dw) + tx
                )
                weight = buffer_ops.buffer_load(
                    weights_rsrc, index, vec_width=1, dtype=T.f32
                )
                buffer_ops.buffer_store(
                    weight, record_rsrc, dst + fx.Int32(weights_dw) + tx
                )
                if expert_invalid:
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

            if tx == fx.Int32(0):
                source = buffer_ops.buffer_load(
                    source_rsrc, safe_row, vec_width=1, dtype=T.i32
                )
                source_invalid = row_invalid | (source < fx.Int32(0)) | (
                    source >= max_source_tokens
                )
                source_valid = row_valid & (source >= fx.Int32(0)) & (
                    source < max_source_tokens
                )
                buffer_ops.buffer_store(
                    source_valid.select(source, fx.Int32(-1)),
                    record_rsrc,
                    dst + fx.Int32(source_dw),
                )
                # Four reserved bytes between source and route mask.
                buffer_ops.buffer_store(
                    fx.Int32(0), record_rsrc, dst + fx.Int32(source_dw + 1)
                )
                mask_words = buffer_ops.buffer_load(
                    mask_rsrc,
                    safe_row * fx.Int32(2),
                    vec_width=2,
                    dtype=T.i32,
                )
                mask = fx.Vector(mask_words).bitcast(fx.Int64)[0]
                mask_invalid = row_invalid | (
                    (mask & fx.Int64(~allowed_mask)) != fx.Int64(0)
                )
                mask_valid = row_valid & (
                    (mask & fx.Int64(~allowed_mask)) == fx.Int64(0)
                )
                stored_mask = mask_valid.select(mask, fx.Int64(0))
                buffer_ops.buffer_store(
                    fx.Vector.from_elements([stored_mask], fx.Int64).bitcast(fx.Int32),
                    record_rsrc,
                    dst + fx.Int32(route_dw),
                )
                if source_invalid:
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))
                if mask_invalid:
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))
                if row_invalid:
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

            for item in range(tx, tail_dw, fx.Int32(256)):
                buffer_ops.buffer_store(
                    fx.Int32(0),
                    record_rsrc,
                    dst + fx.Int32(route_dw + 2) + item,
                )

        fx.rocdl.s_waitcnt(0)
        fx.gpu.barrier()
        if tx == fx.Int32(0):
            comm_ops.fence_system_release()

    @flyc.kernel(
        name=f"megamoe_dispatch_record_unpack_{tag}",
        known_block_size=[256, 1, 1],
    )
    def unpack_kernel(
        records: fx.Int64,
        hidden_out: fx.Int64,
        scales_out: fx.Int64,
        ids_out: fx.Int64,
        weights_out: fx.Int64,
        sources_out: fx.Int64,
        masks_out: fx.Int64,
        record_count: fx.Int32,
        num_experts: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
    ):
        tx = fx.Int32(gpu.thread_id("x"))
        record = fx.Int32(gpu.block_id("x"))
        if record < record_count:
            record_rsrc = buffer_ops.create_buffer_resource_from_addr(records)
            hidden_rsrc = buffer_ops.create_buffer_resource_from_addr(hidden_out)
            scale_rsrc = buffer_ops.create_buffer_resource_from_addr(scales_out)
            ids_rsrc = buffer_ops.create_buffer_resource_from_addr(ids_out)
            weights_rsrc = buffer_ops.create_buffer_resource_from_addr(weights_out)
            source_rsrc = buffer_ops.create_buffer_resource_from_addr(sources_out)
            mask_rsrc = buffer_ops.create_buffer_resource_from_addr(masks_out)
            src = record * fx.Int32(record_dw)

            for dword in range(tx * fx.Int32(4), hidden_dw, fx.Int32(1024)):
                value = buffer_ops.buffer_load(
                    record_rsrc, src + dword, vec_width=4, dtype=T.i32
                )
                buffer_ops.buffer_store(
                    value,
                    hidden_rsrc,
                    record * fx.Int32(hidden_dw) + dword,
                )

            for dword in range(tx * fx.Int32(4), scale_dw, fx.Int32(1024)):
                value = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(hidden_dw) + dword,
                    vec_width=4,
                    dtype=T.i32,
                )
                buffer_ops.buffer_store(
                    value,
                    scale_rsrc,
                    record * fx.Int32(scale_dw) + dword,
                )

            if tx < fx.Int32(topk):
                expert = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(ids_dw) + tx,
                    vec_width=1,
                    dtype=T.i32,
                )
                buffer_ops.buffer_store(
                    expert, ids_rsrc, record * fx.Int32(topk) + tx
                )
                weight = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(weights_dw) + tx,
                    vec_width=1,
                    dtype=T.f32,
                )
                buffer_ops.buffer_store(
                    weight, weights_rsrc, record * fx.Int32(topk) + tx
                )
                if (expert < fx.Int32(0)) | (expert >= num_experts):
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

            if tx == fx.Int32(0):
                source = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(source_dw),
                    vec_width=1,
                    dtype=T.i32,
                )
                buffer_ops.buffer_store(source, source_rsrc, record)
                if (source < fx.Int32(0)) | (source >= max_source_tokens):
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

                reserved = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(source_dw + 1),
                    vec_width=1,
                    dtype=T.i32,
                )
                if reserved != fx.Int32(0):
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

                mask_words = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(route_dw),
                    vec_width=2,
                    dtype=T.i32,
                )
                buffer_ops.buffer_store(
                    mask_words,
                    mask_rsrc,
                    record * fx.Int32(2),
                )
                mask = fx.Vector(mask_words).bitcast(fx.Int64)[0]
                if (mask & fx.Int64(~allowed_mask)) != fx.Int64(0):
                    comm_ops.atomic_add_system(error_flag, fx.Int32(1))

            for item in range(tx, tail_dw, fx.Int32(256)):
                padding = buffer_ops.buffer_load(
                    record_rsrc,
                    src + fx.Int32(route_dw + 2) + item,
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
        hidden_q: fx.Int64,
        scales: fx.Int64,
        topk_ids: fx.Int64,
        topk_weights: fx.Int64,
        source_tokens: fx.Int64,
        route_masks: fx.Int64,
        row_ids: fx.Int64,
        records: fx.Int64,
        record_count: fx.Int32,
        token_count: fx.Int32,
        num_experts: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
        stream: fx.Stream,
    ):
        pack_kernel(
            hidden_q,
            scales,
            topk_ids,
            topk_weights,
            source_tokens,
            route_masks,
            row_ids,
            records,
            record_count,
            token_count,
            num_experts,
            max_source_tokens,
            error_flag,
        ).launch(grid=(record_count, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_unpack(
        records: fx.Int64,
        hidden_out: fx.Int64,
        scales_out: fx.Int64,
        ids_out: fx.Int64,
        weights_out: fx.Int64,
        sources_out: fx.Int64,
        masks_out: fx.Int64,
        record_count: fx.Int32,
        num_experts: fx.Int32,
        max_source_tokens: fx.Int32,
        error_flag: fx.Int64,
        stream: fx.Stream,
    ):
        unpack_kernel(
            records,
            hidden_out,
            scales_out,
            ids_out,
            weights_out,
            sources_out,
            masks_out,
            record_count,
            num_experts,
            max_source_tokens,
            error_flag,
        ).launch(grid=(record_count, 1, 1), block=(256, 1, 1), stream=stream)

    launch_pack.record_bytes = layout.record_bytes
    launch_unpack.record_bytes = layout.record_bytes
    return DispatchRecordModule(launch_pack, launch_unpack, layout)
