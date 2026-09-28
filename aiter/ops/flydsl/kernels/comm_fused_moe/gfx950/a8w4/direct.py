# SPDX-License-Identifier: Apache-2.0
"""Direct BF16 reduce-scatter and shared-expert addition."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import T

from .... import communication_ops_utils as comm_ops
from .collectives import buffer_tensor_from_addr, load_bf16, peer_base, store_bf16
from .config import DIRECT_BLOCK, DirectConfig


@flyc.jit
def _reduce(partials, vector, vector_width: fx.Constexpr[int]):
    values = []
    for source_round in range_constexpr(len(partials)):
        values.append(
            load_bf16(
                partials[source_round],
                vector * fx.Int32(vector_width),
                vector_width,
                cache_modifier=0,
            )
        )
    acc = values[0].to(fx.Float32)
    for source_round in range_constexpr(1, len(values)):
        acc = acc + values[source_round].to(fx.Float32)
    return acc


@functools.cache
def compile_reduce_scatter_add(config: DirectConfig):
    """Reduce one BF16 row shard, add shared output, and protect input reuse."""

    shape = config.shape
    tp = shape.tp_size
    h = shape.model_dim
    shard_rows = config.output_rows
    vectors = config.vectors
    grid = config.grid
    vector_width = config.vector_width

    @fx.struct
    class SharedStorage:
        epoch: fx.Array[fx.Int64, 1, 8]

    @flyc.kernel(
        name=(
            f"comm_fused_moe_direct_{shape.tag}_m{config.m}_g{grid}"
            + (f"_v{vector_width}" if vector_width != 8 else "")
        ),
        known_block_size=[DIRECT_BLOCK, 1, 1],
    )
    def kernel(
        workspace: fx.Pointer,
        workspace_base: fx.Int64,
        output: fx.Pointer,
        shared: fx.Pointer,
        rank: fx.Int32,
    ):
        block = fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        local_base = fx.Int64(ptrtoint(workspace))
        signal_index = block * fx.Int32(tp) + rank
        ready_address = (
            local_base
            + fx.Int64(config.ready_offset)
            + fx.Int64(signal_index) * fx.Int64(8)
        )
        epoch = (
            fx.SharedAllocator()
            .allocate(SharedStorage)
            .epoch.peek()
            .view(fx.make_layout(1, 1))
        )
        if tid == fx.Int32(0):
            epoch[0] = fx.Int64(comm_ops.load_i64_global(ready_address)) + fx.Int64(1)
        gpu.barrier()
        expected = epoch[0]

        if tid < fx.Int32(tp):
            target = peer_base(workspace_base, tid)
            target_signal = block * fx.Int32(tp) + rank
            comm_ops.store_i64_global_system(
                target
                + fx.Int64(config.ready_offset)
                + fx.Int64(target_signal) * fx.Int64(8),
                expected,
            )
            local_signal = block * fx.Int32(tp) + tid
            comm_ops.spin_until_ge_i64(
                local_base
                + fx.Int64(config.ready_offset)
                + fx.Int64(local_signal) * fx.Int64(8),
                expected,
            )
            comm_ops.fence_system_acquire()
        gpu.barrier()

        item = block * fx.Int32(DIRECT_BLOCK) + tid
        stride = fx.Int32(grid * DIRECT_BLOCK)
        output_buffer = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(output)),
            fx.BFloat16,
            shard_rows * h * 2,
        )
        shared_buffer = buffer_tensor_from_addr(
            fx.Int64(ptrtoint(shared)),
            fx.BFloat16,
            shard_rows * h * 2,
        )
        partials = []
        for source_round in range_constexpr(tp):
            source = (rank + fx.Int32(source_round)) % fx.Int32(tp)
            partials.append(
                buffer_tensor_from_addr(
                    rocdl.readfirstlane(T.i64, peer_base(workspace_base, source)),
                    fx.BFloat16,
                    config.partial_bytes,
                )
            )
        # The final CTA may contain inactive lanes for a generic DirectConfig.
        # Keep every lane on a valid address until the CTA-wide barriers finish.
        safe_item = (item < fx.Int32(vectors)).select(item, fx.Int32(0))
        global_vector = rank * fx.Int32(vectors) + safe_item
        first = _reduce(partials, global_vector, vector_width).to(fx.BFloat16)

        if item < fx.Int32(vectors):
            result = first.to(fx.Float32) + load_bf16(
                shared_buffer,
                item * fx.Int32(vector_width),
                vector_width,
                cache_modifier=0,
            ).to(fx.Float32)
            store_bf16(
                output_buffer,
                item * fx.Int32(vector_width),
                result.to(fx.BFloat16),
                vector_width,
            )

        for next_item in range(item + stride, fx.Int32(vectors), stride):
            acc = _reduce(
                partials,
                rank * fx.Int32(vectors) + next_item,
                vector_width,
            ).to(fx.BFloat16)
            acc = acc.to(fx.Float32) + load_bf16(
                shared_buffer,
                next_item * fx.Int32(vector_width),
                vector_width,
                cache_modifier=0,
            ).to(fx.Float32)
            store_bf16(
                output_buffer,
                next_item * fx.Int32(vector_width),
                acc.to(fx.BFloat16),
                vector_width,
            )

        if tid < fx.Int32(tp):
            target = peer_base(workspace_base, tid)
            target_signal = block * fx.Int32(tp) + rank
            comm_ops.store_i64_global_system(
                target
                + fx.Int64(config.done_offset)
                + fx.Int64(target_signal) * fx.Int64(8),
                expected,
            )
            local_signal = block * fx.Int32(tp) + tid
            comm_ops.spin_until_ge_i64(
                local_base
                + fx.Int64(config.done_offset)
                + fx.Int64(local_signal) * fx.Int64(8),
                expected,
            )
        gpu.barrier()

    @flyc.jit
    def launch(
        workspace,
        workspace_base,
        output,
        shared,
        rank,
        stream,
    ):
        kernel(
            workspace,
            workspace_base,
            output,
            shared,
            rank,
        ).launch(
            grid=(grid, 1, 1),
            block=(DIRECT_BLOCK, 1, 1),
            stream=stream,
        )

    return launch
