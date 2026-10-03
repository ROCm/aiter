# SPDX-License-Identifier: Apache-2.0
"""Direct BF16 reduce-scatter and shared-expert addition."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
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
def compile_wait_for_reuse(config: DirectConfig):
    """Wait until every rank finished reading the previous Stage2 partial."""

    tp = config.shape.tp_size

    @fx.struct
    class ReuseWaitStorage:
        epoch: fx.Array[fx.Int64, 1, 8]

    @flyc.kernel(
        name=f"comm_fused_moe_direct_reuse_wait_v2_{config.shape.tag}_m{config.m}",
        known_block_size=[64, 1, 1],
    )
    def kernel(workspace: fx.Pointer, workspace_base: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x"))
        local_base = fx.Int64(ptrtoint(workspace))
        epoch = (
            fx.SharedAllocator()
            .allocate(ReuseWaitStorage)
            .epoch.peek()
            .view(fx.make_layout(1, 1))
        )
        if tid == fx.Int32(0):
            epoch[0] = fx.Int64(
                comm_ops.load_i64_global(
                    local_base + fx.Int64(config.done_epoch_offset)
                )
            )
        gpu.barrier()
        if tid < fx.Int32(tp):
            comm_ops.spin_until_ge_i64(
                peer_base(workspace_base, tid) + fx.Int64(config.done_epoch_offset),
                epoch[0],
            )
            comm_ops.fence_system_acquire()

    @flyc.jit
    def launch(workspace, workspace_base, stream):
        kernel(workspace, workspace_base).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


@functools.cache
def compile_reduce_scatter_add(config: DirectConfig, protect_reuse: bool = True):
    """Reduce one BF16 row shard, add shared output, and protect input reuse."""

    shape = config.shape
    tp = shape.tp_size
    h = shape.model_dim
    shard_rows = config.output_rows
    vectors = config.vectors
    grid = config.grid
    vector_width = config.vector_width

    @fx.struct
    class DirectStorage:
        epoch: fx.Array[fx.Int32, 1, 4]
        arrival: fx.Array[fx.Int32, 1, 4]

    @flyc.kernel(
        name=(
            f"comm_fused_moe_direct_{shape.tag}_m{config.m}_g{grid}"
            + (f"_v{vector_width}" if vector_width != 8 else "")
            + ("_deferred_reuse_v2" if protect_reuse else "_external_sync")
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
        storage = fx.SharedAllocator().allocate(DirectStorage)
        epoch = storage.epoch.peek().view(fx.make_layout(1, 1))
        arrival = storage.arrival.peek().view(fx.make_layout(1, 1))
        ready_epoch_address = local_base + fx.Int64(config.ready_offset)
        ready_gate_address = ready_epoch_address + fx.Int64(4)
        ready_arrival_address = ready_gate_address + fx.Int64(4)
        completion_counter_address = local_base + fx.Int64(
            config.completion_counter_offset
        )
        done_epoch_address = local_base + fx.Int64(config.done_epoch_offset)

        if tid == fx.Int32(0):
            epoch[0] = fx.Int32(
                comm_ops.load_i32_global_agent(ready_gate_address)
            ) + fx.Int32(1)
        gpu.barrier()
        expected = epoch[0]

        # Stage2 completed on this stream before this kernel was launched.  A
        # single CTA publishes that rank-level readiness, waits for the other
        # ranks, and releases the remaining local CTAs through an agent-local
        # gate.  Elect the last locally arriving CTA so every CTA snapshots the
        # old gate before it can advance; otherwise a late-starting CTA could
        # mistake the new gate value for the previous epoch and wait forever.
        if tid == fx.Int32(0):
            arrival[0] = fx.Int32(
                comm_ops.atomic_add_agent_one_as(ready_arrival_address, fx.Int32(1))
            )
        gpu.barrier()
        if arrival[0] == fx.Int32(grid - 1):
            if tid == fx.Int32(0):
                comm_ops.fence_agent_acquire()
                comm_ops.store_i32_global_agent_release(
                    ready_arrival_address, fx.Int32(0)
                )
                comm_ops.store_i32_global_system_release(ready_epoch_address, expected)
            gpu.barrier()
            if tid < fx.Int32(tp):
                comm_ops.spin_until_ge_i32_system(
                    peer_base(workspace_base, tid) + fx.Int64(config.ready_offset),
                    expected,
                    acquire=True,
                )
            gpu.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_system_acquire()
                comm_ops.store_i32_global_agent_release(ready_gate_address, expected)

        if tid == fx.Int32(0):
            comm_ops.spin_until_ge_i32_agent(ready_gate_address, expected)
            comm_ops.fence_agent_acquire()
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
        if item < fx.Int32(vectors):
            reduced = _reduce(
                partials,
                rank * fx.Int32(vectors) + item,
                vector_width,
            ).to(fx.BFloat16)
            shared_value = load_bf16(
                shared_buffer,
                item * fx.Int32(vector_width),
                vector_width,
                cache_modifier=0,
            )
            result = reduced.to(fx.Float32) + shared_value.to(fx.Float32)
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

        if const_expr(protect_reuse):
            # Publish completion after every CTA has consumed its portion of all
            # peer partials.  The next invocation waits on this i64 epoch just
            # before Stage2 overwrites the partial, overlapping the wait with
            # all intervening model work.
            gpu.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                arrival[0] = fx.Int32(
                    comm_ops.atomic_add_agent_one_as(
                        completion_counter_address, fx.Int32(1)
                    )
                )
            gpu.barrier()

            if (arrival[0] == fx.Int32(grid - 1)) & (tid == fx.Int32(0)):
                comm_ops.fence_agent_acquire()
                comm_ops.store_i32_global_agent_release(
                    completion_counter_address, fx.Int32(0)
                )
                comm_ops.store_i64_global_system(done_epoch_address, fx.Int64(expected))

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
