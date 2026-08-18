# SPDX-License-Identifier: MIT
"""Transport-neutral epoch reset/publish kernels for hierarchical MoE."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

from aiter.ops.flydsl.kernels import buffer_ops
from .. import comm_ops

from .hier_sync import publish_generation_system


def build_hier_epoch_module(*, max_m_tiles: int, max_source_tokens: int):
    if max_m_tiles <= 0 or max_source_tokens <= 0:
        raise ValueError("epoch capacities must be positive")

    geometry = f"mt{int(max_m_tiles)}_st{int(max_source_tokens)}"

    @flyc.kernel(
        name=f"megamoe_tile_epoch_reset_{geometry}",
        known_block_size=[256, 1, 1],
    )
    def reset_kernel(
        h1_input_expected: fx.Int64,
        h1_input_ready: fx.Int64,
        h1_output_done: fx.Int64,
        h2_output_done: fx.Int64,
        rank_route_expected: fx.Int64,
        rank_route_ready: fx.Int64,
        active_m_tiles: fx.Int32,
        active_source_tokens: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        stride = fx.Int32(gpu.grid_dim.x) * fx.Int32(256)
        idx = fx.Int32(gpu.block_id("x")) * fx.Int32(256) + tid
        tile_bound = (active_m_tiles < fx.Int32(max_m_tiles)).select(
            active_m_tiles, fx.Int32(max_m_tiles)
        )
        token_bound = (active_source_tokens < fx.Int32(max_source_tokens)).select(
            active_source_tokens, fx.Int32(max_source_tokens)
        )
        input_expected = buffer_ops.create_buffer_resource_from_addr(
            h1_input_expected
        )
        input_ready = buffer_ops.create_buffer_resource_from_addr(h1_input_ready)
        output_done = buffer_ops.create_buffer_resource_from_addr(h1_output_done)
        h2_done = buffer_ops.create_buffer_resource_from_addr(h2_output_done)
        for i in range(idx, tile_bound, stride):
            buffer_ops.buffer_store(fx.Int32(0), input_expected, i)
            buffer_ops.buffer_store(fx.Int32(0), input_ready, i)
            buffer_ops.buffer_store(fx.Int32(0), output_done, i)
            buffer_ops.buffer_store(fx.Int32(0), h2_done, i)
        route_expected = buffer_ops.create_buffer_resource_from_addr(
            rank_route_expected
        )
        route_ready = buffer_ops.create_buffer_resource_from_addr(rank_route_ready)
        for i in range(idx, token_bound, stride):
            buffer_ops.buffer_store(fx.Int32(0), route_expected, i)
            buffer_ops.buffer_store(fx.Int32(0), route_ready, i)

    @flyc.kernel(
        name=f"megamoe_tile_publish_plan_{geometry}",
        known_block_size=[1, 1, 1],
    )
    def publish_plan_kernel(plan_ready: fx.Int64, generation: fx.Int64):
        publish_generation_system(plan_ready, generation)

    @flyc.kernel(
        name=f"megamoe_tile_mark_input_ready_{geometry}",
        known_block_size=[1, 1, 1],
    )
    def mark_input_kernel(input_ready: fx.Int64, tile: fx.Int32, delta: fx.Int32):
        comm_ops.fence_system_release()
        comm_ops.atomic_add_system(
            input_ready + fx.Int64(tile) * fx.Int64(4), delta
        )

    @flyc.jit
    def launch_reset(
        h1_input_expected: fx.Int64,
        h1_input_ready: fx.Int64,
        h1_output_done: fx.Int64,
        h2_output_done: fx.Int64,
        rank_route_expected: fx.Int64,
        rank_route_ready: fx.Int64,
        active_m_tiles: fx.Int32,
        active_source_tokens: fx.Int32,
        blocks: fx.Int32,
        stream: fx.Stream,
    ):
        reset_kernel(
            h1_input_expected,
            h1_input_ready,
            h1_output_done,
            h2_output_done,
            rank_route_expected,
            rank_route_ready,
            active_m_tiles,
            active_source_tokens,
        ).launch(grid=(blocks, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.jit
    def launch_publish_plan(
        plan_ready: fx.Int64, generation: fx.Int64, stream: fx.Stream
    ):
        publish_plan_kernel(plan_ready, generation).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_mark_input(
        input_ready: fx.Int64,
        tile: fx.Int32,
        delta: fx.Int32,
        stream: fx.Stream,
    ):
        mark_input_kernel(input_ready, tile, delta).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    return launch_reset, launch_publish_plan, launch_mark_input
