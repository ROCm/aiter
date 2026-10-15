# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B008, SIM102

"""Flat-route MoE sorting for variable-K/ragged routing.

The dense sorter in :mod:`kernels.moe.moe_sorting_kernel` consumes a
``[tokens, top_k]`` rectangle.  Token-choice rounding instead supplies a flat
edge list ``(token, expert, weight)`` where each token may have a different
number of routes.  This module converts that list to the same metadata contract
consumed by the Sonic grouped GEMMs:

* ``sorted_token_ids`` and ``sorted_weights`` are grouped by expert;
* every expert segment is padded to ``unit_size`` with ``token == tokens``;
* ``sorted_expert_ids`` contains one expert id per padded GEMM tile; and
* ``num_valid_ids == [total_padded_routes, tokens]``.

Four kernels are launched on one stream: clear, expert histogram, padded
prefix, and route scatter.  Atomic cursors make duplicate ``(token, expert)``
edges well-defined: every input edge occupies its own output slot.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .memory import atomic_add

BLOCK_SIZE = 256
UNIT_SIZE = 32
_E16_EXPERT_MAJOR_SINGLE_LAUNCH = True
_E16_EXPERT_MAJOR_MAX_PARTITIONS = 16


_ragged_cf_cache = {}
_expert_major_cf_cache = {}


def _expert_major_identity_fusion_parameters(
    num_experts: int,
    token_indices_identity: bool,
    routes: int,
) -> tuple[bool, int]:
    """Return whether to fuse E16 identity metadata and its runtime CTA count.

    One partition covers one block of average per-expert work.  Rounding that
    count to a power of two keeps the launch topology in a small, stable family
    while still increasing parallelism with the route workload.  The returned
    count is a runtime launcher argument, not a compile specialization.
    """

    single_launch = (
        _E16_EXPERT_MAJOR_SINGLE_LAUNCH
        if num_experts == 16 and token_indices_identity
        else False
    )
    identity_partitions = 1
    if single_launch:
        target_partitions = max(
            1,
            (routes + num_experts * BLOCK_SIZE - 1) // (num_experts * BLOCK_SIZE),
        )
        identity_partitions = min(
            _E16_EXPERT_MAJOR_MAX_PARTITIONS,
            1 << (target_partitions - 1).bit_length(),
        )
    return single_launch, identity_partitions


@functools.lru_cache(maxsize=128)
def _compile_moe_ragged_sorting(
    *,
    num_experts: int,
    unit_size: int = UNIT_SIZE,
    emit_route_ids: bool = False,
    mirror_expert_frequency: bool = False,
):
    """Build the four-kernel flat-route counting sort."""

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if unit_size <= 0:
        raise ValueError(f"unit_size must be positive, got {unit_size}")

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def clear_kernel(
        expert_frequency: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_routes: fx.Int32,
        i32_tokens: fx.Int32,
        i32_max_padded: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        c_num_experts = fx.Int32(num_experts)
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(BLOCK_SIZE)
        freq_rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
        cursor_rsrc = buffer_ops.create_buffer_resource(expert_cursors, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        if const_expr(emit_route_ids):
            route_ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_route_ids, max_size=True
            )
        out_rsrc = buffer_ops.create_buffer_resource(moe_buf_i32, max_size=True)

        frequency_iters = (c_num_experts + stride - c_one) // stride
        for iteration in range(
            fx.Index(0),
            ArithValue(frequency_iters).index_cast(T.index),
            fx.Index(1),
        ):
            index = gid + fx.Int32(iteration) * stride
            if index < c_num_experts:
                buffer_ops.buffer_store(c_zero, freq_rsrc, index)
                buffer_ops.buffer_store(c_zero, cursor_rsrc, index)

        sorted_iters = (i32_max_padded + stride - c_one) // stride
        for iteration in range(
            fx.Index(0),
            ArithValue(sorted_iters).index_cast(T.index),
            fx.Index(1),
        ):
            index = gid + fx.Int32(iteration) * stride
            if index < i32_max_padded:
                # ``tokens`` is outside the valid [0, tokens) range and
                # therefore serves as the padding sentinel after GEMM's
                # low-24-bit decode.
                buffer_ops.buffer_store(i32_tokens, ids_rsrc, index)
                buffer_ops.buffer_store(c_zero, weights_rsrc, index)
                if const_expr(emit_route_ids):
                    buffer_ops.buffer_store(i32_routes, route_ids_rsrc, index)

        output_iters = (i32_moe_buf_elems + stride - c_one) // stride
        for iteration in range(
            fx.Index(0),
            ArithValue(output_iters).index_cast(T.index),
            fx.Index(1),
        ):
            index = gid + fx.Int32(iteration) * stride
            if index < i32_moe_buf_elems:
                buffer_ops.buffer_store(c_zero, out_rsrc, index)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def histogram_kernel(
        expert_indices: fx.Tensor,
        expert_frequency: fx.Tensor,
        i32_routes: fx.Int32,
    ):
        c_num_experts = fx.Int32(num_experts)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        if gid < i32_routes:
            experts_rsrc = buffer_ops.create_buffer_resource(
                expert_indices, max_size=True
            )
            expert = buffer_ops.buffer_load(experts_rsrc, gid, vec_width=1, dtype=T.i32)
            if (expert >= fx.Int32(0)) & (expert < c_num_experts):
                atomic_add(expert_frequency, expert, c_one, dtype_bytes=4)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def padded_prefix_kernel(
        expert_frequency: fx.Tensor,
        expert_frequency_mirror: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        c_unit = fx.Int32(unit_size)
        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        # E is at most O(1k) for the target models.  A single serial prefix is
        # cheaper than another hierarchy of global scans and keeps this phase
        # deterministic; the route-heavy work remains fully parallel.
        if gpu.thread_idx.x == c_zero:
            freq_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency, max_size=True
            )
            if const_expr(mirror_expert_frequency):
                mirror_rsrc = buffer_ops.create_buffer_resource(
                    expert_frequency_mirror,
                    max_size=True,
                )
            cursor_rsrc = buffer_ops.create_buffer_resource(
                expert_cursors, max_size=True
            )
            sorted_e_rsrc = buffer_ops.create_buffer_resource(
                sorted_expert_ids, max_size=True
            )
            nvalid_rsrc = buffer_ops.create_buffer_resource(
                num_valid_ids, max_size=True
            )

            offset = c_zero
            for expert_id in range_constexpr(num_experts):
                expert = fx.Int32(expert_id)
                count = buffer_ops.buffer_load(
                    freq_rsrc, expert, vec_width=1, dtype=T.i32
                )
                if const_expr(mirror_expert_frequency):
                    buffer_ops.buffer_store(count, mirror_rsrc, expert)
                blocks = (count + c_unit - c_one) // c_unit
                padded = (count == c_zero).select(c_zero, blocks * c_unit)
                buffer_ops.buffer_store(offset, cursor_rsrc, expert)
                block_start = offset // c_unit
                for block in range(
                    fx.Index(0),
                    ArithValue(blocks).index_cast(T.index),
                    fx.Index(1),
                ):
                    buffer_ops.buffer_store(
                        expert,
                        sorted_e_rsrc,
                        block_start + fx.Int32(block),
                    )
                offset = offset + padded

            buffer_ops.buffer_store(offset, nvalid_rsrc, c_zero)
            buffer_ops.buffer_store(i32_tokens, nvalid_rsrc, c_one)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def scatter_kernel(
        token_indices: fx.Tensor,
        expert_indices: fx.Tensor,
        route_weights: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        i32_routes: fx.Int32,
    ):
        c_num_experts = fx.Int32(num_experts)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        if gid < i32_routes:
            token_rsrc = buffer_ops.create_buffer_resource(token_indices, max_size=True)
            expert_rsrc = buffer_ops.create_buffer_resource(
                expert_indices, max_size=True
            )
            weights_rsrc = buffer_ops.create_buffer_resource(
                route_weights, max_size=True
            )
            sorted_ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_token_ids, max_size=True
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights, max_size=True
            )
            if const_expr(emit_route_ids):
                sorted_route_rsrc = buffer_ops.create_buffer_resource(
                    sorted_route_ids, max_size=True
                )

            token = buffer_ops.buffer_load(token_rsrc, gid, vec_width=1, dtype=T.i32)
            expert = buffer_ops.buffer_load(expert_rsrc, gid, vec_width=1, dtype=T.i32)
            weight_bits = buffer_ops.buffer_load(
                weights_rsrc, gid, vec_width=1, dtype=T.i32
            )
            if (expert >= fx.Int32(0)) & (expert < c_num_experts):
                position = fx.Int32(
                    atomic_add(expert_cursors, expert, c_one, dtype_bytes=4)
                )
                buffer_ops.buffer_store(token, sorted_ids_rsrc, position)
                buffer_ops.buffer_store(weight_bits, sorted_w_rsrc, position)
                if const_expr(emit_route_ids):
                    buffer_ops.buffer_store(gid, sorted_route_rsrc, position)

    @flyc.jit
    def launch_ragged_sorting(
        token_indices: fx.Tensor,
        expert_indices: fx.Tensor,
        route_weights: fx.Tensor,
        expert_frequency: fx.Tensor,
        expert_frequency_mirror: fx.Tensor,
        expert_cursors: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_routes: fx.Int32,
        i32_tokens: fx.Int32,
        i32_max_padded: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        i32_clear_grid: fx.Int32,
        i32_route_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear = clear_kernel(
            expert_frequency,
            expert_cursors,
            sorted_token_ids,
            sorted_weights,
            sorted_route_ids,
            moe_buf_i32,
            i32_routes,
            i32_tokens,
            i32_max_padded,
            i32_moe_buf_elems,
        )
        clear.launch(
            grid=(i32_clear_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

        histogram = histogram_kernel(expert_indices, expert_frequency, i32_routes)
        histogram.launch(
            grid=(i32_route_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

        prefix = padded_prefix_kernel(
            expert_frequency,
            expert_frequency_mirror,
            expert_cursors,
            sorted_expert_ids,
            num_valid_ids,
            i32_tokens,
        )
        prefix.launch(grid=(1, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

        scatter = scatter_kernel(
            token_indices,
            expert_indices,
            route_weights,
            expert_cursors,
            sorted_token_ids,
            sorted_weights,
            sorted_route_ids,
            i32_routes,
        )
        scatter.launch(
            grid=(i32_route_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_ragged_sorting


def _launch_cached(cache, key, launch_fn, args, stream):
    compiled = cache.get(key)
    stream_arg = fx.Stream(stream)
    if compiled is not None:
        compiled(*args, stream_arg)
        return
    launch_fn(*args, stream=stream)
    cache[key] = flyc.compile(launch_fn, *args, stream_arg)


def moe_ragged_sorting_flydsl(
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    route_weights: torch.Tensor,
    expert_frequency: torch.Tensor,
    expert_cursors: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    *,
    tokens: int,
    max_padded_routes: int,
    unit_size: int = UNIT_SIZE,
    sorted_route_ids: torch.Tensor | None = None,
    expert_frequency_mirror: torch.Tensor | None = None,
):
    """Group a flat route list into the metadata consumed by grouped GEMMs.

    Inputs are canonical device tensors: token/expert indices are contiguous
    int32 and route weights contiguous float32.  Bounds validation intentionally
    stays in the higher-level API so this launch path does not synchronize.
    ``expert_frequency`` receives the exact route occurrence count for each
    expert and is not overwritten by the prefix phase.  An optional distinct
    ``expert_frequency_mirror`` receives the same counts from the existing
    prefix dispatch, avoiding a separate device-copy launch.
    """

    routes = int(route_weights.numel())
    if int(token_indices.numel()) != routes or int(expert_indices.numel()) != routes:
        raise ValueError(
            "token_indices, expert_indices, and route_weights must have equal length"
        )
    if token_indices.dtype != torch.int32 or expert_indices.dtype != torch.int32:
        raise TypeError("token_indices and expert_indices must be int32")
    if route_weights.dtype != torch.float32:
        raise TypeError("route_weights must be float32")
    if sorted_route_ids is not None:
        if sorted_route_ids.dtype != torch.int32:
            raise TypeError("sorted_route_ids must be int32")
        if sorted_route_ids.device != token_indices.device:
            raise ValueError("sorted_route_ids must be on the route tensor device")
        if (
            not sorted_route_ids.is_contiguous()
            or int(sorted_route_ids.numel()) < max_padded_routes
        ):
            raise ValueError(
                "sorted_route_ids must be contiguous with at least "
                f"{max_padded_routes} elements"
            )
    if expert_frequency_mirror is not None:
        if not isinstance(expert_frequency_mirror, torch.Tensor):
            raise TypeError("expert_frequency_mirror must be a torch.Tensor")
        if expert_frequency_mirror.device != token_indices.device:
            raise ValueError(
                "expert_frequency_mirror must be on the route tensor device"
            )
        if (
            expert_frequency_mirror.dtype != torch.int32
            or not expert_frequency_mirror.is_contiguous()
            or tuple(expert_frequency_mirror.shape) != (num_experts,)
        ):
            raise ValueError(
                "expert_frequency_mirror must be contiguous int32 with shape "
                f"({num_experts},)"
            )
        if (
            expert_frequency_mirror.untyped_storage().data_ptr()
            == expert_frequency.untyped_storage().data_ptr()
        ):
            raise ValueError("expert_frequency_mirror must not alias expert_frequency")
    device = token_indices.device
    stream = torch.cuda.current_stream(device)
    moe_buf_i32 = moe_buf.view(torch.int32)
    clear_elems = max(num_experts, max_padded_routes, int(moe_buf_i32.numel()))
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    clear_grid = min(
        max(1, (clear_elems + BLOCK_SIZE - 1) // BLOCK_SIZE),
        num_cu * 2,
    )
    route_grid = max(1, (routes + BLOCK_SIZE - 1) // BLOCK_SIZE)

    launch_fn = _compile_moe_ragged_sorting(
        num_experts=num_experts,
        unit_size=unit_size,
        emit_route_ids=sorted_route_ids is not None,
        mirror_expert_frequency=expert_frequency_mirror is not None,
    )
    sorted_route_ids_arg = (
        expert_cursors if sorted_route_ids is None else sorted_route_ids
    )
    expert_frequency_mirror_arg = (
        expert_frequency if expert_frequency_mirror is None else expert_frequency_mirror
    )
    args = (
        token_indices,
        expert_indices,
        route_weights,
        expert_frequency,
        expert_frequency_mirror_arg,
        expert_cursors,
        sorted_token_ids,
        sorted_weights,
        sorted_route_ids_arg,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf_i32,
        routes,
        int(tokens),
        int(max_padded_routes),
        int(moe_buf_i32.numel()),
        clear_grid,
        route_grid,
    )
    # ``moe_buf`` is intentionally an opaque scratch owner: forward passes its
    # rank-2 output tensor while backward passes a compact rank-1 int32 buffer.
    # FlyDSL specializes Tensor argument rank in the compiled launcher ABI, so
    # those two call sites must not reuse the same cached function.
    cache_key = (
        num_experts,
        unit_size,
        sorted_route_ids is not None,
        expert_frequency_mirror is not None,
        moe_buf_i32.ndim,
        device.index,
    )
    _launch_cached(_ragged_cf_cache, cache_key, launch_fn, args, stream)

    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        expert_frequency,
        moe_buf,
    )


@functools.lru_cache(maxsize=128)
def _compile_moe_expert_major_sorting(
    *,
    num_experts: int,
    unit_size: int,
    emit_route_ids: bool,
    mirror_expert_frequency: bool,
    token_indices_identity: bool,
    clear_output: bool,
    single_launch_identity: bool,
    expert_counts_input: bool = False,
):
    """Build the expert-major metadata adapter.

    ``expert_offsets`` normally describes the already expert-major input as
    half-open route intervals.  In the E16 single-launch identity
    specialization, ``expert_counts_input=True`` instead interprets that
    tensor as per-expert row counts and derives the raw prefix in-kernel.
    Unlike :func:`_compile_moe_ragged_sorting`, this path
    has no histogram or atomic scatter: its first launch derives the padded
    ABI layout, and its second launch copies each route into that layout.  The
    E16 identity specialization instead derives both pieces per expert in one
    partitioned launch.
    Runtime route and identity-partition counts are ordinary scalar arguments
    and are deliberately not part of the compile-cache key.
    """

    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if unit_size <= 0:
        raise ValueError(f"unit_size must be positive, got {unit_size}")
    if expert_counts_input and not single_launch_identity:
        raise ValueError(
            "expert_counts_input requires the single-launch identity kernel"
        )

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def identity_expert_pack_kernel(
        route_weights: fx.Tensor,
        expert_offsets: fx.Tensor,
        expert_frequency: fx.Tensor,
        expert_frequency_mirror: fx.Tensor,
        expert_padded_offsets: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_routes: fx.Int32,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        i32_identity_partitions: fx.Int32,
    ):
        """Build E16 identity-route metadata with one expert-centric launch.

        Every CTA owns one expert.  ``expert_offsets`` either provides
        disjoint expert intervals or, for the E16 counts specialization, the
        16 raw expert lengths.  The CTA derives its padded destination without
        the cross-CTA dependency that required the old prefix launch.
        Lane zero computes the small E16 padded prefix and broadcasts it
        through 16 bytes of LDS while the lanes cooperatively copy routes.
        """

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_unit = fx.Int32(unit_size)
        c_block = fx.Int32(BLOCK_SIZE)
        # Keep the divisor compile-time constant: partition-major block order
        # avoids a dynamic integer divide/modulo in every lane while the grid
        # and per-expert stride remain runtime-selectable.
        expert = gpu.block_idx.x % fx.Int32(num_experts)
        partition = gpu.block_idx.x // fx.Int32(num_experts)
        thread = gpu.thread_idx.x
        offsets_rsrc = buffer_ops.create_buffer_resource(expert_offsets, max_size=True)
        frequency_rsrc = buffer_ops.create_buffer_resource(
            expert_frequency, max_size=True
        )
        padded_offsets_rsrc = buffer_ops.create_buffer_resource(
            expert_padded_offsets, max_size=True
        )
        weights_in_rsrc = buffer_ops.create_buffer_resource(
            route_weights, max_size=True
        )
        ids_out_rsrc = buffer_ops.create_buffer_resource(
            sorted_token_ids, max_size=True
        )
        weights_out_rsrc = buffer_ops.create_buffer_resource(
            sorted_weights, max_size=True
        )
        experts_out_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=True
        )
        valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
        if const_expr(emit_route_ids):
            route_ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_route_ids, max_size=True
            )
        if const_expr(mirror_expert_frequency):
            mirror_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency_mirror, max_size=True
            )
        if const_expr(clear_output):
            output_rsrc = buffer_ops.create_buffer_resource(moe_buf_i32, max_size=True)

        # begin, count, padded row count, and padded output offset.  Only lane
        # zero computes the short E16 prefix; LDS broadcasts it to the CTA.
        shared = fx.SharedAllocator().allocate(16, alignment=16).peek()
        metadata = fx.recast_iter(fx.Int32, shared.ptr)
        if thread == c_zero:
            if const_expr(expert_counts_input):
                count_lane0 = buffer_ops.buffer_load(
                    offsets_rsrc, expert, vec_width=1, dtype=T.i32
                )
                begin_lane0 = c_zero
                padded_offset_lane0 = c_zero
                for prefix_expert_id in range_constexpr(num_experts):
                    prefix_expert = fx.Int32(prefix_expert_id)
                    prefix_count = buffer_ops.buffer_load(
                        offsets_rsrc,
                        prefix_expert,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    prefix_blocks = (prefix_count + c_unit - c_one) // c_unit
                    prefix_padded = (prefix_count == c_zero).select(
                        c_zero, prefix_blocks * c_unit
                    )
                    is_prior = prefix_expert < expert
                    begin_lane0 = begin_lane0 + is_prior.select(
                        prefix_count,
                        c_zero,
                    )
                    padded_offset_lane0 = padded_offset_lane0 + is_prior.select(
                        prefix_padded,
                        c_zero,
                    )
            else:
                begin_lane0 = buffer_ops.buffer_load(
                    offsets_rsrc, expert, vec_width=1, dtype=T.i32
                )
                end_lane0 = buffer_ops.buffer_load(
                    offsets_rsrc, expert + c_one, vec_width=1, dtype=T.i32
                )
                count_lane0 = end_lane0 - begin_lane0
                padded_offset_lane0 = begin_lane0
                prefix_begin = buffer_ops.buffer_load(
                    offsets_rsrc, c_zero, vec_width=1, dtype=T.i32
                )
                for prefix_expert_id in range_constexpr(num_experts):
                    prefix_expert = fx.Int32(prefix_expert_id)
                    prefix_end = buffer_ops.buffer_load(
                        offsets_rsrc,
                        prefix_expert + c_one,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    prefix_count = prefix_end - prefix_begin
                    prefix_blocks = (prefix_count + c_unit - c_one) // c_unit
                    prefix_padded = (prefix_count == c_zero).select(
                        c_zero, prefix_blocks * c_unit
                    )
                    prefix_padding = prefix_padded - prefix_count
                    padded_offset_lane0 = padded_offset_lane0 + (
                        prefix_expert < expert
                    ).select(prefix_padding, c_zero)
                    prefix_begin = prefix_end
            blocks_lane0 = (count_lane0 + c_unit - c_one) // c_unit
            padded_lane0 = (count_lane0 == c_zero).select(c_zero, blocks_lane0 * c_unit)
            fx.ptr_store(begin_lane0, metadata)
            fx.ptr_store(count_lane0, metadata + fx.Int64(1))
            fx.ptr_store(padded_lane0, metadata + fx.Int64(2))
            fx.ptr_store(padded_offset_lane0, metadata + fx.Int64(3))
        gpu.barrier()
        begin = fx.Int32(fx.ptr_load(metadata))
        count = fx.Int32(fx.ptr_load(metadata + fx.Int64(1)))
        padded = fx.Int32(fx.ptr_load(metadata + fx.Int64(2)))
        padded_offset = fx.Int32(fx.ptr_load(metadata + fx.Int64(3)))
        blocks = padded // c_unit

        if (partition == c_zero) & (thread == c_zero):
            buffer_ops.buffer_store(count, frequency_rsrc, expert)
            buffer_ops.buffer_store(padded_offset, padded_offsets_rsrc, expert)
            if const_expr(mirror_expert_frequency):
                buffer_ops.buffer_store(count, mirror_rsrc, expert)
            if expert == fx.Int32(num_experts - 1):
                buffer_ops.buffer_store(padded_offset + padded, valid_rsrc, c_zero)
                buffer_ops.buffer_store(i32_tokens, valid_rsrc, c_one)

        if partition == c_zero:
            block_start = padded_offset // c_unit
            block_iters = (blocks + c_block - c_one) // c_block
            for iteration in range(
                fx.Index(0),
                ArithValue(block_iters).index_cast(T.index),
                fx.Index(1),
            ):
                block = thread + fx.Int32(iteration) * c_block
                if block < blocks:
                    buffer_ops.buffer_store(
                        expert,
                        experts_out_rsrc,
                        block_start + block,
                    )

            padding_rows = padded - count
            padding_iters = (padding_rows + c_block - c_one) // c_block
            for iteration in range(
                fx.Index(0),
                ArithValue(padding_iters).index_cast(T.index),
                fx.Index(1),
            ):
                padding = thread + fx.Int32(iteration) * c_block
                if padding < padding_rows:
                    output_row = padded_offset + count + padding
                    buffer_ops.buffer_store(i32_tokens, ids_out_rsrc, output_row)
                    buffer_ops.buffer_store(
                        fx.Float32(0.0), weights_out_rsrc, output_row
                    )
                    if const_expr(emit_route_ids):
                        buffer_ops.buffer_store(i32_routes, route_ids_rsrc, output_row)

        partition_stride = i32_identity_partitions * c_block
        partition_start = partition * c_block
        route_iters = (count + partition_stride - c_one) // partition_stride
        for iteration in range(
            fx.Index(0),
            ArithValue(route_iters).index_cast(T.index),
            fx.Index(1),
        ):
            local_row = (
                partition_start + thread + fx.Int32(iteration) * partition_stride
            )
            if local_row < count:
                route = begin + local_row
                output_row = padded_offset + local_row
                weight_bits = buffer_ops.buffer_load(
                    weights_in_rsrc, route, vec_width=1, dtype=T.i32
                )
                buffer_ops.buffer_store(route, ids_out_rsrc, output_row)
                buffer_ops.buffer_store(weight_bits, weights_out_rsrc, output_row)
                if const_expr(emit_route_ids):
                    buffer_ops.buffer_store(route, route_ids_rsrc, output_row)

        if const_expr(clear_output):
            global_thread = gpu.block_idx.x * c_block + thread
            global_stride = gpu.grid_dim.x * c_block
            clear_iters = (i32_moe_buf_elems + global_stride - c_one) // global_stride
            for iteration in range(
                fx.Index(0),
                ArithValue(clear_iters).index_cast(T.index),
                fx.Index(1),
            ):
                output_index = global_thread + fx.Int32(iteration) * global_stride
                if output_index < i32_moe_buf_elems:
                    buffer_ops.buffer_store(c_zero, output_rsrc, output_index)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def prefix_and_padding_kernel(
        expert_offsets: fx.Tensor,
        expert_frequency: fx.Tensor,
        expert_frequency_mirror: fx.Tensor,
        expert_padded_offsets: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_routes: fx.Int32,
        i32_tokens: fx.Int32,
    ):
        """Emit counts, padded expert tiles, and only the padding rows.

        A single CTA is intentional: E16's prefix is tiny and serializing it
        makes the padded ABI deterministic without a temporary prefix buffer.
        Real rows are written in parallel by ``pack_kernel``.
        """

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_unit = fx.Int32(unit_size)
        if gpu.thread_idx.x == c_zero:
            offsets_rsrc = buffer_ops.create_buffer_resource(
                expert_offsets, max_size=True
            )
            frequency_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency, max_size=True
            )
            padded_offsets_rsrc = buffer_ops.create_buffer_resource(
                expert_padded_offsets, max_size=True
            )
            ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_token_ids, max_size=True
            )
            weights_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights, max_size=True
            )
            experts_rsrc = buffer_ops.create_buffer_resource(
                sorted_expert_ids, max_size=True
            )
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            if const_expr(emit_route_ids):
                route_ids_rsrc = buffer_ops.create_buffer_resource(
                    sorted_route_ids, max_size=True
                )
            if const_expr(mirror_expert_frequency):
                mirror_rsrc = buffer_ops.create_buffer_resource(
                    expert_frequency_mirror, max_size=True
                )

            padded_offset = c_zero
            for expert_id in range_constexpr(num_experts):
                expert = fx.Int32(expert_id)
                begin = buffer_ops.buffer_load(
                    offsets_rsrc, expert, vec_width=1, dtype=T.i32
                )
                end = buffer_ops.buffer_load(
                    offsets_rsrc, expert + c_one, vec_width=1, dtype=T.i32
                )
                count = end - begin
                blocks = (count + c_unit - c_one) // c_unit
                padded = (count == c_zero).select(c_zero, blocks * c_unit)
                buffer_ops.buffer_store(count, frequency_rsrc, expert)
                buffer_ops.buffer_store(
                    padded_offset,
                    padded_offsets_rsrc,
                    expert,
                )
                if const_expr(mirror_expert_frequency):
                    buffer_ops.buffer_store(count, mirror_rsrc, expert)

                block_start = padded_offset // c_unit
                for block in range(
                    fx.Index(0),
                    ArithValue(blocks).index_cast(T.index),
                    fx.Index(1),
                ):
                    buffer_ops.buffer_store(
                        expert,
                        experts_rsrc,
                        block_start + fx.Int32(block),
                    )

                # The output buffers need initialization only in the tail of
                # every non-empty expert segment.  This bounded serial work
                # replaces the ragged sorter's whole-buffer clear launch.
                for row in range(
                    ArithValue(count).index_cast(T.index),
                    ArithValue(padded).index_cast(T.index),
                    fx.Index(1),
                ):
                    output_row = padded_offset + fx.Int32(row)
                    buffer_ops.buffer_store(i32_tokens, ids_rsrc, output_row)
                    buffer_ops.buffer_store(fx.Float32(0.0), weights_rsrc, output_row)
                    if const_expr(emit_route_ids):
                        buffer_ops.buffer_store(i32_routes, route_ids_rsrc, output_row)
                padded_offset = padded_offset + padded

            buffer_ops.buffer_store(padded_offset, valid_rsrc, c_zero)
            buffer_ops.buffer_store(i32_tokens, valid_rsrc, c_one)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def pack_kernel(
        token_indices: fx.Tensor,
        expert_indices: fx.Tensor,
        route_weights: fx.Tensor,
        expert_offsets: fx.Tensor,
        expert_padded_offsets: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_routes: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
    ):
        """Pack routes directly from contiguous expert-major intervals."""

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        gid = gpu.block_idx.x * fx.Int32(BLOCK_SIZE) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(BLOCK_SIZE)

        if const_expr(clear_output):
            output_rsrc = buffer_ops.create_buffer_resource(moe_buf_i32, max_size=True)
            clear_iters = (i32_moe_buf_elems + stride - c_one) // stride
            for iteration in range(
                fx.Index(0),
                ArithValue(clear_iters).index_cast(T.index),
                fx.Index(1),
            ):
                output_index = gid + fx.Int32(iteration) * stride
                if output_index < i32_moe_buf_elems:
                    buffer_ops.buffer_store(c_zero, output_rsrc, output_index)

        if gid < i32_routes:
            experts_rsrc = buffer_ops.create_buffer_resource(
                expert_indices, max_size=True
            )
            offsets_rsrc = buffer_ops.create_buffer_resource(
                expert_offsets, max_size=True
            )
            padded_offsets_rsrc = buffer_ops.create_buffer_resource(
                expert_padded_offsets, max_size=True
            )
            weights_in_rsrc = buffer_ops.create_buffer_resource(
                route_weights, max_size=True
            )
            ids_out_rsrc = buffer_ops.create_buffer_resource(
                sorted_token_ids, max_size=True
            )
            weights_out_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights, max_size=True
            )
            expert = buffer_ops.buffer_load(experts_rsrc, gid, vec_width=1, dtype=T.i32)
            begin = buffer_ops.buffer_load(
                offsets_rsrc, expert, vec_width=1, dtype=T.i32
            )

            output_offset = buffer_ops.buffer_load(
                padded_offsets_rsrc,
                expert,
                vec_width=1,
                dtype=T.i32,
            )
            output_index = output_offset + gid - begin

            if const_expr(token_indices_identity):
                token = gid
            else:
                tokens_rsrc = buffer_ops.create_buffer_resource(
                    token_indices, max_size=True
                )
                token = buffer_ops.buffer_load(
                    tokens_rsrc, gid, vec_width=1, dtype=T.i32
                )
            weight_bits = buffer_ops.buffer_load(
                weights_in_rsrc, gid, vec_width=1, dtype=T.i32
            )
            buffer_ops.buffer_store(token, ids_out_rsrc, output_index)
            buffer_ops.buffer_store(weight_bits, weights_out_rsrc, output_index)
            if const_expr(emit_route_ids):
                route_ids_rsrc = buffer_ops.create_buffer_resource(
                    sorted_route_ids, max_size=True
                )
                buffer_ops.buffer_store(gid, route_ids_rsrc, output_index)

    @flyc.jit
    def launch_expert_major_sorting(
        token_indices: fx.Tensor,
        expert_indices: fx.Tensor,
        route_weights: fx.Tensor,
        expert_offsets: fx.Tensor,
        expert_frequency: fx.Tensor,
        expert_frequency_mirror: fx.Tensor,
        expert_padded_offsets: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        moe_buf_i32: fx.Tensor,
        i32_routes: fx.Int32,
        i32_tokens: fx.Int32,
        i32_moe_buf_elems: fx.Int32,
        i32_identity_partitions: fx.Int32,
        i32_route_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        if const_expr(single_launch_identity):
            fused = identity_expert_pack_kernel(
                route_weights,
                expert_offsets,
                expert_frequency,
                expert_frequency_mirror,
                expert_padded_offsets,
                sorted_token_ids,
                sorted_weights,
                sorted_route_ids,
                sorted_expert_ids,
                num_valid_ids,
                moe_buf_i32,
                i32_routes,
                i32_tokens,
                i32_moe_buf_elems,
                i32_identity_partitions,
            )
            fused.launch(
                grid=(
                    fx.Int64(i32_identity_partitions) * fx.Int64(num_experts),
                    1,
                    1,
                ),
                block=(BLOCK_SIZE, 1, 1),
                stream=stream,
            )
            return

        prefix = prefix_and_padding_kernel(
            expert_offsets,
            expert_frequency,
            expert_frequency_mirror,
            expert_padded_offsets,
            sorted_token_ids,
            sorted_weights,
            sorted_route_ids,
            sorted_expert_ids,
            num_valid_ids,
            i32_routes,
            i32_tokens,
        )
        prefix.launch(grid=(1, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)

        pack = pack_kernel(
            token_indices,
            expert_indices,
            route_weights,
            expert_offsets,
            expert_padded_offsets,
            sorted_token_ids,
            sorted_weights,
            sorted_route_ids,
            moe_buf_i32,
            i32_routes,
            i32_moe_buf_elems,
        )
        pack.launch(
            grid=(i32_route_grid, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_expert_major_sorting


def moe_expert_major_sorting_flydsl(
    token_indices: torch.Tensor | None,
    expert_indices: torch.Tensor | None,
    route_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_frequency: torch.Tensor,
    expert_padded_offsets: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    *,
    tokens: int,
    max_padded_routes: int,
    unit_size: int = UNIT_SIZE,
    sorted_route_ids: torch.Tensor | None = None,
    expert_frequency_mirror: torch.Tensor | None = None,
    token_indices_identity: bool = False,
    clear_output: bool = False,
    route_policy_size: int | None = None,
    expert_counts_input: bool = False,
):
    """Build grouped-GEMM metadata from an already expert-major route list.

    ``expert_offsets`` is normally contiguous int32 ``[E + 1]`` and defines
    the half-open input interval for each expert.  With
    ``expert_counts_input=True`` it is instead contiguous int32 ``[E]`` raw
    counts; this mode is restricted to implicit E16 identity ids.  The caller
    guarantees that the intervals/counts partition ``[0, R)``.  Keeping that
    value check outside this asynchronous launch is deliberate: it avoids
    device-to-host reads on the dynamic-R hot path.

    ``expert_padded_offsets`` is caller-owned contiguous int32 ``[E]`` scratch
    connecting the prefix and parallel pack launches.  The emitted tensors
    have the exact flat ragged-sorter ABI: padded token
    ids use ``tokens`` as sentinel, padded weights are zero, optional route ids
    use ``R`` as sentinel, and ``num_valid_ids == [P, tokens]`` where
    ``P = sum_e ceil(count_e / unit_size) * unit_size``.  Generic calls launch
    one prefix/padding CTA and one parallel pack CTA; E16 identity calls use a
    single partitioned expert-centric kernel.  That E16 identity specialization
    also accepts ``token_indices=expert_indices=None``: route and expert ids are
    derived from the identity/expert-major contract without materializing the
    two input vectors.  No other specialization accepts omitted ids.
    Set ``clear_output`` only when ``moe_buf`` owns an output/scratch tensor
    that must be zeroed as part of the second CTA launch.

    ``route_policy_size`` is accepted for API compatibility with the shared
    E16 tuning policy.  Sorter storage, work loops, and the runtime CTA count
    always use the actual route count; the hint is validated but never enters
    a compiler/cache key or inflates a short rank's launch.
    """

    routes = int(route_weights.numel())
    if num_experts <= 0 or unit_size <= 0:
        raise ValueError("num_experts and unit_size must be positive")
    if route_weights.ndim != 1:
        raise ValueError("route_weights must be one-dimensional")
    implicit_identity_ids = token_indices is None and expert_indices is None
    if (token_indices is None) != (expert_indices is None):
        raise ValueError(
            "token_indices and expert_indices must either both be tensors or "
            "both be None"
        )
    if not implicit_identity_ids:
        assert token_indices is not None
        assert expert_indices is not None
        if token_indices.ndim != 1 or expert_indices.ndim != 1:
            raise ValueError("token_indices and expert_indices must be one-dimensional")
        if (
            int(token_indices.numel()) != routes
            or int(expert_indices.numel()) != routes
        ):
            raise ValueError(
                "token_indices, expert_indices, and route_weights must have equal length"
            )
    device = route_weights.device
    tensors = {
        "expert_offsets": expert_offsets,
        "expert_frequency": expert_frequency,
        "expert_padded_offsets": expert_padded_offsets,
        "sorted_token_ids": sorted_token_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "moe_buf": moe_buf,
    }
    if not implicit_identity_ids:
        assert token_indices is not None
        assert expert_indices is not None
        tensors["token_indices"] = token_indices
        tensors["expert_indices"] = expert_indices
    if not route_weights.is_cuda:
        raise ValueError("route tensors must be CUDA/ROCm tensors")
    if route_weights.dtype != torch.float32 or not route_weights.is_contiguous():
        raise ValueError("route_weights must be contiguous float32")
    if not implicit_identity_ids:
        assert token_indices is not None
        assert expert_indices is not None
        if token_indices.dtype != torch.int32 or expert_indices.dtype != torch.int32:
            raise TypeError("token_indices and expert_indices must be int32")
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the route tensor device")
    if not isinstance(expert_counts_input, bool):
        raise TypeError("expert_counts_input must be bool")
    expected_boundaries_shape = (
        (num_experts,) if expert_counts_input else (num_experts + 1,)
    )
    if (
        expert_offsets.dtype != torch.int32
        or tuple(expert_offsets.shape) != expected_boundaries_shape
    ):
        input_name = "expert_counts" if expert_counts_input else "expert_offsets"
        raise ValueError(
            f"{input_name} must be contiguous int32 with shape "
            f"{expected_boundaries_shape}"
        )
    if expert_frequency.dtype != torch.int32 or tuple(expert_frequency.shape) != (
        num_experts,
    ):
        raise ValueError(
            f"expert_frequency must be contiguous int32 with shape ({num_experts},)"
        )
    if expert_padded_offsets.dtype != torch.int32 or tuple(
        expert_padded_offsets.shape
    ) != (num_experts,):
        raise ValueError(
            f"expert_padded_offsets must be contiguous int32 with shape ({num_experts},)"
        )
    if (
        sorted_token_ids.ndim != 1
        or sorted_token_ids.dtype != torch.int32
        or int(sorted_token_ids.numel()) < max_padded_routes
    ):
        raise ValueError(
            "sorted_token_ids must be int32 with at least max_padded_routes elements"
        )
    if (
        sorted_weights.ndim != 1
        or sorted_weights.dtype != torch.float32
        or int(sorted_weights.numel()) < max_padded_routes
    ):
        raise ValueError(
            "sorted_weights must be float32 with at least max_padded_routes elements"
        )
    if (
        sorted_expert_ids.ndim != 1
        or sorted_expert_ids.dtype != torch.int32
        or int(sorted_expert_ids.numel())
        < (max_padded_routes + unit_size - 1) // unit_size
    ):
        raise ValueError("sorted_expert_ids has insufficient int32 block capacity")
    if (
        num_valid_ids.ndim != 1
        or num_valid_ids.dtype != torch.int32
        or int(num_valid_ids.numel()) < 2
    ):
        raise ValueError("num_valid_ids must have at least two int32 elements")
    if sorted_route_ids is not None:
        if (
            sorted_route_ids.device != device
            or sorted_route_ids.ndim != 1
            or sorted_route_ids.dtype != torch.int32
            or not sorted_route_ids.is_contiguous()
            or int(sorted_route_ids.numel()) < max_padded_routes
        ):
            raise ValueError(
                "sorted_route_ids must be contiguous int32 with at least max_padded_routes elements"
            )
    if expert_frequency_mirror is not None:
        if (
            expert_frequency_mirror.device != device
            or expert_frequency_mirror.dtype != torch.int32
            or not expert_frequency_mirror.is_contiguous()
            or tuple(expert_frequency_mirror.shape) != (num_experts,)
        ):
            raise ValueError(
                f"expert_frequency_mirror must be contiguous int32 with shape ({num_experts},)"
            )
        if (
            expert_frequency_mirror.untyped_storage().data_ptr()
            == expert_frequency.untyped_storage().data_ptr()
        ):
            raise ValueError("expert_frequency_mirror must not alias expert_frequency")
    if not isinstance(token_indices_identity, bool) or not isinstance(
        clear_output, bool
    ):
        raise TypeError("token_indices_identity and clear_output must be bool")
    if route_policy_size is not None:
        if isinstance(route_policy_size, bool) or not isinstance(
            route_policy_size, int
        ):
            raise TypeError("route_policy_size must be None or an integer")
        if route_policy_size <= 0:
            raise ValueError(
                "route_policy_size must be positive when supplied, got "
                f"{route_policy_size}"
            )

    stream = torch.cuda.current_stream(device)
    moe_buf_i32 = moe_buf.view(torch.int32)
    route_grid = max(1, (routes + BLOCK_SIZE - 1) // BLOCK_SIZE)
    single_launch_identity, identity_partitions = (
        _expert_major_identity_fusion_parameters(
            num_experts,
            token_indices_identity,
            routes,
        )
    )
    if implicit_identity_ids and not single_launch_identity:
        raise NotImplementedError(
            "implicit expert-major ids require the E16 single-launch identity "
            "metadata specialization"
        )
    if expert_counts_input and not implicit_identity_ids:
        raise ValueError("expert_counts_input requires omitted token/expert ids")
    launch_fn = _compile_moe_expert_major_sorting(
        num_experts=num_experts,
        unit_size=unit_size,
        emit_route_ids=sorted_route_ids is not None,
        mirror_expert_frequency=expert_frequency_mirror is not None,
        token_indices_identity=token_indices_identity,
        clear_output=clear_output,
        single_launch_identity=single_launch_identity,
        expert_counts_input=expert_counts_input,
    )
    sorted_route_ids_arg = (
        expert_frequency if sorted_route_ids is None else sorted_route_ids
    )
    expert_frequency_mirror_arg = (
        expert_frequency if expert_frequency_mirror is None else expert_frequency_mirror
    )
    # Preserve the existing compiled launcher ABI.  In the guarded
    # single-launch identity branch these first two arguments are not
    # dereferenced; use the already validated int32 offsets tensor instead of
    # allocating fake route-id vectors.  The check immediately above prevents
    # this placeholder from ever reaching the generic pack branch.
    token_indices_arg = expert_offsets if implicit_identity_ids else token_indices
    expert_indices_arg = expert_offsets if implicit_identity_ids else expert_indices
    assert token_indices_arg is not None
    assert expert_indices_arg is not None
    args = (
        token_indices_arg,
        expert_indices_arg,
        route_weights,
        expert_offsets,
        expert_frequency,
        expert_frequency_mirror_arg,
        expert_padded_offsets,
        sorted_token_ids,
        sorted_weights,
        sorted_route_ids_arg,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf_i32,
        routes,
        int(tokens),
        int(moe_buf_i32.numel()),
        identity_partitions,
        route_grid,
    )
    cache_key = (
        num_experts,
        unit_size,
        sorted_route_ids is not None,
        expert_frequency_mirror is not None,
        token_indices_identity,
        clear_output,
        single_launch_identity,
        expert_counts_input,
        moe_buf_i32.ndim,
        device.index,
    )
    _launch_cached(_expert_major_cf_cache, cache_key, launch_fn, args, stream)
    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        expert_frequency,
        moe_buf,
    )


__all__ = ["moe_expert_major_sorting_flydsl", "moe_ragged_sorting_flydsl"]
