# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# ruff: noqa: B008

"""Device-side work compaction for grouped SonicMoE GEMMs.

The route sorters pad every non-empty expert segment to ``sorted_block_m``.
Launching one GEMM workgroup for every padded sub-tile wastes work for sparse
experts, while assigning one workgroup to an expert serializes long segments.
This module builds a compact list containing only the real ``block_m`` tiles:

``descriptor = global sorted-row offset // block_m``.

One device thread handles one expert.  It locates that expert's first sorted
metadata block, atomically reserves ``ceil(frequency / block_m)`` descriptor
slots, and writes the corresponding global M-block indices.  Descriptor order
is deliberately unspecified; grouped GEMMs must treat it as a work queue.

For GEMM M tiles wider than the sorter's metadata unit, the legacy scalar
descriptor is insufficient: a tile can start at a row that is not aligned to
``block_m`` and the final tile can contain fewer than ``block_m`` real rows.
The exact queue API therefore stores a self-contained record for every tile:

``queue = [count, (expert, first_sorted_row, valid_rows) * capacity]``.

Unlike an active-expert queue, this schedule exposes every M tile as an
independent work item.  A grouped GEMM can consequently map one CTA to one
``(M tile, N tile)`` without serializing all M tiles of a hot expert.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .memory import atomic_add

_BLOCK_THREADS = 256
_MAX_SIGNED_I32 = (1 << 31) - 1


def compact_m_tile_descriptor_upper_bound(
    routes: int,
    num_experts: int,
    block_m: int,
    *,
    max_expert_rows: int | None = None,
) -> int:
    """Return a host-known upper bound for ``sum_e ceil(freq[e] / block_m)``.

    ``max_expert_rows`` can tighten the bound when the routing contract limits
    each expert's count.  Fixed-K routing with unique experts per token may pass
    ``tokens``; ragged routing, which permits duplicate ``(token, expert)``
    edges, should leave it unset.
    """

    for name, value in (
        ("routes", routes),
        ("num_experts", num_experts),
        ("block_m", block_m),
    ):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if routes > _MAX_SIGNED_I32:
        raise ValueError(f"routes exceeds signed int32 metadata capacity: {routes}")
    if max_expert_rows is not None:
        if not isinstance(max_expert_rows, int):
            raise TypeError(
                f"max_expert_rows must be an int or None, got {type(max_expert_rows).__name__}"
            )
        if max_expert_rows < 0:
            raise ValueError(
                f"max_expert_rows must be non-negative, got {max_expert_rows}"
            )
        if routes > num_experts * max_expert_rows:
            raise ValueError(
                f"routes={routes} cannot fit num_experts={num_experts} with "
                f"max_expert_rows={max_expert_rows}"
            )

    if routes == 0:
        return 0
    active_experts = min(num_experts, routes)
    # For A non-empty experts, sum ceil(c_e/BM) is at most
    # floor((sum(c_e) + A*(BM-1))/BM).
    bound = (routes + active_experts * (block_m - 1)) // block_m
    if max_expert_rows is not None:
        per_expert_tiles = (max_expert_rows + block_m - 1) // block_m
        bound = min(bound, active_experts * per_expert_tiles)
    if bound > _MAX_SIGNED_I32:
        raise ValueError(f"descriptor count exceeds signed int32: {bound}")
    return bound


def fixed_compact_m_tile_descriptor_upper_bound(
    tokens: int,
    num_experts: int,
    topk: int,
    block_m: int,
) -> int:
    """Upper bound for fixed-K routing with unique expert IDs per token."""

    for name, value in (("tokens", tokens), ("topk", topk)):
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if topk > num_experts:
        raise ValueError(
            f"fixed-K unique routing requires topk <= num_experts, got topk={topk}, "
            f"num_experts={num_experts}"
        )
    return compact_m_tile_descriptor_upper_bound(
        tokens * topk,
        num_experts,
        block_m,
        max_expert_rows=tokens,
    )


def ragged_compact_m_tile_descriptor_upper_bound(
    routes: int,
    num_experts: int,
    block_m: int,
) -> int:
    """Upper bound for flat ragged routing, including duplicate edges."""

    return compact_m_tile_descriptor_upper_bound(routes, num_experts, block_m)


def exact_m_tile_queue_upper_bound(
    routes: int,
    num_experts: int,
    block_m: int,
    *,
    max_expert_rows: int | None = None,
) -> int:
    """Return the required record capacity for an exact M-tile queue."""

    return compact_m_tile_descriptor_upper_bound(
        routes,
        num_experts,
        block_m,
        max_expert_rows=max_expert_rows,
    )


def exact_m_tile_queue_elements(
    routes: int,
    num_experts: int,
    block_m: int,
    *,
    max_expert_rows: int | None = None,
) -> int:
    """Return int32 storage elements for the exact queue's counter and records."""

    capacity = exact_m_tile_queue_upper_bound(
        routes,
        num_experts,
        block_m,
        max_expert_rows=max_expert_rows,
    )
    return 1 + 3 * capacity


@functools.lru_cache(maxsize=128)
def compile_exact_m_tile_queue_builder(
    num_experts: int,
    block_m: int,
    sorted_block_m: int,
    device_index: int,
    emit_active_experts: bool = False,
    emit_hot_splits: bool = False,
):
    """Compile a device-side exact M-tile queue builder.

    The returned launcher writes the counter-first ABI
    ``[count, (expert, first_row, valid_rows) * capacity]``.  ``first_row`` is
    an exact sorted-row offset, so ``block_m`` need not divide
    ``sorted_block_m``.  When ``emit_active_experts`` is true, the same scan
    also fills ``[active_count, (expert, first_row) * active_capacity]`` for
    grouped TN contractions.  ``emit_hot_splits`` additionally emits compact
    split-K and hot-expert queues during the same E16 device scan.  Split row
    thresholds are runtime launcher arguments so changing a dynamic route
    capacity never creates another compiled kernel family.
    """

    del device_index
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if sorted_block_m <= 0:
        raise ValueError(f"sorted_block_m must be positive, got {sorted_block_m}")
    if emit_hot_splits and (not emit_active_experts or num_experts > _BLOCK_THREADS):
        raise ValueError("hot split emission requires active experts and one workgroup")

    max_metadata_blocks = _MAX_SIGNED_I32 // sorted_block_m
    lower_bound_steps = max(1, max_metadata_blocks.bit_length())

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_queue_kernel(
        queue: fx.Tensor,
        active_expert_storage: fx.Tensor,
    ):
        if gpu.thread_idx.x == fx.Int32(0):
            queue_rsrc = buffer_ops.create_buffer_resource(queue, max_size=True)
            buffer_ops.buffer_store(fx.Int32(0), queue_rsrc, fx.Int32(0))
            if const_expr(emit_active_experts):
                active_rsrc = buffer_ops.create_buffer_resource(
                    active_expert_storage,
                    max_size=True,
                )
                buffer_ops.buffer_store(fx.Int32(0), active_rsrc, fx.Int32(0))

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clamp_queue_counts_kernel(
        queue: fx.Tensor,
        i32_queue_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
    ):
        if gpu.thread_idx.x == fx.Int32(0):
            queue_rsrc = buffer_ops.create_buffer_resource(queue, max_size=True)
            required_records = fx.Int32(
                buffer_ops.buffer_load(
                    queue_rsrc,
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            stored_records = (required_records < i32_queue_capacity).select(
                required_records,
                i32_queue_capacity,
            )
            buffer_ops.buffer_store(stored_records, queue_rsrc, fx.Int32(0))
            if const_expr(emit_active_experts):
                active_rsrc = buffer_ops.create_buffer_resource(
                    active_expert_storage,
                    max_size=True,
                )
                required_active = fx.Int32(
                    buffer_ops.buffer_load(
                        active_rsrc,
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                stored_active = (required_active < i32_active_expert_capacity).select(
                    required_active,
                    i32_active_expert_capacity,
                )
                buffer_ops.buffer_store(
                    stored_active,
                    active_rsrc,
                    fx.Int32(0),
                )

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def build_queue_kernel(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        queue: fx.Tensor,
        i32_queue_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
        hot_split_storage: fx.Tensor,
        i32_hot_split_capacity: fx.Int32,
        hot_expert_storage: fx.Tensor,
        i32_hot_expert_capacity: fx.Int32,
        i32_split_rows: fx.Int32,
        i32_min_hot_rows: fx.Int32,
        i32_split_activation_rows: fx.Int32,
        i32_sparse_split_max_experts: fx.Int32,
    ):
        if const_expr(num_experts <= _BLOCK_THREADS):
            # The Qwen3 E16 target fits in one workgroup, so clear and build in
            # one dispatch.  The barrier makes the counter visible before any
            # expert reserves records.
            if gpu.thread_idx.x == fx.Int32(0):
                queue_rsrc = buffer_ops.create_buffer_resource(queue, max_size=True)
                buffer_ops.buffer_store(fx.Int32(0), queue_rsrc, fx.Int32(0))
                if const_expr(emit_active_experts):
                    active_rsrc = buffer_ops.create_buffer_resource(
                        active_expert_storage,
                        max_size=True,
                    )
                    buffer_ops.buffer_store(fx.Int32(0), active_rsrc, fx.Int32(0))
                if const_expr(emit_hot_splits):
                    split_rsrc = buffer_ops.create_buffer_resource(
                        hot_split_storage,
                        max_size=True,
                    )
                    hot_rsrc = buffer_ops.create_buffer_resource(
                        hot_expert_storage,
                        max_size=True,
                    )
                    buffer_ops.buffer_store(fx.Int32(0), split_rsrc, fx.Int32(0))
                    # A negative hot count is a workgroup-local disabled
                    # sentinel.  Count experts which are actually eligible
                    # for splitting rather than all non-empty experts: one
                    # hot expert plus many tiny tails should retain the sparse
                    # split path.  Dense cohorts wait for the shard-wide
                    # crossover before any member is split.
                    frequency_rsrc = buffer_ops.create_buffer_resource(
                        expert_frequency,
                        max_size=True,
                    )
                    cohort_count = fx.Int32(0)
                    max_frequency = fx.Int32(0)
                    eligible_hot_count = fx.Int32(0)
                    required_split_count = fx.Int32(0)
                    # Treat experts within one regular M tile of the split
                    # cutoff as one cohort.  This keeps small route-count
                    # jitter (for example 8193/8191/8190 rows) from enabling
                    # an isolated split launch, while tiny long-tail experts
                    # still do not suppress the sparse hot1/hot2 path.
                    cohort_floor = i32_min_hot_rows - fx.Int32(block_m)
                    cohort_min_rows = (cohort_floor > fx.Int32(0)).select(
                        cohort_floor,
                        fx.Int32(1),
                    )
                    for candidate_expert in range_constexpr(num_experts):
                        candidate_frequency = fx.Int32(
                            buffer_ops.buffer_load(
                                frequency_rsrc,
                                fx.Int32(candidate_expert),
                                vec_width=1,
                                dtype=T.i32,
                            )
                        )
                        in_hot_cohort = candidate_frequency >= cohort_min_rows
                        cohort_count = cohort_count + in_hot_cohort.select(
                            fx.Int32(1),
                            fx.Int32(0),
                        )
                        eligible_hot = candidate_frequency >= i32_min_hot_rows
                        eligible_hot_count = eligible_hot_count + eligible_hot.select(
                            fx.Int32(1), fx.Int32(0)
                        )
                        candidate_partitions = (
                            candidate_frequency + i32_split_rows - fx.Int32(1)
                        ) // i32_split_rows
                        required_split_count = (
                            required_split_count
                            + eligible_hot.select(
                                candidate_partitions,
                                fx.Int32(0),
                            )
                        )
                        max_frequency = (candidate_frequency > max_frequency).select(
                            candidate_frequency, max_frequency
                        )
                    capacities_sufficient = (
                        eligible_hot_count <= i32_hot_expert_capacity
                    ) & (required_split_count <= i32_hot_split_capacity)
                    split_enabled = (
                        (max_frequency >= i32_min_hot_rows)
                        & (
                            (cohort_count <= i32_sparse_split_max_experts)
                            | (max_frequency >= i32_split_activation_rows)
                        )
                        & capacities_sufficient
                    )
                    buffer_ops.buffer_store(
                        split_enabled.select(fx.Int32(0), fx.Int32(-1)),
                        hot_rsrc,
                        fx.Int32(0),
                    )
            gpu.barrier()
        expert = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if expert < fx.Int32(num_experts):
            frequency_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency, max_size=True
            )
            frequency = fx.Int32(
                buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)
            )
            if frequency > fx.Int32(0):
                valid_rsrc = buffer_ops.create_buffer_resource(
                    num_valid_ids, max_size=True
                )
                padded_rows = fx.Int32(
                    buffer_ops.buffer_load(
                        valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                    )
                )
                metadata_blocks = padded_rows // fx.Int32(sorted_block_m)
                expert_ids_rsrc = buffer_ops.create_buffer_resource(
                    sorted_expert_ids,
                    max_size=True,
                )

                lo = fx.Int32(0)
                hi = metadata_blocks
                for _ in range_constexpr(lower_bound_steps):
                    searching = lo < hi
                    mid = (lo + hi) // fx.Int32(2)
                    safe_mid = searching.select(mid, fx.Int32(0))
                    mid_expert = fx.Int32(
                        buffer_ops.buffer_load(
                            expert_ids_rsrc,
                            safe_mid,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    move_right = searching & (mid_expert < expert)
                    lo = move_right.select(mid + fx.Int32(1), lo)
                    move_left = searching & (mid_expert >= expert)
                    hi = move_left.select(mid, hi)

                in_metadata = lo < metadata_blocks
                safe_lo = in_metadata.select(lo, fx.Int32(0))
                found_expert = fx.Int32(
                    buffer_ops.buffer_load(
                        expert_ids_rsrc,
                        safe_lo,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                if in_metadata & (found_expert == expert):
                    if const_expr(emit_active_experts):
                        active_slot = atomic_add(
                            active_expert_storage,
                            fx.Int32(0),
                            fx.Int32(1),
                            dtype_bytes=4,
                        )
                        active_index = fx.Int32(active_slot)
                        if active_index < i32_active_expert_capacity:
                            active_rsrc = buffer_ops.create_buffer_resource(
                                active_expert_storage,
                                max_size=True,
                            )
                            active_offset = fx.Int32(1) + active_index * fx.Int32(2)
                            buffer_ops.buffer_store(expert, active_rsrc, active_offset)
                            buffer_ops.buffer_store(
                                lo * fx.Int32(sorted_block_m),
                                active_rsrc,
                                active_offset + fx.Int32(1),
                            )
                    tile_count = (frequency + fx.Int32(block_m - 1)) // fx.Int32(
                        block_m
                    )
                    reservation = atomic_add(
                        queue,
                        fx.Int32(0),
                        tile_count,
                        dtype_bytes=4,
                    )
                    output_start = fx.Int32(reservation)
                    first_sorted_row = lo * fx.Int32(sorted_block_m)
                    if const_expr(emit_hot_splits):
                        hot_rsrc = buffer_ops.create_buffer_resource(
                            hot_expert_storage,
                            max_size=True,
                        )
                        split_enabled = fx.Int32(
                            buffer_ops.buffer_load(
                                hot_rsrc,
                                fx.Int32(0),
                                vec_width=1,
                                dtype=T.i32,
                            )
                        ) >= fx.Int32(0)
                        if split_enabled & (frequency >= i32_min_hot_rows):
                            partition_count = (
                                frequency + i32_split_rows - fx.Int32(1)
                            ) // i32_split_rows
                            first_partition = fx.Int32(
                                atomic_add(
                                    hot_split_storage,
                                    fx.Int32(0),
                                    partition_count,
                                    dtype_bytes=4,
                                )
                            )
                            hot_slot = fx.Int32(
                                atomic_add(
                                    hot_expert_storage,
                                    fx.Int32(0),
                                    fx.Int32(1),
                                    dtype_bytes=4,
                                )
                            )
                            if hot_slot < i32_hot_expert_capacity:
                                hot_offset = fx.Int32(1) + hot_slot * fx.Int32(3)
                                buffer_ops.buffer_store(expert, hot_rsrc, hot_offset)
                                buffer_ops.buffer_store(
                                    first_partition,
                                    hot_rsrc,
                                    hot_offset + fx.Int32(1),
                                )
                                buffer_ops.buffer_store(
                                    partition_count,
                                    hot_rsrc,
                                    hot_offset + fx.Int32(2),
                                )
                            split_rsrc = buffer_ops.create_buffer_resource(
                                hot_split_storage,
                                max_size=True,
                            )
                            for local_partition in range(0, partition_count, 1):
                                partition = first_partition + fx.Int32(local_partition)
                                if partition < i32_hot_split_capacity:
                                    local_row = (
                                        fx.Int32(local_partition) * i32_split_rows
                                    )
                                    remaining = frequency - local_row
                                    valid_rows = (remaining < i32_split_rows).select(
                                        remaining, i32_split_rows
                                    )
                                    split_offset = fx.Int32(1) + partition * fx.Int32(3)
                                    buffer_ops.buffer_store(
                                        expert,
                                        split_rsrc,
                                        split_offset,
                                    )
                                    buffer_ops.buffer_store(
                                        first_sorted_row + local_row,
                                        split_rsrc,
                                        split_offset + fx.Int32(1),
                                    )
                                    buffer_ops.buffer_store(
                                        valid_rows,
                                        split_rsrc,
                                        split_offset + fx.Int32(2),
                                    )
                    queue_rsrc = buffer_ops.create_buffer_resource(queue, max_size=True)
                    for local_tile in range(0, tile_count, 1):
                        local_row = fx.Int32(local_tile) * fx.Int32(block_m)
                        output_index = output_start + fx.Int32(local_tile)
                        if output_index < i32_queue_capacity:
                            record_offset = fx.Int32(1) + output_index * fx.Int32(3)
                            remaining_rows = frequency - local_row
                            valid_rows = (remaining_rows < fx.Int32(block_m)).select(
                                remaining_rows,
                                fx.Int32(block_m),
                            )
                            buffer_ops.buffer_store(expert, queue_rsrc, record_offset)
                            buffer_ops.buffer_store(
                                first_sorted_row + local_row,
                                queue_rsrc,
                                record_offset + fx.Int32(1),
                            )
                            buffer_ops.buffer_store(
                                valid_rows,
                                queue_rsrc,
                                record_offset + fx.Int32(2),
                            )

        if const_expr(num_experts <= _BLOCK_THREADS):
            # Every reservation belongs to this workgroup.  Clamp the published
            # counts only after all record writes complete so consumers never
            # observe more records than fit in the caller-provided capacities.
            gpu.barrier()
            if gpu.thread_idx.x == fx.Int32(0):
                queue_rsrc = buffer_ops.create_buffer_resource(queue, max_size=True)
                required_records = fx.Int32(
                    buffer_ops.buffer_load(
                        queue_rsrc,
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                stored_records = (required_records < i32_queue_capacity).select(
                    required_records,
                    i32_queue_capacity,
                )
                buffer_ops.buffer_store(stored_records, queue_rsrc, fx.Int32(0))
                if const_expr(emit_active_experts):
                    active_rsrc = buffer_ops.create_buffer_resource(
                        active_expert_storage,
                        max_size=True,
                    )
                    required_active = fx.Int32(
                        buffer_ops.buffer_load(
                            active_rsrc,
                            fx.Int32(0),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    stored_active = (
                        required_active < i32_active_expert_capacity
                    ).select(
                        required_active,
                        i32_active_expert_capacity,
                    )
                    buffer_ops.buffer_store(
                        stored_active,
                        active_rsrc,
                        fx.Int32(0),
                    )
                if const_expr(emit_hot_splits):
                    split_rsrc = buffer_ops.create_buffer_resource(
                        hot_split_storage,
                        max_size=True,
                    )
                    hot_rsrc = buffer_ops.create_buffer_resource(
                        hot_expert_storage,
                        max_size=True,
                    )
                    required_splits = fx.Int32(
                        buffer_ops.buffer_load(
                            split_rsrc,
                            fx.Int32(0),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    required_hot = fx.Int32(
                        buffer_ops.buffer_load(
                            hot_rsrc,
                            fx.Int32(0),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    )
                    nonnegative_hot = (required_hot >= fx.Int32(0)).select(
                        required_hot,
                        fx.Int32(0),
                    )
                    buffer_ops.buffer_store(
                        (required_splits < i32_hot_split_capacity).select(
                            required_splits,
                            i32_hot_split_capacity,
                        ),
                        split_rsrc,
                        fx.Int32(0),
                    )
                    buffer_ops.buffer_store(
                        (nonnegative_hot < i32_hot_expert_capacity).select(
                            nonnegative_hot,
                            i32_hot_expert_capacity,
                        ),
                        hot_rsrc,
                        fx.Int32(0),
                    )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        queue: fx.Tensor,
        i32_queue_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
        hot_split_storage: fx.Tensor,
        i32_hot_split_capacity: fx.Int32,
        hot_expert_storage: fx.Tensor,
        i32_hot_expert_capacity: fx.Int32,
        i32_split_rows: fx.Int32,
        i32_min_hot_rows: fx.Int32,
        i32_split_activation_rows: fx.Int32,
        i32_sparse_split_max_experts: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        if num_experts > _BLOCK_THREADS:
            clear_queue_kernel(queue, active_expert_storage).launch(
                grid=(1, 1, 1),
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )
        build_queue_kernel(
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            queue,
            i32_queue_capacity,
            active_expert_storage,
            i32_active_expert_capacity,
            hot_split_storage,
            i32_hot_split_capacity,
            hot_expert_storage,
            i32_hot_expert_capacity,
            i32_split_rows,
            i32_min_hot_rows,
            i32_split_activation_rows,
            i32_sparse_split_max_experts,
        ).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        if num_experts > _BLOCK_THREADS:
            # Multiple workgroups cannot finalize a shared count in-kernel.
            # A same-stream one-thread dispatch publishes the saturated count
            # after every builder workgroup and record write has completed.
            clamp_queue_counts_kernel(
                queue,
                i32_queue_capacity,
                active_expert_storage,
                i32_active_expert_capacity,
            ).launch(
                grid=(1, 1, 1),
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch


def build_exact_m_tile_queue(
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    queue: torch.Tensor,
    *,
    block_m: int,
    sorted_block_m: int,
    queue_capacity: int | None = None,
    active_expert_storage: torch.Tensor | None = None,
    active_expert_capacity: int | None = None,
    hot_split_storage: torch.Tensor | None = None,
    hot_expert_storage: torch.Tensor | None = None,
    split_rows: int | None = None,
    min_hot_rows: int | None = None,
    split_activation_rows: int | None = None,
    sparse_split_max_experts: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Asynchronously build a self-contained exact M-tile work queue.

    ``queue`` is contiguous int32 storage with room for one counter plus three
    integers per record.  A short ``queue_capacity`` truncates record writes
    and the published counter to the number of stored records.  Optional
    ``active_expert_storage`` uses the existing
    ``[count, (expert, first_row) * capacity]`` ABI and is populated by the
    same device scan.  Optional ``hot_split_storage`` and
    ``hot_expert_storage`` emit split-K metadata in that same scan.
    """

    tensors = {
        "expert_frequency": expert_frequency,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "queue": queue,
    }
    device = expert_frequency.device
    if device.type != "cuda":
        raise ValueError("exact M-tile queue construction requires a ROCm device")
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if expert_frequency.ndim != 1 or expert_frequency.numel() <= 0:
        raise ValueError("expert_frequency must be a non-empty rank-1 tensor")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be rank 1")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain at least one element")
    if queue.ndim != 1 or queue.numel() < 1 or (queue.numel() - 1) % 3:
        raise ValueError(
            "queue must use the counter-first "
            "[count, (expert, first_row, valid_rows) * capacity] ABI"
        )
    if active_expert_storage is not None:
        if active_expert_storage.device != device:
            raise ValueError(
                f"active_expert_storage must be on {device}, got {active_expert_storage.device}"
            )
        if active_expert_storage.dtype != torch.int32:
            raise ValueError(
                "active_expert_storage must have dtype torch.int32, "
                f"got {active_expert_storage.dtype}"
            )
        if not active_expert_storage.is_contiguous():
            raise ValueError("active_expert_storage must be contiguous")
        if (
            active_expert_storage.ndim != 1
            or active_expert_storage.numel() < 1
            or (active_expert_storage.numel() - 1) % 2
        ):
            raise ValueError(
                "active_expert_storage must use the counter-first "
                "[count, (expert, first_row) * capacity] ABI"
            )
    elif active_expert_capacity is not None:
        raise ValueError("active_expert_capacity requires active_expert_storage")

    emit_hot_splits = hot_split_storage is not None or hot_expert_storage is not None
    if emit_hot_splits:
        if hot_split_storage is None or hot_expert_storage is None:
            raise ValueError(
                "hot split and hot expert storage must be supplied together"
            )
        if active_expert_storage is None:
            raise ValueError("hot split emission requires active_expert_storage")
        if split_rows is None or min_hot_rows is None:
            raise ValueError("hot split emission requires split_rows and min_hot_rows")
        if split_rows <= 0 or min_hot_rows < split_rows:
            raise ValueError(
                "hot split emission requires 0 < split_rows <= min_hot_rows"
            )
        split_activation_rows = (
            min_hot_rows if split_activation_rows is None else split_activation_rows
        )
        sparse_split_max_experts = (
            int(expert_frequency.numel())
            if sparse_split_max_experts is None
            else sparse_split_max_experts
        )
        if (
            not isinstance(split_activation_rows, int)
            or isinstance(split_activation_rows, bool)
            or split_activation_rows < min_hot_rows
        ):
            raise ValueError(
                "split_activation_rows must be an int at least min_hot_rows"
            )
        if (
            not isinstance(sparse_split_max_experts, int)
            or isinstance(sparse_split_max_experts, bool)
            or not 0 <= sparse_split_max_experts <= int(expert_frequency.numel())
        ):
            raise ValueError("sparse_split_max_experts must be in [0, num_experts]")
        for name, tensor, record_width in (
            ("hot_split_storage", hot_split_storage, 3),
            ("hot_expert_storage", hot_expert_storage, 3),
        ):
            if tensor.device != device or tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be int32 on {device}")
            if (
                not tensor.is_contiguous()
                or tensor.ndim != 1
                or tensor.numel() < 1
                or (tensor.numel() - 1) % record_width
            ):
                raise ValueError(f"{name} has an invalid counter-first ABI")
        hot_split_capacity = (hot_split_storage.numel() - 1) // 3
        hot_expert_capacity = (hot_expert_storage.numel() - 1) // 3
        hot_split_arg = hot_split_storage
        hot_expert_arg = hot_expert_storage
    else:
        if (
            split_rows is not None
            or min_hot_rows is not None
            or split_activation_rows is not None
            or sparse_split_max_experts is not None
        ):
            raise ValueError("split thresholds require hot split storage")
        hot_split_capacity = 0
        hot_expert_capacity = 0
        hot_split_arg = queue
        hot_expert_arg = queue
        split_rows = 0
        min_hot_rows = 0
        split_activation_rows = 0
        sparse_split_max_experts = 0
    if not isinstance(block_m, int) or isinstance(block_m, bool) or block_m <= 0:
        raise ValueError(f"block_m must be a positive int, got {block_m}")
    if (
        not isinstance(sorted_block_m, int)
        or isinstance(sorted_block_m, bool)
        or sorted_block_m <= 0
    ):
        raise ValueError(f"sorted_block_m must be a positive int, got {sorted_block_m}")

    storage_capacity = (queue.numel() - 1) // 3
    capacity = storage_capacity if queue_capacity is None else queue_capacity
    if not isinstance(capacity, int) or isinstance(capacity, bool):
        raise TypeError(
            f"queue_capacity must be an int or None, got {type(capacity).__name__}"
        )
    if capacity < 0 or capacity > storage_capacity or capacity > _MAX_SIGNED_I32:
        raise ValueError(
            "queue_capacity must be in "
            f"[0, {min(storage_capacity, _MAX_SIGNED_I32)}], got {capacity}"
        )

    if active_expert_storage is None:
        active_capacity = 0
        active_storage_arg = queue
    else:
        active_storage_capacity = (active_expert_storage.numel() - 1) // 2
        active_capacity = (
            active_storage_capacity
            if active_expert_capacity is None
            else active_expert_capacity
        )
        if not isinstance(active_capacity, int) or isinstance(active_capacity, bool):
            raise TypeError(
                "active_expert_capacity must be an int or None, "
                f"got {type(active_capacity).__name__}"
            )
        if (
            active_capacity < 0
            or active_capacity > active_storage_capacity
            or active_capacity > _MAX_SIGNED_I32
        ):
            raise ValueError(
                "active_expert_capacity must be in "
                f"[0, {min(active_storage_capacity, _MAX_SIGNED_I32)}], "
                f"got {active_capacity}"
            )
        active_storage_arg = active_expert_storage

    if stream is None:
        stream = torch.cuda.current_stream(device)
    launcher = compile_exact_m_tile_queue_builder(
        int(expert_frequency.numel()),
        block_m,
        sorted_block_m,
        device.index or 0,
        active_expert_storage is not None,
        emit_hot_splits,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        queue,
        capacity,
        active_storage_arg,
        active_capacity,
        hot_split_arg,
        hot_split_capacity,
        hot_expert_arg,
        hot_expert_capacity,
        split_rows,
        min_hot_rows,
        split_activation_rows,
        sparse_split_max_experts,
        stream,
    )
    return queue


@functools.lru_cache(maxsize=128)
def compile_compact_m_tile_descriptor_builder(
    num_experts: int,
    block_m: int,
    sorted_block_m: int,
    device_index: int,
    emit_active_experts: bool = False,
):
    """Compile the counter-clear plus descriptor-build launch sequence.

    When ``emit_active_experts`` is true, the same expert scan also writes a
    shared ``[count, (expert, first_sorted_row) * capacity]`` queue.  This lets
    output-stationary grouped weight-gradient kernels reuse the W1/dX compact
    scheduler without paying for a second active-expert builder.
    """

    del device_index
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if block_m <= 0:
        raise ValueError(f"block_m must be positive, got {block_m}")
    if sorted_block_m <= 0:
        raise ValueError(f"sorted_block_m must be positive, got {sorted_block_m}")
    if sorted_block_m % block_m:
        raise ValueError(
            f"sorted_block_m ({sorted_block_m}) must be divisible by block_m ({block_m})"
        )

    subtiles_per_sorted_block = sorted_block_m // block_m
    max_metadata_blocks = _MAX_SIGNED_I32 // sorted_block_m
    lower_bound_steps = max(1, max_metadata_blocks.bit_length())

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_counter_kernel(
        total_tiles: fx.Tensor,
        active_expert_storage: fx.Tensor,
    ):
        if gpu.thread_idx.x == fx.Int32(0):
            total_rsrc = buffer_ops.create_buffer_resource(total_tiles, max_size=True)
            buffer_ops.buffer_store(fx.Int32(0), total_rsrc, fx.Int32(0))
            if const_expr(emit_active_experts):
                active_rsrc = buffer_ops.create_buffer_resource(
                    active_expert_storage,
                    max_size=True,
                )
                buffer_ops.buffer_store(fx.Int32(0), active_rsrc, fx.Int32(0))

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def build_descriptor_kernel(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        descriptors: fx.Tensor,
        total_tiles: fx.Tensor,
        i32_descriptor_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
    ):
        expert = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if expert < fx.Int32(num_experts):
            frequency_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency, max_size=True
            )
            frequency = fx.Int32(
                buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)
            )
            if frequency > fx.Int32(0):
                valid_rsrc = buffer_ops.create_buffer_resource(
                    num_valid_ids, max_size=True
                )
                padded_rows = fx.Int32(
                    buffer_ops.buffer_load(
                        valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                    )
                )
                metadata_blocks = padded_rows // fx.Int32(sorted_block_m)
                expert_ids_rsrc = buffer_ops.create_buffer_resource(
                    sorted_expert_ids, max_size=True
                )

                lo = fx.Int32(0)
                hi = metadata_blocks
                for _ in range_constexpr(lower_bound_steps):
                    searching = lo < hi
                    mid = (lo + hi) // fx.Int32(2)
                    safe_mid = searching.select(mid, fx.Int32(0))
                    mid_expert = fx.Int32(
                        buffer_ops.buffer_load(
                            expert_ids_rsrc, safe_mid, vec_width=1, dtype=T.i32
                        )
                    )
                    move_right = searching & (mid_expert < expert)
                    lo = move_right.select(mid + fx.Int32(1), lo)
                    move_left = searching & (mid_expert >= expert)
                    hi = move_left.select(mid, hi)

                in_metadata = lo < metadata_blocks
                safe_lo = in_metadata.select(lo, fx.Int32(0))
                found_expert = fx.Int32(
                    buffer_ops.buffer_load(
                        expert_ids_rsrc, safe_lo, vec_width=1, dtype=T.i32
                    )
                )
                if in_metadata & (found_expert == expert):
                    if const_expr(emit_active_experts):
                        active_slot = atomic_add(
                            active_expert_storage,
                            fx.Int32(0),
                            fx.Int32(1),
                            dtype_bytes=4,
                        )
                        active_index = fx.Int32(active_slot)
                        if active_index < i32_active_expert_capacity:
                            active_rsrc = buffer_ops.create_buffer_resource(
                                active_expert_storage,
                                max_size=True,
                            )
                            active_offset = fx.Int32(1) + active_index * fx.Int32(2)
                            buffer_ops.buffer_store(expert, active_rsrc, active_offset)
                            buffer_ops.buffer_store(
                                lo * fx.Int32(sorted_block_m),
                                active_rsrc,
                                active_offset + fx.Int32(1),
                            )
                    tile_count = (frequency + fx.Int32(block_m - 1)) // fx.Int32(
                        block_m
                    )
                    reservation = atomic_add(
                        total_tiles,
                        fx.Int32(0),
                        tile_count,
                        dtype_bytes=4,
                    )
                    output_start = fx.Int32(reservation)
                    first_m_block = lo * fx.Int32(subtiles_per_sorted_block)
                    descriptors_rsrc = buffer_ops.create_buffer_resource(
                        descriptors, max_size=True
                    )
                    for local_tile in range(0, tile_count, 1):
                        output_index = output_start + fx.Int32(local_tile)
                        if output_index < i32_descriptor_capacity:
                            buffer_ops.buffer_store(
                                first_m_block + fx.Int32(local_tile),
                                descriptors_rsrc,
                                output_index,
                            )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        descriptors: fx.Tensor,
        total_tiles: fx.Tensor,
        i32_descriptor_capacity: fx.Int32,
        active_expert_storage: fx.Tensor,
        i32_active_expert_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_counter_kernel(total_tiles, active_expert_storage).launch(
            grid=(1, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        build_descriptor_kernel(
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            descriptors,
            total_tiles,
            i32_descriptor_capacity,
            active_expert_storage,
            i32_active_expert_capacity,
        ).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def build_compact_m_tile_descriptors(
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    descriptors: torch.Tensor,
    total_tiles: torch.Tensor,
    *,
    block_m: int,
    sorted_block_m: int,
    descriptor_capacity: int | None = None,
    active_expert_storage: torch.Tensor | None = None,
    active_expert_capacity: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clear counters and asynchronously build compact M descriptors.

    Supplying ``active_expert_storage`` additionally emits the counter-first
    active-expert queue used by grouped dW1/dW2, within the same clear/build
    kernel pair.  The optional storage ABI is
    ``[count, expert0, first_row0, expert1, first_row1, ...]``.
    """

    tensors = {
        "expert_frequency": expert_frequency,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
        "descriptors": descriptors,
        "total_tiles": total_tiles,
    }
    device = expert_frequency.device
    if device.type != "cuda":
        raise ValueError(
            "compact M-tile descriptor construction requires a ROCm device"
        )
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if expert_frequency.ndim != 1 or expert_frequency.numel() <= 0:
        raise ValueError("expert_frequency must be a non-empty rank-1 tensor")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be rank 1")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain at least one element")
    if descriptors.ndim != 1:
        raise ValueError("descriptors must be rank 1")
    if total_tiles.ndim != 1 or total_tiles.numel() < 1:
        raise ValueError("total_tiles must contain at least one element")
    if active_expert_storage is not None:
        if active_expert_storage.device != device:
            raise ValueError(
                f"active_expert_storage must be on {device}, got {active_expert_storage.device}"
            )
        if active_expert_storage.dtype != torch.int32:
            raise ValueError(
                "active_expert_storage must have dtype torch.int32, "
                f"got {active_expert_storage.dtype}"
            )
        if not active_expert_storage.is_contiguous():
            raise ValueError("active_expert_storage must be contiguous")
        if (
            active_expert_storage.ndim != 1
            or active_expert_storage.numel() < 1
            or (active_expert_storage.numel() - 1) % 2
        ):
            raise ValueError(
                "active_expert_storage must use the counter-first "
                "[count, (expert, first_row) * capacity] ABI"
            )
    elif active_expert_capacity is not None:
        raise ValueError("active_expert_capacity requires active_expert_storage")
    if block_m <= 0 or sorted_block_m <= 0 or sorted_block_m % block_m:
        raise ValueError(
            f"expected positive block_m dividing sorted_block_m, got block_m={block_m}, "
            f"sorted_block_m={sorted_block_m}"
        )

    capacity = (
        descriptors.numel() if descriptor_capacity is None else descriptor_capacity
    )
    if not isinstance(capacity, int):
        raise TypeError(
            f"descriptor_capacity must be an int or None, got {type(capacity).__name__}"
        )
    if capacity < 0 or capacity > descriptors.numel() or capacity > _MAX_SIGNED_I32:
        raise ValueError(
            f"descriptor_capacity must be in [0, {min(descriptors.numel(), _MAX_SIGNED_I32)}], got {capacity}"
        )
    if active_expert_storage is None:
        active_capacity = 0
        active_storage_arg = total_tiles
    else:
        storage_capacity = (active_expert_storage.numel() - 1) // 2
        active_capacity = (
            storage_capacity
            if active_expert_capacity is None
            else active_expert_capacity
        )
        if not isinstance(active_capacity, int):
            raise TypeError(
                "active_expert_capacity must be an int or None, "
                f"got {type(active_capacity).__name__}"
            )
        if (
            active_capacity < 0
            or active_capacity > storage_capacity
            or active_capacity > _MAX_SIGNED_I32
        ):
            raise ValueError(
                "active_expert_capacity must be in "
                f"[0, {min(storage_capacity, _MAX_SIGNED_I32)}], got {active_capacity}"
            )
        active_storage_arg = active_expert_storage
    if stream is None:
        stream = torch.cuda.current_stream(device)
    device_index = device.index or 0
    launcher = compile_compact_m_tile_descriptor_builder(
        int(expert_frequency.numel()),
        block_m,
        sorted_block_m,
        device_index,
        active_expert_storage is not None,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        descriptors,
        total_tiles,
        capacity,
        active_storage_arg,
        active_capacity,
        stream,
    )
    return descriptors, total_tiles


__all__ = [
    "build_compact_m_tile_descriptors",
    "build_exact_m_tile_queue",
    "compact_m_tile_descriptor_upper_bound",
    "compile_compact_m_tile_descriptor_builder",
    "compile_exact_m_tile_queue_builder",
    "exact_m_tile_queue_elements",
    "exact_m_tile_queue_upper_bound",
    "fixed_compact_m_tile_descriptor_upper_bound",
    "ragged_compact_m_tile_descriptor_upper_bound",
]
