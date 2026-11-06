# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""GPU work planning for FlyDSL paged attention with mixed context lengths."""

from dataclasses import dataclass
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .tensor_shim import _run_compiled, ptr_arg, ptr_buf_tensor

DEFAULT_MAX_BATCH_SIZE = 4096
# int32 slots per arrival counter: one 128-byte line per (sequence, KV head).
ARRIVAL_COUNTER_STRIDE = 32
# Two warp scans and one reduction use at most 512 bytes at 1024 wave32 threads.
_PLAN_SCAN_LDS_BYTES = 512


def _lds_max_batch(lds_bytes):
    """Largest batch whose task ends and scan storage fit ``lds_bytes``."""
    return (lds_bytes - _PLAN_SCAN_LDS_BYTES) // 16 * 4


def _block_threads(capacity):
    return max(64, min(1024, 1 << (capacity - 1).bit_length()))


@cache
def compile_pa_decode_plan(
    max_batch,
    threads,
    windowed,
    cluster_tasks=False,
    tile_tokens=256,
    single_partition=False,
):
    """Compile a batch bucket and block size for dense or windowed plans.

    Bucket 1 uses a scan-free kernel: its task count is simply the minimum
    of visible tiles, capacity and partition cap. Other buckets retain scans.
    Actual batch, task capacity, partition cap and window are runtime values.
    ``single_partition`` specializes a validated cap of one: a single
    count scan compacts live rows and no tile-budget allocation is needed.
    ``cluster_tasks`` adds aligned records for small gfx1250 cluster plans.
    """
    if single_partition and cluster_tasks:
        raise ValueError("single-partition plans cannot have cluster records")
    if max_batch == 1:
        return _compile_single_sequence_plan(
            threads, windowed, cluster_tasks, tile_tokens
        )

    if single_partition:
        return _compile_single_partition_plan(max_batch, threads, windowed, tile_tokens)

    scan_tiles = fx.coop.BlockScan[fx.Int64, threads]
    scan_counts = fx.coop.BlockScan[fx.Int32, threads]
    reduce_rows = fx.coop.BlockReduce[fx.Int32, threads]

    @fx.struct
    class TaskStorage:
        ends: fx.Array[fx.Int32, max_batch, 16]

    def visible_tiles(ctx, window):
        """Return the first visible tile and the visible tile count."""
        length = fx.max(fx.Int64(ctx), fx.Int64(0))
        first = fx.Int64(0)
        if windowed:
            first = fx.max(length - fx.Int64(window), fx.Int64(0)) // tile_tokens
        return first, (length + tile_tokens - 1) // tile_tokens - first

    def split(tiles, lower, upper, max_parts):
        """Return a row's task count from its cumulative slot bounds."""
        # Tiles of an int32 length fit Int32, so clamp the partition cap there.
        return fx.min(
            fx.Int32(fx.min(fx.Int64(tiles > 0) + upper - lower, tiles)), max_parts
        )

    # Lookup reaches rows below max_batch, so it needs one fewer level than
    # max_batch.bit_length() at power-of-two buckets.
    lookup_bits = (max_batch - 1).bit_length()

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def pa_decode_plan_kernel(
        lengths_ptr: fx.Pointer,
        work_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        window: fx.Int32,
        cluster_ptr: fx.Pointer,
    ):
        tid = fx.thread_idx.x
        lengths = ptr_buf_tensor(lengths_ptr)
        work = ptr_buf_tensor(work_ptr)
        reduce_info = ptr_buf_tensor(reduce_ptr)
        smem = fx.SharedAllocator()
        tile_storage = smem.allocate(scan_tiles.SharedStorage).peek()
        count_storage = smem.allocate(scan_counts.SharedStorage).peek()
        row_storage = smem.allocate(reduce_rows.SharedStorage).peek()
        task_ends = smem.allocate(TaskStorage).peek().ends
        # Each thread owns consecutive rows, so scans of per-thread totals
        # preserve sequence order. Buckets that fit the block give each thread
        # at most one row; larger ones walk rows with runtime bounds, so padding
        # rows cost nothing and register use does not grow with the bucket.
        single = max_batch <= threads
        if fx.const_expr(single):
            row = fx.Int32(tid)
            row_ctx = fx.Int32(0)
            if row < batch:
                row_ctx = lengths[row]
            _, thread_tiles = visible_tiles(row_ctx, window)
            thread_rows = fx.Int32(thread_tiles > 0)
        else:
            rows = (batch + threads - 1) // threads
            row_begin = fx.min(fx.Int32(tid) * rows, batch)
            row_end = fx.min(row_begin + rows, batch)
            thread_tiles = fx.Int64(0)
            thread_rows = fx.Int32(0)
            for seq in range(row_begin, row_end):
                _, tiles = visible_tiles(lengths[seq], window)
                thread_tiles = thread_tiles + tiles
                thread_rows = thread_rows + fx.Int32(tiles > 0)

        cumulative, total = scan_tiles.exclusive_with_aggregate(
            thread_tiles, fx.ReductionOp.ADD, storage=tile_storage
        )
        total = fx.max(total, fx.Int64(1))
        remaining = fx.Int64(
            capacity - reduce_rows(thread_rows, fx.ReductionOp.ADD, storage=row_storage)
        )
        if fx.const_expr(single):
            # Empty and padding rows get no tasks; skip their Int64 divisions.
            thread_tasks = fx.Int32(0)
            if thread_tiles > 0:
                thread_tasks = split(
                    thread_tiles,
                    cumulative * remaining // total,
                    (cumulative + thread_tiles) * remaining // total,
                    max_parts,
                )
        else:
            # Each row's lower bound is the previous row's upper bound.
            lower = cumulative * remaining // total
            thread_tasks = fx.Int32(0)
            for seq in range(row_begin, row_end):
                _, tiles = visible_tiles(lengths[seq], window)
                cumulative = cumulative + tiles
                upper = cumulative * remaining // total
                count = split(tiles, lower, upper, max_parts)
                lower = upper
                task_ends[seq] = count
                thread_tasks = thread_tasks + count
        row_start, total_tasks = scan_counts.exclusive_with_aggregate(
            thread_tasks, fx.ReductionOp.ADD, storage=count_storage
        )

        if fx.const_expr(single):
            if row < batch:
                reduce_info[row * 2] = row_start
                reduce_info[row * 2 + 1] = thread_tasks
                task_ends[row] = row_start + thread_tasks
        else:
            # Rows are thread-private here; convert counts to ends in place.
            for seq in range(row_begin, row_end):
                count = task_ends[seq]
                reduce_info[seq * 2] = row_start
                reduce_info[seq * 2 + 1] = count
                row_start = row_start + count
                task_ends[seq] = row_start
        fx.gpu.barrier()

        if fx.const_expr(cluster_tasks):
            # One aligned record per cluster CTA avoids a dependent lookup of
            # reduce_info followed by work_info in the attention kernel.
            records = fx.recast_iter(fx.Int32, cluster_ptr)
            cluster_parts = fx.min(max_parts, fx.Int32(8))
            for index in range(tid, batch * cluster_parts, threads):
                seq, part = index // cluster_parts, index % cluster_parts
                start = fx.Int32(0)
                if seq > 0:
                    start = task_ends[seq - 1]
                count = task_ends[seq] - start
                if part >= count:
                    record = fx.Vector.from_elements(
                        [
                            seq,
                            fx.Int32(0),
                            fx.Int32(0),
                            lengths[seq],
                            start,
                            count,
                            fx.Int32(0),
                            fx.Int32(0),
                        ],
                        dtype=fx.Int32,
                    )
                    fx.ptr_store(record, fx.add_offset(records, index * 8))

        # Spread packed task writes across all threads, including small batches
        # with many partitions. Upper-bound search skips repeated ends from empty
        # rows and uses one LDS int32 per sequence (16 KiB at the default cap).
        for slot in range(tid, capacity, threads):
            if slot < total_tasks:
                seq = fx.Int32(-1)
                for bit in fx.range_constexpr(lookup_bits - 1, -1, -1):
                    candidate = seq + (1 << bit)
                    if candidate < batch:  # noqa: SIM102 -- guard the LDS read
                        if task_ends[candidate] <= slot:
                            seq = candidate
                seq = seq + 1
                start = fx.Int32(0)
                if seq > 0:
                    start = task_ends[seq - 1]
                count = task_ends[seq] - start
                ctx = lengths[seq]
                first, num_tiles = visible_tiles(ctx, window)
                part = fx.Int64(slot - start)
                begin = fx.Int32(first + part * num_tiles // count)
                end = fx.Int32(first + (part + 1) * num_tiles // count)
                work[slot * 4] = seq
                work[slot * 4 + 1] = begin
                work[slot * 4 + 2] = end
                work[slot * 4 + 3] = ctx
                if fx.const_expr(cluster_tasks):  # noqa: SIM102 -- constexpr scope
                    # Reuse the packed task's split arithmetic for active CTAs;
                    # the loop above only initializes idle records.
                    if part < cluster_parts:
                        record = fx.Vector.from_elements(
                            [
                                seq,
                                begin,
                                end,
                                ctx,
                                start,
                                count,
                                fx.Int32(0),
                                fx.Int32(0),
                            ],
                            dtype=fx.Int32,
                        )
                        fx.ptr_store(
                            record,
                            fx.add_offset(
                                records, (fx.Int64(seq) * cluster_parts + part) * 8
                            ),
                        )
            else:
                # Padding is disjoint from active records, even for empty batches.
                for field in fx.range_constexpr(4):
                    work[slot * 4 + field] = fx.Int32(0)

    @flyc.jit
    def launch(
        lengths: fx.Pointer,
        work: fx.Pointer,
        reduce_info: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        window: fx.Int32,
        cluster_info: fx.Pointer,
        stream: fx.Stream,
    ):
        pa_decode_plan_kernel(
            lengths, work, reduce_info, batch, capacity, max_parts, window, cluster_info
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


@cache
def _compile_single_sequence_plan(threads, windowed, cluster_tasks, tile_tokens):
    """One sequence has no scan or proportional budget allocation."""

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def pa_decode_single_sequence_plan_kernel(
        lengths_ptr: fx.Pointer,
        work_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        window: fx.Int32,
        cluster_ptr: fx.Pointer,
    ):
        tid = fx.thread_idx.x
        original = fx.Int32(fx.recast_iter(fx.Int32, lengths_ptr)[0])
        length = fx.max(fx.Int64(original), fx.Int64(0))
        first = fx.Int64(0)
        if fx.const_expr(windowed):
            first = fx.max(length - fx.Int64(window), fx.Int64(0)) // tile_tokens
        tiles = (length + tile_tokens - 1) // tile_tokens - first
        count = fx.Int32(fx.min(tiles, fx.Int64(fx.min(capacity, max_parts))))
        zero = fx.Int32(0)
        if tid == 0:
            fx.ptr_store(
                fx.Vector.from_elements([zero, count], dtype=fx.Int32),
                fx.recast_iter(fx.Int32, reduce_ptr),
            )
        work = fx.recast_iter(fx.Int32, work_ptr)
        for slot in range(tid, capacity, threads):
            record = fx.Vector.filled(4, 0, fx.Int32)
            if slot < count:
                begin = fx.Int32(first + fx.Int64(slot) * tiles // count)
                end = fx.Int32(first + fx.Int64(slot + 1) * tiles // count)
                record = fx.Vector.from_elements(
                    [zero, begin, end, original], dtype=fx.Int32
                )
            fx.ptr_store(record, fx.add_offset(work, slot * 4))
        if fx.const_expr(cluster_tasks):
            stride = fx.min(max_parts, fx.Int32(8))
            if tid < stride:
                begin, end = zero, zero
                if tid < count:
                    begin = fx.Int32(first + fx.Int64(tid) * tiles // count)
                    end = fx.Int32(first + fx.Int64(tid + 1) * tiles // count)
                record = fx.Vector.from_elements(
                    [zero, begin, end, original, zero, count, zero, zero],
                    dtype=fx.Int32,
                )
                fx.ptr_store(
                    record,
                    fx.add_offset(fx.recast_iter(fx.Int32, cluster_ptr), tid * 8),
                )

    @flyc.jit
    def launch(
        lengths: fx.Pointer,
        work: fx.Pointer,
        reduce_info: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        window: fx.Int32,
        cluster_info: fx.Pointer,
        stream: fx.Stream,
    ):
        pa_decode_single_sequence_plan_kernel(
            lengths, work, reduce_info, capacity, max_parts, window, cluster_info
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


@cache
def _compile_single_partition_plan(max_batch, threads, windowed, tile_tokens):
    """Compact live sequences with one count scan; no tile-budget division."""
    scan = fx.coop.BlockScan[fx.Int32, threads]

    def bounds(original, window):
        length = fx.max(fx.Int64(original), fx.Int64(0))
        first = fx.Int64(0)
        if windowed:
            first = fx.max(length - fx.Int64(window), fx.Int64(0)) // tile_tokens
        end = (length + tile_tokens - 1) // tile_tokens
        return fx.Int32(first), fx.Int32(end)

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def pa_decode_single_partition_plan_kernel(
        lengths_ptr: fx.Pointer,
        work_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        window: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lengths = fx.recast_iter(fx.Int32, lengths_ptr)
        work = fx.recast_iter(fx.Int32, work_ptr)
        reduce_info = fx.recast_iter(fx.Int32, reduce_ptr)
        storage = fx.SharedAllocator().allocate(scan.SharedStorage).peek()
        single = max_batch <= threads
        if fx.const_expr(single):
            row = fx.Int32(tid)
            original = fx.Int32(0)
            if row < batch:
                original = fx.Int32(lengths[row])
            first, end = bounds(original, window)
            tasks = fx.Int32(end > first)
        else:
            rows = (batch + threads - 1) // threads
            begin = fx.min(fx.Int32(tid) * rows, batch)
            limit = fx.min(begin + rows, batch)
            tasks = fx.Int32(0)
            for seq in range(begin, limit):
                first, end = bounds(lengths[seq], window)
                tasks = tasks + fx.Int32(end > first)
        start, total = scan.exclusive_with_aggregate(
            tasks, fx.ReductionOp.ADD, storage=storage
        )
        if fx.const_expr(single):
            if row < batch:
                fx.ptr_store(
                    fx.Vector.from_elements([start, tasks], dtype=fx.Int32),
                    fx.add_offset(reduce_info, row * 2),
                )
                if tasks > 0:
                    fx.ptr_store(
                        fx.Vector.from_elements(
                            [row, first, end, original], dtype=fx.Int32
                        ),
                        fx.add_offset(work, start * 4),
                    )
        else:
            for seq in range(begin, limit):
                original = fx.Int32(lengths[seq])
                first, end = bounds(original, window)
                count = fx.Int32(end > first)
                fx.ptr_store(
                    fx.Vector.from_elements([start, count], dtype=fx.Int32),
                    fx.add_offset(reduce_info, seq * 2),
                )
                if count > 0:
                    fx.ptr_store(
                        fx.Vector.from_elements(
                            [seq, first, end, original], dtype=fx.Int32
                        ),
                        fx.add_offset(work, start * 4),
                    )
                start = start + count
        # Padding starts beyond every live store, including all-empty refreshes.
        for slot in range(total + tid, capacity, threads):
            fx.ptr_store(
                fx.Vector.filled(4, 0, fx.Int32), fx.add_offset(work, slot * 4)
            )

    @flyc.jit
    def launch(
        lengths: fx.Pointer,
        work: fx.Pointer,
        reduce_info: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        window: fx.Int32,
        cluster_info: fx.Pointer,
        stream: fx.Stream,
    ):
        pa_decode_single_partition_plan_kernel(
            lengths, work, reduce_info, batch, capacity, window
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


@dataclass(frozen=True)
class PADecodePlan:
    """Reusable GPU metadata; refresh it whenever context lengths change.

    ``work_info`` is [capacity, 4]: sequence, begin/end absolute 256-token tile,
    original context length. Tile ranges use ``tile_tokens`` units, are half-open
    and cover the MTP union.
    ``reduce_info`` is [batch, 2]: first packed task, actual task count.
    ``cluster_info``, when present, is [batch, min(max_partitions, 8), 8] with the
    first four work fields followed by first task, task count and two padding
    words. Idle CTAs have zero bounds. Refresh updates every record in place.
    Wider plans keep only their first eight records per sequence, carrying
    the actual task count; clusters wider than this stride use work_info.
    ``arrival_counters`` is [batch, num_kv_heads, ARRIVAL_COUNTER_STRIDE]
    int32 and zero between decodes: gfx1250 decodes that merge partitions in
    the tile kernel count finished tasks in slot 0 and the last one resets it,
    so decodes running concurrently must not share a plan. ``None`` keeps the
    shared reducer.
    """

    work_info: torch.Tensor
    reduce_info: torch.Tensor
    num_kv_heads: int
    max_partitions: int
    sliding_window: int = 0
    query_length: int = 1
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE
    arrival_counters: torch.Tensor | None = None
    # Optional [batch, min(max_partitions, 8), 8] records for resident clusters.
    # Fields: sequence, begin/end tile, context length, first task, count, 0, 0.
    cluster_info: torch.Tensor | None = None
    tile_tokens: int = 256

    @property
    def capacity(self) -> int:
        return self.work_info.shape[0]

    @property
    def cluster_task_stride(self) -> int:
        return 0 if self.cluster_info is None else self.cluster_info.shape[1]

    def validate(self, batch_size: int, num_kv_heads: int, device: torch.device):
        if type(self.max_batch_size) is not int or self.max_batch_size < 1:
            raise ValueError("plan max_batch_size must be a positive int")
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError(f"plan supports batches in [1, {self.max_batch_size}]")
        if self.num_kv_heads != num_kv_heads:
            raise ValueError("plan KV head count does not match the cache")
        num_compute_units = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
        if not 1 <= self.max_partitions <= num_compute_units:
            raise ValueError(f"plan max_partitions must be in [1, {num_compute_units}]")
        if self.sliding_window < 0 or self.query_length < 1:
            raise ValueError("invalid plan sliding_window or query_length")
        if self.reduce_info.shape != (batch_size, 2):
            raise ValueError("reduce_info must have shape [batch_size, 2]")
        if self.work_info.ndim != 2 or self.work_info.shape[1] != 4:
            raise ValueError("work_info must have shape [capacity, 4]")
        if not batch_size <= self.capacity <= batch_size * self.max_partitions:
            raise ValueError("plan capacity must be in [batch, batch * max_partitions]")
        if type(self.tile_tokens) is not int or self.tile_tokens not in (128, 256):
            raise ValueError("plan tile_tokens must be 128 or 256")
        if self.tile_tokens == 128 and torch.cuda.get_device_properties(
            device
        ).gcnArchName.split(":", 1)[0] not in ("gfx1250", "gfx1250-strict"):
            raise ValueError("128-token plans require gfx1250")
        metadata = [self.work_info, self.reduce_info]
        if self.cluster_info is not None:
            if self.cluster_info.shape != (batch_size, min(self.max_partitions, 8), 8):
                raise ValueError(
                    "cluster_info must have shape [batch, min(max_partitions, 8), 8]"
                )
            if self.max_partitions < 2:
                raise ValueError("cluster_info requires max_partitions >= 2")
            metadata.append(self.cluster_info)
        if self.arrival_counters is not None:
            counter_shape = (batch_size, num_kv_heads, ARRIVAL_COUNTER_STRIDE)
            if self.arrival_counters.shape != counter_shape:
                raise ValueError(
                    "arrival_counters must have shape "
                    "[batch_size, num_kv_heads, ARRIVAL_COUNTER_STRIDE]"
                )
            metadata.append(self.arrival_counters)
        for tensor in metadata:
            if tensor.device != device or tensor.dtype != torch.int32:
                raise ValueError("plan metadata must be int32 on the query device")
            if not tensor.is_contiguous():
                raise ValueError("plan metadata must be contiguous")


def plan_pa_decode(
    context_lengths: torch.Tensor,
    num_kv_heads: int,
    *,
    max_partitions: int | None = None,
    max_batch_size: int | None = None,
    workgroup_budget: int | None = None,
    sliding_window: int = 0,
    query_length: int = 1,
    plan: PADecodePlan | None = None,
    tile_tokens: int | None = None,
) -> PADecodePlan:
    """Build/refresh GPU work metadata on the current stream without readback.

    Allocate outside graph capture; pass ``plan`` to refresh buffers in place.
    The budget counts task slots across KV heads before query splitting.

    ``max_partitions`` is in [1, device CU count], defaulting to CU count;
    omitting it on refresh preserves the existing limit.
    ``tile_tokens`` selects 128/256-token metadata units. On gfx1250, new
    single-query full-attention plans with batch <=3 and cap >=32 default to 128 to expose
    more small-context tasks. Other new plans default to 256; refresh retains
    the existing unit and must not change it. Explicit 128 requires gfx1250.
    This changes task division, not scratch capacity or the partition cap.
    ``max_batch_size`` defaults to 4096 for a new plan and is retained on refresh.
    Kernels use batch buckets with upper bounds 1, 2, 4, ... clipped to the
    device's LDS capacity, independent of ``max_batch_size``. Each bucket
    requires four LDS bytes per sequence plus scan storage. Actual batch, task
    capacity, partition cap and window are runtime values; capacity only
    specializes the block size (64--1024), and the window whether it is enabled.
    ``python -m aiter.aot.flydsl.pa_decode_plan`` precompiles every variant.

    Lengths include MTP tokens. Positive windows include each query's position;
    nonpositive windows disable them. Plans cover the MTP window union: match
    the window and query length when decoding or refreshing. Dense plans ignore
    query length.
    """
    if not isinstance(sliding_window, int):
        raise TypeError("sliding_window must be an int")
    sliding_window = max(sliding_window, 0)
    if not isinstance(query_length, int):
        raise TypeError("query_length must be an int")
    if query_length < 1:
        raise ValueError("query_length must be positive")
    if context_lengths.device.type != "cuda" or context_lengths.dtype != torch.int32:
        raise ValueError("context_lengths must be a CUDA int32 tensor")
    if context_lengths.ndim != 1 or not context_lengths.is_contiguous():
        raise ValueError("context_lengths must be a contiguous vector")
    if max_batch_size is None:
        max_batch_size = DEFAULT_MAX_BATCH_SIZE if plan is None else plan.max_batch_size
    if type(max_batch_size) is not int:
        raise TypeError("max_batch_size must be an int")
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be positive")
    batch = context_lengths.numel()
    if batch < 1 or batch > max_batch_size:
        raise ValueError(f"plan supports batches in [1, {max_batch_size}]")
    if num_kv_heads < 1:
        raise ValueError("num_kv_heads must be positive")
    dev = context_lengths.device
    properties = torch.cuda.get_device_properties(dev)
    num_compute_units = properties.multi_processor_count
    shared_memory_per_block = getattr(properties, "shared_memory_per_block", None)
    if shared_memory_per_block is None:
        shared_memory_per_block = get_lds_capacity_bytes(properties.gcnArchName)
    task_end_bytes = (batch * 4 + 15) // 16 * 16
    if task_end_bytes + _PLAN_SCAN_LDS_BYTES > shared_memory_per_block:
        raise ValueError(
            f"batch={batch} exceeds planner LDS capacity: requires up to "
            f"{task_end_bytes + _PLAN_SCAN_LDS_BYTES} bytes, device supports "
            f"{shared_memory_per_block} bytes per block"
        )
    # A cap below the power-of-two bound has the same lookup depth and row path,
    # so buckets ignore it. Clip the last bucket so rounding up cannot reject a
    # batch that fits LDS.
    lds_max_batch = _lds_max_batch(shared_memory_per_block)
    max_batch = min(lds_max_batch, 1 << (batch - 1).bit_length())
    if max_partitions is None:
        max_partitions = num_compute_units if plan is None else plan.max_partitions
    if not 1 <= max_partitions <= num_compute_units:
        raise ValueError(f"max_partitions must be in [1, {num_compute_units}]")
    if tile_tokens is None:
        tile_tokens = (
            plan.tile_tokens
            if plan is not None
            else (
                128
                if properties.gcnArchName.split(":", 1)[0]
                in ("gfx1250", "gfx1250-strict")
                and batch <= 3
                and query_length == 1
                and max_partitions >= 32
                and sliding_window == 0
                else 256
            )
        )
    if type(tile_tokens) is not int or tile_tokens not in (128, 256):
        raise ValueError("tile_tokens must be 128 or 256")
    if tile_tokens == 128 and properties.gcnArchName.split(":", 1)[0] not in (
        "gfx1250",
        "gfx1250-strict",
    ):
        raise ValueError("128-token plans require gfx1250")
    if plan is not None and tile_tokens != plan.tile_tokens:
        raise ValueError("tile_tokens must match the reused plan")
    if plan is None:
        if workgroup_budget is None:
            workgroup_budget = 2 * num_compute_units
        if workgroup_budget < 1:
            raise ValueError("workgroup_budget must be positive")
        capacity = min(
            batch * max_partitions,
            max(batch, (workgroup_budget + num_kv_heads - 1) // num_kv_heads),
        )
        plan = PADecodePlan(
            torch.empty((capacity, 4), dtype=torch.int32, device=dev),
            torch.empty((batch, 2), dtype=torch.int32, device=dev),
            num_kv_heads,
            max_partitions,
            sliding_window,
            query_length,
            max_batch_size,
            # Zeroed once here: every decode that counts arrivals resets them.
            torch.zeros(
                (batch, num_kv_heads, ARRIVAL_COUNTER_STRIDE),
                dtype=torch.int32,
                device=dev,
            ),
            tile_tokens=tile_tokens,
            cluster_info=(
                torch.empty(
                    (batch, min(max_partitions, 8), 8), dtype=torch.int32, device=dev
                )
                if properties.gcnArchName.split(":", 1)[0]
                in ("gfx1250", "gfx1250-strict")
                and max_partitions >= 2
                and batch * num_kv_heads <= num_compute_units * 15 // 16
                else None
            ),
        )
    else:
        if workgroup_budget is not None:
            raise ValueError("workgroup_budget is fixed when reusing a plan")
        if max_partitions != plan.max_partitions:
            raise ValueError("max_partitions must match the reused plan")
        if max_batch_size != plan.max_batch_size:
            raise ValueError("max_batch_size must match the reused plan")
        if sliding_window != plan.sliding_window:
            raise ValueError("sliding_window must match the reused plan")
        if sliding_window > 0 and query_length != plan.query_length:
            raise ValueError("query_length must match the reused plan")
    plan.validate(batch, num_kv_heads, dev)
    # The first visible token is length - window. Clamping to int32 keeps
    # every visible range, since no length exceeds it.
    window = min(query_length - 1 + sliding_window, 2**31 - 1) if sliding_window else 0
    with torch.cuda.device(dev):
        launch = compile_pa_decode_plan(
            max_batch,
            _block_threads(plan.capacity),
            sliding_window > 0,
            plan.cluster_info is not None,
            plan.tile_tokens,
            plan.max_partitions == 1,
        )
        _run_compiled(
            launch,
            ptr_arg(context_lengths, fx.Int32),
            ptr_arg(plan.work_info, fx.Int32),
            ptr_arg(plan.reduce_info, fx.Int32),
            batch,
            plan.capacity,
            plan.max_partitions,
            window,
            (
                ptr_arg(plan.cluster_info, fx.Int32)
                if plan.cluster_info is not None
                else flyc.from_c_void_p(fx.Int32, 0)
            ),
            torch.cuda.current_stream(dev),
        )
    return plan
