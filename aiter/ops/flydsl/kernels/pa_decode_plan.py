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
# Two warp scans and one reduction use at most 512 bytes at 1024 wave32 threads.
_PLAN_SCAN_LDS_BYTES = 512


@cache
def compile_pa_decode_plan(max_batch, threads, max_parts, sliding_window, query_length):
    """Compile a batch bucket; actual batch and task capacity are runtime values."""
    scan_tiles = fx.coop.BlockScan[fx.Int64, threads]
    scan_counts = fx.coop.BlockScan[fx.Int32, threads]
    reduce_rows = fx.coop.BlockReduce[fx.Int32, threads]

    @fx.struct
    class TaskStorage:
        ends: fx.Array[fx.Int32, max_batch, 16]

    def visible_tiles(ctx):
        """Return the first visible tile and the visible tile count."""
        length = fx.max(fx.Int64(ctx), fx.Int64(0))
        first = fx.Int64(0)
        if sliding_window > 0:
            first = (
                fx.max(length - (query_length - 1) - sliding_window, fx.Int64(0)) // 256
            )
        return first, (length + 255) // 256 - first

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def pa_decode_plan_kernel(
        lengths_ptr: fx.Pointer,
        work_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
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
        # Each thread walks its consecutive rows with runtime bounds, so scans
        # of per-thread totals preserve sequence order, padding rows cost
        # nothing, and register use does not grow with the batch bucket.
        rows = (batch + threads - 1) // threads
        row_begin = fx.min(fx.Int32(tid) * rows, batch)
        row_end = fx.min(row_begin + rows, batch)
        thread_tiles = fx.Int64(0)
        thread_rows = fx.Int32(0)
        for seq in range(row_begin, row_end):
            _, tiles = visible_tiles(lengths[seq])
            thread_tiles = thread_tiles + tiles
            thread_rows = thread_rows + fx.Int32(tiles > 0)

        cumulative, total = scan_tiles.exclusive_with_aggregate(
            thread_tiles, fx.ReductionOp.ADD, storage=tile_storage
        )
        total = fx.max(total, fx.Int64(1))
        remaining = fx.Int64(
            capacity - reduce_rows(thread_rows, fx.ReductionOp.ADD, storage=row_storage)
        )
        # Each row's lower bound is the previous row's upper bound.
        lower = cumulative * remaining // total
        thread_tasks = fx.Int32(0)
        for seq in range(row_begin, row_end):
            _, tiles = visible_tiles(lengths[seq])
            cumulative = cumulative + tiles
            upper = cumulative * remaining // total
            count = fx.Int32(
                fx.min(
                    fx.min(fx.Int64(tiles > 0) + upper - lower, tiles),
                    fx.Int64(max_parts),
                )
            )
            lower = upper
            task_ends[seq] = count
            thread_tasks = thread_tasks + count
        row_start, total_tasks = scan_counts.exclusive_with_aggregate(
            thread_tasks, fx.ReductionOp.ADD, storage=count_storage
        )

        # Rows are thread-private here; convert counts to ends in place.
        for seq in range(row_begin, row_end):
            count = task_ends[seq]
            reduce_info[seq * 2] = row_start
            reduce_info[seq * 2 + 1] = count
            row_start = row_start + count
            task_ends[seq] = row_start
        fx.gpu.barrier()

        # Spread packed task writes across all threads, including small batches
        # with many partitions. Upper-bound search skips repeated ends from empty
        # rows and uses one LDS int32 per sequence (16 KiB at the default cap).
        for slot in range(tid, capacity, threads):
            if slot < total_tasks:
                seq = fx.Int32(-1)
                for bit in fx.range_constexpr(max_batch.bit_length() - 1, -1, -1):
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
                first, num_tiles = visible_tiles(ctx)
                part = fx.Int64(slot - start)
                work[slot * 4] = seq
                work[slot * 4 + 1] = fx.Int32(first + part * num_tiles // count)
                work[slot * 4 + 2] = fx.Int32(first + (part + 1) * num_tiles // count)
                work[slot * 4 + 3] = ctx
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
        stream: fx.Stream,
    ):
        pa_decode_plan_kernel(lengths, work, reduce_info, batch, capacity).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


@dataclass(frozen=True)
class PADecodePlan:
    """Reusable GPU metadata; refresh it whenever context lengths change.

    ``work_info`` is [capacity, 4]: sequence, begin/end absolute 256-token tile,
    original context length. Tile ranges are half-open and cover the MTP union.
    ``reduce_info`` is [batch, 2]: first packed task, actual task count.
    """

    work_info: torch.Tensor
    reduce_info: torch.Tensor
    num_kv_heads: int
    max_partitions: int
    sliding_window: int = 0
    query_length: int = 1
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE

    @property
    def capacity(self) -> int:
        return self.work_info.shape[0]

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
        for tensor in (self.work_info, self.reduce_info):
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
) -> PADecodePlan:
    """Build/refresh GPU work metadata on the current stream without readback.

    Allocate outside graph capture; pass ``plan`` to refresh buffers in place.
    The budget counts task slots across KV heads before query splitting.

    ``max_partitions`` is in [1, device CU count], defaulting to CU count;
    omitting it on refresh preserves the existing limit.
    ``max_batch_size`` defaults to 4096 for a new plan and is retained on refresh.
    Kernels use batch buckets with upper bounds 64, 128, 256, ... clipped to
    ``max_batch_size`` and the device's LDS capacity. Each bucket requires four
    LDS bytes per sequence plus scan storage. Actual batch and task capacity
    are runtime values; capacity only specializes the block size (64--1024).

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
    # Clip the last bucket so rounding up cannot reject a batch that fits LDS.
    lds_max_batch = (shared_memory_per_block - _PLAN_SCAN_LDS_BYTES) // 16 * 4
    max_batch = min(
        max_batch_size, lds_max_batch, max(64, 1 << (batch - 1).bit_length())
    )
    if max_partitions is None:
        max_partitions = num_compute_units if plan is None else plan.max_partitions
    if not 1 <= max_partitions <= num_compute_units:
        raise ValueError(f"max_partitions must be in [1, {num_compute_units}]")
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
    with torch.cuda.device(dev):
        threads = max(64, min(1024, 1 << (plan.capacity - 1).bit_length()))
        launch = compile_pa_decode_plan(
            max_batch,
            threads,
            plan.max_partitions,
            # Clamp windows to int32 lengths; dense plans share one specialization.
            min(sliding_window, 2**31 - 1),
            query_length if sliding_window > 0 else 1,
        )
        _run_compiled(
            launch,
            ptr_arg(context_lengths, fx.Int32),
            ptr_arg(plan.work_info, fx.Int32),
            ptr_arg(plan.reduce_info, fx.Int32),
            batch,
            plan.capacity,
            torch.cuda.current_stream(dev),
        )
    return plan
