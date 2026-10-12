# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""GPU work planning for FlyDSL paged attention with mixed context lengths."""

from dataclasses import dataclass
from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from .tensor_shim import _run_compiled, ptr_arg, ptr_buf_tensor


@cache
def compile_pa_decode_plan(batch, capacity, max_parts, sliding_window, query_length):
    """Allocate packed tasks with one block-wide scan, without GPU readback."""
    threads = max(64, min(1024, 1 << (capacity - 1).bit_length()))
    items = (batch + threads - 1) // threads
    scan_tiles = fx.coop.BlockScan[fx.Int64, threads]
    scan_counts = fx.coop.BlockScan[fx.Int32, threads]
    reduce_rows = fx.coop.BlockReduce[fx.Int32, threads]

    @fx.struct
    class TaskStorage:
        ends: fx.Array[fx.Int32, batch, 16]

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def pa_decode_plan_kernel(
        lengths_ptr: fx.Pointer,
        work_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
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
        tiles = []
        nonempty = []
        # Consecutive per-thread items preserve sequence order in vector scans.
        for item in fx.range_constexpr(items):
            seq = tid * items + item
            ctx = fx.Int32(0)
            if seq < batch:
                ctx = lengths[seq]
            length = fx.max(fx.Int64(ctx), fx.Int64(0))
            first = fx.Int64(0)
            if fx.const_expr(sliding_window > 0):
                first = (
                    fx.max(length - (query_length - 1) - sliding_window, fx.Int64(0))
                    // 256
                )
            count = (length + 255) // 256 - first
            tiles.append(count)
            nonempty.append(fx.Int32(count > 0))

        cumulative, total = scan_tiles.inclusive_with_aggregate(
            fx.Vector.from_elements(tiles), fx.ReductionOp.ADD, storage=tile_storage
        )
        total = fx.max(total, fx.Int64(1))
        remaining = capacity - reduce_rows(
            fx.Vector.from_elements(nonempty), fx.ReductionOp.ADD, storage=row_storage
        )
        counts = []
        for item in fx.range_constexpr(items):
            upper = cumulative[item] * fx.Int64(remaining) // total
            lower = (cumulative[item] - tiles[item]) * fx.Int64(remaining) // total
            count = fx.min(
                fx.min(fx.Int64(nonempty[item]) + upper - lower, tiles[item]),
                fx.Int64(max_parts),
            )
            counts.append(fx.Int32(count))
        starts, total_tasks = scan_counts.exclusive_with_aggregate(
            fx.Vector.from_elements(counts), fx.ReductionOp.ADD, storage=count_storage
        )

        for item in fx.range_constexpr(items):
            seq = tid * items + item
            if seq < batch:
                start = starts[item]
                count = counts[item]
                reduce_info[seq * 2] = start
                reduce_info[seq * 2 + 1] = count
                task_ends[seq] = start + count
        fx.gpu.barrier()

        # Spread packed task writes across all threads, including small batches
        # with many partitions. Upper-bound search skips repeated ends from empty
        # rows and needs at most 16 KiB of LDS for the largest supported batch.
        for slot in range(tid, capacity, threads):
            if slot < total_tasks:
                seq = fx.Int32(-1)
                for bit in fx.range_constexpr(batch.bit_length() - 1, -1, -1):
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
                length = fx.max(fx.Int64(ctx), fx.Int64(0))
                first = fx.Int64(0)
                if fx.const_expr(sliding_window > 0):
                    first = (
                        fx.max(
                            length - (query_length - 1) - sliding_window, fx.Int64(0)
                        )
                        // 256
                    )
                num_tiles = (length + 255) // 256 - first
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
        stream: fx.Stream,
    ):
        pa_decode_plan_kernel(lengths, work, reduce_info).launch(
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

    @property
    def capacity(self) -> int:
        return self.work_info.shape[0]

    def validate(self, batch_size: int, num_kv_heads: int, device: torch.device):
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

    Lengths include MTP tokens. Positive windows include each query's position;
    0 and -1 disable them. Plans cover the MTP window union: match the window
    and query length when decoding or refreshing. Dense plans ignore query length.
    """
    if not isinstance(sliding_window, int):
        raise TypeError("sliding_window must be an int")
    if sliding_window < -1:
        raise ValueError("sliding_window must be -1, 0, or positive")
    sliding_window = max(sliding_window, 0)
    if not isinstance(query_length, int):
        raise TypeError("query_length must be an int")
    if query_length < 1:
        raise ValueError("query_length must be positive")
    if context_lengths.device.type != "cuda" or context_lengths.dtype != torch.int32:
        raise ValueError("context_lengths must be a CUDA int32 tensor")
    if context_lengths.ndim != 1 or not context_lengths.is_contiguous():
        raise ValueError("context_lengths must be a contiguous vector")
    batch = context_lengths.numel()
    if batch < 1 or batch > 4096:
        raise ValueError("plan supports batches in [1, 4096]")
    if num_kv_heads < 1:
        raise ValueError("num_kv_heads must be positive")
    dev = context_lengths.device
    num_compute_units = torch.cuda.get_device_properties(dev).multi_processor_count
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
        )
    else:
        if workgroup_budget is not None:
            raise ValueError("workgroup_budget is fixed when reusing a plan")
        if max_partitions != plan.max_partitions:
            raise ValueError("max_partitions must match the reused plan")
        if sliding_window != plan.sliding_window:
            raise ValueError("sliding_window must match the reused plan")
        if sliding_window > 0 and query_length != plan.query_length:
            raise ValueError("query_length must match the reused plan")
    plan.validate(batch, num_kv_heads, dev)
    with torch.cuda.device(dev):
        launch = compile_pa_decode_plan(
            batch,
            plan.capacity,
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
            torch.cuda.current_stream(dev),
        )
    return plan
