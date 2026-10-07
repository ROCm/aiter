# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""GPU work planning for FlyDSL paged attention with mixed context lengths."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _plan_pa_decode(
    lengths,
    work,
    reduce_info,
    B: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    QUERY_LENGTH: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    seq = tl.program_id(0)
    b = tl.arange(0, BLOCK_B)
    ctx = tl.maximum(tl.load(lengths + b, b < B, other=0), 0).to(tl.int64)
    first_tile = tl.full((BLOCK_B,), 0, tl.int64)
    if SLIDING_WINDOW > 0:
        # Absolute tiles cover the MTP window union without rebasing cache offsets.
        first_tile = tl.maximum(ctx - (QUERY_LENGTH - 1) - SLIDING_WINDOW, 0) // 256
    tiles = tl.where(b < B, (ctx + 255) // 256 - first_tile, 0)
    nonempty = (tiles > 0).to(tl.int32)
    total = tl.maximum(tl.sum(tiles, 0), 1)
    remaining = CAPACITY - tl.sum(nonempty, 0)
    cumulative = tl.cumsum(tiles, 0)
    # Reserve one task per nonempty row, then apportion the rest using integer
    # prefixes so rounding cannot overfill the budget.
    upper = cumulative * remaining // total
    lower = (cumulative - tiles) * remaining // total
    counts = tl.minimum(tl.minimum(nonempty + upper - lower, tiles), MAX_PARTS)
    count = tl.sum(tl.where(b == seq, counts, 0), 0).to(tl.int32)
    start = tl.sum(tl.where(b < seq, counts, 0), 0).to(tl.int32)
    seq_ctx = tl.load(lengths + seq)
    seq_length = tl.maximum(seq_ctx, 0).to(tl.int64)
    seq_first_tile = tl.full((), 0, tl.int64)
    if SLIDING_WINDOW > 0:
        seq_first_tile = (
            tl.maximum(seq_length - (QUERY_LENGTH - 1) - SLIDING_WINDOW, 0) // 256
        )
    seq_tiles = (seq_length + 255) // 256 - seq_first_tile
    total_tasks = tl.sum(counts, 0).to(tl.int32)
    tl.store(reduce_info + seq * 2, start)
    tl.store(reduce_info + seq * 2 + 1, count)

    part = tl.arange(0, BLOCK_P)
    active = part < count
    begin = seq_first_tile + part.to(tl.int64) * seq_tiles // tl.maximum(count, 1)
    end = seq_first_tile + (part.to(tl.int64) + 1) * seq_tiles // tl.maximum(count, 1)
    slot = start + part
    tl.store(work + slot * 4, seq, active)
    tl.store(work + slot * 4 + 1, begin, active)
    tl.store(work + slot * 4 + 2, end, active)
    tl.store(work + slot * 4 + 3, seq_ctx, active)

    # Zero unused slots for fixed-capacity graph launches. Padding stores must
    # stay disjoint from active records, including for all-empty batches.
    pad = seq * BLOCK_P + part
    padding = (pad >= total_tasks) & (pad < CAPACITY)
    for field in tl.static_range(4):
        tl.store(work + pad * 4 + field, 0, padding)


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
        _plan_pa_decode[(batch,)](
            context_lengths,
            plan.work_info,
            plan.reduce_info,
            batch,
            plan.capacity,
            plan.max_partitions,
            # Clamp windows to int32 lengths; dense plans share one specialization.
            min(sliding_window, 2**31 - 1),
            query_length if sliding_window > 0 else 1,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(plan.max_partitions),
            num_warps=4,
        )
    return plan
