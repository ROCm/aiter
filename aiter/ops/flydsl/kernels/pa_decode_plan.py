# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""GPU work planning for FlyDSL paged attention with mixed context lengths."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .pa_decode_reduce import MAX_CONTEXT_PARTITIONS


@triton.jit
def _plan_pa_decode(
    lengths,
    work,
    reduce_info,
    B: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    seq = tl.program_id(0)
    b = tl.arange(0, BLOCK_B)
    ctx = tl.load(lengths + b, b < B, other=0)
    tiles = tl.where(b < B, tl.maximum(ctx, 0).to(tl.int64) + 255, 0) // 256
    nonempty = (tiles > 0).to(tl.int32)
    total = tl.maximum(tl.sum(tiles, 0), 1)
    remaining = CAPACITY - tl.sum(nonempty, 0)
    cumulative = tl.cumsum(tiles, 0)
    # Integer prefix apportionment telescopes to the budget, even when the
    # lengths differ by orders of magnitude. Each nonempty request gets one
    # task first; clamp extras to useful tiles and the reducer's per-row limit.
    upper = cumulative * remaining // total
    lower = (cumulative - tiles) * remaining // total
    counts = tl.minimum(tl.minimum(nonempty + upper - lower, tiles), MAX_PARTS)
    count = tl.sum(tl.where(b == seq, counts, 0), 0).to(tl.int32)
    start = tl.sum(tl.where(b < seq, counts, 0), 0).to(tl.int32)
    seq_ctx = tl.load(lengths + seq)
    seq_tiles = (tl.maximum(seq_ctx, 0).to(tl.int64) + 255) // 256
    total_tasks = tl.sum(counts, 0).to(tl.int32)
    tl.store(reduce_info + seq * 2, start)
    tl.store(reduce_info + seq * 2 + 1, count)

    part = tl.arange(0, BLOCK_P)
    active = part < count
    begin = part.to(tl.int64) * seq_tiles // tl.maximum(count, 1)
    end = (part.to(tl.int64) + 1) * seq_tiles // tl.maximum(count, 1)
    slot = start + part
    tl.store(work + slot * 4, seq, active)
    tl.store(work + slot * 4 + 1, begin, active)
    tl.store(work + slot * 4 + 2, end, active)
    tl.store(work + slot * 4 + 3, seq_ctx, active)

    # The launch capacity is static for graph capture. Clear every padded work
    # record so the attention kernel can skip it without touching stale scratch.
    # These stores and the active stores are disjoint, including when every
    # context is empty.
    pad = seq * BLOCK_P + part
    padding = (pad >= total_tasks) & (pad < CAPACITY)
    for field in tl.static_range(4):
        tl.store(work + pad * 4 + field, 0, padding)


@dataclass(frozen=True)
class PADecodePlan:
    """Reusable GPU metadata; refresh it whenever context lengths change.

    ``work_info`` is [capacity, 4]: sequence, first/last 256-token tile, length.
    ``reduce_info`` is [batch, 2]: first packed task, actual task count.
    Scratch uses [KV heads, capacity, query rows (, head dim)].
    """

    work_info: torch.Tensor
    reduce_info: torch.Tensor
    num_kv_heads: int
    max_partitions: int

    @property
    def capacity(self) -> int:
        return self.work_info.shape[0]

    def validate(self, batch_size: int, num_kv_heads: int, device: torch.device):
        if self.num_kv_heads != num_kv_heads:
            raise ValueError("plan KV head count does not match the cache")
        if not 1 <= self.max_partitions <= MAX_CONTEXT_PARTITIONS:
            raise ValueError("invalid plan max_partitions")
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
    max_partitions: int = MAX_CONTEXT_PARTITIONS,
    workgroup_budget: int | None = None,
    plan: PADecodePlan | None = None,
) -> PADecodePlan:
    """Build/update a plan on the current stream without GPU-to-CPU readback.

    Allocate once outside graph capture, then pass ``plan=...`` to refresh the
    same metadata in place. Include this refresh in end-to-end measurements.
    The budget counts CTAs over all KV heads, with fused query positions.
    This is opt-in: uniform or short-context workloads may favor static splits.
    """
    if context_lengths.device.type != "cuda" or context_lengths.dtype != torch.int32:
        raise ValueError("context_lengths must be a CUDA int32 tensor")
    if context_lengths.ndim != 1 or not context_lengths.is_contiguous():
        raise ValueError("context_lengths must be a contiguous vector")
    batch = context_lengths.numel()
    if batch < 1 or batch > 4096:
        raise ValueError("plan supports batches in [1, 4096]")
    if num_kv_heads < 1:
        raise ValueError("num_kv_heads must be positive")
    if not 1 <= max_partitions <= MAX_CONTEXT_PARTITIONS:
        raise ValueError(f"max_partitions must be in [1, {MAX_CONTEXT_PARTITIONS}]")
    dev = context_lengths.device
    if plan is None:
        if workgroup_budget is None:
            workgroup_budget = (
                2 * torch.cuda.get_device_properties(dev).multi_processor_count
            )
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
        )
    else:
        if workgroup_budget is not None:
            raise ValueError("workgroup_budget is fixed when reusing a plan")
        if max_partitions != plan.max_partitions:
            raise ValueError("max_partitions must match the reused plan")
    plan.validate(batch, num_kv_heads, dev)
    with torch.cuda.device(dev):
        _plan_pa_decode[(batch,)](
            context_lengths,
            plan.work_info,
            plan.reduce_info,
            batch,
            plan.capacity,
            plan.max_partitions,
            triton.next_power_of_2(batch),
            triton.next_power_of_2(plan.max_partitions),
            num_warps=4,
        )
    return plan
