# SPDX-License-Identifier: MIT
"""Explicitly opted-in stable MoE sorting for the measured M3 prefill contract.

Three kernels: per-token-tile histogram, global offsets, parallel stable scatter.
Contract: contiguous unique top-k routes, no expert mask/local-token indirection,
no accumulation buffer. Matches the route-reduce M3 prefill contract and emits
the same packed token IDs, padding, optional a4w4 indices and reverse mapping.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _histogram(Ids, Counts, T: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
               NT: tl.constexpr, TILE: tl.constexpr, BI: tl.constexpr,
               BE: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tl.arange(0, BI)
    positions = tile * TILE * K + offsets
    ids = tl.load(Ids + positions,
                  mask=(offsets < TILE * K) & (positions < T * K), other=E)
    histogram = tl.histogram(ids, BE)
    expert = tl.arange(0, BE)
    tl.store(Counts + expert * NT + tile, histogram, mask=expert < E)


@triton.jit
def _prefix(Counts, TileOffsets, ExpertOffsets, ExpertCounts, NumValid,
            T: tl.constexpr, E: tl.constexpr, NT: tl.constexpr,
            UNIT: tl.constexpr, BE: tl.constexpr, BN: tl.constexpr):
    expert = tl.arange(0, BE)
    tile = tl.arange(0, BN)
    counts = tl.load(Counts + expert[:, None] * NT + tile[None, :],
                     mask=(expert[:, None] < E) & (tile[None, :] < NT), other=0)
    tile_ends = tl.cumsum(counts, axis=1)
    totals = tl.sum(counts, axis=1)
    padded = tl.cdiv(totals, UNIT) * UNIT
    expert_ends = tl.cumsum(padded, axis=0)
    expert_starts = expert_ends - padded
    starts = tile_ends - counts + expert_starts[:, None]
    tl.store(TileOffsets + expert[:, None] * NT + tile[None, :], starts,
             mask=(expert[:, None] < E) & (tile[None, :] < NT))
    tl.store(ExpertOffsets + expert, expert_starts, mask=expert < E)
    tl.store(ExpertCounts + expert, totals, mask=expert < E)
    tl.store(NumValid, tl.sum(padded, axis=0))
    tl.store(NumValid + 1, T)


@triton.jit
def _scatter(Ids, Weights, TileOffsets, ExpertOffsets, ExpertCounts,
             SortedIds, SortedWeights, SortedExperts, MIndices, Reverse,
             T: tl.constexpr, K: tl.constexpr, E: tl.constexpr, NT: tl.constexpr,
             UNIT: tl.constexpr, TILE: tl.constexpr, BI: tl.constexpr,
             AUX: tl.constexpr):
    expert = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tl.arange(0, BI)
    positions = tile * TILE * K + offsets
    valid = (offsets < TILE * K) & (positions < T * K)
    ids = tl.load(Ids + positions, mask=valid, other=-1)
    selected = valid & (ids == expert)
    local_end = tl.cumsum(selected.to(tl.int32), axis=0)
    tile_start = tl.load(TileOffsets + expert * NT + tile)
    destination = tile_start + local_end - 1
    token = positions // K
    slot = positions % K
    packed = token | (slot << 24)
    weight = tl.load(Weights + positions, mask=selected, other=0)
    tl.store(SortedIds + destination, packed, mask=selected)
    tl.store(SortedWeights + destination, weight, mask=selected)
    if AUX:
        tl.store(MIndices + destination, token, mask=selected)
        tl.store(Reverse + positions, destination, mask=selected)

    expert_start = tl.load(ExpertOffsets + expert)
    count = tl.load(ExpertCounts + expert)
    padded_count = tl.cdiv(count, UNIT) * UNIT
    # Disjoint metadata writes. No atomics and no device-wide synchronization.
    block_index = tile * BI + offsets
    tl.store(SortedExperts + expert_start // UNIT + block_index, expert,
             mask=block_index < padded_count // UNIT)
    if tile == 0:
        pad_pos = expert_start + count + offsets
        is_pad = offsets < padded_count - count
        tl.store(SortedIds + pad_pos, T | (K << 24), mask=is_pad)
        tl.store(SortedWeights + pad_pos, 0., mask=is_pad)
        if AUX:
            tl.store(MIndices + pad_pos, T, mask=is_pad)


def tiled_sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
               block_size, *, output_aux=False, tile=256, num_warps=4):
    if (topk_ids.ndim != 2 or topk_ids.dtype != torch.int32
            or topk_weights.dtype != torch.float32
            or topk_ids.shape != topk_weights.shape
            or not topk_ids.is_contiguous() or not topk_weights.is_contiguous()):
        raise ValueError("Requires matching contiguous int32 IDs / float32 weights")
    if tile not in (128, 256, 512, 1024, 2048):
        raise ValueError("Unsupported token tile")
    tokens, topk = topk_ids.shape
    if tokens <= 0 or tokens >= 2**24 or topk >= 256:
        raise ValueError("Packed token format overflow")
    if not 0 < block_size <= tile:
        raise ValueError("Invalid MoE tile")
    device = topk_ids.device
    tiles = triton.cdiv(tokens, tile)
    capacity = triton.cdiv(tokens * topk + num_experts * block_size - topk,
                           block_size) * block_size
    sorted_ids = torch.empty(capacity, device=device, dtype=torch.int32)
    sorted_weights = torch.empty(capacity, device=device, dtype=torch.float32)
    sorted_experts = torch.empty(capacity // block_size, device=device, dtype=torch.int32)
    num_valid = torch.empty(2, device=device, dtype=torch.int32)
    moe_buf = torch.empty((0, 0), device=device, dtype=moebuf_dtype)
    counts = torch.empty((num_experts, tiles), device=device, dtype=torch.int32)
    offsets = torch.empty_like(counts)
    expert_offsets = torch.empty(num_experts, device=device, dtype=torch.int32)
    expert_counts = torch.empty_like(expert_offsets)
    m_indices = torch.empty_like(sorted_ids) if output_aux else sorted_ids
    reverse = torch.empty_like(topk_ids).flatten() if output_aux else sorted_ids
    bins = triton.next_power_of_2(num_experts + 1)
    items = triton.next_power_of_2(tile * topk)
    _histogram[(tiles,)](topk_ids, counts, tokens, topk, num_experts, tiles,
                         tile, items, bins, num_warps=num_warps)
    _prefix[(1,)](counts, offsets, expert_offsets, expert_counts, num_valid,
                   tokens, num_experts, tiles, block_size,
                   triton.next_power_of_2(num_experts), triton.next_power_of_2(tiles),
                   num_warps=8)
    _scatter[(num_experts, tiles)](
        topk_ids, topk_weights, offsets, expert_offsets, expert_counts,
        sorted_ids, sorted_weights, sorted_experts, m_indices, reverse,
        tokens, topk, num_experts, tiles, block_size, tile, items,
        bool(output_aux), num_warps=num_warps, num_stages=1)
    result = (sorted_ids, sorted_weights, sorted_experts, num_valid, moe_buf)
    return (*result, m_indices, reverse) if output_aux else result


_reported = False


def try_m3_tiled_sort(
    topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, block_size,
    *, expert_mask=None, num_local_tokens=None, dispatch_policy=0,
    return_local_topk_ids=False, accumulate=True, flat=False, output_aux=False,
):
    """Return None for any unmeasured or incompatible sorting contract.

    Top-k routes must be unique within each token, as required by the native
    byte-mesh sorter. This preserves original top-k slots and expert/token order.
    """
    if not (
        topk_ids.shape == (32768, 5)
        and topk_weights.shape == topk_ids.shape
        and topk_ids.dtype == torch.int32
        and topk_weights.dtype == torch.float32
        and topk_ids.is_cuda
        and topk_weights.device == topk_ids.device
        and topk_ids.is_contiguous()
        and topk_weights.is_contiguous()
        and num_experts == 129
        and model_dim == 6144
        and moebuf_dtype == torch.bfloat16
        and block_size == 64
        and expert_mask is None
        and num_local_tokens is None
        and dispatch_policy == 0
        and not return_local_topk_ids
        and not accumulate
        and not flat
        and output_aux in (False, "opus")
    ):
        return None
    from aiter import logger
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        return None
    global _reported
    if not _reported:
        logger.info(
            "M3 tiled MoE sort: tokens=32768, experts=129, topk=5, "
            "block=64, tile=256, aux=%s", output_aux
        )
        _reported = True
    return tiled_sort(
        topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
        block_size, output_aux=output_aux,
    )
