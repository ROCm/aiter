# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Explicitly opted-in stable MoE sorting for the measured M3 prefill contract.

Three kernels: per-token-tile histogram, global offsets, parallel stable scatter.
Contract: contiguous unique top-k routes, no expert mask/local-token indirection,
no accumulation buffer. Matches the route-reduce M3 prefill contract and emits
the same packed token IDs, padding, optional a4w4 indices and reverse mapping.
"""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_sorting_tiled import (
    _histogram,
    _prefix,
    _scatter,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch


def tiled_sort(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    *,
    output_aux=False,
    tile=None,
    num_warps=None,
):
    """Stably group token routes into padded expert blocks on gfx950.

    IDs and weights must be matching contiguous CUDA matrices, respectively
    int32 and float32, with unique in-range expert IDs per token. ``block_size``
    controls expert padding; ``model_dim`` is retained for sorter compatibility
    and ``moebuf_dtype`` sets the empty, non-accumulating buffer dtype.
    ``tile`` and ``num_warps`` optionally override the JSON launch defaults.

    Return sorted packed IDs, weights, block expert IDs, valid/token counts and
    an empty MoE buffer. ``output_aux`` also returns token indices and a reverse
    mapping. Expert masks, local-token indirection and accumulation are not
    supported; the M3 dispatcher adds the measured-shape guard.
    """
    if (
        topk_ids.ndim != 2
        or topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or topk_ids.shape != topk_weights.shape
        or not topk_ids.is_contiguous()
        or not topk_weights.is_contiguous()
        or not topk_ids.is_cuda
        or topk_weights.device != topk_ids.device
    ):
        raise ValueError("Requires matching contiguous int32 IDs / float32 weights")
    config = get_moe_dispatch("SORTING-TILED", get_arch(), "triton").get("any")
    if config is None:
        raise ValueError("No tiled sorting configuration for this architecture")
    tile = config["TILE"] if tile is None else tile
    num_warps = config["num_warps"] if num_warps is None else num_warps
    if tile not in (128, 256, 512, 1024, 2048):
        raise ValueError("Unsupported token tile")
    tokens, topk = topk_ids.shape
    if tokens <= 0 or tokens >= 2**24 or not 0 < topk < 256:
        raise ValueError("Packed token format overflow")
    if not 0 < num_experts or not 0 < block_size <= tile:
        raise ValueError("Invalid MoE tile")
    device = topk_ids.device
    tiles = triton.cdiv(tokens, tile)
    capacity = (
        triton.cdiv(tokens * topk + num_experts * block_size - topk, block_size)
        * block_size
    )
    sorted_ids = torch.empty(capacity, device=device, dtype=torch.int32)
    sorted_weights = torch.empty(capacity, device=device, dtype=torch.float32)
    sorted_experts = torch.empty(
        capacity // block_size, device=device, dtype=torch.int32
    )
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
    _histogram[(tiles,)](
        topk_ids,
        counts,
        tokens,
        topk,
        num_experts,
        tiles,
        tile,
        items,
        bins,
        num_warps=num_warps,
    )
    _prefix[(1,)](
        counts,
        offsets,
        expert_offsets,
        expert_counts,
        num_valid,
        tokens,
        num_experts,
        tiles,
        block_size,
        triton.next_power_of_2(num_experts),
        triton.next_power_of_2(tiles),
        num_warps=config["prefix_num_warps"],
    )
    _scatter[(num_experts, tiles)](
        topk_ids,
        topk_weights,
        offsets,
        expert_offsets,
        expert_counts,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        m_indices,
        reverse,
        tokens,
        topk,
        num_experts,
        tiles,
        block_size,
        tile,
        items,
        bool(output_aux),
        num_warps=num_warps,
        num_stages=config["num_stages"],
    )
    result = (sorted_ids, sorted_weights, sorted_experts, num_valid, moe_buf)
    return (*result, m_indices, reverse) if output_aux else result


_reported = False


def try_m3_tiled_sort(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    *,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
    return_local_topk_ids=False,
    accumulate=True,
    flat=False,
    output_aux=False,
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
            "M3 tiled MoE sort: tokens=32768, experts=129, topk=5, " "block=64, aux=%s",
            output_aux,
        )
        _reported = True
    return tiled_sort(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        output_aux=output_aux,
    )
