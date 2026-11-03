# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_routing.topk_align import (
    _topk_hist_kernel,
    _topk_prefix_kernel,
    _topk_scatter_kernel,
)
from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.moe_config_utils import get_moe_dispatch
from aiter.utility.graph_alloc import persistent_alloc

_TOPK_ALIGN_SCRATCH = {}


def _max_tiles(route_count: int, num_experts: int, block_m: int) -> int:
    if route_count <= num_experts:
        return route_count
    return (
        triton.cdiv(max(route_count - num_experts + 1, 0), block_m)
        + num_experts
        - 1
    )


def get_topk_align_config():
    return dict(get_moe_dispatch("MOE-TOPK-ALIGN", get_arch(), "triton")["any"])


def _scratch(device, route_count: int, num_experts: int, block_m: int):
    stream = torch.cuda.current_stream(device)
    key = (device, stream.cuda_stream, route_count, num_experts, block_m)
    scratch = _TOPK_ALIGN_SCRATCH.get(key)
    if scratch is not None:
        return scratch

    max_blocks = _max_tiles(route_count, num_experts, block_m)
    with persistent_alloc(device):
        scratch = {
            "hist": torch.empty(num_experts, dtype=torch.int32, device=device),
            "raw_offsets": torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            "tile_offsets": torch.empty(
                num_experts + 1, dtype=torch.int32, device=device
            ),
            "cursor": torch.empty(num_experts, dtype=torch.int32, device=device),
            "total_blocks": torch.empty(1, dtype=torch.int32, device=device),
            "block_pid_map": torch.empty(max_blocks, dtype=torch.int32, device=device),
            "sorted_routes": torch.empty(route_count, dtype=torch.int32, device=device),
            "sorted_weights": torch.empty(
                route_count, dtype=torch.float32, device=device
            ),
            "inverse": torch.empty(route_count, dtype=torch.int32, device=device),
        }
    _TOPK_ALIGN_SCRATCH[key] = scratch
    return scratch


def align_topk_ids(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_m: int,
):
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    topk_ids = topk_ids.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    tokens, topk = topk_ids.shape
    route_count = tokens * topk
    device = topk_ids.device
    cfg = get_topk_align_config()

    scratch = _scratch(device, route_count, num_experts, block_m)
    hist = scratch["hist"]
    raw_offsets = scratch["raw_offsets"]
    tile_offsets = scratch["tile_offsets"]
    cursor = scratch["cursor"]
    total_blocks = scratch["total_blocks"]
    max_blocks = _max_tiles(route_count, num_experts, block_m)
    block_pid_map = scratch["block_pid_map"]
    sorted_routes = scratch["sorted_routes"]
    sorted_weights = scratch["sorted_weights"]
    inverse = scratch["inverse"]
    hist.zero_()

    block = cfg["route_block"]
    num_warps = cfg["num_warps"]
    _topk_hist_kernel[(triton.cdiv(route_count, block),)](
        topk_ids,
        hist,
        route_count,
        BLOCK=block,
        num_warps=num_warps,
    )
    block_e = triton.next_power_of_2(num_experts)
    _topk_prefix_kernel[(1,)](
        hist,
        raw_offsets,
        tile_offsets,
        cursor,
        total_blocks,
        block_pid_map,
        BLOCK_M=block_m,
        NUM_EXPERTS=num_experts,
        MAX_BLOCKS=max_blocks,
        BLOCK_E=block_e,
        BLOCK_B=triton.next_power_of_2(max_blocks),
        num_warps=num_warps,
    )
    _topk_scatter_kernel[(triton.cdiv(route_count, block),)](
        topk_ids,
        topk_weights,
        cursor,
        raw_offsets,
        tile_offsets,
        block_pid_map,
        sorted_routes,
        sorted_weights,
        inverse,
        route_count,
        BLOCK_M=block_m,
        BLOCK=block,
        num_warps=num_warps,
    )

    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=sorted_weights,
        expt_hist=hist,
        n_expts_tot=num_experts,
        n_expts_act=topk,
        expt_data=ExptData(hist, raw_offsets, tile_offsets, block_pid_map),
        topk_ids=topk_ids,
    )
    return routing_data, sorted_routes, inverse, sorted_weights
