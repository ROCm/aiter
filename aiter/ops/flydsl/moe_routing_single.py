# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bounded single-token routing for the explicit gfx942 A16W4 MoE path."""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.routing_single import (
    _build_routing_single_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


@functools.lru_cache(maxsize=128)
def _routing_module(routes: int, block_m: int, experts: int):
    # Cache the geometry-specific JIT factory, not a device-specific compiled
    # handle. tensor_shim owns compiled specializations and keys every tensor's
    # device/dtype/rank/unit-stride metadata as well as the stream contract.
    return _build_routing_single_module(routes, block_m, experts)


def prepare_single_token_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    block_m: int,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return stable sorted IDs/weights/experts and two counts on the input GPU.

    IDs are contiguous int32 [1, R], weights contiguous float32 [1, R],
    1 <= R <= 16, and BM is a supported A16W4 row tile. Invalid expert IDs
    are excluded; duplicate experts and slot order are preserved. Metadata
    capacity is fixed at R*BM / R / 2, so no device-to-host read is needed.
    The helper neither clears a MoE output nor changes the router weights.
    """
    if get_gfx_runtime() != "gfx942":
        raise NotImplementedError("single-token A16W4 routing requires gfx942")
    if topk_ids.ndim != 2 or topk_ids.shape[0] != 1:
        raise ValueError("topk_ids must have shape [1, R]")
    routes = int(topk_ids.shape[1])
    if not 1 <= routes <= 16:
        raise ValueError("single-token routing requires 1 <= R <= 16")
    if (
        topk_ids.dtype != torch.int32
        or topk_weights.dtype != torch.float32
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError("routing requires matching int32 IDs and float32 weights")
    if (
        topk_ids.device.type != "cuda"
        or topk_weights.device != topk_ids.device
        or not topk_ids.is_contiguous()
        or not topk_weights.is_contiguous()
    ):
        raise ValueError("routing inputs must be contiguous on the same GPU")
    if not isinstance(experts, int) or not 0 < experts < 2**31:
        raise ValueError("experts must be a positive int32 count")
    if block_m not in (16, 32, 64, 128):
        raise ValueError("block_m must be one of 16, 32, 64, 128")
    device = topk_ids.device
    if stream is None:
        stream = torch.cuda.current_stream(device)
    if stream.device != device:
        raise ValueError("routing stream must belong to the input GPU")
    sorted_ids = torch.empty(routes * block_m, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(routes * block_m, dtype=torch.float32, device=device)
    sorted_experts = torch.empty(routes, dtype=torch.int32, device=device)
    counts = torch.empty(2, dtype=torch.int32, device=device)
    _run_compiled(
        _routing_module(routes, block_m, experts),
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_experts,
        counts,
        stream,
    )
    return sorted_ids, sorted_weights, sorted_experts, counts
