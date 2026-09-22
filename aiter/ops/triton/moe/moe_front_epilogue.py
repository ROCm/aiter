# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public wrapper for a merged MoE-front epilogue."""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_front_epilogue import (
    _moe_front_bf16_epilogue_kernel,
)

__all__ = ["moe_front_bf16_epilogue"]

_SHARED_GATE_UP = 1536
_SHARED_INTERMEDIATE = 768
_NUM_EXPERTS = 896
_ROUTED_LATENT = 3584
_FRONT = _SHARED_GATE_UP + _NUM_EXPERTS + _ROUTED_LATENT
_DEFAULT_TILE = 512


def moe_front_bf16_epilogue(
    front: torch.Tensor,
    *,
    shared_out: torch.Tensor | None = None,
    router_out: torch.Tensor | None = None,
    routed_out: torch.Tensor | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    tile: int = _DEFAULT_TILE,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a merged MoE-front accumulator into shared, router, and routed inputs.

    Args:
        front: Contiguous CUDA FP32 tensor with shape ``[M, 6016]``.
        shared_out: Optional contiguous CUDA BF16 output with shape ``[M, 768]``.
        router_out: Optional contiguous CUDA FP32 output with shape ``[M, 896]``.
        routed_out: Optional contiguous CUDA BF16 output with shape ``[M, 3584]``.
        situ_beta: Positive SiTU gate beta.
        situ_linear_beta: Positive SiTU linear beta.
        tile: Epilogue tile width. Must be 128, 256, or 512 and divide the
            shared gate/up and routed dimensions.
        num_warps: Triton launch warp count. Must be 4 or 8.

    Returns:
        ``(shared_out, router_out, routed_out)``. Provided output buffers are
        reused; omitted buffers are allocated on ``front.device``.

    Raises:
        ValueError: If an input, output buffer, or launch option violates the
            shape, dtype, device, contiguity, or value contract.
    """

    if (
        front.dim() != 2
        or front.shape[1] != _FRONT
        or front.dtype != torch.float32
        or front.device.type != "cuda"
        or not front.is_contiguous()
    ):
        raise ValueError(
            f"front must be contiguous CUDA FP32 [M, {_FRONT}], got "
            f"{tuple(front.shape)}/{front.dtype}/{front.device}"
        )
    if situ_beta <= 0.0 or situ_linear_beta <= 0.0:
        raise ValueError("SiTU beta values must be positive")
    if tile not in (128, 256, 512):
        raise ValueError("tile must be one of 128, 256, or 512")
    if _SHARED_GATE_UP % tile or _ROUTED_LATENT % tile:
        raise ValueError("tile must divide the shared and routed dimensions")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")

    m = front.shape[0]
    device = front.device
    if shared_out is None:
        shared_out = torch.empty(
            (m, _SHARED_INTERMEDIATE),
            dtype=torch.bfloat16,
            device=device,
        )
    if router_out is None:
        router_out = torch.empty(
            (m, _NUM_EXPERTS),
            dtype=torch.float32,
            device=device,
        )
    if routed_out is None:
        routed_out = torch.empty(
            (m, _ROUTED_LATENT),
            dtype=torch.bfloat16,
            device=device,
        )

    expected = (
        (shared_out, (m, _SHARED_INTERMEDIATE), torch.bfloat16),
        (router_out, (m, _NUM_EXPERTS), torch.float32),
        (routed_out, (m, _ROUTED_LATENT), torch.bfloat16),
    )
    for output, shape, dtype in expected:
        if (
            tuple(output.shape) != shape
            or output.dtype != dtype
            or output.device != device
            or not output.is_contiguous()
        ):
            raise ValueError(
                f"expected contiguous {shape}/{dtype} on {device}, got "
                f"{tuple(output.shape)}/{output.dtype}/{output.device}"
            )

    shared_tiles = _SHARED_GATE_UP // tile
    router_tiles = triton.cdiv(_NUM_EXPERTS, tile)
    routed_tiles = _ROUTED_LATENT // tile
    total_tiles = shared_tiles + router_tiles + routed_tiles
    _moe_front_bf16_epilogue_kernel[(m * total_tiles,)](
        front,
        shared_out,
        router_out,
        routed_out,
        front.stride(0),
        shared_out.stride(0),
        router_out.stride(0),
        routed_out.stride(0),
        SITU_BETA=float(situ_beta),
        SITU_LINEAR_BETA=float(situ_linear_beta),
        SHARED_GATE_UP=_SHARED_GATE_UP,
        SHARED_INTERMEDIATE=_SHARED_INTERMEDIATE,
        NUM_EXPERTS=_NUM_EXPERTS,
        TILE=tile,
        SHARED_TILES=shared_tiles,
        ROUTER_TILES=router_tiles,
        TOTAL_TILES=total_tiles,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return shared_out, router_out, routed_out
