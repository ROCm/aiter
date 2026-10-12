# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public wrapper for a merged MoE SiTU epilogue."""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_situ_epilogue import (
    _moe_situ_epilogue_kernel,
)

__all__ = ["moe_situ_epilogue"]

_DEFAULT_TILE = 512


def moe_situ_epilogue(
    merged_projection: torch.Tensor,
    *,
    shared_intermediate_size: int,
    num_experts: int,
    routed_latent_size: int,
    shared_out: torch.Tensor | None = None,
    router_out: torch.Tensor | None = None,
    routed_out: torch.Tensor | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    tile: int = _DEFAULT_TILE,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply SiTU and split a merged MoE projection into its three branches.

    ``merged_projection`` has the column layout ``[shared gate, shared up,
    router logits, routed latent]``. The shared gate and up values are rounded
    to BF16 before the SiTU transforms, then multiplied and written as BF16.
    Router logits remain FP32, while the routed latent is written as BF16.

    Args:
        merged_projection: Contiguous CUDA FP32 tensor with shape
            ``[M, 2 * shared_intermediate_size + num_experts + routed_latent_size]``.
        shared_intermediate_size: Width of each shared gate/up projection and
            of the shared output.
        num_experts: Width of the router-logit output.
        routed_latent_size: Width of the routed-latent output.
        shared_out: Optional contiguous CUDA BF16 output with shape
            ``[M, shared_intermediate_size]``.
        router_out: Optional contiguous CUDA FP32 output with shape
            ``[M, num_experts]``.
        routed_out: Optional contiguous CUDA BF16 output with shape
            ``[M, routed_latent_size]``.
        situ_beta: Positive SiTU gate beta.
        situ_linear_beta: Positive SiTU linear beta.
        tile: Epilogue tile width. Must be 128, 256, or 512.
        num_warps: Triton launch warp count. Must be 4 or 8.

    Returns:
        ``(shared_out, router_out, routed_out)``. Provided output buffers are
        reused; omitted buffers are allocated on ``merged_projection.device``.

    Raises:
        ValueError: If an input, output buffer, or launch option violates the
            shape, dtype, device, contiguity, or value contract.
    """

    dimensions = {
        "shared_intermediate_size": shared_intermediate_size,
        "num_experts": num_experts,
        "routed_latent_size": routed_latent_size,
    }
    for name, value in dimensions.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    projection_size = 2 * shared_intermediate_size + num_experts + routed_latent_size
    if (
        merged_projection.dim() != 2
        or merged_projection.shape[1] != projection_size
        or merged_projection.dtype != torch.float32
        or merged_projection.device.type != "cuda"
        or not merged_projection.is_contiguous()
    ):
        raise ValueError(
            "merged_projection must be contiguous CUDA FP32 "
            f"[M, {projection_size}], got {tuple(merged_projection.shape)}/"
            f"{merged_projection.dtype}/{merged_projection.device}"
        )
    if situ_beta <= 0.0 or situ_linear_beta <= 0.0:
        raise ValueError("SiTU beta values must be positive")
    if tile not in (128, 256, 512):
        raise ValueError("tile must be one of 128, 256, or 512")
    if num_warps not in (4, 8):
        raise ValueError("num_warps must be 4 or 8")

    m = merged_projection.shape[0]
    device = merged_projection.device
    if shared_out is None:
        shared_out = torch.empty(
            (m, shared_intermediate_size),
            dtype=torch.bfloat16,
            device=device,
        )
    if router_out is None:
        router_out = torch.empty(
            (m, num_experts),
            dtype=torch.float32,
            device=device,
        )
    if routed_out is None:
        routed_out = torch.empty(
            (m, routed_latent_size),
            dtype=torch.bfloat16,
            device=device,
        )

    expected = (
        (shared_out, (m, shared_intermediate_size), torch.bfloat16),
        (router_out, (m, num_experts), torch.float32),
        (routed_out, (m, routed_latent_size), torch.bfloat16),
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

    shared_tiles = triton.cdiv(shared_intermediate_size, tile // 2)
    router_tiles = triton.cdiv(num_experts, tile)
    routed_tiles = triton.cdiv(routed_latent_size, tile)
    total_tiles = shared_tiles + router_tiles + routed_tiles
    _moe_situ_epilogue_kernel[(m * total_tiles,)](
        merged_projection,
        shared_out,
        router_out,
        routed_out,
        merged_projection.stride(0),
        shared_out.stride(0),
        router_out.stride(0),
        routed_out.stride(0),
        SITU_BETA=float(situ_beta),
        SITU_LINEAR_BETA=float(situ_linear_beta),
        SHARED_INTERMEDIATE=shared_intermediate_size,
        NUM_EXPERTS=num_experts,
        ROUTED_LATENT=routed_latent_size,
        TILE=tile,
        SHARED_TILES=shared_tiles,
        ROUTER_TILES=router_tiles,
        TOTAL_TILES=total_tiles,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
    return shared_out, router_out, routed_out
