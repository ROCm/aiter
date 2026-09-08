# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-hot and large-M merged MoE front for Kimi-K3."""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from aiter import hipb_create_extension
from aiter.ops.gradlib import _hipb_mm

__all__ = [
    "kimi_k3_moe_front_large_m_bf16",
    "kimi_k3_moe_front_bf16_epilogue",
    "merge_kimi_k3_moe_front_weights",
]

_HIDDEN = 7168
_SHARED_GATE_UP = 1536
_SHARED_INTERMEDIATE = 768
_NUM_EXPERTS = 896
_ROUTED_LATENT = 3584
_FRONT = _SHARED_GATE_UP + _NUM_EXPERTS + _ROUTED_LATENT
_DEFAULT_TILE = 512


def merge_kimi_k3_moe_front_weights(
    shared_gate_up: torch.Tensor,
    router: torch.Tensor,
    routed_down: torch.Tensor,
) -> torch.Tensor:
    """Pack Kimi-K3 front weights while preserving native row layouts."""

    expected = (
        (shared_gate_up, _SHARED_GATE_UP, "shared_gate_up"),
        (router, _NUM_EXPERTS, "router"),
        (routed_down, _ROUTED_LATENT, "routed_down"),
    )
    for weight, rows, name in expected:
        if (
            weight.dim() != 2
            or tuple(weight.shape) != (rows, _HIDDEN)
            or weight.dtype != torch.bfloat16
            or weight.device.type != "cuda"
        ):
            raise ValueError(
                f"{name} must be CUDA BF16 [{rows}, {_HIDDEN}], got "
                f"shape={tuple(weight.shape)} dtype={weight.dtype} "
                f"device={weight.device}"
            )
    if len({weight.device for weight, _, _ in expected}) != 1:
        raise ValueError("All Kimi-K3 front weights must be on the same device")

    return torch.cat(
        (shared_gate_up, router, routed_down),
        dim=0,
    ).contiguous()


@functools.lru_cache(maxsize=1)
def _initialize_hipblaslt() -> None:
    hipb_create_extension()


@functools.lru_cache(maxsize=16)
def _large_m_front_gemm_config(m: int) -> dict:
    from aiter.tuned_gemm import get_GEMM_A16W16_config

    return get_GEMM_A16W16_config(
        m,
        _FRONT,
        _HIDDEN,
        False,
        str(torch.bfloat16),
        str(torch.float32),
    )


def _large_m_front_gemm(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    front_out: torch.Tensor,
) -> None:
    config = _large_m_front_gemm_config(hidden_states.shape[0])
    if config["libtype"] == "hipblaslt":
        _initialize_hipblaslt()
        _hipb_mm(
            hidden_states,
            merged_weight.t(),
            int(config["solidx"]),
            front_out,
            None,
            None,
            None,
            None,
            False,
            False,
        )
        return

    torch.mm(
        hidden_states,
        merged_weight.t(),
        out=front_out,
        out_dtype=torch.float32,
    )


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _kimi_k3_moe_front_bf16_epilogue_kernel(
    front_ptr,
    shared_ptr,
    router_ptr,
    routed_ptr,
    stride_front_m,
    stride_shared_m,
    stride_router_m,
    stride_routed_m,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
    SHARED_GATE_UP: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TILE: tl.constexpr,
    SHARED_TILES: tl.constexpr,
    ROUTER_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // TOTAL_TILES
    tile = pid % TOTAL_TILES

    if tile < SHARED_TILES:
        pair_offsets = tile * (TILE // 2) + tl.arange(0, TILE // 2)
        gate_cols = pair_offsets
        up_cols = pair_offsets + SHARED_INTERMEDIATE
        gate = (
            tl.load(front_ptr + row * stride_front_m + gate_cols)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        up = (
            tl.load(front_ptr + row * stride_front_m + up_cols)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        gate = SITU_BETA * _tanh(gate / SITU_BETA) * tl.sigmoid(gate)
        up = SITU_LINEAR_BETA * _tanh(up / SITU_LINEAR_BETA)
        tl.store(
            shared_ptr + row * stride_shared_m + pair_offsets,
            gate * up,
        )
    elif tile < SHARED_TILES + ROUTER_TILES:
        offsets = (tile - SHARED_TILES) * TILE + tl.arange(0, TILE)
        mask = offsets < NUM_EXPERTS
        values = tl.load(
            front_ptr + row * stride_front_m + SHARED_GATE_UP + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            router_ptr + row * stride_router_m + offsets,
            values,
            mask=mask,
        )
    else:
        offsets = (tile - SHARED_TILES - ROUTER_TILES) * TILE + tl.arange(0, TILE)
        values = tl.load(
            front_ptr + row * stride_front_m + SHARED_GATE_UP + NUM_EXPERTS + offsets
        )
        tl.store(routed_ptr + row * stride_routed_m + offsets, values)


def kimi_k3_moe_front_bf16_epilogue(
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
    """Split `[M,6016]` FP32 accumulators into native Kimi MoE inputs."""

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
    _kimi_k3_moe_front_bf16_epilogue_kernel[(m * total_tiles,)](
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
        num_warps=num_warps,
    )
    return shared_out, router_out, routed_out


def kimi_k3_moe_front_large_m_bf16(
    hidden_states: torch.Tensor,
    merged_weight: torch.Tensor,
    *,
    front_out: torch.Tensor | None = None,
    shared_out: torch.Tensor | None = None,
    router_out: torch.Tensor | None = None,
    routed_out: torch.Tensor | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
    tile: int = _DEFAULT_TILE,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the large-M merged front and leave MoE quantization to AITER."""

    if (
        hidden_states.dim() != 2
        or hidden_states.shape[1] != _HIDDEN
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.device.type != "cuda"
        or not hidden_states.is_contiguous()
    ):
        raise ValueError(f"hidden_states must be contiguous CUDA BF16 [M, {_HIDDEN}]")
    if (
        tuple(merged_weight.shape) != (_FRONT, _HIDDEN)
        or merged_weight.dtype != torch.bfloat16
        or merged_weight.device != hidden_states.device
        or not merged_weight.is_contiguous()
    ):
        raise ValueError(
            f"merged_weight must be contiguous CUDA BF16 [{_FRONT}, {_HIDDEN}]"
        )

    m = hidden_states.shape[0]
    if front_out is None:
        front_out = torch.empty(
            (m, _FRONT),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    elif (
        tuple(front_out.shape) != (m, _FRONT)
        or front_out.dtype != torch.float32
        or front_out.device != hidden_states.device
        or not front_out.is_contiguous()
    ):
        raise ValueError(f"front_out must be contiguous CUDA FP32 [M, {_FRONT}]")

    _large_m_front_gemm(hidden_states, merged_weight, front_out)
    return kimi_k3_moe_front_bf16_epilogue(
        front_out,
        shared_out=shared_out,
        router_out=router_out,
        routed_out=routed_out,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        tile=tile,
        num_warps=num_warps,
    )
