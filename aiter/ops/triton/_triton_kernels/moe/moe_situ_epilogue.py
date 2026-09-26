# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for a merged MoE SiTU epilogue."""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.activation import _tanh
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_moe_situ_epilogue_repr = make_kernel_repr(
    "_moe_situ_epilogue_kernel",
    [
        "SHARED_INTERMEDIATE",
        "NUM_EXPERTS",
        "ROUTED_LATENT",
        "TILE",
        "SHARED_TILES",
        "ROUTER_TILES",
        "TOTAL_TILES",
        "NUM_WARPS",
    ],
)


@triton.jit(repr=_moe_situ_epilogue_repr)
def _moe_situ_epilogue_kernel(
    projection_ptr,
    shared_ptr,
    router_ptr,
    routed_ptr,
    stride_projection_m,
    stride_shared_m,
    stride_router_m,
    stride_routed_m,
    SITU_BETA: tl.constexpr,
    SITU_LINEAR_BETA: tl.constexpr,
    SHARED_INTERMEDIATE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    ROUTED_LATENT: tl.constexpr,
    TILE: tl.constexpr,
    SHARED_TILES: tl.constexpr,
    ROUTER_TILES: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // TOTAL_TILES
    tile_idx = pid % TOTAL_TILES
    shared_gate_up = 2 * SHARED_INTERMEDIATE

    if tile_idx < SHARED_TILES:
        pair_offsets = tile_idx * (TILE // 2) + tl.arange(0, TILE // 2)
        mask = pair_offsets < SHARED_INTERMEDIATE
        gate_cols = pair_offsets
        up_cols = pair_offsets + SHARED_INTERMEDIATE
        gate = (
            tl.load(
                projection_ptr + row * stride_projection_m + gate_cols,
                mask=mask,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        up = (
            tl.load(
                projection_ptr + row * stride_projection_m + up_cols,
                mask=mask,
                other=0.0,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        gate = SITU_BETA * _tanh(gate / SITU_BETA) * tl.sigmoid(gate)
        up = SITU_LINEAR_BETA * _tanh(up / SITU_LINEAR_BETA)
        tl.store(
            shared_ptr + row * stride_shared_m + pair_offsets,
            gate * up,
            mask=mask,
        )
    elif tile_idx < SHARED_TILES + ROUTER_TILES:
        offsets = (tile_idx - SHARED_TILES) * TILE + tl.arange(0, TILE)
        mask = offsets < NUM_EXPERTS
        values = tl.load(
            projection_ptr + row * stride_projection_m + shared_gate_up + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            router_ptr + row * stride_router_m + offsets,
            values,
            mask=mask,
        )
    else:
        offsets = (tile_idx - SHARED_TILES - ROUTER_TILES) * TILE + tl.arange(0, TILE)
        mask = offsets < ROUTED_LATENT
        values = tl.load(
            projection_ptr
            + row * stride_projection_m
            + shared_gate_up
            + NUM_EXPERTS
            + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            routed_ptr + row * stride_routed_m + offsets,
            values,
            mask=mask,
        )
