# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton kernel for a merged MoE-front epilogue."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_moe_front_bf16_epilogue_repr = make_kernel_repr(
    "_moe_front_bf16_epilogue_kernel",
    [
        "TILE",
        "SHARED_TILES",
        "ROUTER_TILES",
        "TOTAL_TILES",
        "NUM_WARPS",
    ],
)


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit(repr=_moe_front_bf16_epilogue_repr)
def _moe_front_bf16_epilogue_kernel(
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
    NUM_WARPS: tl.constexpr,
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
