# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DSA Q/K publication in OPUS gfx950's MXFP4 layout.

No weight quantization: Q is the BF16 projection, K the BF16 pre-LayerNorm
projection. DCP can publish all queries while skipping negative K slots.
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit
def _rope(
    x,
    Positions,
    Cos,
    Sin,
    token,
    CS_LD: tl.constexpr,
    MAX_POS: tl.constexpr,
    ROPE: tl.constexpr,
    NEOX: tl.constexpr,
):
    d = tl.arange(0, 128)
    if ROPE:
        half: tl.constexpr = ROPE // 2
        partner = d ^ (half if NEOX else 1)
        other = tl.gather(x, tl.broadcast_to(partner[None, :], x.shape), 1)
        pair = d % half if NEOX else d // 2
        lead = d < half if NEOX else d % 2 == 0
        pos = tl.minimum(tl.maximum(tl.load(Positions + token), 0), MAX_POS - 1)
        cos = tl.load(Cos + pos * CS_LD + pair, d < ROPE, 1).to(tl.float32)
        sin = tl.load(Sin + pos * CS_LD + pair, d < ROPE, 0).to(tl.float32)
        rotated = tl.where(lead[None, :], x * cos - other * sin, x * cos + other * sin)
        x = tl.where(d[None, :] < ROPE, rotated.to(tl.bfloat16).to(tl.float32), x)
    return x


@triton.jit(
    repr=make_kernel_repr(
        "indexer_fp4_opus", ["HEADS", "ROPE", "NEOX", "ALL_Q", "num_warps"]
    )
)
def _publish(
    Q,
    K,
    Weights,
    WeightsOut,
    NW,
    NB,
    Slots,
    Cache,
    CacheScale,
    QOut,
    QScale,
    Positions,
    Cos,
    Sin,
    WS: tl.constexpr,
    WH: tl.constexpr,
    WOS: tl.constexpr,
    WOH: tl.constexpr,
    ALL_Q: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    KS: tl.constexpr,
    SLOT_COUNT: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    ROPE: tl.constexpr,
    NEOX: tl.constexpr,
    CS_LD: tl.constexpr,
    MAX_POS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    d = tl.arange(0, 128)
    b = tl.arange(0, 64)
    g = tl.arange(0, 4)
    slot = tl.load(Slots + token, token < SLOT_COUNT, -1).to(tl.int64)
    if task < HEADS and (ALL_Q or slot >= 0):
        q = tl.load(Q + token * QS + task * QH + d).to(tl.float32)[None, :]
        q = _rope(q, Positions, Cos, Sin, token, CS_LD, MAX_POS, ROPE, NEOX)
        packed, scales = _mxfp4_quant_op(q, 128, 1, 32, SCALING_MODE=1, USE_ASM=True)
        tl.store(QOut + (token * HEADS + task) * 64 + b, packed.reshape(64))
        weight = tl.load(Weights + token * WS + task * WH)
        tl.store(WeightsOut + token * WOS + task * WOH, weight)
        # OPUS's query scale [T,2,32,H/16], including H=32's two bytes/lane.
        at = (
            (g % 2 * 32 + task % 32) * (HEADS // 16)
            + g // 2 * (HEADS // 32)
            + task // 32
        )
        tl.store(QScale + token * HEADS * 4 + at, scales.reshape(4))
    elif task == HEADS:
        if slot >= 0:
            x = tl.load(K + token * KS + d).to(tl.float32)
            mean = tl.sum(x, 0) * (1.0 / 128)
            x = x - mean
            inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 128) + EPS)
            gamma = tl.load(NW + d).to(tl.float32)
            beta = tl.load(NB + d).to(tl.float32)
            if not ROPE:
                gamma = gamma.to(tl.bfloat16).to(tl.float32)
                beta = beta.to(tl.bfloat16).to(tl.float32)
                x = (x * inv) * gamma + beta
            else:
                x = tl.fma(x * inv, gamma, beta)
            x = x.to(tl.bfloat16).to(tl.float32)[None, :]
            x = _rope(x, Positions, Cos, Sin, token, CS_LD, MAX_POS, ROPE, NEOX)
            packed, scales = _mxfp4_quant_op(
                x, 128, 1, 32, SCALING_MODE=1, USE_ASM=True
            )
            group, row = slot // 64, slot % 64
            tl.store(
                Cache + group * 4096 + (b // 16 * 64 + row) * 16 + b % 16,
                packed.reshape(64),
            )
            at = g % 2 * 128 + row % 32 * 4 + g // 2 * 2 + row // 32
            tl.store(CacheScale + group * 256 + at, scales.reshape(4))
