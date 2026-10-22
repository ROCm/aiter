# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-tier Triton kernels for the fused Qwen3-Next attention prologue.

Fuses per-head Q/K GemmaRMSNorm ``x*(1+w)``, partial NeoX RoPE (rotary_dim 64,
reuse_freqs), and the ``[Q||gate]`` split. One program per ``(token, role)``;
roles write disjoint regions with no scratch or barrier. The small-M winner of
the schedule bake-off (flat ~31-33us up to M~=12288).
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_decoupled_heads_repr = make_kernel_repr(
    "_decoupled_heads",
    ["HEADS"],
)


@triton.jit
def _normalized_head(packed, weight, trig, position, eps):
    # Striped channels keep the non-rotary work local to each lane.
    channel = tl.max_contiguous(tl.arange(0, 256), 1)
    source = tl.load(packed + channel).to(tl.float32)
    gain = tl.load(weight + channel).to(tl.float32)
    cosine = tl.load(trig + position * 64 + channel % 32, channel < 64, 0).to(
        tl.float32
    )
    sine = tl.load(trig + position * 64 + 32 + channel % 32, channel < 64, 0).to(
        tl.float32
    )
    variance = tl.sum(source * source, 0) / 256.0
    inv_rms = tl.rsqrt(variance + eps)
    # RMSNorm rounds to BF16 before the rotary arithmetic.
    normalized = (source * inv_rms * (1.0 + gain)).to(tl.bfloat16)
    values = normalized.to(tl.float32)
    partner = tl.gather(normalized, channel ^ 32, axis=0).to(tl.float32)
    partner_sine = partner * sine
    rotated = tl.where(
        channel < 32,
        values * cosine - partner_sine,
        values * cosine + partner_sine,
    )
    return tl.where(channel < 64, rotated, values).to(tl.bfloat16)


@triton.jit
def _normalized_key_tile(packed, weight, trig, position, eps, tile):
    # Each 64-channel writer independently reduces the complete K head.
    full_channel = tl.max_contiguous(tl.arange(0, 256), 1)
    full_source = tl.load(packed + full_channel).to(tl.float32)
    variance = tl.sum(full_source * full_source, 0) / 256.0
    inv_rms = tl.rsqrt(variance + eps)
    local_channel = tl.arange(0, 64)
    channel = tile * 64 + local_channel
    source = tl.load(packed + channel).to(tl.float32)
    gain = tl.load(weight + channel).to(tl.float32)
    cosine = tl.load(trig + position * 64 + channel % 32, channel < 64, 0).to(
        tl.float32
    )
    sine = tl.load(trig + position * 64 + 32 + channel % 32, channel < 64, 0).to(
        tl.float32
    )
    # RMSNorm rounds to BF16 before the rotary arithmetic.
    normalized = (source * inv_rms * (1.0 + gain)).to(tl.bfloat16)
    values = normalized.to(tl.float32)
    partner = tl.gather(normalized, local_channel ^ 32, axis=0).to(tl.float32)
    partner_sine = partner * sine
    rotated = tl.where(
        channel < 32,
        values * cosine - partner_sine,
        values * cosine + partner_sine,
    )
    return tl.where(channel < 64, rotated, values).to(tl.bfloat16)


@triton.jit(repr=_decoupled_heads_repr)
def _decoupled_heads(
    positions,
    packed,
    trig,
    eps,
    q_weight,
    k_weight,
    qk,
    gate,
    HEADS: tl.constexpr,
    POSITION_STRIDE: tl.constexpr,
):
    token = tl.program_id(0)
    job = tl.program_id(1)
    tl.assume(token >= 0)
    tl.assume(job >= 0)
    channel = tl.max_contiguous(tl.arange(0, 256), 1)
    token_base = token * (HEADS + 1) * 512
    # Q first, then four K tiles, then two gate workers.
    # The roles write disjoint regions and need no scratch or synchronization.
    if job < HEADS:
        head = job
        position = tl.load(positions + token * POSITION_STRIDE)
        query = _normalized_head(
            packed + token_base + head * 512, q_weight, trig, position, eps
        )
        tl.store(
            qk + (token * (HEADS + 1) + head) * 256 + channel,
            query,
            cache_modifier=".wt",
        )
    elif job < HEADS + 4:
        tile = job - HEADS
        position = tl.load(positions + token * POSITION_STRIDE)
        key_channel = tile * 64 + tl.arange(0, 64)
        key = _normalized_key_tile(
            packed + token_base + HEADS * 512, k_weight, trig, position, eps, tile
        )
        tl.store(
            qk + (token * (HEADS + 1) + HEADS) * 256 + key_channel,
            key,
            cache_modifier=".wt",
        )
    else:
        side = job - HEADS - 4
        tl.assume(side >= 0)
        gate_offsets = side * (HEADS * 128) + tl.arange(0, HEADS * 128)
        source_offsets = gate_offsets // 256 * 512 + 256 + gate_offsets % 256
        payload = tl.load(packed + token_base + source_offsets)
        tl.store(
            gate + token * HEADS * 256 + gate_offsets, payload, cache_modifier=".wt"
        )
