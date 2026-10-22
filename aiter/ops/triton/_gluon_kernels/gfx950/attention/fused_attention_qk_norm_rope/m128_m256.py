# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mid-tier Gluon kernels for the fused Qwen3-Next attention prologue (gfx950).

Q/K GemmaRMSNorm ``x*(1+w)`` + partial NeoX RoPE (rotary_dim 64) + ``[Q||gate]``
split, one CTA per ``(token, role)``. M128 uses one head per wave; M256 reuses
the rotary data across four Q heads. Wins the schedule bake-off for
``12288 < M <= 24576`` (including the 16384 chunk).
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_attention_prologue_repr = make_kernel_repr(
    "_attention_prologue_m128_m256",
    ["QUERY_HEADS", "HEADS_PER_WAVE"],
)


@gluon.jit
def _normalize_and_rotate(source, weight, offsets, cos_sin_ptr, position, eps):
    # Accept either one head or a group of heads; reduce only the channels.
    variance = gl.sum(source * source, axis=-1) / 256.0
    if len(source.shape) == 2:
        variance = variance[:, None]
        weight = weight[None, :]
    # The BF16 rounding boundary before RoPE is part of the numerical contract.
    normalized = (source * gl.rsqrt(variance + eps) * (1.0 + weight)).to(gl.bfloat16)
    values = normalized.to(gl.float32)
    if len(source.shape) == 2:
        partner_indices = gl.full(
            (source.shape[0], 1), 0, gl.int32, source.type.layout
        ) + (offsets[None, :] ^ 32)
    else:
        partner_indices = offsets ^ 32
    partner = gl.gather(values, partner_indices, axis=len(source.shape) - 1)
    frequency = offsets % 32
    cosine = gl.load(cos_sin_ptr + position * 64 + frequency).to(gl.float32)
    sine = gl.load(cos_sin_ptr + position * 64 + 32 + frequency).to(gl.float32)
    if len(source.shape) == 2:
        offsets = offsets[None, :]
        cosine = cosine[None, :]
        sine = sine[None, :]
    rotated = gl.where(
        offsets < 32,
        values * cosine - partner * sine,
        values * cosine + partner * sine,
    )
    return gl.where(offsets < 64, rotated, values).to(gl.bfloat16)


@gluon.jit
def _process_kv(
    kv_input_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    key_ptr,
    token,
    eps,
    QUERY_HEADS: gl.constexpr,
):
    position = gl.load(positions_ptr + token).to(gl.int64)
    layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
    offsets = gl.arange(0, 256, layout=layout)
    packed_row = kv_input_ptr + token * ((QUERY_HEADS + 1) * 512)
    source = gl.load(packed_row + offsets).to(gl.float32)
    weight = gl.load(weight_ptr + offsets).to(gl.float32)
    result = _normalize_and_rotate(source, weight, offsets, cos_sin_ptr, position, eps)
    gl.store(key_ptr + token * 256 + offsets, result)


@gluon.jit
def _process_queries(
    packed_ptr,
    weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    query_ptr,
    gate_ptr,
    token,
    group,
    eps,
    QUERY_HEADS: gl.constexpr,
    HEADS_PER_WAVE: gl.constexpr,
):
    position = gl.load(positions_ptr + token).to(gl.int64)
    if HEADS_PER_WAVE == 1:
        layout: gl.constexpr = gl.BlockedLayout([2], [64], [1], [0])
        offsets = gl.arange(0, 256, layout=layout)
        heads = group
        columns = offsets
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [1, 1], [1, 0])
        heads = group * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, layout))
        heads = heads[:, None]
        offsets = gl.arange(0, 256, layout=gl.SliceLayout(0, layout))
        columns = offsets[None, :]
    source_offset = token * ((QUERY_HEADS + 1) * 512) + heads * 512 + columns
    source = gl.load(packed_ptr + source_offset).to(gl.float32)
    weight = gl.load(weight_ptr + offsets).to(gl.float32)
    result = _normalize_and_rotate(source, weight, offsets, cos_sin_ptr, position, eps)
    gate = gl.load(packed_ptr + source_offset + 256)
    output_offset = (token * QUERY_HEADS + heads) * 256 + columns
    gl.store(query_ptr + output_offset, result)
    gl.store(gate_ptr + output_offset, gate)


@gluon.jit(repr=_attention_prologue_repr)
def _attention_prologue(
    packed_ptr,
    kv_input_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    query_ptr,
    gate_ptr,
    key_ptr,
    eps,
    QUERY_HEADS: gl.constexpr,
    HEADS_PER_WAVE: gl.constexpr,
):
    token = gl.program_id(0)
    grid_head = gl.program_id(1)
    # Schedule the longer KV work first; the branch is uniform across the CTA.
    if grid_head == 0:
        _process_kv(
            kv_input_ptr,
            k_weight_ptr,
            positions_ptr,
            cos_sin_ptr,
            key_ptr,
            token,
            eps,
            QUERY_HEADS,
        )
    else:
        _process_queries(
            packed_ptr,
            q_weight_ptr,
            cos_sin_ptr,
            positions_ptr,
            query_ptr,
            gate_ptr,
            token,
            grid_head - 1,
            eps,
            QUERY_HEADS,
            HEADS_PER_WAVE,
        )
