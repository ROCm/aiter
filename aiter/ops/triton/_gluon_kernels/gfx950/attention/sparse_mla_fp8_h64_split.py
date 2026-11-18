# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA decode for 64 heads, fp8 q and an fp8_scalar cache,
12 or 24 query rows. sparse_mla_fwd (attention/sparse_mla.py) launches it.

One split kernel per (row, head tile, key split): it gathers 256 selected rows
per tile into LDS, runs the QK and rope dots on the fp8 matrix core and the PV
dot in fp16 or bf16, and stores FP32 partial numerators and (max, sum) records.
One merge kernel per (row, head block, value block) combines the splits.
Per-row knobs come from the "_sparse_mla_fp8_h64" config.

Index, scale and output contracts are those of sparse_mla_fp8_h64_small.

Ported from the Artemis glm52_fp8_ep8 kernel pack (GLM-5.2 FP8 EP8 decode).
"""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _key_reduce(
    value,
    storage,
    HEAD_TILE: gl.constexpr,
    MAXIMUM: gl.constexpr,
    COMPACT_FOLD: gl.constexpr,
):
    """Row max (MAXIMUM) or row sum of a [HEAD_TILE, 256] score tile across the key waves."""
    KEY_WAVES: gl.constexpr = 256 // HEAD_TILE
    if HEAD_TILE == 32:
        local = gl.reshape(value, [HEAD_TILE, KEY_WAVES, 32])
        if MAXIMUM:
            partial = gl.max(local, 2)
        else:
            partial = gl.sum(local, 2)
    else:
        local = gl.reshape(value, [HEAD_TILE, 2, KEY_WAVES, 32])
        if MAXIMUM:
            partial = gl.max(gl.max(local, 3), 1)
        else:
            partial = gl.sum(gl.sum(local, 3), 1)
    shared = storage.slice(0 if MAXIMUM else KEY_WAVES, KEY_WAVES, dim=1)
    shared.store(partial)
    if HEAD_TILE == 32:
        if COMPACT_FOLD:
            fold_layout: gl.constexpr = gl.BlockedLayout(
                [1, 4], [32, 2], [1, 8], [0, 1]
            )
        else:
            fold_layout: gl.constexpr = gl.BlockedLayout(
                [1, 8], [32, 2], [1, 8], [0, 1]
            )
    else:
        fold_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2]],
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 0]],
            warp_bases=[[0, 0], [0, 0], [32, 0]],
            block_bases=[],
            shape=[64, 4],
        )
    partials = shared.load(fold_layout)
    if MAXIMUM:
        result = gl.max(partials, 1)
    else:
        result = gl.sum(partials, 1)
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[HEAD_TILE // 32, KEY_WAVES],
    )
    return gl.convert_layout(result, gl.SliceLayout(1, matrix))


@gluon.jit
def _softmax_exp(value):
    return gl.exp2(value * 1.4426950408889634)


@gluon.jit
def _stage_query(
    Query,
    row,
    head_block,
    HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    VALUE: gl.constexpr,
    load_layout: gl.constexpr,
    rope_load_layout: gl.constexpr,
    score_lhs: gl.constexpr,
):
    heads = head_block * 32 + gl.arange(0, 32, gl.SliceLayout(1, load_layout))
    channels = gl.arange(0, VALUE, gl.SliceLayout(0, load_layout))
    rope_heads = head_block * 32 + gl.arange(0, 32, gl.SliceLayout(1, rope_load_layout))
    rope_channels = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
    query_offsets = (row * HEADS + heads[:, None]) * WIDTH + channels[None, :]
    rope_offsets = (
        (row * HEADS + rope_heads[:, None]) * WIDTH + VALUE + rope_channels[None, :]
    )
    query_mask = gl.full((32, 1), True, gl.int1, load_layout)
    rope_mask = rope_channels[None, :] < WIDTH - VALUE
    query = gl.amd.cdna4.buffer_load(Query, query_offsets, query_mask, 0.0)
    rope = gl.amd.cdna4.buffer_load(Query, rope_offsets, rope_mask, 0.0)
    query_shared = gl.allocate_shared_memory(
        gl.float8e4nv,
        [32, VALUE],
        gl.PaddedSharedLayout.with_identity_for([[VALUE, 16]], [32, VALUE], [1, 0]),
    )
    query_shared.store(query.to(gl.float8e4nv))
    return (query_shared, gl.convert_layout(rope.to(gl.float8e4nv), score_lhs))


@gluon.jit
def _stage_persistent_query(
    Query,
    row,
    head_block,
    HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    VALUE: gl.constexpr,
    load_layout: gl.constexpr,
    rope_load_layout: gl.constexpr,
    score_lhs: gl.constexpr,
):
    heads = head_block * 32 + gl.arange(0, 32, gl.SliceLayout(1, load_layout))
    channels = gl.arange(0, 256, gl.SliceLayout(0, load_layout))
    offsets = (row * HEADS + heads[:, None]) * WIDTH + channels[None, :]
    mask = heads[:, None] < HEADS
    first = gl.amd.cdna4.buffer_load(Query, offsets, mask, 0.0).to(gl.float8e4nv)
    second = gl.amd.cdna4.buffer_load(Query, offsets + 256, mask, 0.0).to(gl.float8e4nv)
    first_shared = gl.allocate_shared_memory(
        gl.float8e4nv, [32, 256], gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    )
    second_shared = gl.allocate_shared_memory(
        gl.float8e4nv, [32, 256], gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    )
    first_shared.store(first)
    second_shared.store(second)
    rope_heads = head_block * 32 + gl.arange(0, 32, gl.SliceLayout(1, rope_load_layout))
    rope_channels = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
    rope = gl.amd.cdna4.buffer_load(
        Query,
        (row * HEADS + rope_heads[:, None]) * WIDTH + VALUE + rope_channels[None, :],
        (rope_heads[:, None] < HEADS) & (rope_channels[None, :] < WIDTH - VALUE),
        0.0,
    ).to(gl.float8e4nv)
    rope_shared = gl.allocate_shared_memory(
        gl.float8e4nv, [32, 64], gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    )
    rope_shared.store(rope)
    return (first_shared.load(score_lhs), second_shared, rope_shared.load(score_lhs))


@gluon.jit
def _wide_query_scores(
    Query,
    cache_shared,
    key_rope,
    row,
    HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    VALUE: gl.constexpr,
    load_layout: gl.constexpr,
    rope_load_layout: gl.constexpr,
    score_matrix: gl.constexpr,
    score_lhs: gl.constexpr,
    score_rhs: gl.constexpr,
):
    heads = gl.arange(0, 64, gl.SliceLayout(1, load_layout))
    channels = gl.arange(0, 256, gl.SliceLayout(0, load_layout))
    query_shared = gl.allocate_shared_memory(
        gl.float8e4nv, [64, 256], gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    )
    rope_heads = gl.arange(0, 64, gl.SliceLayout(1, rope_load_layout))
    rope_channels = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
    query_rope = gl.amd.cdna4.buffer_load(
        Query,
        (row * HEADS + rope_heads[:, None]) * WIDTH + VALUE + rope_channels[None, :],
        rope_channels[None, :] < WIDTH - VALUE,
        0.0,
    )
    rope_shared = gl.allocate_shared_memory(
        gl.float8e4nv, [64, 64], gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
    )
    rope_shared.store(query_rope.to(gl.float8e4nv))
    scores = gl.zeros((64, 256), gl.float32, score_matrix)
    for slab in gl.static_range(2):
        query = gl.amd.cdna4.buffer_load(
            Query,
            (row * HEADS + heads[:, None]) * WIDTH + slab * 256 + channels[None, :],
        )
        query_shared.store(query.to(gl.float8e4nv))
        operand = query_shared.load(score_lhs)
        keys = (
            cache_shared.slice(slab * 256, 256, dim=1).permute((1, 0)).load(score_rhs)
        )
        scores = gl.amd.cdna4.mfma_scaled(
            operand, None, "e4m3", keys, None, "e4m3", scores
        )
    rope_operand = rope_shared.load(score_lhs)
    return gl.amd.cdna4.mfma_scaled(
        rope_operand, None, "e4m3", key_rope, None, "e4m3", scores
    )


@gluon.jit
def _value_product_half(
    probability_shared, cache_shared, accumulator, START: gl.constexpr
):
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    for slab in gl.static_range(8):
        probability = probability_shared.slice(slab * 32, 32, dim=1).load(lhs)
        values = (
            cache_shared.slice(slab * 32, 32, dim=0)
            .slice(START, 256, dim=1)
            .load(rhs)
            .to(gl.bfloat16)
        )
        accumulator = gl.amd.cdna4.mfma(probability, values, accumulator)
    return accumulator


@gluon.jit
def _value_products(
    probability_shared,
    cache_shared,
    accumulator0,
    accumulator1,
    HEAD_TILE: gl.constexpr,
    FULL_KEYS: gl.constexpr,
    PV_BF16: gl.constexpr,
):
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[HEAD_TILE // 32, 256 // HEAD_TILE],
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    PV_DTYPE: gl.constexpr = gl.bfloat16 if PV_BF16 else gl.float16
    if FULL_KEYS:
        probability = probability_shared.load(lhs)
        values0 = cache_shared.slice(0, 256, dim=1).load(rhs).to(PV_DTYPE)
        accumulator0 = gl.amd.cdna4.mfma(probability, values0, accumulator0)
        values1 = cache_shared.slice(256, 256, dim=1).load(rhs).to(PV_DTYPE)
        accumulator1 = gl.amd.cdna4.mfma(probability, values1, accumulator1)
    else:
        for slab in gl.static_range(2):
            probability = probability_shared.slice(slab * 128, 128, dim=1).load(lhs)
            cache_slab = cache_shared.slice(slab * 128, 128, dim=0)
            values0 = cache_slab.slice(0, 256, dim=1).load(rhs).to(PV_DTYPE)
            accumulator0 = gl.amd.cdna4.mfma(probability, values0, accumulator0)
            values1 = cache_slab.slice(256, 256, dim=1).load(rhs).to(PV_DTYPE)
            accumulator1 = gl.amd.cdna4.mfma(probability, values1, accumulator1)
    return (accumulator0, accumulator1)


@gluon.jit
def _panel_value_products(
    probability_shared,
    cache0,
    cache1,
    accumulator0,
    accumulator1,
    PV_KEYS: gl.constexpr,
):
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    for slab in gl.static_range(256 // PV_KEYS):
        probability = probability_shared.slice(slab * PV_KEYS, PV_KEYS, dim=1).load(lhs)
        values0 = cache0.slice(slab * PV_KEYS, PV_KEYS, dim=0).load(rhs).to(gl.bfloat16)
        accumulator0 = gl.amd.cdna4.mfma(probability, values0, accumulator0)
        values1 = cache1.slice(slab * PV_KEYS, PV_KEYS, dim=0).load(rhs).to(gl.bfloat16)
        accumulator1 = gl.amd.cdna4.mfma(probability, values1, accumulator1)
    return (accumulator0, accumulator1)


@gluon.jit
def _prefetch_panel_value_products(
    probability_shared,
    cache0,
    cache1,
    accumulator0,
    accumulator1,
    PV_KEYS: gl.constexpr,
    Cache,
    Slots,
    count,
    split,
    WIDTH: gl.constexpr,
    TOPK: gl.constexpr,
    CACHE_BUFFER: gl.constexpr,
    load_layout: gl.constexpr,
    PREFETCH_AT: gl.constexpr,
):
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    for slab in gl.static_range(256 // PV_KEYS):
        if slab == PREFETCH_AT:
            channels = gl.arange(0, 256, gl.SliceLayout(0, load_layout))
            keys = (split * 2 + 1) * 256 + gl.arange(
                0, 256, gl.SliceLayout(1, load_layout)
            )
            slot = gl.load(Slots + keys, keys < count, -1)
            valid = slot >= 0
            if CACHE_BUFFER:
                base = gl.maximum(slot, 0) * WIDTH
                next_latent = gl.amd.cdna4.buffer_load(
                    Cache, base[:, None] + channels[None, :], valid[:, None], 0.0
                )
            else:
                base = gl.maximum(slot, 0).to(gl.int64) * WIDTH
                next_latent = gl.load(
                    Cache + base[:, None] + channels[None, :], valid[:, None], 0.0
                )
        probability = probability_shared.slice(slab * PV_KEYS, PV_KEYS, dim=1).load(lhs)
        values0 = cache0.slice(slab * PV_KEYS, PV_KEYS, dim=0).load(rhs).to(gl.bfloat16)
        accumulator0 = gl.amd.cdna4.mfma(probability, values0, accumulator0)
        values1 = cache1.slice(slab * PV_KEYS, PV_KEYS, dim=0).load(rhs).to(gl.bfloat16)
        accumulator1 = gl.amd.cdna4.mfma(probability, values1, accumulator1)
    cache0.store(next_latent)
    return (accumulator0, accumulator1)


@gluon.jit
def _write_partial(
    Partial,
    numerator,
    row,
    head_block,
    split,
    HEADS: gl.constexpr,
    SPLITS: gl.constexpr,
    START: gl.constexpr,
    HEAD_TILE: gl.constexpr,
    RECORD_VALUE: gl.constexpr,
):
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
    )
    local_heads = gl.arange(0, HEAD_TILE, gl.SliceLayout(1, matrix))
    channels = START + gl.arange(0, 256, gl.SliceLayout(0, matrix))
    record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
    groups = channels[None, :] // 8
    if HEAD_TILE == 64:
        groups = groups ^ split * 4
    offsets = (groups * HEAD_TILE + local_heads[:, None]) * 8 + channels[None, :] % 8
    gl.amd.cdna4.buffer_store(
        numerator.to(gl.float32),
        Partial + record * RECORD_VALUE * HEAD_TILE,
        offsets,
        cache=".cs",
    )


_attention_split_repr = make_kernel_repr(
    "_sparse_mla_fp8_h64_split_partials",
    ["SPLITS", "CACHE_BUFFER", "HEAD_TILE", "RECORD_VALUE", "FULL_PV_KEYS", "PV_BF16"],
)


@gluon.jit(repr=_attention_split_repr)
def _attention_split(
    Query,
    Cache,
    Slots,
    Indptr,
    Partial,
    Stats,
    scale,
    QScale,
    KScale,
    HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    VALUE: gl.constexpr,
    TOPK: gl.constexpr,
    SPLITS: gl.constexpr,
    CACHE_BUFFER: gl.constexpr,
    LDS_VECTOR: gl.constexpr,
    HEAD_TILE: gl.constexpr = 32,
    RECORD_VALUE: gl.constexpr = 512,
    EARLY_STATS: gl.constexpr = False,
    FULL_PV_KEYS: gl.constexpr = True,
    PACKED_STATS: gl.constexpr = False,
    BALANCED_GATHER: gl.constexpr = False,
    PAD_PROBABILITY: gl.constexpr = False,
    PV_BF16: gl.constexpr = False,
    EARLY_STATS_PLAIN: gl.constexpr = False,
):
    row, head_block, split = (gl.program_id(0), gl.program_id(1), gl.program_id(2))
    start = gl.load(Indptr + row)
    count = gl.load(Indptr + row + 1) - start
    Slots = Slots + start
    scale = scale * gl.load(QScale) * gl.load(KScale)
    TILES: gl.constexpr = triton.cdiv(TOPK, SPLITS * 256)
    STATS_PITCH: gl.constexpr = (
        RECORD_VALUE * HEAD_TILE // 2 if PACKED_STATS else 2 * HEAD_TILE
    )
    PV_DTYPE: gl.constexpr = gl.bfloat16 if PV_BF16 else gl.float16
    if BALANCED_GATHER:
        load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 2], [1, 0])
    else:
        load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [8, 1], [1, 0])
    rope_load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [8, 1], [1, 0])
    matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16],
        transposed=True,
        warps_per_cta=[HEAD_TILE // 32, 256 // HEAD_TILE],
    )
    score_matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=True,
        warps_per_cta=[HEAD_TILE // 32, 256 // HEAD_TILE],
    )
    score_lhs: gl.constexpr = gl.DotOperandLayout(0, score_matrix, 32)
    score_rhs: gl.constexpr = gl.DotOperandLayout(1, score_matrix, 32)
    if PAD_PROBABILITY or (HEAD_TILE == 32 and TILES == 1):
        probability_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[256, 8]], [HEAD_TILE, 256], [1, 0]
        )
    else:
        probability_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
    channels = gl.arange(0, 256 if TILES > 1 else VALUE, gl.SliceLayout(0, load_layout))
    direct_rope_layout: gl.constexpr = gl.BlockedLayout(
        [32, 1], [2, 32], [HEAD_TILE // 32, 256 // HEAD_TILE], [1, 0]
    )
    rope_channels = gl.arange(0, 64, gl.SliceLayout(1, direct_rope_layout))
    key_offsets = gl.arange(0, 256, gl.SliceLayout(1, load_layout))
    if TILES > 1:
        cache0 = gl.allocate_shared_memory(
            Cache.dtype.element_ty,
            [256, 256],
            gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0]),
        )
        cache1 = gl.allocate_shared_memory(
            Cache.dtype.element_ty,
            [256, 256],
            gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0]),
        )
    elif HEAD_TILE == 32:
        cache_shared = gl.allocate_shared_memory(
            Cache.dtype.element_ty,
            [256, VALUE],
            gl.PaddedSharedLayout.with_identity_for(
                [[VALUE, 16]], [256, VALUE], [1, 0]
            ),
        )
    else:
        cache_shared = gl.allocate_shared_memory(
            Cache.dtype.element_ty,
            [256, VALUE],
            gl.SwizzledSharedLayout(LDS_VECTOR, 1, 16, [1, 0]),
        )
    reduction_shared = gl.allocate_shared_memory(
        gl.float32,
        [HEAD_TILE, 2 * (256 // HEAD_TILE)],
        gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
    )
    maximum = gl.full(
        (HEAD_TILE,), -float("inf"), gl.float32, gl.SliceLayout(1, matrix)
    )
    denominator = gl.full((HEAD_TILE,), 0, gl.float32, gl.SliceLayout(1, matrix))
    value_matrix: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
    )
    numerator0 = gl.zeros((HEAD_TILE, 256), gl.float32, value_matrix)
    numerator1 = gl.zeros((HEAD_TILE, 256), gl.float32, value_matrix)
    for tile in gl.static_range(TILES):
        indices = (split * TILES + tile) * 256 + key_offsets
        slot = gl.load(Slots + indices, indices < count, -1)
        valid = slot >= 0
        if CACHE_BUFFER:
            base = gl.maximum(slot, 0) * WIDTH
        else:
            base = gl.maximum(slot, 0).to(gl.int64) * WIDTH
        if TILES == 1 or tile == 0:
            if CACHE_BUFFER:
                latent = gl.amd.cdna4.buffer_load(
                    Cache, base[:, None] + channels[None, :], valid[:, None], 0.0
                )
            else:
                latent = gl.load(
                    Cache + base[:, None] + channels[None, :], valid[:, None], 0.0
                )
        if TILES > 1:
            if tile == 0:
                cache0.store(latent)
            if CACHE_BUFFER:
                latent1 = gl.amd.cdna4.buffer_load(
                    Cache, base[:, None] + 256 + channels[None, :], valid[:, None], 0.0
                )
            else:
                latent1 = gl.load(
                    Cache + base[:, None] + 256 + channels[None, :], valid[:, None], 0.0
                )
            cache1.store(latent1)
        else:
            cache_shared.store(latent)
        rope_keys = (split * TILES + tile) * 256 + gl.arange(
            0, 256, gl.SliceLayout(0, direct_rope_layout)
        )
        rope_slots = gl.load(
            Slots + rope_keys, (rope_keys < TOPK) & (rope_keys < count), -1
        )
        rope_valid = rope_slots >= 0
        if CACHE_BUFFER:
            rope_base = gl.maximum(rope_slots, 0) * WIDTH
        else:
            rope_base = gl.maximum(rope_slots, 0).to(gl.int64) * WIDTH
        rope_offsets = rope_base[None, :] + VALUE + rope_channels[:, None]
        rope_mask = rope_valid[None, :] & (rope_channels[:, None] < WIDTH - VALUE)
        if CACHE_BUFFER:
            rope = gl.amd.cdna4.buffer_load(Cache, rope_offsets, rope_mask, 0.0)
        else:
            rope = gl.load(Cache + rope_offsets, rope_mask, 0.0)
        if HEAD_TILE == 64:
            key_rope = gl.convert_layout(rope, score_rhs)
            scores = _wide_query_scores(
                Query,
                cache_shared,
                key_rope,
                row,
                HEADS,
                WIDTH,
                VALUE,
                load_layout,
                rope_load_layout,
                score_matrix,
                score_lhs,
                score_rhs,
            )
        else:
            if tile == 0:
                if TILES > 1:
                    persistent_query, query_storage, query_rope = (
                        _stage_persistent_query(
                            Query,
                            row,
                            head_block,
                            HEADS,
                            WIDTH,
                            VALUE,
                            load_layout,
                            rope_load_layout,
                            score_lhs,
                        )
                    )
                else:
                    query_storage, query_rope = _stage_query(
                        Query,
                        row,
                        head_block,
                        HEADS,
                        WIDTH,
                        VALUE,
                        load_layout,
                        rope_load_layout,
                        score_lhs,
                    )
            if TILES > 1:
                key_rope = gl.convert_layout(rope, score_rhs)
                scores = gl.zeros((HEAD_TILE, 256), gl.float32, score_matrix)
                key0 = cache0.permute((1, 0)).load(score_rhs)
                scores = gl.amd.cdna4.mfma_scaled(
                    persistent_query, None, "e4m3", key0, None, "e4m3", scores
                )
                query1 = query_storage.load(score_lhs)
                key1 = cache1.permute((1, 0)).load(score_rhs)
                scores = gl.amd.cdna4.mfma_scaled(
                    query1, None, "e4m3", key1, None, "e4m3", scores
                )
            else:
                key = cache_shared.permute((1, 0)).load(score_rhs)
                key_rope = gl.convert_layout(rope, score_rhs)
                query = query_storage.load(score_lhs)
                scores = gl.zeros((HEAD_TILE, 256), gl.float32, score_matrix)
                scores = gl.amd.cdna4.mfma_scaled(
                    query, None, "e4m3", key, None, "e4m3", scores
                )
            scores = gl.amd.cdna4.mfma_scaled(
                query_rope, None, "e4m3", key_rope, None, "e4m3", scores
            )
        scores = gl.convert_layout(scores, matrix) * scale
        score_keys = (split * TILES + tile) * 256 + gl.arange(
            0, 256, gl.SliceLayout(0, matrix)
        )
        score_slots = gl.load(Slots + score_keys, score_keys < count, -1)
        valid_scores = score_slots >= 0
        scores = gl.where(valid_scores[None, :], scores, -float("inf"))
        tile_max = _key_reduce(
            scores,
            reduction_shared,
            HEAD_TILE,
            True,
            HEAD_TILE == 32 and (TILES > 1 or not BALANCED_GATHER),
        )
        if tile == 0:
            next_max = tile_max
        else:
            next_max = gl.maximum(maximum, tile_max)
        safe_max = gl.where(next_max == -float("inf"), 0, next_max)
        probability = gl.where(
            valid_scores[None, :], _softmax_exp(scores - safe_max[:, None]), 0
        )
        if HEAD_TILE == 32:
            probability_shared = gl.allocate_shared_memory(
                PV_DTYPE, [HEAD_TILE, 256], probability_layout
            )
            probability_shared.store(probability.to(PV_DTYPE))
        tile_sum = _key_reduce(
            probability,
            reduction_shared,
            HEAD_TILE,
            False,
            HEAD_TILE == 32 and (TILES > 1 or not BALANCED_GATHER),
        )
        if tile == 0:
            denominator = tile_sum
        else:
            correction = _softmax_exp(maximum - safe_max)
            denominator = denominator * correction + tile_sum
            numerator0 = numerator0 * correction[:, None]
            numerator1 = numerator1 * correction[:, None]
        maximum = next_max
        if EARLY_STATS and tile == TILES - 1:
            stats_heads = head_block * HEAD_TILE + gl.arange(
                0, HEAD_TILE, gl.SliceLayout(1, matrix)
            )
            stats_record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
            stats_base_early = Stats + stats_record * STATS_PITCH
            stats_mask_early = gl.full(
                (HEAD_TILE,), True, gl.int1, gl.SliceLayout(1, matrix)
            )
            if EARLY_STATS_PLAIN:
                gl.store(
                    stats_base_early + stats_heads % HEAD_TILE,
                    maximum,
                    stats_mask_early,
                )
                gl.store(
                    stats_base_early + stats_heads % HEAD_TILE + HEAD_TILE,
                    denominator,
                    stats_mask_early,
                )
            else:
                gl.amd.cdna4.buffer_store(
                    maximum, stats_base_early, stats_heads % HEAD_TILE, stats_mask_early
                )
                gl.amd.cdna4.buffer_store(
                    denominator,
                    stats_base_early,
                    stats_heads % HEAD_TILE + HEAD_TILE,
                    stats_mask_early,
                )
        if HEAD_TILE != 32:
            probability_shared = gl.allocate_shared_memory(
                gl.bfloat16, [HEAD_TILE, 256], gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
            )
            probability_shared.store(probability.to(gl.bfloat16))
        if TILES == 1 and HEAD_TILE == 64:
            numerator0 = _value_product_half(
                probability_shared, cache_shared, numerator0, 0
            )
            _write_partial(
                Partial,
                numerator0,
                row,
                head_block,
                split,
                HEADS,
                SPLITS,
                0,
                HEAD_TILE,
                RECORD_VALUE,
            )
            numerator1 = _value_product_half(
                probability_shared, cache_shared, numerator1, 256
            )
            _write_partial(
                Partial,
                numerator1,
                row,
                head_block,
                split,
                HEADS,
                SPLITS,
                256,
                HEAD_TILE,
                RECORD_VALUE,
            )
        elif TILES > 1:
            if tile == 0:
                numerator0, numerator1 = _prefetch_panel_value_products(
                    probability_shared,
                    cache0,
                    cache1,
                    numerator0,
                    numerator1,
                    64,
                    Cache,
                    Slots,
                    count,
                    split,
                    WIDTH,
                    TOPK,
                    CACHE_BUFFER,
                    load_layout,
                    2,
                )
            else:
                numerator0, numerator1 = _panel_value_products(
                    probability_shared, cache0, cache1, numerator0, numerator1, 64
                )
        else:
            numerator0, numerator1 = _value_products(
                probability_shared,
                cache_shared,
                numerator0,
                numerator1,
                HEAD_TILE,
                TILES == 1 and HEAD_TILE == 32 and FULL_PV_KEYS,
                PV_BF16,
            )
    heads = head_block * HEAD_TILE + gl.arange(0, HEAD_TILE, gl.SliceLayout(1, matrix))
    record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
    if TILES > 1 or HEAD_TILE == 32:
        _write_partial(
            Partial,
            numerator0,
            row,
            head_block,
            split,
            HEADS,
            SPLITS,
            0,
            HEAD_TILE,
            RECORD_VALUE,
        )
        _write_partial(
            Partial,
            numerator1,
            row,
            head_block,
            split,
            HEADS,
            SPLITS,
            256,
            HEAD_TILE,
            RECORD_VALUE,
        )
    if not EARLY_STATS:
        stats_mask = heads < HEADS
        if TILES == 1:
            stats_mask = gl.full((HEAD_TILE,), True, gl.int1, gl.SliceLayout(1, matrix))
        stats_base = Stats + record * STATS_PITCH
        stats_offsets = heads % HEAD_TILE
        gl.amd.cdna4.buffer_store(maximum, stats_base, stats_offsets, stats_mask)
        gl.amd.cdna4.buffer_store(
            denominator, stats_base, stats_offsets + HEAD_TILE, stats_mask
        )


_attention_merge_repr = make_kernel_repr(
    "_sparse_mla_fp8_h64_split_merge",
    [
        "SPLITS",
        "BC",
        "RECORD_HEADS",
        "RECORD_VALUE",
        "MERGE_HEADS",
        "NORMALIZE_WEIGHTS",
    ],
)


@gluon.jit(repr=_attention_merge_repr)
def _attention_merge(
    Partial,
    Stats,
    Output,
    KScale,
    out_stride0,
    out_stride1,
    HEADS: gl.constexpr,
    VALUE: gl.constexpr,
    SPLITS: gl.constexpr,
    BC: gl.constexpr,
    BUFFER_PARTIAL: gl.constexpr,
    RECORD_HEADS: gl.constexpr = 32,
    RECORD_VALUE: gl.constexpr = 512,
    MERGE_HEADS: gl.constexpr = 32,
    CACHE_POLICY: gl.constexpr = "",
    PACKED_STATS: gl.constexpr = False,
    BUFFER_OUTPUT: gl.constexpr = False,
    NORMALIZE_WEIGHTS: gl.constexpr = False,
    LOCAL_HEAD_INDEX: gl.constexpr = False,
):
    row, head_block, channel_block = (
        gl.program_id(0),
        gl.program_id(1),
        gl.program_id(2),
    )
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8],
        [1, 32, 2],
        [1, MERGE_HEADS // 32, gl.num_warps() * 32 // MERGE_HEADS],
        [1, 2, 0],
    )
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    splits = gl.arange(0, SPLITS, gl.SliceLayout(1, stat_layout))
    if LOCAL_HEAD_INDEX:
        local_heads = gl.arange(0, MERGE_HEADS, gl.SliceLayout(0, stat_layout))
    else:
        heads = head_block * MERGE_HEADS + gl.arange(
            0, MERGE_HEADS, gl.SliceLayout(0, stat_layout)
        )
    channels = channel_block * BC + gl.arange(
        0, BC, gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    record_base = (
        row * (HEADS // RECORD_HEADS) + head_block * MERGE_HEADS // RECORD_HEADS
    ) * SPLITS
    STATS_PITCH: gl.constexpr = (
        RECORD_VALUE * RECORD_HEADS // 2 if PACKED_STATS else 2 * RECORD_HEADS
    )
    stats_base = Stats + record_base * STATS_PITCH
    partial_base = Partial + record_base * RECORD_VALUE * RECORD_HEADS
    if LOCAL_HEAD_INDEX:
        heads = head_block * MERGE_HEADS + local_heads
        head_record = heads % MERGE_HEADS // RECORD_HEADS
        head_in_record = heads % RECORD_HEADS
        stat_index = (
            head_record[None, :] * SPLITS * STATS_PITCH
            + splits[:, None] * STATS_PITCH
            + head_in_record[None, :]
        )
    else:
        head_record = heads % MERGE_HEADS // RECORD_HEADS
        stat_index = (
            head_record[None, :] * SPLITS * STATS_PITCH
            + splits[:, None] * STATS_PITCH
            + heads[None, :] % RECORD_HEADS
        )
    maximums = gl.load(stats_base + stat_index)
    denominators = gl.load(stats_base + stat_index + RECORD_HEADS)
    maximum = gl.max(maximums, 0)
    safe_max = gl.where(maximum == -float("inf"), 0, maximum)
    correction = _softmax_exp(maximums - safe_max[None, :])
    denominator = gl.sum(denominators * correction, 0)
    value_scale = gl.load(KScale)
    if NORMALIZE_WEIGHTS:
        inverse_denominator = value_scale / gl.where(denominator > 0, denominator, 1)
        correction = correction * inverse_denominator[None, :]
    groups = channels[None, None, :] // 8
    if RECORD_HEADS == 64:
        groups = groups ^ splits[:, None, None] * 4
    if LOCAL_HEAD_INDEX:
        offsets = (
            head_record[None, :, None] * SPLITS * RECORD_VALUE * RECORD_HEADS
            + splits[:, None, None] * RECORD_VALUE * RECORD_HEADS
            + (groups * RECORD_HEADS + head_in_record[None, :, None]) * 8
            + channels[None, None, :] % 8
        )
    else:
        offsets = (
            head_record[None, :, None] * SPLITS * RECORD_VALUE * RECORD_HEADS
            + splits[:, None, None] * RECORD_VALUE * RECORD_HEADS
            + (groups * RECORD_HEADS + heads[None, :, None] % RECORD_HEADS) * 8
            + channels[None, None, :] % 8
        )
    if BUFFER_PARTIAL:
        partials = gl.amd.cdna4.buffer_load(
            partial_base, offsets, cache=CACHE_POLICY
        ).to(gl.float32)
    else:
        partials = gl.load(partial_base + offsets, cache_modifier=CACHE_POLICY).to(
            gl.float32
        )
    numerator = gl.sum(partials * correction[:, :, None], 0)
    if NORMALIZE_WEIGHTS:
        result = numerator
    else:
        inverse_denominator = value_scale / gl.where(denominator > 0, denominator, 1)
        norm = gl.convert_layout(
            inverse_denominator, gl.SliceLayout(1, gl.SliceLayout(0, layout))
        )
        result = numerator * norm[:, None]
    output_layout: gl.constexpr = gl.SliceLayout(0, layout)
    out_heads = head_block * MERGE_HEADS + gl.arange(
        0, MERGE_HEADS, gl.SliceLayout(1, output_layout)
    )
    out_channels = channel_block * BC + gl.arange(
        0, BC, gl.SliceLayout(0, output_layout)
    )
    output_offsets = out_heads[:, None] * out_stride1 + out_channels[None, :]
    if BUFFER_OUTPUT:
        gl.amd.cdna4.buffer_store(
            result.to(Output.dtype.element_ty),
            Output + row * out_stride0,
            output_offsets,
            cache=".wt",
        )
    else:
        gl.store(
            Output + row * out_stride0 + output_offsets,
            result.to(Output.dtype.element_ty),
        )
