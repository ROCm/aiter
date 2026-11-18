# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA decode for 64 heads, fp8 q and an fp8_scalar cache,
16 or 32 query rows. sparse_mla_fwd (attention/sparse_mla.py) launches it.

The split kernel follows sparse_mla_fp8_h64_split, with two other PV forms: a
128-key wide PV (16 rows) and two 256-column cache panels whose second tile is
gathered during the first tile's PV (32 rows). Per-row knobs come from the
"_sparse_mla_fp8_h64" config.

Index, scale and output contracts are those of sparse_mla_fp8_h64_small.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla_fp8_h64_split import (
    _key_reduce,
    _softmax_exp,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.constexpr_function
def _matrix_layout(head_tile, instruction_k=16):
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, instruction_k],
        transposed=True,
        warps_per_cta=[head_tile // 32, 256 // head_tile],
    )


@gluon.jit
def _gather_cache(
    Cache,
    slots,
    channels,
    mask,
    WIDTH: gl.constexpr,
    CACHE_BUFFER: gl.constexpr,
    NATIVE_ZERO: gl.constexpr = False,
    CLAMP_SLOT: gl.constexpr = True,
):
    if CACHE_BUFFER:
        if CLAMP_SLOT:
            offsets = gl.maximum(slots, 0) * WIDTH + channels
        else:
            offsets = slots * WIDTH + channels
        if NATIVE_ZERO:
            return gl.amd.cdna4.buffer_load(Cache, offsets, mask)
        return gl.amd.cdna4.buffer_load(Cache, offsets, mask, 0.0)
    offsets = gl.maximum(slots, 0).to(gl.int64) * WIDTH + channels
    return gl.load(Cache + offsets, mask, 0.0)


@gluon.jit
def _fp8_tile_storage(
    rows: gl.constexpr,
    channels: gl.constexpr,
    dtype: gl.constexpr = gl.float8e4nv,
    PAD: gl.constexpr = 0,
):
    if PAD:
        return gl.allocate_shared_memory(
            dtype,
            [rows, channels],
            gl.PaddedSharedLayout.with_identity_for(
                [[channels, PAD]], [rows, channels], [1, 0]
            ),
        )
    else:
        return gl.allocate_shared_memory(
            dtype, [rows, channels], gl.SwizzledSharedLayout(16, 1, 16, [1, 0])
        )


@gluon.jit
def _stage_query(
    Query,
    row,
    head_block,
    HEADS: gl.constexpr,
    WIDTH: gl.constexpr,
    VALUE: gl.constexpr,
    TILES: gl.constexpr,
    load_layout: gl.constexpr,
    rope_load_layout: gl.constexpr,
    score_lhs: gl.constexpr,
    QUERY_PAD: gl.constexpr = 0,
):
    heads = head_block * 32 + gl.arange(0, 32, gl.SliceLayout(1, load_layout))
    CHANNELS: gl.constexpr = 256 if TILES == 2 else VALUE
    channels = gl.arange(0, CHANNELS, gl.SliceLayout(0, load_layout))
    if TILES == 2:
        first = gl.amd.cdna4.buffer_load(
            Query, (row * HEADS + heads[:, None]) * WIDTH + channels[None, :]
        )
        second = gl.amd.cdna4.buffer_load(
            Query, (row * HEADS + heads[:, None]) * WIDTH + 256 + channels[None, :]
        )
        first_shared = _fp8_tile_storage(32, 256)
        second_shared = _fp8_tile_storage(32, 256)
        first_shared.store(first.to(gl.float8e4nv))
        second_shared.store(second.to(gl.float8e4nv))
        rope_heads = head_block * 32 + gl.arange(
            0, 32, gl.SliceLayout(1, rope_load_layout)
        )
        rope_channels = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
        rope = gl.amd.cdna4.buffer_load(
            Query,
            (row * HEADS + rope_heads[:, None]) * WIDTH + 512 + rope_channels[None, :],
        )
        rope_shared = _fp8_tile_storage(32, 64)
        rope_shared.store(rope.to(gl.float8e4nv))
        return (
            first_shared.load(score_lhs),
            second_shared,
            rope_shared.load(score_lhs),
        )
    else:
        rope_heads = head_block * 32 + gl.arange(
            0, 32, gl.SliceLayout(1, rope_load_layout)
        )
        rope_channels = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
        query_offsets = (row * HEADS + heads[:, None]) * WIDTH + channels[None, :]
        rope_offsets = (
            (row * HEADS + rope_heads[:, None]) * WIDTH + VALUE + rope_channels[None, :]
        )
        rope_mask = rope_channels[None, :] < WIDTH - VALUE
        query = gl.amd.cdna4.buffer_load(Query, query_offsets)
        rope = gl.amd.cdna4.buffer_load(Query, rope_offsets, rope_mask, 0.0)
        query_shared = _fp8_tile_storage(32, VALUE, PAD=QUERY_PAD)
        query_shared.store(query.to(gl.float8e4nv))
        return (
            query_shared,
            None,
            gl.convert_layout(rope.to(gl.float8e4nv), score_lhs),
        )


@gluon.jit
def _fp8_qk(query, key, scores):
    return gl.amd.cdna4.mfma_scaled(query, None, "e4m3", key, None, "e4m3", scores)


@gluon.jit
def _narrow_query_scores(query_first, query_rope, cache_shared, key_rope):
    matrix: gl.constexpr = _matrix_layout(32, 64)
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 32)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 32)
    scores = gl.zeros((32, 256), gl.float32, matrix)
    key = cache_shared.permute((1, 0)).load(rhs)
    query = query_first.load(lhs)
    scores = _fp8_qk(query, key, scores)
    return _fp8_qk(query_rope, key_rope, scores)


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
    query_shared = _fp8_tile_storage(64, 256)
    rope_heads = gl.arange(0, 64, gl.SliceLayout(1, rope_load_layout))
    rope_channels = gl.arange(0, 64, gl.SliceLayout(0, rope_load_layout))
    query_rope = gl.amd.cdna4.buffer_load(
        Query,
        (row * HEADS + rope_heads[:, None]) * WIDTH + VALUE + rope_channels[None, :],
        rope_channels[None, :] < WIDTH - VALUE,
        0.0,
    )
    rope_shared = _fp8_tile_storage(64, 64)
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
        scores = _fp8_qk(operand, keys, scores)
    rope_operand = rope_shared.load(score_lhs)
    return _fp8_qk(rope_operand, key_rope, scores)


@gluon.jit
def _value_products(
    probability_shared,
    cache_shared,
    accumulator0,
    accumulator1,
    HEAD_TILE: gl.constexpr,
    PV_MODE: gl.constexpr,
    HALF: gl.constexpr = 0,
    PV_DTYPE: gl.constexpr = gl.float16,
):
    if PV_MODE == "serial32":
        matrix: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
        )
    else:
        matrix: gl.constexpr = _matrix_layout(HEAD_TILE)
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    if PV_MODE != "wide128":
        accumulator0 = gl.zeros((HEAD_TILE, 256), gl.float32, matrix)
    KEY_SLAB: gl.constexpr = 32 if PV_MODE == "serial32" else 128
    for slab in gl.static_range(256 // KEY_SLAB):
        probability = probability_shared.slice(slab * KEY_SLAB, KEY_SLAB, dim=1).load(
            lhs
        )
        values = cache_shared.slice(slab * KEY_SLAB, KEY_SLAB, dim=0)
        if PV_MODE == "wide128":
            values0 = values.load(rhs).to(PV_DTYPE)
        else:
            values0 = values.slice(HALF * 256, 256, dim=1).load(rhs).to(PV_DTYPE)
        accumulator0 = gl.amd.cdna4.mfma(probability, values0, accumulator0)
    return (accumulator0, accumulator1)


@gluon.jit
def _panel_scores(query_first, query_second, query_rope, cache0, cache1, key_rope):
    matrix: gl.constexpr = _matrix_layout(32, 64)
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 32)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 32)
    scores = gl.zeros((32, 256), gl.float32, matrix)
    scores = _fp8_qk(query_first, cache0.permute((1, 0)).load(rhs), scores)
    scores = _fp8_qk(query_second.load(lhs), cache1.permute((1, 0)).load(rhs), scores)
    return _fp8_qk(query_rope, key_rope, scores)


@gluon.jit
def _panel_products(
    probability_shared,
    cache0,
    cache1,
    numerator0,
    numerator1,
    PV_DTYPE: gl.constexpr = gl.float16,
):
    matrix: gl.constexpr = _matrix_layout(32)
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    for slab in gl.static_range(4):
        probability = probability_shared.slice(slab * 64, 64, dim=1).load(lhs)
        values0 = cache0.slice(slab * 64, 64, dim=0).load(rhs).to(PV_DTYPE)
        numerator0 = gl.amd.cdna4.mfma(probability, values0, numerator0)
        values1 = cache1.slice(slab * 64, 64, dim=0).load(rhs).to(PV_DTYPE)
        numerator1 = gl.amd.cdna4.mfma(probability, values1, numerator1)
    return (numerator0, numerator1)


@gluon.jit
def _panel_products_refill(
    probability_shared,
    cache0,
    cache1,
    numerator0,
    numerator1,
    Cache,
    Slots,
    count,
    split,
    TOPK: gl.constexpr,
    WIDTH: gl.constexpr,
    CACHE_BUFFER: gl.constexpr,
    PV_DTYPE: gl.constexpr,
):
    matrix: gl.constexpr = _matrix_layout(32)
    lhs: gl.constexpr = gl.DotOperandLayout(0, matrix, 8)
    rhs: gl.constexpr = gl.DotOperandLayout(1, matrix, 8)
    gather_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [8, 1], [1, 0])
    keys = gl.arange(0, 64, gl.SliceLayout(1, gather_layout))
    channels = gl.arange(0, 256, gl.SliceLayout(0, gather_layout))
    for slab in gl.static_range(4):
        probability = probability_shared.slice(slab * 64, 64, dim=1).load(lhs)
        values0 = cache0.slice(slab * 64, 64, dim=0).load(rhs).to(PV_DTYPE)
        values1 = cache1.slice(slab * 64, 64, dim=0).load(rhs).to(PV_DTYPE)
        refill_keys = split * 512 + 256 + slab * 64 + keys
        slots = gl.load(Slots + refill_keys, refill_keys < count, -1)
        valid = slots >= 0
        next0 = _gather_cache(
            Cache,
            slots[:, None],
            channels[None, :],
            valid[:, None],
            WIDTH,
            CACHE_BUFFER,
            True,
        )
        next1 = _gather_cache(
            Cache,
            slots[:, None],
            256 + channels[None, :],
            valid[:, None],
            WIDTH,
            CACHE_BUFFER,
            True,
        )
        numerator0 = gl.amd.cdna4.mfma(probability, values0, numerator0)
        numerator1 = gl.amd.cdna4.mfma(probability, values1, numerator1)
        gl.barrier()
        cache0.slice(slab * 64, 64, dim=0).store(next0)
        cache1.slice(slab * 64, 64, dim=0).store(next1)
    return (numerator0, numerator1)


@gluon.jit
def _stage_probability(
    probability,
    HEAD_TILE: gl.constexpr,
    PADDED: gl.constexpr,
    PV_DTYPE: gl.constexpr = gl.float16,
):
    if PADDED:
        storage = gl.allocate_shared_memory(
            PV_DTYPE,
            [HEAD_TILE, 256],
            gl.PaddedSharedLayout.with_identity_for(
                [[256, 8]], [HEAD_TILE, 256], [1, 0]
            ),
        )
    else:
        storage = gl.allocate_shared_memory(
            PV_DTYPE, [HEAD_TILE, 256], gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
        )
    storage.store(probability.to(PV_DTYPE))
    return storage


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
    RECORD_WIDTH: gl.constexpr,
    COLOR_STEP: gl.constexpr,
    VALUE_AXIS: gl.constexpr = False,
):
    if VALUE_AXIS:
        matrix: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[32, 32, 16], transposed=True, warps_per_cta=[1, 8]
        )
    else:
        matrix: gl.constexpr = _matrix_layout(HEAD_TILE)
    local_heads = gl.arange(0, HEAD_TILE, gl.SliceLayout(1, matrix))
    channels = START + gl.arange(0, numerator.shape[1], gl.SliceLayout(0, matrix))
    record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
    groups = channels // 8
    if COLOR_STEP:
        groups = groups ^ split * COLOR_STEP
    offsets = (groups[None, :] * HEAD_TILE + local_heads[:, None]) * 8 + channels[
        None, :
    ] % 8
    gl.amd.cdna4.buffer_store(
        numerator.to(gl.float32),
        Partial + record * RECORD_WIDTH * HEAD_TILE,
        offsets,
        cache=".cs",
    )


_attention_split_repr = make_kernel_repr(
    "_sparse_mla_fp8_h64_panel_partials",
    ["SPLITS", "CACHE_BUFFER", "HEAD_TILE", "RECORD_WIDTH", "PV_MODE", "REFILL"],
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
    HEAD_TILE: gl.constexpr,
    RECORD_WIDTH: gl.constexpr,
    PV_MODE: gl.constexpr,
    COLOR_STEP: gl.constexpr,
    PACK_STATS: gl.constexpr = False,
    PAD_PROB: gl.constexpr = False,
    PAD_QUERY_CACHE: gl.constexpr = False,
    COMPACT_REDUCE: gl.constexpr = False,
    PV_BF16: gl.constexpr = False,
    BALANCED_GATHER: gl.constexpr = False,
    REFILL: gl.constexpr = False,
):
    row, head_block, split = (gl.program_id(0), gl.program_id(1), gl.program_id(2))
    start = gl.load(Indptr + row)
    count = gl.load(Indptr + row + 1) - start
    Slots = Slots + start
    scale = scale * gl.load(QScale) * gl.load(KScale)
    if PACK_STATS:
        Stats = Partial.to(gl.pointer_type(gl.float32))
    gl.static_assert(TOPK % (SPLITS * 256) == 0)
    TILES: gl.constexpr = TOPK // (SPLITS * 256)
    gl.static_assert(
        PV_MODE == "serial"
        or PV_MODE == "serial32"
        or PV_MODE == "panels64"
        or (PV_MODE == "wide128")
    )
    PANEL_CACHE: gl.constexpr = PV_MODE == "panels64"
    gl.static_assert(not REFILL or PANEL_CACHE)
    gl.static_assert(TILES == 1 or PANEL_CACHE)
    gl.static_assert(not PACK_STATS or (TILES == 1 and RECORD_WIDTH >= VALUE + 4))
    gl.static_assert(RECORD_WIDTH >= VALUE)
    gl.static_assert(RECORD_WIDTH * HEAD_TILE % 2 == 0)
    gl.static_assert(
        not PACK_STATS or (RECORD_WIDTH - VALUE) * HEAD_TILE >= 4 * HEAD_TILE
    )
    gl.static_assert(not PAD_QUERY_CACHE or (TILES == 1 and HEAD_TILE == 32))
    STAGING_PAD: gl.constexpr = 16 if PAD_QUERY_CACHE else 0
    PV_DTYPE: gl.constexpr = gl.bfloat16 if PV_BF16 else gl.float16
    if BALANCED_GATHER:
        load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [8, 8], [4, 2], [1, 0])
    else:
        load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [4, 16], [8, 1], [1, 0])
    rope_load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [8, 1], [1, 0])
    matrix: gl.constexpr = _matrix_layout(HEAD_TILE)
    score_matrix: gl.constexpr = _matrix_layout(HEAD_TILE, 64)
    score_lhs: gl.constexpr = gl.DotOperandLayout(0, score_matrix, 32)
    score_rhs: gl.constexpr = gl.DotOperandLayout(1, score_matrix, 32)
    channels = gl.arange(0, VALUE, gl.SliceLayout(0, load_layout))
    direct_rope_layout: gl.constexpr = gl.BlockedLayout(
        [32, 1], [2, 32], [HEAD_TILE // 32, 256 // HEAD_TILE], [1, 0]
    )
    rope_channels = gl.arange(0, 64, gl.SliceLayout(1, direct_rope_layout))
    key_offsets = gl.arange(0, 256, gl.SliceLayout(1, load_layout))
    if PANEL_CACHE:
        gl.static_assert(TILES == 2 and HEAD_TILE == 32)
        cache0 = _fp8_tile_storage(256, 256, Cache.dtype.element_ty)
        cache1 = _fp8_tile_storage(256, 256, Cache.dtype.element_ty)
    else:
        cache_shared = _fp8_tile_storage(
            256, VALUE, Cache.dtype.element_ty, STAGING_PAD
        )
    if HEAD_TILE == 32:
        reduction_shared = gl.allocate_shared_memory(
            gl.float32,
            [HEAD_TILE, 2 * (256 // HEAD_TILE)],
            gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
        )
    else:
        reduction_shared: gl.constexpr = None
    maximum = gl.full(
        (HEAD_TILE,), -float("inf"), gl.float32, gl.SliceLayout(1, matrix)
    )
    denominator = gl.full((HEAD_TILE,), 0, gl.float32, gl.SliceLayout(1, matrix))
    if PV_MODE == "wide128":
        numerator0 = gl.zeros((HEAD_TILE, VALUE), gl.float32, matrix)
        numerator1: gl.constexpr = None
    elif PV_MODE != "serial" and PV_MODE != "serial32":
        numerator0 = gl.zeros((HEAD_TILE, 256), gl.float32, matrix)
        numerator1 = gl.zeros((HEAD_TILE, 256), gl.float32, matrix)
    for tile in gl.static_range(TILES):
        indices = (split * TILES + tile) * 256 + key_offsets
        slot = gl.load(Slots + indices, indices < count, -1)
        valid = slot >= 0
        if PANEL_CACHE:
            panel_channels = gl.arange(0, 256, gl.SliceLayout(0, load_layout))
            if not REFILL or tile == 0:
                latent0 = _gather_cache(
                    Cache,
                    slot[:, None],
                    panel_channels[None, :],
                    valid[:, None],
                    WIDTH,
                    CACHE_BUFFER,
                    True,
                )
                cache0.store(latent0)
                latent1 = _gather_cache(
                    Cache,
                    slot[:, None],
                    256 + panel_channels[None, :],
                    valid[:, None],
                    WIDTH,
                    CACHE_BUFFER,
                    True,
                )
                cache1.store(latent1)
        else:
            latent = _gather_cache(
                Cache,
                slot[:, None],
                channels[None, :],
                valid[:, None],
                WIDTH,
                CACHE_BUFFER,
            )
            cache_shared.store(latent)
        rope_keys = (split * TILES + tile) * 256 + gl.arange(
            0, 256, gl.SliceLayout(0, direct_rope_layout)
        )
        rope_slots = gl.load(
            Slots + rope_keys, (rope_keys < TOPK) & (rope_keys < count), -1
        )
        rope_valid = rope_slots >= 0
        rope_mask = rope_valid[None, :] & (rope_channels[:, None] < WIDTH - VALUE)
        rope = _gather_cache(
            Cache,
            rope_slots[None, :],
            VALUE + rope_channels[:, None],
            rope_mask,
            WIDTH,
            CACHE_BUFFER,
            PANEL_CACHE,
            not PANEL_CACHE,
        )
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
                query_first, query_second, query_rope = _stage_query(
                    Query,
                    row,
                    head_block,
                    HEADS,
                    WIDTH,
                    VALUE,
                    TILES,
                    load_layout,
                    rope_load_layout,
                    score_lhs,
                    STAGING_PAD,
                )
            key_rope = gl.convert_layout(rope, score_rhs)
            if PANEL_CACHE:
                scores = _panel_scores(
                    query_first, query_second, query_rope, cache0, cache1, key_rope
                )
            else:
                scores = _narrow_query_scores(
                    query_first, query_rope, cache_shared, key_rope
                )
        scores = gl.convert_layout(scores, matrix) * scale
        score_keys = (split * TILES + tile) * 256 + gl.arange(
            0, 256, gl.SliceLayout(0, matrix)
        )
        score_slots = gl.load(Slots + score_keys, score_keys < count, -1)
        valid_scores = score_slots >= 0
        scores = gl.where(valid_scores[None, :], scores, -float("inf"))
        if HEAD_TILE == 64:
            reduction_shared = gl.allocate_shared_memory(
                gl.float32,
                [HEAD_TILE, 2 * (256 // HEAD_TILE)],
                gl.SwizzledSharedLayout(1, 1, 1, [0, 1]),
            )
        tile_max = _key_reduce(
            scores, reduction_shared, HEAD_TILE, True, COMPACT_REDUCE
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
            probability_shared = _stage_probability(
                probability, HEAD_TILE, PAD_PROB, PV_DTYPE
            )
        tile_sum = _key_reduce(
            probability, reduction_shared, HEAD_TILE, False, COMPACT_REDUCE
        )
        if tile == 0:
            denominator = tile_sum
        else:
            correction = _softmax_exp(maximum - safe_max)
            denominator = denominator * correction + tile_sum
            numerator0 = numerator0 * correction[:, None]
            numerator1 = numerator1 * correction[:, None]
        maximum = next_max
        if PANEL_CACHE and tile == TILES - 1:
            record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
            stats_base = Stats + record * 2 * HEAD_TILE
            stats_offsets = gl.arange(0, HEAD_TILE, gl.SliceLayout(1, matrix))
            gl.amd.cdna4.buffer_store(maximum, stats_base, stats_offsets)
            gl.amd.cdna4.buffer_store(
                denominator, stats_base, stats_offsets + HEAD_TILE
            )
        if HEAD_TILE != 32:
            probability_shared = _stage_probability(
                probability, HEAD_TILE, PAD_PROB, PV_DTYPE
            )
        if PV_MODE == "serial" or PV_MODE == "serial32":
            for half in gl.static_range(2):
                numerator, _ = _value_products(
                    probability_shared,
                    cache_shared,
                    None,
                    None,
                    HEAD_TILE,
                    PV_MODE,
                    half,
                    PV_DTYPE,
                )
                _write_partial(
                    Partial,
                    numerator,
                    row,
                    head_block,
                    split,
                    HEADS,
                    SPLITS,
                    half * 256,
                    HEAD_TILE,
                    RECORD_WIDTH,
                    COLOR_STEP,
                    PV_MODE == "serial32",
                )
        elif PANEL_CACHE:
            if REFILL and tile == 0:
                numerator0, numerator1 = _panel_products_refill(
                    probability_shared,
                    cache0,
                    cache1,
                    numerator0,
                    numerator1,
                    Cache,
                    Slots,
                    count,
                    split,
                    TOPK,
                    WIDTH,
                    CACHE_BUFFER,
                    PV_DTYPE,
                )
            else:
                numerator0, numerator1 = _panel_products(
                    probability_shared, cache0, cache1, numerator0, numerator1, PV_DTYPE
                )
        else:
            numerator0, numerator1 = _value_products(
                probability_shared,
                cache_shared,
                numerator0,
                numerator1,
                HEAD_TILE,
                PV_MODE,
                0,
                PV_DTYPE,
            )
    heads = head_block * HEAD_TILE + gl.arange(0, HEAD_TILE, gl.SliceLayout(1, matrix))
    record = (row * (HEADS // HEAD_TILE) + head_block) * SPLITS + split
    if PV_MODE == "wide128":
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
            RECORD_WIDTH,
            COLOR_STEP,
        )
    elif PV_MODE != "serial" and PV_MODE != "serial32":
        for half in gl.static_range(2):
            numerator = numerator0 if half == 0 else numerator1
            _write_partial(
                Partial,
                numerator,
                row,
                head_block,
                split,
                HEADS,
                SPLITS,
                half * 256,
                HEAD_TILE,
                RECORD_WIDTH,
                COLOR_STEP,
            )
    stats_mask = heads < HEADS
    if TILES == 1:
        stats_mask = gl.full((HEAD_TILE,), True, gl.int1, gl.SliceLayout(1, matrix))
    if PACK_STATS:
        stats_base = (
            Stats + record * (RECORD_WIDTH * HEAD_TILE // 2) + VALUE * HEAD_TILE // 2
        )
    else:
        stats_base = Stats + record * 2 * HEAD_TILE
    stats_offsets = heads % HEAD_TILE
    if not PANEL_CACHE:
        gl.amd.cdna4.buffer_store(maximum, stats_base, stats_offsets, stats_mask)
        gl.amd.cdna4.buffer_store(
            denominator, stats_base, stats_offsets + HEAD_TILE, stats_mask
        )


_attention_merge_repr = make_kernel_repr(
    "_sparse_mla_fp8_h64_panel_merge",
    ["SPLITS", "BC", "RECORD_HEADS", "MERGE_HEADS", "RECORD_WIDTH", "PACK_STATS"],
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
    RECORD_HEADS: gl.constexpr,
    MERGE_HEADS: gl.constexpr,
    RECORD_WIDTH: gl.constexpr,
    L2_PARTIALS: gl.constexpr,
    COLOR_STEP: gl.constexpr,
    PACK_STATS: gl.constexpr = False,
    BUFFER_MERGE: gl.constexpr = False,
    BUFFER_OUTPUT: gl.constexpr = False,
):
    row, channel_block = (gl.program_id(0), gl.program_id(2))
    if PACK_STATS:
        Stats = Partial.to(gl.pointer_type(gl.float32))
    if MERGE_HEADS == 32:
        head_block = gl.program_id(1)
        head_start = head_block * 32
    else:
        head_start: gl.constexpr = 0
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 8],
        [1, 32, 2],
        [1, MERGE_HEADS // 32, gl.num_warps() // (MERGE_HEADS // 32)],
        [1, 2, 0],
    )
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    splits = gl.arange(0, SPLITS, gl.SliceLayout(1, stat_layout))
    heads = head_start + gl.arange(0, MERGE_HEADS, gl.SliceLayout(0, stat_layout))
    channels = channel_block * BC + gl.arange(
        0, BC, gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    if MERGE_HEADS == 32:
        record_base = (
            row * (HEADS // RECORD_HEADS) + head_block // (RECORD_HEADS // 32)
        ) * SPLITS
    else:
        record_base = row * (HEADS // RECORD_HEADS) * SPLITS
    STAT_STRIDE: gl.constexpr = (
        RECORD_WIDTH * RECORD_HEADS // 2 if PACK_STATS else 2 * RECORD_HEADS
    )
    gl.static_assert(RECORD_WIDTH >= VALUE)
    gl.static_assert(RECORD_WIDTH * RECORD_HEADS % 2 == 0)
    gl.static_assert(
        not PACK_STATS or (RECORD_WIDTH - VALUE) * RECORD_HEADS >= 4 * RECORD_HEADS
    )
    stats_base = Stats + record_base * STAT_STRIDE
    if PACK_STATS:
        stats_base = stats_base + VALUE * RECORD_HEADS // 2
    partial_base = Partial + record_base * RECORD_WIDTH * RECORD_HEADS
    stat_index = splits[:, None] * STAT_STRIDE + heads[None, :] % RECORD_HEADS
    if MERGE_HEADS == 64:
        stat_index = stat_index + heads[None, :] // RECORD_HEADS * SPLITS * STAT_STRIDE
    maximums = gl.load(stats_base + stat_index)
    denominators = gl.load(stats_base + stat_index + RECORD_HEADS)
    maximum = gl.max(maximums, 0)
    safe_max = gl.where(maximum == -float("inf"), 0, maximum)
    correction = _softmax_exp(maximums - safe_max[None, :])
    denominator = gl.sum(denominators * correction, 0)
    record_offsets = splits[:, None, None] * RECORD_WIDTH * RECORD_HEADS
    if MERGE_HEADS == 64:
        record_offsets = (
            record_offsets
            + heads[None, :, None]
            // RECORD_HEADS
            * SPLITS
            * RECORD_WIDTH
            * RECORD_HEADS
        )
    groups = channels[None, None, :] // 8
    if COLOR_STEP:
        groups = groups ^ splits[:, None, None] * COLOR_STEP
    offsets = (
        record_offsets
        + (groups * RECORD_HEADS + heads[None, :, None] % RECORD_HEADS) * 8
        + channels[None, None, :] % 8
    )
    if MERGE_HEADS == 32 or BUFFER_MERGE:
        partials = gl.amd.cdna4.buffer_load(
            partial_base, offsets, cache=".cg" if L2_PARTIALS else ""
        ).to(gl.float32)
    else:
        partials = gl.load(partial_base + offsets).to(gl.float32)
    numerator = gl.sum(partials * correction[:, :, None], 0)
    inverse_denominator = gl.load(KScale) / gl.where(denominator > 0, denominator, 1)
    norm = gl.convert_layout(
        inverse_denominator, gl.SliceLayout(1, gl.SliceLayout(0, layout))
    )
    result = numerator * norm[:, None]
    output_layout: gl.constexpr = gl.SliceLayout(0, layout)
    out_heads = head_start + gl.arange(0, MERGE_HEADS, gl.SliceLayout(1, output_layout))
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
