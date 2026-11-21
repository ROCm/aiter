# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA decode for 64 heads, fp8 q and an fp8_scalar cache,
1 to 8 query rows. sparse_mla_fwd (attention/sparse_mla.py) launches it.

One partials kernel per (row, head group, key split) and one merge kernel per
(row, head group, value block). The partials kernel gathers the selected rows
into LDS with async copies, runs the QK and rope dots on the fp8 matrix core,
and feeds the PV dot a bf16 high/low split of the probabilities. Per-row tile
and form knobs come from the "_sparse_mla_fp8_h64" config.

Each row reads at most 2048 indices from kv_indices[kv_indptr[row]:], and a
-1 index or an index past the row's count is masked out. The cache rows are
576 B ([P, 576] fp8). The scales fold in as sm_scale * q_scale * kv_scale on
the scores and kv_scale on the output.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import _core as _gl_core
from triton.language.core import range as loop_range

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Triton 3.8 spells shared_memory_descriptor.reinterpret as _reinterpret.
_HAS_REINTERPRET = gl.constexpr(
    hasattr(_gl_core.shared_memory_descriptor, "reinterpret")
)


@gluon.jit
def _reinterpret(
    descriptor, dtype: gl.constexpr, shape: gl.constexpr, layout: gl.constexpr
):
    if _HAS_REINTERPRET:
        return descriptor.reinterpret(dtype, shape, layout)
    else:
        return descriptor._reinterpret(dtype, shape, layout)


@gluon.jit
def _row_slots(Slots, Indptr, row):
    """-> (pointer to the row's first index, the row's index count)."""
    start = gl.load(Indptr + row)
    return Slots + start, gl.load(Indptr + row + 1) - start


@gluon.jit
def _load_cache(KV, offset, mask, BUFFER_CACHE: gl.constexpr):
    if BUFFER_CACHE:
        return gl.amd.cdna4.buffer_load(KV, offset, mask, 0.0)
    else:
        return gl.load(KV + offset, mask, 0.0)


@gluon.jit
def _load_slots(Slots, count, key, layout: gl.constexpr, BUFFER_SLOTS: gl.constexpr):
    if BUFFER_SLOTS:
        return gl.amd.cdna4.buffer_load(Slots, key, key < count, -1)
    else:
        return gl.load(Slots + key, key < count, -1)


@gluon.jit
def _probability_operands(
    probability, pv_layout: gl.constexpr, PACKED_PV: gl.constexpr
):
    HEAD_BLOCK: gl.constexpr = probability.shape[0]
    KEY_BLOCK: gl.constexpr = probability.shape[1]
    high = probability.to(gl.bfloat16)
    residual = (probability - high.to(gl.float32)).to(gl.bfloat16)
    operand_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, 8)
    if PACKED_PV and HEAD_BLOCK == 8:
        joined = gl.reshape(
            gl.permute(gl.join(high, residual), (0, 2, 1)), (2 * HEAD_BLOCK, KEY_BLOCK)
        )
    else:
        joined = gl.reshape(
            gl.permute(gl.join(high, residual), (2, 0, 1)), (2 * HEAD_BLOCK, KEY_BLOCK)
        )
    if PACKED_PV and KEY_BLOCK == 256:
        shared_layout: gl.constexpr = (
            gl.amd.cdna4.compute_efficient_padded_shared_layout(
                operand_layout, [2 * HEAD_BLOCK, KEY_BLOCK], gl.bfloat16
            )
        )
    else:
        shared_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, [1, 0])
    shared = gl.allocate_shared_memory(
        gl.bfloat16, (2 * HEAD_BLOCK, KEY_BLOCK), shared_layout, joined
    )
    if PACKED_PV:
        packed = shared.load(operand_layout)
        return (packed, packed)
    else:
        return (
            shared.slice(0, HEAD_BLOCK, dim=0).load(operand_layout),
            shared.slice(HEAD_BLOCK, HEAD_BLOCK, dim=0).load(operand_layout),
        )


@gluon.jit
def _read_shared(descriptor, layout: gl.constexpr, PUBLISHED: gl.constexpr):
    if PUBLISHED:
        return gl.amd.cdna4.async_copy.load_shared_relaxed(descriptor, layout)
    else:
        return descriptor.load(layout)


@gluon.jit
def _scratch_offset(
    split,
    head,
    column,
    SPLITS: gl.constexpr,
    SPLIT_INNER: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
):
    if SPLIT_INNER:
        offset = (
            column // 16 * SPLITS * HEAD_BLOCK * 16
            + split * HEAD_BLOCK * 16
            + head * 16
            + column % 16
        )
    else:
        offset = (
            split * 64 * 512 + column // 16 * HEAD_BLOCK * 16 + head * 16 + column % 16
        )
    return offset


@gluon.jit
def _split_softmax(scores, present):
    scores = gl.where(present[None, :], scores, -float("inf"))
    maximum = gl.max(scores, 1)
    safe_max = gl.where(maximum == -float("inf"), 0.0, maximum)
    probability = gl.where(
        present[None, :],
        gl.exp2((scores - safe_max[:, None]) * 1.4426950408889634),
        0.0,
    )
    return (probability, maximum, gl.sum(probability, 1))


@gluon.constexpr_function
def _value_shared_layout(key_block, column_block):
    phases = column_block // 16
    bases = [[0, 0, 1 << bit] for bit in range(column_block.bit_length() - 1)]
    bases += [
        [0, 1 << bit, (1 << bit) % phases * 16]
        for bit in range(key_block.bit_length() - 1)
    ]
    bases += [[1 << bit, 0, 0] for bit in range((512 // column_block).bit_length() - 1)]
    return gl.SharedLinearLayout(bases, alignment=16)


@gluon.jit
def _coordinates(
    SPLITS: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    SPLIT_MAJOR_GRID: gl.constexpr,
    FLAT_ROWS: gl.constexpr,
    FLAT_ASSUME: gl.constexpr,
):
    if FLAT_ROWS:
        if FLAT_ASSUME:
            linear = gl.program_id(0)
            gl.assume(linear >= 0)
            row = linear % FLAT_ROWS
            split = linear // FLAT_ROWS % SPLITS
            head_group = linear // (FLAT_ROWS * SPLITS)
        else:
            program = gl.program_id(0).to(gl.uint32)
            row = program % FLAT_ROWS
            if SPLIT_MAJOR_GRID:
                split = program // FLAT_ROWS % SPLITS
                head_group = program // (FLAT_ROWS * SPLITS)
            else:
                head_group = program // FLAT_ROWS % (64 // HEAD_BLOCK)
                split = program // (FLAT_ROWS * (64 // HEAD_BLOCK))
    else:
        row = gl.program_id(0)
        if SPLIT_MAJOR_GRID:
            split = gl.program_id(1)
            head_group = gl.program_id(2)
        else:
            head_group = gl.program_id(1)
            split = gl.program_id(2)
    return (row, split, head_group)


_attention_partials_repr = make_kernel_repr(
    "_sparse_mla_fp8_h64_small_partials",
    [
        "ROWS",
        "SPLITS",
        "HEAD_BLOCK",
        "KEY_BLOCK",
        "PV_BLOCK",
        "BUFFER_CACHE",
        "QK_TILE",
    ],
)


@gluon.jit(repr=_attention_partials_repr)
def _attention_partials(
    Q,
    KV,
    Slots,
    Indptr,
    Partial,
    Stats,
    Scale,
    QScale,
    KScale,
    ROWS: gl.constexpr,
    SPLITS: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    KEY_BLOCK: gl.constexpr,
    PV_BLOCK: gl.constexpr,
    BUFFER_CACHE: gl.constexpr,
    SCORE_VECTOR: gl.constexpr,
    NATIVE_SCORES: gl.constexpr,
    SPLIT_MAJOR_GRID: gl.constexpr,
    QUERY_PHASES: gl.constexpr,
    SPLIT_INNER_SCRATCH: gl.constexpr,
    NATIVE_ROPE: gl.constexpr,
    PV_UNROLL: gl.constexpr,
    EARLY_STATISTICS: gl.constexpr,
    PACK_STATISTICS: gl.constexpr,
    LATE_QUERY_READ: gl.constexpr,
    PACKED_PV: gl.constexpr,
    PV_TILES: gl.constexpr,
    GROUP_STATISTICS: gl.constexpr,
    QK_TILE: gl.constexpr,
    TRANSPOSED_PV: gl.constexpr,
    FLAT_ROWS: gl.constexpr,
    FLAT_ASSUME: gl.constexpr,
    STATS_IN_PARTIAL: gl.constexpr,
    PRESENCE_OUTSIDE: gl.constexpr,
    QUERY_ROW_BASE: gl.constexpr,
    BASE_DIRECT: gl.constexpr,
    GATHER_DIRECT: gl.constexpr,
    NARROW_FORM: gl.constexpr,
    WIDE_STAGE: gl.constexpr,
    QUERY_STAGE_STORE: gl.constexpr,
    SCORE_STATS_LAYOUT: gl.constexpr,
    LATE_PARTIAL_BASE: gl.constexpr,
):
    if STATS_IN_PARTIAL:
        Stats = Partial + ROWS * 64 * SPLITS * 512
    Scale = Scale * gl.load(QScale) * gl.load(KScale)
    if BUFFER_CACHE or not PRESENCE_OUTSIDE:
        _attention_body(
            Q,
            KV,
            Slots,
            Indptr,
            Partial,
            Stats,
            Scale,
            SPLITS,
            HEAD_BLOCK,
            KEY_BLOCK,
            PV_BLOCK,
            BUFFER_CACHE,
            SCORE_VECTOR,
            NATIVE_SCORES,
            SPLIT_MAJOR_GRID,
            QUERY_PHASES,
            SPLIT_INNER_SCRATCH,
            NATIVE_ROPE,
            PV_UNROLL,
            EARLY_STATISTICS,
            PACK_STATISTICS,
            LATE_QUERY_READ,
            PACKED_PV,
            PV_TILES,
            GROUP_STATISTICS,
            QK_TILE,
            TRANSPOSED_PV,
            FLAT_ROWS,
            FLAT_ASSUME,
            PRESENCE_OUTSIDE,
            QUERY_ROW_BASE,
            BASE_DIRECT,
            GATHER_DIRECT,
            NARROW_FORM,
            WIDE_STAGE,
            QUERY_STAGE_STORE,
            SCORE_STATS_LAYOUT,
            LATE_PARTIAL_BASE,
        )
    else:
        row, split, head_group = _coordinates(
            SPLITS, HEAD_BLOCK, SPLIT_MAJOR_GRID, FLAT_ROWS, FLAT_ASSUME
        )
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        key = split * KEY_BLOCK + gl.arange(0, KEY_BLOCK, layout)
        row_slots, count = _row_slots(Slots, Indptr, row)
        slot = gl.load(row_slots + key, key < count, -1)
        if gl.sum((slot >= 0).to(gl.int32), 0) > 0:
            _attention_body(
                Q,
                KV,
                Slots,
                Indptr,
                Partial,
                Stats,
                Scale,
                SPLITS,
                HEAD_BLOCK,
                KEY_BLOCK,
                PV_BLOCK,
                BUFFER_CACHE,
                SCORE_VECTOR,
                NATIVE_SCORES,
                SPLIT_MAJOR_GRID,
                QUERY_PHASES,
                SPLIT_INNER_SCRATCH,
                NATIVE_ROPE,
                PV_UNROLL,
                EARLY_STATISTICS,
                PACK_STATISTICS,
                LATE_QUERY_READ,
                PACKED_PV,
                PV_TILES,
                GROUP_STATISTICS,
                QK_TILE,
                TRANSPOSED_PV,
                FLAT_ROWS,
                FLAT_ASSUME,
                PRESENCE_OUTSIDE,
                QUERY_ROW_BASE,
                BASE_DIRECT,
                GATHER_DIRECT,
                NARROW_FORM,
                WIDE_STAGE,
                QUERY_STAGE_STORE,
                SCORE_STATS_LAYOUT,
                LATE_PARTIAL_BASE,
            )
        else:
            _publish_empty_record(
                Stats, row, split, head_group, SPLITS, HEAD_BLOCK, GROUP_STATISTICS
            )


@gluon.jit
def _attention_body(
    Q,
    KV,
    Slots,
    Indptr,
    Partial,
    Stats,
    Scale,
    SPLITS: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    KEY_BLOCK: gl.constexpr,
    PV_BLOCK: gl.constexpr,
    BUFFER_CACHE: gl.constexpr,
    SCORE_VECTOR: gl.constexpr,
    NATIVE_SCORES: gl.constexpr,
    SPLIT_MAJOR_GRID: gl.constexpr,
    QUERY_PHASES: gl.constexpr,
    SPLIT_INNER_SCRATCH: gl.constexpr,
    NATIVE_ROPE: gl.constexpr,
    PV_UNROLL: gl.constexpr,
    EARLY_STATISTICS: gl.constexpr,
    PACK_STATISTICS: gl.constexpr,
    LATE_QUERY_READ: gl.constexpr,
    PACKED_PV: gl.constexpr,
    PV_TILES: gl.constexpr,
    GROUP_STATISTICS: gl.constexpr,
    QK_TILE: gl.constexpr,
    TRANSPOSED_PV: gl.constexpr,
    FLAT_ROWS: gl.constexpr,
    FLAT_ASSUME: gl.constexpr,
    PRESENCE_OUTSIDE: gl.constexpr,
    QUERY_ROW_BASE: gl.constexpr,
    BASE_DIRECT: gl.constexpr,
    GATHER_DIRECT: gl.constexpr,
    NARROW_FORM: gl.constexpr,
    WIDE_STAGE: gl.constexpr,
    QUERY_STAGE_STORE: gl.constexpr,
    SCORE_STATS_LAYOUT: gl.constexpr,
    LATE_PARTIAL_BASE: gl.constexpr,
):
    row, split, head_group = _coordinates(
        SPLITS, HEAD_BLOCK, SPLIT_MAJOR_GRID, FLAT_ROWS, FLAT_ASSUME
    )
    row_slots, count = _row_slots(Slots, Indptr, row)
    rope_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 4], [4, 1], [1, 0])
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 128], transposed=True, warps_per_cta=[1, 4]
    )
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 16] if PACKED_PV and HEAD_BLOCK == 16 else [16, 16, 32],
        transposed=TRANSPOSED_PV,
        warps_per_cta=[1, 4],
        tiles_per_warp=[1, PV_TILES],
    )
    rope_dot_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, 4]
    )
    ROPE_BLOCK: gl.constexpr = 128 if not NATIVE_ROPE else 64
    key = split * KEY_BLOCK + gl.arange(
        0, KEY_BLOCK, layout=gl.SliceLayout(1, rope_layout)
    )
    rr = gl.arange(0, ROPE_BLOCK, layout=gl.SliceLayout(0, rope_layout))
    BUFFER_SLOTS: gl.constexpr = HEAD_BLOCK == 8 or KEY_BLOCK == 128
    slot = _load_slots(
        row_slots, count, key, gl.SliceLayout(1, rope_layout), BUFFER_SLOTS
    )
    valid = slot >= 0
    if NATIVE_SCORES:
        score_layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 4], [0, 1])
    else:
        score_layout: gl.constexpr = gl.BlockedLayout(
            [1, SCORE_VECTOR], [4, 16], [4, 1], [1, 0]
        )
    if BUFFER_CACHE or PRESENCE_OUTSIDE:
        populated = True
    else:
        presence_layout: gl.constexpr = gl.BlockedLayout(
            [KEY_BLOCK // 64], [64], [4], [0]
        )
        presence_key = split * KEY_BLOCK + gl.arange(0, KEY_BLOCK, presence_layout)
        presence = (
            _load_slots(row_slots, count, presence_key, presence_layout, True) >= 0
        )
        populated = gl.sum(presence.to(gl.int32), 0) > 0
    if populated:
        latent_query, rope_query = _load_query(
            Q, row, head_group, HEAD_BLOCK, ROPE_BLOCK, QUERY_ROW_BASE
        )
        if not BUFFER_CACHE:
            latent_query = latent_query.to(KV.dtype.element_ty)
            rope_query = rope_query.to(KV.dtype.element_ty)
        base = _cache_row_offset(slot[:, None], BUFFER_CACHE, BASE_DIRECT, NARROW_FORM)
        STAGE_IN_BUFFER_STAGER: gl.constexpr = BUFFER_CACHE or not WIDE_STAGE
        if STAGE_IN_BUFFER_STAGER and KEY_BLOCK >= 128:
            value_linear = _stage_cache_async(
                KV,
                row_slots,
                count,
                split,
                KEY_BLOCK,
                PV_BLOCK,
                BUFFER_SLOTS,
                BUFFER_CACHE,
                GATHER_DIRECT,
            )
        kr = _load_cache(
            KV,
            base + 512 + rr[None, :],
            valid[:, None] & (rr[None, :] < 64),
            BUFFER_CACHE,
        )
        if STAGE_IN_BUFFER_STAGER and KEY_BLOCK < 128:
            value_linear = _stage_cache_async(
                KV,
                row_slots,
                count,
                split,
                KEY_BLOCK,
                PV_BLOCK,
                BUFFER_SLOTS,
                BUFFER_CACHE,
                GATHER_DIRECT,
            )
        elif not STAGE_IN_BUFFER_STAGER:
            value_linear = _stage_wide_cache_async(
                KV, row_slots, count, split, KEY_BLOCK, PV_BLOCK, NARROW_FORM
            )
        value_shared = _reinterpret(
            value_linear,
            KV.dtype.element_ty,
            (512 // PV_BLOCK, KEY_BLOCK, PV_BLOCK),
            gl.SwizzledSharedLayout(16, 1, PV_BLOCK // 16, [1, 0]),
        )
        cache_shared = value_linear.permute((1, 0, 2)).reshape((KEY_BLOCK, 512))
        if BUFFER_CACHE:
            latent_query = latent_query.to(KV.dtype.element_ty)
            rope_query = rope_query.to(KV.dtype.element_ty)
        latent_shared, rope_shared = _stage_query(
            latent_query,
            rope_query,
            HEAD_BLOCK,
            ROPE_BLOCK,
            QUERY_PHASES,
            NATIVE_ROPE or KEY_BLOCK == 256,
            QK_TILE,
            QUERY_STAGE_STORE,
        )
        if BUFFER_CACHE:
            present = _score_presence(
                row_slots, count, split, KEY_BLOCK, score_layout, BUFFER_SLOTS
            )
        if not LATE_QUERY_READ:
            q_dot, qr_dot = _query_operands(
                latent_shared,
                rope_shared,
                qk_layout,
                rope_dot_layout,
                NATIVE_ROPE,
                False,
                QK_TILE,
            )
        gl.amd.cdna4.async_copy.wait_group(0)
        gl.barrier()
        if LATE_QUERY_READ:
            q_dot, qr_dot = _query_operands(
                latent_shared,
                rope_shared,
                qk_layout,
                rope_dot_layout,
                NATIVE_ROPE,
                True,
                QK_TILE,
            )
        scores = _qk_scores(
            q_dot,
            qr_dot,
            kr,
            cache_shared,
            value_shared,
            HEAD_BLOCK,
            KEY_BLOCK,
            qk_layout,
            rope_dot_layout,
            NATIVE_ROPE,
            QK_TILE,
        )
        scores = gl.convert_layout(scores, score_layout) * Scale
        if not BUFFER_CACHE:
            present = _score_presence(
                row_slots, count, split, KEY_BLOCK, score_layout, BUFFER_SLOTS
            )
        probability, maximum, denominator = _split_softmax(scores, present)
        p_high_dot, p_low_dot = _probability_operands(probability, pv_layout, PACKED_PV)
        if GROUP_STATISTICS:
            scalar_record = (
                row * 64 + head_group * HEAD_BLOCK
            ) * SPLITS + split * HEAD_BLOCK
        else:
            scalar_record = (row * SPLITS + split) * 64 + head_group * HEAD_BLOCK
        if EARLY_STATISTICS:
            _publish_statistics(
                Stats,
                scalar_record,
                maximum,
                denominator,
                HEAD_BLOCK,
                score_layout,
                PACK_STATISTICS,
                SCORE_STATS_LAYOUT,
            )
        _publish_numerators(
            Partial,
            value_shared,
            p_high_dot,
            p_low_dot,
            row,
            split,
            head_group,
            SPLITS,
            HEAD_BLOCK,
            PV_BLOCK,
            pv_layout,
            PV_UNROLL,
            SPLIT_INNER_SCRATCH,
            PACKED_PV,
            TRANSPOSED_PV,
            LATE_PARTIAL_BASE,
        )
        if not EARLY_STATISTICS:
            _publish_statistics(
                Stats,
                scalar_record,
                maximum,
                denominator,
                HEAD_BLOCK,
                score_layout,
                PACK_STATISTICS,
                SCORE_STATS_LAYOUT,
            )
    else:
        head_layout: gl.constexpr = gl.SliceLayout(1, score_layout)
        maximum = gl.full((HEAD_BLOCK,), -float("inf"), gl.float32, head_layout)
        denominator = gl.full((HEAD_BLOCK,), 0.0, gl.float32, head_layout)
        if GROUP_STATISTICS:
            scalar_record = (
                row * 64 + head_group * HEAD_BLOCK
            ) * SPLITS + split * HEAD_BLOCK
        else:
            scalar_record = (row * SPLITS + split) * 64 + head_group * HEAD_BLOCK
        _publish_statistics(
            Stats,
            scalar_record,
            maximum,
            denominator,
            HEAD_BLOCK,
            score_layout,
            PACK_STATISTICS,
            SCORE_STATS_LAYOUT,
        )


@gluon.jit
def _cache_row_offset(
    slot,
    BUFFER_CACHE: gl.constexpr,
    DIRECT: gl.constexpr,
    NARROW_FORM: gl.constexpr = 0,
):
    if BUFFER_CACHE:
        if DIRECT:
            return slot * 576
        return gl.maximum(slot, 0) * 576
    elif NARROW_FORM == 1:
        return (gl.maximum(slot, 0).to(gl.uint32) * 576).to(gl.int64)
    elif NARROW_FORM == 2:
        return (slot.to(gl.uint32) * 576).to(gl.int64)
    else:
        return gl.maximum(slot, 0).to(gl.int64) * 576


@gluon.jit
def _load_query(
    Q,
    row,
    head_group,
    HEAD_BLOCK: gl.constexpr,
    ROPE_BLOCK: gl.constexpr,
    QUERY_ROW_BASE: gl.constexpr = False,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [4, 16], [2, 2] if HEAD_BLOCK == 8 else [4, 1], [1, 0]
    )
    if QUERY_ROW_BASE:
        local_head = gl.arange(0, HEAD_BLOCK, layout=gl.SliceLayout(1, layout))
        latent_column = gl.arange(0, 512, layout=gl.SliceLayout(0, layout))
        rope_column = gl.arange(0, ROPE_BLOCK, layout=gl.SliceLayout(0, layout))
        head_base = row * 64 + head_group * HEAD_BLOCK
        query_row = (head_base + local_head[:, None]) * 576
        rope_offset = query_row + 512 + rope_column[None, :]
        latent_mask = gl.full((HEAD_BLOCK, 512), True, gl.int1, layout)
        rope_mask = rope_column[None, :] < 64
        latent = gl.amd.cdna4.buffer_load(
            Q, query_row + latent_column[None, :], latent_mask, 0.0
        )
        rope = gl.amd.cdna4.buffer_load(Q, rope_offset, rope_mask, 0.0)
    else:
        head = head_group * HEAD_BLOCK + gl.arange(
            0, HEAD_BLOCK, layout=gl.SliceLayout(1, layout)
        )
        latent_column = gl.arange(0, 512, layout=gl.SliceLayout(0, layout))
        rope_column = gl.arange(0, ROPE_BLOCK, layout=gl.SliceLayout(0, layout))
        query_row = (row * 64 + head[:, None]) * 576
        latent_mask = gl.full((HEAD_BLOCK, 512), True, gl.int1, layout)
        rope_mask = rope_column[None, :] < 64
        latent = gl.amd.cdna4.buffer_load(
            Q, query_row + latent_column[None, :], latent_mask, 0.0
        )
        rope = gl.amd.cdna4.buffer_load(
            Q, query_row + 512 + rope_column[None, :], rope_mask, 0.0
        )
    return (latent, rope)


_merge_head_groups_repr = make_kernel_repr(
    "_sparse_mla_fp8_h64_small_merge",
    [
        "ROWS",
        "SPLITS",
        "VALUE_BLOCK",
        "HEAD_BLOCK",
        "SPLIT_INNER_SCRATCH",
        "GROUP_STATISTICS",
    ],
)


@gluon.jit(repr=_merge_head_groups_repr)
def _merge_head_groups(
    Partial,
    Stats,
    Out,
    KScale,
    out_stride0,
    out_stride1,
    ROWS: gl.constexpr,
    SPLITS: gl.constexpr,
    VALUE_BLOCK: gl.constexpr,
    REDUCE_THREADS: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    MERGE_WARPS: gl.constexpr,
    HEAD_THREADS: gl.constexpr,
    SPLIT_INNER_SCRATCH: gl.constexpr,
    GROUP_STATISTICS: gl.constexpr,
    PRUNE_EMPTY: gl.constexpr,
    STATS_IN_PARTIAL: gl.constexpr,
):
    if STATS_IN_PARTIAL:
        Stats = Partial + ROWS * 64 * SPLITS * 512
    row = gl.program_id(0)
    head_group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4],
        [REDUCE_THREADS, HEAD_THREADS, 64 // (REDUCE_THREADS * HEAD_THREADS)],
        [1, MERGE_WARPS, 1],
        [2, 1, 0],
    )
    stat_layout: gl.constexpr = gl.SliceLayout(2, layout)
    s = gl.arange(0, SPLITS, gl.SliceLayout(1, stat_layout))
    h = head_group * HEAD_BLOCK + gl.arange(
        0, HEAD_BLOCK, gl.SliceLayout(0, stat_layout)
    )
    d = gl.program_id(2) * VALUE_BLOCK + gl.arange(
        0, VALUE_BLOCK, gl.SliceLayout(0, gl.SliceLayout(1, layout))
    )
    if GROUP_STATISTICS:
        stat_base = (row * 64 + head_group * HEAD_BLOCK) * SPLITS * 2
        stat_offset = s[:, None] * HEAD_BLOCK * 2 + h[None, :] % HEAD_BLOCK * 2
    else:
        stat_base = (row * SPLITS * 64 + head_group * HEAD_BLOCK) * 2
        stat_offset = s[:, None] * 64 * 2 + h[None, :] % HEAD_BLOCK * 2
    if HEAD_BLOCK == 8:
        pair_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1, 2],
            [REDUCE_THREADS, HEAD_THREADS, 64 // (REDUCE_THREADS * HEAD_THREADS)],
            [1, MERGE_WARPS, 1],
            [2, 1, 0],
        )
        pair_split = gl.arange(
            0, SPLITS, gl.SliceLayout(1, gl.SliceLayout(2, pair_layout))
        )
        pair_head = gl.arange(
            0, HEAD_BLOCK, gl.SliceLayout(0, gl.SliceLayout(2, pair_layout))
        )
        pair_field = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(1, pair_layout)))
        if GROUP_STATISTICS:
            pair_offset = (
                pair_split[:, None, None] * HEAD_BLOCK * 2
                + pair_head[None, :, None] * 2
                + pair_field[None, None, :]
            )
        else:
            pair_offset = (
                pair_split[:, None, None] * 64 * 2
                + pair_head[None, :, None] * 2
                + pair_field[None, None, :]
            )
        pairs = gl.amd.cdna4.buffer_load(Stats + stat_base, pair_offset)
        maxima, denoms = gl.split(pairs)
        maxima = gl.convert_layout(maxima, stat_layout)
        denoms = gl.convert_layout(denoms, stat_layout)
    else:
        maxima = gl.amd.cdna4.buffer_load(
            Stats + stat_base, stat_offset, s[:, None] < SPLITS, -float("inf")
        )
        denoms = gl.amd.cdna4.buffer_load(
            Stats + stat_base, stat_offset + 1, s[:, None] < SPLITS, 0.0
        )
    maximum = gl.max(maxima, 0)
    safe_max = gl.where(maximum == -float("inf"), 0.0, maximum)
    correction = gl.exp2((maxima - safe_max[None, :]) * 1.4426950408889634)
    denominator = gl.sum(denoms * correction, 0)
    if SPLIT_INNER_SCRATCH:
        group_base = (row * 64 + head_group * HEAD_BLOCK) * SPLITS * 512
    else:
        group_base = (row * SPLITS * 64 + head_group * HEAD_BLOCK) * 512
    offset = _scratch_offset(
        s[:, None, None],
        h[None, :, None] % HEAD_BLOCK,
        d[None, None, :],
        SPLITS,
        SPLIT_INNER_SCRATCH,
        HEAD_BLOCK,
    )
    partial_mask = s[:, None, None] < SPLITS
    if PRUNE_EMPTY:
        partial_mask = partial_mask & (denoms[:, :, None] > 0.0)
    partial = gl.amd.cdna4.buffer_load(
        Partial + group_base, offset, partial_mask, 0.0, cache=".cg"
    )
    numerator = gl.sum(partial * correction[:, :, None], 0)
    out_layout: gl.constexpr = gl.SliceLayout(0, layout)
    reciprocal = gl.load(KScale) / gl.where(denominator > 0.0, denominator, 1.0)
    reciprocal = gl.convert_layout(reciprocal, gl.SliceLayout(1, out_layout))
    result = numerator * reciprocal[:, None]
    oh = gl.arange(0, HEAD_BLOCK, gl.SliceLayout(1, out_layout))
    od = gl.program_id(2) * VALUE_BLOCK + gl.arange(
        0, VALUE_BLOCK, gl.SliceLayout(0, out_layout)
    )
    out_base = row * out_stride0 + head_group * HEAD_BLOCK * out_stride1
    gl.amd.cdna4.buffer_store(
        result.to(Out.dtype.element_ty),
        Out + out_base,
        oh[:, None] * out_stride1 + od[None, :],
    )


@gluon.jit
def _publish_empty_record(
    Stats,
    row,
    split,
    head_group,
    SPLITS: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    GROUP_STATISTICS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    head_layout: gl.constexpr = gl.SliceLayout(1, layout)
    if GROUP_STATISTICS:
        record = (row * 64 + head_group * HEAD_BLOCK) * SPLITS + split * HEAD_BLOCK
    else:
        record = (row * SPLITS + split) * 64 + head_group * HEAD_BLOCK
    _publish_statistics(
        Stats,
        record,
        gl.full((HEAD_BLOCK,), -float("inf"), gl.float32, head_layout),
        gl.full((HEAD_BLOCK,), 0.0, gl.float32, head_layout),
        HEAD_BLOCK,
        layout,
        True,
        True,
    )


@gluon.jit
def _publish_numerators(
    Partial,
    value_shared,
    p_high,
    p_residual,
    row,
    split,
    head_group,
    SPLITS: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    PV_BLOCK: gl.constexpr,
    pv_layout: gl.constexpr,
    PV_UNROLL: gl.constexpr,
    SPLIT_INNER_SCRATCH: gl.constexpr,
    PACKED_PV: gl.constexpr,
    TRANSPOSED_PV: gl.constexpr,
    LATE_PARTIAL_BASE: gl.constexpr = False,
):
    if not PACKED_PV:
        store_layout: gl.constexpr = (
            pv_layout
            if TRANSPOSED_PV
            else gl.BlockedLayout([4, 1], [4, 16], [1, 4], [1, 0])
        )
    for section in loop_range(
        0, 512 // PV_BLOCK, loop_unroll_factor=PV_UNROLL, disable_licm=True
    ):
        value = _read_shared(
            value_shared.index(section), gl.DotOperandLayout(1, pv_layout, 8), True
        ).to(gl.bfloat16)
        if PACKED_PV:
            packed = gl.zeros((2 * HEAD_BLOCK, PV_BLOCK), gl.float32, layout=pv_layout)
            packed = gl.amd.cdna4.mfma(p_high, value, packed)
            if HEAD_BLOCK == 16:
                high, residual = gl.split(
                    gl.permute(gl.reshape(packed, (2, HEAD_BLOCK, PV_BLOCK)), (1, 2, 0))
                )
            else:
                high, residual = gl.split(
                    gl.permute(gl.reshape(packed, (HEAD_BLOCK, 2, PV_BLOCK)), (0, 2, 1))
                )
            numerator = high + residual
            store_layout: gl.constexpr = numerator.type.layout
        else:
            numerator = gl.zeros((HEAD_BLOCK, PV_BLOCK), gl.float32, layout=pv_layout)
            numerator = gl.amd.cdna4.mfma(p_residual, value, numerator)
            numerator = gl.amd.cdna4.mfma(p_high, value, numerator)
        numerator = gl.convert_layout(numerator, store_layout)
        local_head = gl.arange(0, HEAD_BLOCK, layout=gl.SliceLayout(1, store_layout))
        store_mask = gl.full((HEAD_BLOCK, 1), True, gl.int1, layout=store_layout)
        column = section * PV_BLOCK + gl.arange(
            0, PV_BLOCK, layout=gl.SliceLayout(0, store_layout)
        )
        if SPLIT_INNER_SCRATCH:
            group_record = (row * 64 + head_group * HEAD_BLOCK) * SPLITS
        else:
            group_record = (row * SPLITS + split) * 64 + head_group * HEAD_BLOCK
        if not LATE_PARTIAL_BASE:
            fragment_base = group_record * 512
        fragment_offset = _scratch_offset(
            split if SPLIT_INNER_SCRATCH else 0,
            local_head[:, None],
            column[None, :],
            SPLITS,
            SPLIT_INNER_SCRATCH,
            HEAD_BLOCK,
        )
        if LATE_PARTIAL_BASE:
            fragment_base = group_record * 512
        gl.amd.cdna4.buffer_store(
            numerator, Partial + fragment_base, fragment_offset, store_mask, cache=".cs"
        )


@gluon.jit
def _publish_statistics(
    Stats,
    record,
    maximum,
    denominator,
    HEAD_BLOCK: gl.constexpr,
    score_layout: gl.constexpr,
    PACKED: gl.constexpr,
    SCORE_STATS_LAYOUT: gl.constexpr = False,
):
    if PACKED:
        if SCORE_STATS_LAYOUT:
            layout: gl.constexpr = gl.BlockedLayout(
                [1, 2],
                score_layout.threads_per_warp,
                score_layout.warps_per_cta,
                score_layout.order,
            )
        else:
            layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
        values = gl.convert_layout(gl.join(maximum, denominator), layout)
        head = gl.arange(0, HEAD_BLOCK, gl.SliceLayout(1, layout))
        field = gl.arange(0, 2, gl.SliceLayout(0, layout))
        gl.amd.cdna4.buffer_store(
            values, Stats + record * 2, head[:, None] * 2 + field[None, :]
        )
    else:
        head_layout: gl.constexpr = gl.SliceLayout(1, score_layout)
        head = gl.arange(0, HEAD_BLOCK, layout=head_layout)
        mask = gl.full((HEAD_BLOCK,), True, gl.int1, layout=head_layout)
        gl.amd.cdna4.buffer_store(maximum, Stats + record * 2, head * 2, mask)
        gl.amd.cdna4.buffer_store(denominator, Stats + record * 2, head * 2 + 1, mask)


@gluon.jit
def _qk_scores(
    latent_query,
    rope_query,
    rope_key,
    cache_shared,
    value_shared,
    HEAD_BLOCK: gl.constexpr,
    KEY_BLOCK: gl.constexpr,
    qk_layout: gl.constexpr,
    rope_layout: gl.constexpr,
    NATIVE_ROPE: gl.constexpr,
    QK_TILE: gl.constexpr,
):
    scores = gl.zeros((HEAD_BLOCK, KEY_BLOCK), gl.float32, layout=qk_layout)
    if QK_TILE == 512:
        scores = gl.amd.cdna4.mfma_scaled(
            latent_query,
            None,
            "e4m3",
            _read_shared(
                cache_shared.permute((1, 0)),
                gl.DotOperandLayout(1, qk_layout, 16),
                True,
            ),
            None,
            "e4m3",
            scores,
        )
    else:
        assert QK_TILE == value_shared.shape[2]
        for section in loop_range(
            0, 512 // QK_TILE, loop_unroll_factor=1, disable_licm=True
        ):
            query = _read_shared(
                latent_query.index(section), gl.DotOperandLayout(0, qk_layout, 16), True
            )
            key = _read_shared(
                value_shared.index(section).permute((1, 0)),
                gl.DotOperandLayout(1, qk_layout, 16),
                True,
            )
            scores = gl.amd.cdna4.mfma_scaled(
                query, None, "e4m3", key, None, "e4m3", scores
            )
    if NATIVE_ROPE:
        scores = gl.amd.cdna4.mfma(
            rope_query,
            gl.convert_layout(
                gl.permute(rope_key, (1, 0)), gl.DotOperandLayout(1, rope_layout, 8)
            ),
            gl.convert_layout(scores, rope_layout),
        )
        scores = gl.convert_layout(scores, qk_layout)
    else:
        scores = gl.amd.cdna4.mfma_scaled(
            rope_query,
            None,
            "e4m3",
            gl.convert_layout(
                gl.permute(rope_key, (1, 0)), gl.DotOperandLayout(1, qk_layout, 16)
            ),
            None,
            "e4m3",
            scores,
        )
    return scores


@gluon.jit
def _query_operands(
    latent_shared,
    rope_shared,
    qk_layout: gl.constexpr,
    rope_layout: gl.constexpr,
    NATIVE_ROPE: gl.constexpr,
    PUBLISHED: gl.constexpr,
    QK_TILE: gl.constexpr,
):
    if QK_TILE == 512:
        latent = _read_shared(
            latent_shared, gl.DotOperandLayout(0, qk_layout, 16), PUBLISHED
        )
    else:
        latent = latent_shared
    if NATIVE_ROPE:
        rope = _read_shared(
            rope_shared, gl.DotOperandLayout(0, rope_layout, 8), PUBLISHED
        )
    else:
        rope = _read_shared(
            rope_shared, gl.DotOperandLayout(0, qk_layout, 16), PUBLISHED
        )
    return (latent, rope)


@gluon.jit
def _score_presence(
    Slots,
    count,
    split,
    KEY_BLOCK: gl.constexpr,
    score_layout: gl.constexpr,
    BUFFER_SLOTS: gl.constexpr,
):
    key = split * KEY_BLOCK + gl.arange(
        0, KEY_BLOCK, layout=gl.SliceLayout(0, score_layout)
    )
    return (
        _load_slots(Slots, count, key, gl.SliceLayout(0, score_layout), BUFFER_SLOTS)
        >= 0
    )


@gluon.jit
def _stage_cache_async(
    KV,
    Slots,
    count,
    split,
    KEY_BLOCK: gl.constexpr,
    PV_BLOCK: gl.constexpr,
    BUFFER_SLOTS: gl.constexpr,
    BUFFER_CACHE: gl.constexpr,
    GATHER_DIRECT: gl.constexpr,
):
    copy_layout: gl.constexpr = gl.BlockedLayout([16], [64], [4], [0])
    copy_columns: gl.constexpr = PV_BLOCK if KEY_BLOCK == 128 else 512
    if BUFFER_SLOTS:
        gather_layout: gl.constexpr = gl.BlockedLayout(
            [1, 16], [1024 // PV_BLOCK, PV_BLOCK // 16], [4, 1], [1, 0]
        )
        gather_rows: gl.constexpr = KEY_BLOCK * (copy_columns // PV_BLOCK)
        physical_row = gl.arange(0, gather_rows, gl.SliceLayout(1, gather_layout))
        physical_column = gl.arange(0, PV_BLOCK, gl.SliceLayout(0, gather_layout))
        row_key = physical_row % KEY_BLOCK
        row_slot = _load_slots(
            Slots,
            count,
            split * KEY_BLOCK + row_key,
            gl.SliceLayout(1, gather_layout),
            True,
        )
        swizzle = row_key % (PV_BLOCK // 16) * 16
        row_offset = (
            _cache_row_offset(row_slot[:, None], BUFFER_CACHE, GATHER_DIRECT)
            + (physical_column[None, :] ^ swizzle[:, None])
            + physical_row[:, None] // KEY_BLOCK * PV_BLOCK
        )
        copy_slot = row_slot[:, None] + gl.full(
            (gather_rows, PV_BLOCK), 0, gl.int32, gather_layout
        )
        slot = gl.convert_layout(
            gl.reshape(copy_slot, (KEY_BLOCK * copy_columns,)), copy_layout
        )
        offset = gl.convert_layout(
            gl.reshape(row_offset, (KEY_BLOCK * copy_columns,)), copy_layout
        )
    else:
        physical = gl.arange(0, KEY_BLOCK * copy_columns, layout=copy_layout)
        key = physical // PV_BLOCK % KEY_BLOCK
        column = physical % PV_BLOCK ^ key % (PV_BLOCK // 16) * 16
        column += physical // (KEY_BLOCK * PV_BLOCK) * PV_BLOCK
        slot = gl.load(
            Slots + split * KEY_BLOCK + key, split * KEY_BLOCK + key < count, -1
        )
        offset = _cache_row_offset(slot, BUFFER_CACHE, False) + column
    offset = gl.max_contiguous(gl.multiple_of(offset, 16), 16)
    flat = gl.allocate_shared_memory(
        gl.uint8, (KEY_BLOCK * 512,), gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    raw_cache = KV.to(gl.pointer_type(gl.uint8))
    if KEY_BLOCK == 128:
        for section in gl.static_range(512 // PV_BLOCK):
            destination = flat.slice(
                section * KEY_BLOCK * PV_BLOCK, KEY_BLOCK * PV_BLOCK, dim=0
            )
            if BUFFER_CACHE:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    destination, raw_cache, offset + section * PV_BLOCK, mask=slot >= 0
                )
            else:
                gl.amd.cdna4.async_copy.global_load_to_shared(
                    destination,
                    raw_cache + offset + section * PV_BLOCK,
                    mask=slot >= 0,
                    other=0,
                )
    elif BUFFER_CACHE:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            flat, raw_cache, offset, mask=slot >= 0
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(
            flat, raw_cache + offset, mask=slot >= 0, other=0
        )
    gl.amd.cdna4.async_copy.commit_group()
    return _reinterpret(
        flat,
        KV.dtype.element_ty,
        (512 // PV_BLOCK, KEY_BLOCK, PV_BLOCK),
        _value_shared_layout(KEY_BLOCK, PV_BLOCK),
    )


@gluon.jit
def _stage_wide_cache_async(
    KV,
    Slots,
    count,
    split,
    KEY_BLOCK: gl.constexpr,
    PV_BLOCK: gl.constexpr,
    NARROW_FORM: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([16], [64], [4], [0])
    physical = gl.arange(0, KEY_BLOCK * 512, layout)
    key = physical // PV_BLOCK % KEY_BLOCK
    column = physical % PV_BLOCK ^ key % (PV_BLOCK // 16) * 16
    column += physical // (KEY_BLOCK * PV_BLOCK) * PV_BLOCK
    slot = gl.load(Slots + split * KEY_BLOCK + key, split * KEY_BLOCK + key < count, -1)
    if NARROW_FORM:
        offset = (
            (gl.maximum(slot, 0) if NARROW_FORM == 1 else slot).to(gl.uint32) * 576
            + column
        ).to(gl.int64)
    else:
        offset = gl.maximum(slot, 0).to(gl.int64) * 576 + column
    offset = gl.max_contiguous(gl.multiple_of(offset, 16), 16)
    flat = gl.allocate_shared_memory(
        gl.uint8, (KEY_BLOCK * 512,), gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    gl.amd.cdna4.async_copy.global_load_to_shared(
        flat, KV.to(gl.pointer_type(gl.uint8)) + offset, mask=slot >= 0, other=0
    )
    gl.amd.cdna4.async_copy.commit_group()
    return _reinterpret(
        flat,
        KV.dtype.element_ty,
        (512 // PV_BLOCK, KEY_BLOCK, PV_BLOCK),
        _value_shared_layout(KEY_BLOCK, PV_BLOCK),
    )


@gluon.jit
def _stage_full_query(
    latent,
    rope,
    HEAD_BLOCK: gl.constexpr,
    ROPE_BLOCK: gl.constexpr,
    QUERY_PHASES: gl.constexpr,
    SEPARATE_ROPE: gl.constexpr,
):
    query_layout: gl.constexpr = gl.SwizzledSharedLayout(16, 1, QUERY_PHASES, [1, 0])
    if SEPARATE_ROPE:
        query_shared = gl.allocate_shared_memory(
            latent.dtype, (HEAD_BLOCK, 512), query_layout, latent
        )
        rope_shared = gl.allocate_shared_memory(
            rope.dtype,
            (HEAD_BLOCK, ROPE_BLOCK),
            gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
            rope,
        )
    else:
        query_shared = gl.allocate_shared_memory(
            latent.dtype, (HEAD_BLOCK, 1024), query_layout
        )
        query_shared.slice(0, 512, dim=1).store(latent)
        query_shared.slice(512, ROPE_BLOCK, dim=1).store(rope)
        rope_shared = query_shared.slice(512, ROPE_BLOCK, dim=1)
    return (query_shared.slice(0, 512, dim=1), rope_shared)


@gluon.jit
def _stage_query(
    latent,
    rope,
    HEAD_BLOCK: gl.constexpr,
    ROPE_BLOCK: gl.constexpr,
    QUERY_PHASES: gl.constexpr,
    SEPARATE_ROPE: gl.constexpr,
    QK_TILE: gl.constexpr,
    QUERY_STAGE_STORE: gl.constexpr = False,
):
    if QUERY_STAGE_STORE:
        if QK_TILE == 128:
            tiled = gl.permute(gl.reshape(latent, (HEAD_BLOCK, 4, 128)), (1, 0, 2))
            query_tiles = gl.allocate_shared_memory(
                latent.dtype,
                (4, HEAD_BLOCK, 128),
                gl.SwizzledSharedLayout(16, 1, 8, [2, 1, 0]),
                tiled,
            )
            query_shared = _reinterpret(
                query_tiles,
                latent.dtype,
                (4, HEAD_BLOCK, 128),
                gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
            )
        else:
            query_layout: gl.constexpr = gl.SwizzledSharedLayout(
                16, 1, QUERY_PHASES, [1, 0]
            )
            query_shared = gl.allocate_shared_memory(
                latent.dtype, (HEAD_BLOCK, 512), query_layout
            )
            query_shared.store(latent)
        rope_shared = gl.allocate_shared_memory(
            rope.dtype,
            (HEAD_BLOCK, ROPE_BLOCK),
            gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
            rope,
        )
        return (query_shared, rope_shared)
    elif QK_TILE == 128:
        tiled = gl.permute(gl.reshape(latent, (HEAD_BLOCK, 4, 128)), (1, 0, 2))
        query_tiles = gl.allocate_shared_memory(
            latent.dtype,
            (4, HEAD_BLOCK, 128),
            gl.SwizzledSharedLayout(16, 1, 8, [2, 1, 0]),
            tiled,
        )
        tiled_rope = gl.allocate_shared_memory(
            rope.dtype,
            (HEAD_BLOCK, ROPE_BLOCK),
            gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
            rope,
        )
        return (
            _reinterpret(
                query_tiles,
                latent.dtype,
                (4, HEAD_BLOCK, 128),
                gl.SwizzledSharedLayout(16, 1, 8, [1, 0]),
            ),
            tiled_rope,
        )
    else:
        return _stage_full_query(
            latent, rope, HEAD_BLOCK, ROPE_BLOCK, QUERY_PHASES, SEPARATE_ROPE
        )
