# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Large-batch Gluon kernels for the fused Qwen3-Next attention prologue (gfx950).

Query and KV CTAs have independent wave-local tiles and memory schedules. Small
query tiles share RoPE data and keep rotary partners in registers. Per-role
buffer I/O is enabled only when all reachable byte offsets fit int32. The BF16
rounding before/after RoPE is preserved. Wins the schedule bake-off for
``rows > 24576`` (a large fixed cost the big ragged batches amortize). The
host-side launch policy lives in the wrapper.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_asymmetric_repr = make_kernel_repr(
    "_fused_asymmetric",
    ["ROWS", "HEADS", "Q_BLOCK", "K_BLOCK", "GROUP", "WARPS"],
)


@gluon.jit
def _load(ptr, offset, mask, BUFFER: gl.constexpr, CACHE: gl.constexpr = ""):
    if BUFFER:
        return buffer_load(ptr, offset.to(gl.int32), mask, other=0, cache=CACHE)
    return gl.load(ptr + offset, mask, other=0, cache_modifier=CACHE)


@gluon.jit
def _load_index(
    ptr,
    offset,
    mask,
    BUFFER: gl.constexpr,
    NARROW: gl.constexpr,
    LOW_WORD: gl.constexpr,
):
    if LOW_WORD and NARROW and ptr.dtype.element_ty == gl.int64:
        # Valid indices fit uint32 when the complete table/cache byte span
        # fits int32. Read only the useful low word of int64 metadata.
        result = _load(ptr.to(gl.pointer_type(gl.uint32)), offset * 2, mask, BUFFER)
    else:
        result = _load(ptr, offset, mask, BUFFER)
    return result.to(gl.uint32 if NARROW else gl.int64)


@gluon.jit
def _store(ptr, offset, value, mask, BUFFER: gl.constexpr, CACHE: gl.constexpr):
    if BUFFER:
        buffer_store(
            value.to(ptr.dtype.element_ty), ptr, offset.to(gl.int32), mask, cache=CACHE
        )
    else:
        gl.store(ptr + offset, value, mask, cache_modifier=CACHE)


@gluon.jit
def _load_rope(
    ptr,
    position,
    channel,
    valid,
    ROTARY_DIM: gl.constexpr,
    BUFFER: gl.constexpr,
    JOINED: gl.constexpr,
):
    # KV tiles in selected buckets fetch the contiguous cosine/sine record
    # once, then distribute its halves with wave-local gathers.
    half: gl.constexpr = ROTARY_DIM // 2
    if JOINED:
        table = _load(
            ptr,
            position[:, None] * ROTARY_DIM + (channel[None, :] % ROTARY_DIM),
            valid[:, None],
            BUFFER,
        )
        block: gl.constexpr = table.type.shape[0]
        index = channel[None, :] % half + gl.full(
            (block, 1), 0, gl.int32, table.type.layout
        )
        cosine = gl.gather(table, index, axis=1).to(gl.float32)
        sine = gl.gather(table, index + half, axis=1).to(gl.float32)
    else:
        frequency = channel % half
        cosine = _load(
            ptr,
            position[:, None] * ROTARY_DIM + frequency[None, :],
            valid[:, None],
            BUFFER,
        ).to(gl.float32)
        sine = _load(
            ptr,
            position[:, None] * ROTARY_DIM + half + frequency[None, :],
            valid[:, None],
            BUFFER,
        ).to(gl.float32)
    return cosine, sine


@gluon.jit
def _rotate(
    normalized,
    channel,
    cosine,
    sine,
    ROTARY_DIM: gl.constexpr,
    COMPACT: gl.constexpr,
):
    layout: gl.constexpr = normalized.type.layout
    rank: gl.constexpr = len(normalized.type.shape)
    half: gl.constexpr = ROTARY_DIM // 2
    values = normalized.to(gl.float32)
    # Only the rotary prefix is consumed; XOR avoids a general indexed shuffle.
    if rank == 2:
        block: gl.constexpr = normalized.type.shape[0]
        channel_index = channel[None, :]
        partner_channel = channel ^ half
        partner_index = partner_channel[None, :] + gl.full(
            (block, 1), 0, gl.int32, layout
        )
    else:
        tokens: gl.constexpr = normalized.type.shape[0]
        heads: gl.constexpr = normalized.type.shape[1]
        channel_index = channel.expand_dims(0).expand_dims(0)
        partner_index = (channel_index ^ half) + gl.full(
            (tokens, heads, 1), 0, gl.int32, layout
        )
    if COMPACT:
        partner = gl.gather(normalized, partner_index, axis=rank - 1).to(gl.float32)
    else:
        partner = gl.gather(values, partner_index, axis=rank - 1)
    rotated = gl.where(
        channel_index < half,
        values * cosine - partner * sine,
        values * cosine + partner * sine,
    )
    if COMPACT:
        # The non-rotary tail is already BF16 and needs no round trip.
        return gl.where(channel_index < ROTARY_DIM, rotated.to(gl.bfloat16), normalized)
    return gl.where(channel_index < ROTARY_DIM, rotated, values).to(gl.bfloat16)


@gluon.jit
def _process_tile(
    packed_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    query_ptr,
    gate_ptr,
    key_ptr,
    eps,
    first,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    POS_STRIDE: gl.constexpr,
    QUERY_STRIDE: gl.constexpr,
    KEY_STRIDE: gl.constexpr,
    IS_KEY: gl.constexpr,
    BLOCK: gl.constexpr,
    VECTOR: gl.constexpr,
    LANES: gl.constexpr,
    WARPS: gl.constexpr,
    FULL_TILES: gl.constexpr,
    IO_POLICY: gl.constexpr,
    NARROW: gl.constexpr,
    CACHE: gl.constexpr,
    PREFETCH: gl.constexpr,
    EARLY_AUX: gl.constexpr,
):
    # Buffer-addressing bits separate packed reads, metadata/table reads, and
    # returned-output stores.
    ROLE_BUFFER: gl.constexpr = (IO_POLICY >> 5) if IS_KEY else IO_POLICY
    PACKED_BUFFER: gl.constexpr = ROLE_BUFFER & 1
    META_BUFFER: gl.constexpr = ROLE_BUFFER & 2
    TABLE_BUFFER: gl.constexpr = ROLE_BUFFER & 4
    OUTPUT_BUFFER: gl.constexpr = ROLE_BUFFER & 8
    SOURCE_CACHE: gl.constexpr = (
        ".ca"
        if IS_KEY and ((IO_POLICY >> 18) & 1)
        else (".cg" if (IO_POLICY >> 11) & 1 else "")
    )
    LOW_WORD: gl.constexpr = (IO_POLICY >> 12) & 1
    COMPACT_ROPE: gl.constexpr = (IO_POLICY >> 13) & 1
    REBASE_KEY: gl.constexpr = IS_KEY and ((IO_POLICY >> 14) & 1)
    JOINED_TABLE: gl.constexpr = IS_KEY and ((IO_POLICY >> 16) & 1)
    POSITION_NARROW: gl.constexpr = NARROW and not ((IO_POLICY >> 21) & 1)
    # No reduction crosses a wave boundary, including half/quarter-wave heads.
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VECTOR], [64 // LANES, LANES], [WARPS, 1], [1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    channel = gl.arange(0, 256, gl.SliceLayout(0, layout))
    item = first + gl.arange(0, BLOCK, row_layout)
    if IS_KEY:
        token = item
        head = gl.full((BLOCK,), HEADS, gl.int32, row_layout)
    else:
        token = item // HEADS
        head = item % HEADS
    if FULL_TILES:
        valid = gl.full((BLOCK,), True, gl.int1, row_layout)
    else:
        valid = token < ROWS
    if NARROW:
        token = token.to(gl.uint32)
    else:
        token = token.to(gl.int64)
    packed_offset = (
        token[:, None] * ((HEADS + 1) * 512) + head[:, None] * 512 + channel[None, :]
    )
    if REBASE_KEY:
        # Uniform token contributions belong in scalar 64-bit base pointers.
        # RoPE positions keep their original runtime indices.
        packed_ptr += first.to(gl.int64) * ((HEADS + 1) * 512)
        packed_offset -= first * ((HEADS + 1) * 512)
        key_ptr += first.to(gl.int64) * KEY_STRIDE
        output_token = token - first
    else:
        output_token = token
    position_offset = (
        token.to(gl.int64) if (IO_POLICY >> 19) & 1 else token
    ) * POS_STRIDE
    half: gl.constexpr = ROTARY_DIM // 2
    frequency = channel % half
    if PREFETCH & 17:
        position = _load_index(
            positions_ptr,
            position_offset,
            valid,
            META_BUFFER,
            POSITION_NARROW,
            LOW_WORD,
        )
    if PREFETCH & 1:
        if JOINED_TABLE:
            cosine, sine = _load_rope(
                cos_sin_ptr, position, channel, valid, ROTARY_DIM, TABLE_BUFFER, True
            )
        else:
            cosine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
            sine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + half + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
    if not IS_KEY and PREFETCH & 2:
        companion = _load(
            packed_ptr, packed_offset + 256, valid[:, None], PACKED_BUFFER, SOURCE_CACHE
        )

    source = _load(
        packed_ptr, packed_offset, valid[:, None], PACKED_BUFFER, SOURCE_CACHE
    ).to(gl.float32)
    weight = gl.load(weight_ptr + channel).to(gl.float32)
    if not IS_KEY and EARLY_AUX == 2:
        # Release the gate before forming normalized values.
        if not PREFETCH & 2:
            companion = _load(
                packed_ptr,
                packed_offset + 256,
                valid[:, None],
                PACKED_BUFFER,
                SOURCE_CACHE,
            )
        output_offset = (
            output_token[:, None] * QUERY_STRIDE
            + head[:, None] * 256
            + channel[None, :]
        )
        _store(gate_ptr, output_offset, companion, valid[:, None], OUTPUT_BUFFER, CACHE)
    # This BF16 boundary is part of the numerical contract.
    variance = gl.sum(source * source, axis=1) / 256.0
    normalized = (
        source * gl.rsqrt(variance[:, None] + eps) * (1.0 + weight[None, :])
    ).to(gl.bfloat16)

    if not PREFETCH & 17:
        position = _load_index(
            positions_ptr,
            position_offset,
            valid,
            META_BUFFER,
            POSITION_NARROW,
            LOW_WORD,
        )
    if not PREFETCH & 1:
        if JOINED_TABLE:
            cosine, sine = _load_rope(
                cos_sin_ptr, position, channel, valid, ROTARY_DIM, TABLE_BUFFER, True
            )
        else:
            cosine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
            sine = _load(
                cos_sin_ptr,
                position[:, None] * ROTARY_DIM + half + frequency[None, :],
                valid[:, None],
                TABLE_BUFFER,
            ).to(gl.float32)
    if not IS_KEY and not PREFETCH & 2 and EARLY_AUX != 2:
        companion = _load(
            packed_ptr, packed_offset + 256, valid[:, None], PACKED_BUFFER, SOURCE_CACHE
        )

    if not IS_KEY:
        output_offset = (
            output_token[:, None] * QUERY_STRIDE
            + head[:, None] * 256
            + channel[None, :]
        )
        if EARLY_AUX == 1:
            _store(
                gate_ptr, output_offset, companion, valid[:, None], OUTPUT_BUFFER, CACHE
            )

    result = _rotate(normalized, channel, cosine, sine, ROTARY_DIM, COMPACT_ROPE)
    if IS_KEY:
        _store(
            key_ptr,
            output_token[:, None] * KEY_STRIDE + channel[None, :],
            result,
            valid[:, None],
            OUTPUT_BUFFER,
            CACHE,
        )
    else:
        _store(query_ptr, output_offset, result, valid[:, None], OUTPUT_BUFFER, CACHE)
        if not EARLY_AUX:
            _store(
                gate_ptr, output_offset, companion, valid[:, None], OUTPUT_BUFFER, CACHE
            )


@gluon.jit
def _process_query_tokens(
    packed_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    query_ptr,
    gate_ptr,
    eps,
    first,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    POS_STRIDE: gl.constexpr,
    QUERY_STRIDE: gl.constexpr,
    BLOCK: gl.constexpr,
    VECTOR: gl.constexpr,
    LANES: gl.constexpr,
    WARPS: gl.constexpr,
    FULL_TILES: gl.constexpr,
    IO_POLICY: gl.constexpr,
    NARROW: gl.constexpr,
    CACHE: gl.constexpr,
    EARLY_AUX: gl.constexpr,
):
    # Explicit [token, head, channel] ownership removes the head dimension
    # from RoPE loads, while every RMS reduction remains wave-local.
    tokens_per_tile: gl.constexpr = BLOCK // HEADS
    head_lanes: gl.constexpr = min(HEADS, 64 // LANES)
    head_waves: gl.constexpr = min(WARPS, HEADS // head_lanes)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, VECTOR],
        [64 // (head_lanes * LANES), head_lanes, LANES],
        [WARPS // head_waves, head_waves, 1],
        [2, 1, 0],
    )
    token_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    head_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    channel_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(0, layout))
    table_layout: gl.constexpr = gl.SliceLayout(1, layout)
    token = first // HEADS + gl.arange(0, tokens_per_tile, token_layout)
    head = gl.arange(0, HEADS, head_layout)
    channel = gl.arange(0, 256, channel_layout)
    if FULL_TILES:
        valid = gl.full((tokens_per_tile,), True, gl.int1, token_layout)
    else:
        valid = token < ROWS
    token = token.to(gl.uint32 if NARROW else gl.int64)
    token3 = token.expand_dims(1).expand_dims(2)
    head3 = head.expand_dims(0).expand_dims(2)
    channel3 = channel.expand_dims(0).expand_dims(0)
    valid3 = valid.expand_dims(1).expand_dims(2)
    packed_offset = token3 * ((HEADS + 1) * 512) + head3 * 512 + channel3
    output_offset = token3 * QUERY_STRIDE + head3 * 256 + channel3
    PACKED_BUFFER: gl.constexpr = IO_POLICY & 1
    META_BUFFER: gl.constexpr = IO_POLICY & 2
    TABLE_BUFFER: gl.constexpr = IO_POLICY & 4
    OUTPUT_BUFFER: gl.constexpr = IO_POLICY & 8
    table_token = gl.convert_layout(token, gl.SliceLayout(1, table_layout))
    table_valid = gl.convert_layout(valid, gl.SliceLayout(1, table_layout))
    table_channel = gl.arange(0, 256, gl.SliceLayout(0, table_layout))
    position_offset = (
        table_token.to(gl.int64) if (IO_POLICY >> 19) & 1 else table_token
    ) * POS_STRIDE
    position = _load_index(
        positions_ptr,
        position_offset,
        table_valid,
        META_BUFFER,
        NARROW and not ((IO_POLICY >> 21) & 1),
        False,
    )
    cosine, sine = _load_rope(
        cos_sin_ptr,
        position,
        table_channel,
        table_valid,
        ROTARY_DIM,
        TABLE_BUFFER,
        False,
    )
    source = _load(packed_ptr, packed_offset, valid3, PACKED_BUFFER).to(gl.float32)
    weight = gl.load(weight_ptr + channel).to(gl.float32)
    if EARLY_AUX == 2:
        companion = _load(packed_ptr, packed_offset + 256, valid3, PACKED_BUFFER)
        _store(gate_ptr, output_offset, companion, valid3, OUTPUT_BUFFER, CACHE)
    variance = gl.sum(source * source, axis=2) / 256.0
    normalized = (
        source
        * gl.rsqrt(variance.expand_dims(2) + eps)
        * (1.0 + weight.expand_dims(0).expand_dims(0))
    ).to(gl.bfloat16)
    if EARLY_AUX != 2:
        companion = _load(packed_ptr, packed_offset + 256, valid3, PACKED_BUFFER)
    if EARLY_AUX == 1:
        _store(gate_ptr, output_offset, companion, valid3, OUTPUT_BUFFER, CACHE)
    result = _rotate(
        normalized,
        channel,
        cosine.expand_dims(1),
        sine.expand_dims(1),
        ROTARY_DIM,
        (IO_POLICY >> 13) & 1,
    )
    _store(query_ptr, output_offset, result, valid3, OUTPUT_BUFFER, CACHE)
    if not EARLY_AUX:
        _store(gate_ptr, output_offset, companion, valid3, OUTPUT_BUFFER, CACHE)


@gluon.jit(repr=_fused_asymmetric_repr)
def _fused_asymmetric(
    packed_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    query_ptr,
    gate_ptr,
    key_ptr,
    eps,
    ROWS: gl.constexpr,
    HEADS: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    POS_STRIDE: gl.constexpr,
    QUERY_STRIDE: gl.constexpr,
    KEY_STRIDE: gl.constexpr,
    Q_BLOCK: gl.constexpr,
    K_BLOCK: gl.constexpr,
    Q_VECTOR: gl.constexpr,
    K_VECTOR: gl.constexpr,
    Q_LANES: gl.constexpr,
    K_LANES: gl.constexpr,
    WARPS: gl.constexpr,
    GROUP: gl.constexpr,
    PARTITIONS: gl.constexpr,
    KEY_FIRST: gl.constexpr,
    IO_POLICY: gl.constexpr,
    NARROW: gl.constexpr,
    CACHE: gl.constexpr,
    Q_PREFETCH: gl.constexpr,
    K_PREFETCH: gl.constexpr,
    Q_EARLY: gl.constexpr,
    K_EARLY: gl.constexpr,
):
    offset_type: gl.constexpr = gl.uint32 if NARROW else gl.int64
    pid = gl.program_id(0).to(offset_type)
    q_tiles: gl.constexpr = GROUP * HEADS // Q_BLOCK
    k_tiles: gl.constexpr = GROUP // K_BLOCK
    total: gl.constexpr = gl.cdiv(ROWS, GROUP) * (q_tiles + k_tiles)
    if PARTITIONS > 1:
        # Balanced partition transpose: the first total % PARTITIONS partitions
        # have one extra task. This stays bijective for arbitrary grid lengths.
        partition = pid % PARTITIONS
        pid = (
            partition * (total // PARTITIONS)
            + gl.minimum(partition, total % PARTITIONS)
            + pid // PARTITIONS
        )
    group = pid // (q_tiles + k_tiles)
    role = pid % (q_tiles + k_tiles)
    if KEY_FIRST:
        is_key = role < k_tiles
        k_first = group * GROUP + role * K_BLOCK
        q_first = group * GROUP * HEADS + (role - k_tiles) * Q_BLOCK
    else:
        is_key = role >= q_tiles
        k_first = group * GROUP + (role - q_tiles) * K_BLOCK
        q_first = group * GROUP * HEADS + role * Q_BLOCK
    full_tiles: gl.constexpr = ROWS % GROUP == 0
    # This branch is uniform across the entire CTA.
    if is_key:
        _process_tile(
            packed_ptr,
            k_weight_ptr,
            positions_ptr,
            cos_sin_ptr,
            query_ptr,
            gate_ptr,
            key_ptr,
            eps,
            k_first,
            ROWS,
            HEADS,
            ROTARY_DIM,
            POS_STRIDE,
            QUERY_STRIDE,
            KEY_STRIDE,
            True,
            K_BLOCK,
            K_VECTOR,
            K_LANES,
            WARPS,
            full_tiles,
            IO_POLICY,
            NARROW,
            CACHE,
            K_PREFETCH,
            K_EARLY,
        )
    else:
        if ((IO_POLICY >> 17) & 1) and Q_BLOCK >= HEADS:
            _process_query_tokens(
                packed_ptr,
                q_weight_ptr,
                positions_ptr,
                cos_sin_ptr,
                query_ptr,
                gate_ptr,
                eps,
                q_first,
                ROWS,
                HEADS,
                ROTARY_DIM,
                POS_STRIDE,
                QUERY_STRIDE,
                Q_BLOCK,
                Q_VECTOR,
                Q_LANES,
                WARPS,
                full_tiles,
                IO_POLICY,
                NARROW,
                CACHE,
                Q_EARLY,
            )
        else:
            _process_tile(
                packed_ptr,
                q_weight_ptr,
                positions_ptr,
                cos_sin_ptr,
                query_ptr,
                gate_ptr,
                key_ptr,
                eps,
                q_first,
                ROWS,
                HEADS,
                ROTARY_DIM,
                POS_STRIDE,
                QUERY_STRIDE,
                KEY_STRIDE,
                False,
                Q_BLOCK,
                Q_VECTOR,
                Q_LANES,
                WARPS,
                full_tiles,
                IO_POLICY,
                NARROW,
                CACHE,
                Q_PREFETCH,
                Q_EARLY,
            )
