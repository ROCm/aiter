# SPDX-License-Identifier: MIT
"""The KV seam between the projections and attention, in one launch.

RoPE, FP8 quantization and the window row used to be three kernels whose
per-launch cost on MI355X was 5.5-6.8us each whatever shape they ran --
dispatch, not work. Q's rotation rides along because it is the one piece of
the seam that cannot move: it waits on `wq_b`, so it has to run here anyway,
and a program this far under the launch floor does not notice the extra rows.

The RMSNorm ahead of this stays where it is. It shares a launch with Q's, so
pulling its half in saves nothing and would leave Q's on a different kernel
than the one whose bytes the checkpoint was validated against.

The row this writes is `packed_rows`': `head_dim` FP8 values then
`head_dim // 32` E8M0 scales. Where it lands is the cache's, not this file's.
"""

from aiter.ops.mono_standard.codegen import POLICY_ID, compiled_a05

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.gfx1250 import scaled_downcast

from aiter.ops.mono_standard.atom_helpers import (
    AMAX_FLOOR,
    FP8_MAX,
    FP8_TL_DTYPE,
    ceil_pow2_code,
)
from aiter.ops.mono_standard.atom_helpers import window_row

# A token's query heads rotated by one program rather than one each. The
# per-head programs each move one 64-element tail, so a launch of hundreds of
# tokens is nearly all program scheduling; on MI455X the whole launch took
# 2.00 -> 1.51 us at 32 tokens and 16.2 -> 2.2 us at 768, never slower from 1
# token up, with the same bytes out. False only for the tests' per-head
# reference.
_HEAD_BLOCK = True

# C16 mono launch contract. These are compile-time choices, not new API args.
_PERSISTENT_TGS = 256
_PERSISTENT_GRID = (16, 16, 1)
_NUM_WARPS = 4
# Keep allocatable scalar registers below s104/s105. Final ISA must still be
# audited: explicit/implicit machine-register writes are not proved by a hint.
_LLVM_FN_ATTRS = (('amdgpu-num-sgpr', '104'), ('amdgpu-sched-strategy', 'iterative-ilp'), ('mono-kernel-contract', POLICY_ID))


@triton.jit
def _rotate_head(query, token, slot, position, cos, sin, HEADS, D, ROPE):
    """One query head's RoPE tail, in place. GPT-J pairs, as V4 rotates."""
    half: tl.constexpr = ROPE // 2
    pair = tl.arange(0, half)
    cosine = tl.load(cos + position * half + pair).to(tl.float32)
    sine = tl.load(sin + position * half + pair).to(tl.float32)
    head = query + (token * HEADS + slot) * D + (D - ROPE)
    tail = tl.arange(0, ROPE)
    even, odd = tl.split(tl.reshape(tl.load(head + tail).to(tl.float32), (half, 2)))
    tl.store(
        head + tail,
        tl.reshape(
            tl.join(even * cosine - odd * sine, even * sine + odd * cosine), (ROPE,)
        ).to(query.dtype.element_ty),
    )


@triton.jit
def _rotate_heads(query, token, position, cos, sin, HEADS, D, ROPE):
    """Every query head's RoPE tail of one token, in place: `_rotate_head`
    over a `[HEADS, ROPE]` tile, the cos / sin row loaded once."""
    half: tl.constexpr = ROPE // 2
    pair = tl.arange(0, half)
    cosine = tl.load(cos + position * half + pair).to(tl.float32)
    sine = tl.load(sin + position * half + pair).to(tl.float32)
    heads = tl.arange(0, HEADS)
    tail = tl.arange(0, ROPE)
    ptr = query + (token * HEADS + heads[:, None]) * D + (D - ROPE) + tail[None, :]
    even, odd = tl.split(tl.reshape(tl.load(ptr).to(tl.float32), (HEADS, half, 2)))
    tl.store(
        ptr,
        tl.reshape(
            tl.join(
                even * cosine[None, :] - odd * sine[None, :],
                even * sine[None, :] + odd * cosine[None, :],
            ),
            (HEADS, ROPE),
        ).to(query.dtype.element_ty),
    )


@triton.jit
def _quantized_row(row, position, cos, sin, D, ROPE):
    """Rotate and quantize one row (a normed KV row or a query head); returns
    its bytes, E8M0 codes and power-of-two steps."""
    half: tl.constexpr = ROPE // 2
    dim = tl.arange(0, D)
    activation = row.dtype.element_ty
    normed = tl.load(row + dim).to(tl.float32)

    even, odd = tl.split(tl.reshape(normed, (D // 2, 2)))
    phase = tl.arange(0, D // 2) - (D - ROPE) // 2
    # A NoPE pair reads no table entry, and `other=` leaves it alone: cos 1
    # with sin 0 is the identity rotation.
    cosine = tl.load(cos + position * half + phase, phase >= 0, other=1.0).to(
        tl.float32
    )
    sine = tl.load(sin + position * half + phase, phase >= 0, other=0.0).to(tl.float32)
    # Back through the activation dtype first: the kernels this replaces each
    # returned BF16, so the amax has to see the rounded row.
    groups = (
        tl.reshape(
            tl.join(even * cosine - odd * sine, even * sine + odd * cosine),
            (D // 32, 32),
        )
        .to(activation)
        .to(tl.float32)
    )

    amax = tl.maximum(tl.max(tl.abs(groups), 1), AMAX_FLOOR)
    code = ceil_pow2_code(amax * (1.0 / FP8_MAX))
    step = (code << 23).to(tl.float32, bitcast=True)
    quantized = tl.minimum(tl.maximum(groups / step[:, None], -FP8_MAX), FP8_MAX).to(
        FP8_TL_DTYPE
    )
    return quantized, code, step


@triton.jit
def _rope_window_kernel(
    query,
    kv,
    cos,
    sin,
    positions,
    qat,
    values,
    scales,
    values_stride,
    scales_stride,
    query_values,
    query_scales,
    pool,
    batches,
    cu,
    slots,
    ring_start,
    kv_stride,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    ROPE: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    ROW_UNITS: tl.constexpr,
    EMIT_QAT: tl.constexpr,
    EMIT_PACKED: tl.constexpr,
    EMIT_QUERY_FP8: tl.constexpr,
    WRITE_RING: tl.constexpr,
    PACKED_RING: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    TOKENS: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """Persistent programs visit logical (token, slot) tasks; the last is KV.

    With `HEAD_BLOCK` (the in-place BF16 query) slot 0 rotates all of a
    token's heads and slot 1 runs the whole KV chain: one query program per
    token. Otherwise (the MXFP8 query, odd shapes) `HEADS` slots each rotate
    or quantize one query head and slot `HEADS` is the KV. Either way the KV
    chain has its own program rather than sharing the token's, which keeps a
    decode bucket -- a few dozen rows -- wide enough to fill the device; the
    two branches are otherwise independent.
    """
    # The head-blocked query is the in-place BF16 rotation only; the host
    # keeps an MXFP8-emitting query on a program a head.
    tl.static_assert(not (HEAD_BLOCK and EMIT_QUERY_FP8))
    # A fixed physical grid; empty workers execute no loads/stores. TOKENS
    # and SLOTS specialize the program and add no runtime kernel arguments.
    for task in range((tl.program_id(0) + 16 * tl.program_id(1)), TOKENS * SLOTS, 256):
        token = task // SLOTS
        slot = task % SLOTS
        position = tl.load(positions + token)
        dim = tl.arange(0, D)
        if HEAD_BLOCK:
            query_slot = slot == 0
        else:
            query_slot = slot < HEADS
        if query_slot:
            if HEAD_BLOCK:
                _rotate_heads(query, token, position, cos, sin, HEADS, D, ROPE)
            elif EMIT_QUERY_FP8:
                # The FP8 pool's attention reads the query as MXFP8, by the KV
                # row's rule; the BF16 head is not written.
                head = token * HEADS + slot
                quantized, code, _ = _quantized_row(
                    query + head * D, position, cos, sin, D, ROPE
                )
                tl.store(query_values + head * D + dim, tl.reshape(quantized, (D,)))
                tl.store(query_scales + head * (D // 32) + tl.arange(0, D // 32), code)
            else:
                _rotate_head(query, token, slot, position, cos, sin, HEADS, D, ROPE)
        else:
            quantized, code, step = _quantized_row(
                kv + token * kv_stride, position, cos, sin, D, ROPE
            )

            if EMIT_QAT:
                tl.store(
                    qat + token * D + dim,
                    tl.reshape(quantized.to(tl.float32) * step[:, None], (D,)),
                )
            if EMIT_PACKED:
                tl.store(
                    values + token * values_stride + dim, tl.reshape(quantized, (D,))
                )
                tl.store(scales + token * scales_stride + tl.arange(0, D // 32), code)
            if WRITE_RING:
                # Two exits, the ones both window writers already have: a padding
                # row owns no request, and a row the ring has scrolled past has
                # nowhere to land. The second is also what keeps two tokens of one
                # request off the same row -- it admits exactly the last
                # `RING_SLOTS` of them. Logical task bounds stay at the captured
                # token count; the physical grid is always 256 TG.
                batch = tl.load(batches + token)
                if batch >= 0 and token >= tl.load(cu + batch + 1) - RING_SLOTS:
                    base = (
                        window_row(
                            tl.load(slots + batch).to(tl.int64),
                            position,
                            ring_start,
                            RING_SLOTS,
                            SLOT_ROWS,
                            RING_STRIDE,
                            RUN_ROWS,
                        )
                        * ROW_UNITS
                    )
                    if PACKED_RING:
                        tl.store(
                            pool + base + dim,
                            tl.reshape(quantized, (D,)).to(tl.uint8, bitcast=True),
                        )
                        tl.store(pool + base + D + tl.arange(0, D // 32), code)
                    else:
                        tl.store(
                            pool + base + dim,
                            tl.reshape(quantized.to(tl.float32) * step[:, None], (D,)),
                        )


@gluon.jit
def _rope_window_fp8_rows(
    query,
    kv,
    cos,
    sin,
    positions,
    qat,
    values,
    scales,
    values_stride,
    scales_stride,
    query_values,
    query_scales,
    pool,
    batches,
    cu,
    slots,
    ring_start,
    kv_stride,
    HEADS: gl.constexpr,
    D: gl.constexpr,
    ROPE: gl.constexpr,
    RING_SLOTS: gl.constexpr,
    SLOT_ROWS: gl.constexpr,
    RING_STRIDE: gl.constexpr,
    RUN_ROWS: gl.constexpr,
    ROW_UNITS: gl.constexpr,
    EMIT_QAT: gl.constexpr,
    EMIT_PACKED: gl.constexpr,
    EMIT_QUERY_FP8: gl.constexpr,
    WRITE_RING: gl.constexpr,
    PACKED_RING: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    TOKENS: gl.constexpr,
    SLOTS: gl.constexpr,
    row_start,
    IS_QUERY: gl.constexpr,
    BATCH: gl.constexpr,
):
    # Explicit row-per-wave placement keeps pair rearrangement and group32
    # reduction inside a wave. Preserve native's FP32 rotation, BF16 rounding,
    # amax floor, E8M0 ceil and FP8 conversion in exactly that order.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [4, 1], [1, 0])
    rows = gl.arange(0, BATCH, layout=gl.SliceLayout(1, layout))
    dim = gl.arange(0, D, layout=gl.SliceLayout(0, layout))
    is_query: gl.constexpr = IS_QUERY
    row_index = row_start + rows
    token = row_index // HEADS if IS_QUERY else row_index
    valid = row_index < (TOKENS * HEADS if IS_QUERY else TOKENS)
    is_kv = valid & (not IS_QUERY)
    head = row_index
    position = gl.load(positions + token, valid, other=0)
    row = query + head * D if IS_QUERY else kv + token * kv_stride
    if WRITE_RING and not IS_QUERY:
        batch = gl.load(batches + token, is_kv, other=-1)
        has_batch = is_kv & (batch >= 0)
        end = gl.load(cu + batch + 1, has_batch, other=0)
        write = has_batch & (token >= end - RING_SLOTS)
        ring_slot = gl.load(slots + batch, write, other=0).to(gl.int64)
        ring_pos = position % RING_SLOTS
        base = (
            ring_slot * SLOT_ROWS
            + ring_start
            + (ring_pos // RING_STRIDE) * RUN_ROWS
            + ring_pos % RING_STRIDE
        ) * ROW_UNITS
    normed = gl.load(row[:, None] + dim[None, :], valid[:, None], other=0).to(
        gl.float32
    )
    even, odd = gl.split(gl.reshape(normed, (BATCH, D // 2, 2)))
    phase = (
        gl.arange(0, D // 2, layout=gl.SliceLayout(0, even.type.layout))
        - (D - ROPE) // 2
    )
    pair_position = gl.convert_layout(position, gl.SliceLayout(1, even.type.layout))
    pair_valid = gl.convert_layout(valid, gl.SliceLayout(1, even.type.layout))
    table = pair_position[:, None] * (ROPE // 2) + phase[None, :]
    table_mask = pair_valid[:, None] & (phase[None, :] >= 0)
    cosine = gl.load(cos + table, table_mask, other=1.0).to(gl.float32)
    sine = gl.load(sin + table, table_mask, other=0.0).to(gl.float32)
    groups = gl.reshape(
        gl.join(even * cosine - odd * sine, even * sine + odd * cosine),
        (BATCH, D // 32, 32),
    ).to(query.dtype.element_ty)
    # Reduce rounded BF16 values with hardware max-number operations.
    # gl.max ignores NaNs; the final maximum applies the native finite floor.
    amax = gl.maximum(gl.max(gl.abs(groups.to(gl.float32)), 2), AMAX_FLOOR)
    bits = (amax * (1.0 / FP8_MAX)).to(gl.uint32, bitcast=True)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(gl.uint32)
    group = gl.arange(0, D // 32, layout=gl.SliceLayout(0, code.type.layout))
    scale_rows: gl.constexpr = gl.SliceLayout(1, code.type.layout)
    scale_head = gl.convert_layout(head, scale_rows)
    scale_token = gl.convert_layout(token, scale_rows)
    scale_query = gl.convert_layout(valid & is_query, scale_rows)
    scale_kv = gl.convert_layout(is_kv, scale_rows)
    step = (code << 23).to(gl.float32, bitcast=True)
    rounded_bits = groups.to(gl.bfloat16).to(gl.uint16, bitcast=True)
    magnitude = rounded_bits & 0x7FFF
    # Native turns an infinite scale into signed zero for finite inputs;
    # infinity / infinity becomes NaN before the clamp. E8M0 255 itself is
    # reserved by hardware, so use scale 1 for this already-normalized case.
    hardware_bits = gl.where(
        code[:, :, None] == 255,
        gl.where(magnitude < 0x7F80, rounded_bits & 0x8000, 0x7FC0),
        rounded_bits,
    ).to(gl.uint16)
    hardware_scale = gl.where(code == 255, 127, code).to(gl.uint8)
    quantized = scaled_downcast(
        hardware_bits.to(gl.bfloat16, bitcast=True),
        hardware_scale[:, :, None],
        "e4m3",
        axis=2,
    )
    # Native min/max maps NaN to -448 (0xfe). Repair four packed bytes
    # together, including either NaN sign; all finite bytes are unchanged.
    pairs0, pairs1 = gl.split(
        gl.reshape(quantized.to(gl.uint8, bitcast=True), (BATCH, D // 4, 2, 2))
    )
    b0, b2 = gl.split(pairs0)
    b1, b3 = gl.split(pairs1)
    packed = (
        b0.to(gl.uint32)
        | (b1.to(gl.uint32) << 8)
        | (b2.to(gl.uint32) << 16)
        | (b3.to(gl.uint32) << 24)
    )
    nan_mask = ((packed & 0x7F7F7F7F) + 0x01010101) & 0x80808080
    packed = (packed | nan_mask) ^ (nan_mask >> 7)
    fixed = gl.join(
        gl.join(packed.to(gl.uint8), (packed >> 16).to(gl.uint8)),
        gl.join((packed >> 8).to(gl.uint8), (packed >> 24).to(gl.uint8)),
    )
    quantized = gl.reshape(fixed, (BATCH, D // 32, 32)).to(FP8_TL_DTYPE, bitcast=True)
    payload = gl.convert_layout(gl.reshape(quantized, (BATCH, D)), layout)
    if is_query:
        gl.store(
            query_values + head[:, None] * D + dim[None, :],
            payload,
            (valid & is_query)[:, None],
        )
        gl.store(
            query_scales + scale_head[:, None] * (D // 32) + group[None, :],
            code,
            scale_query[:, None],
        )
    else:

        if EMIT_QAT:
            gl.store(
                qat + token[:, None] * D + dim[None, :],
                gl.convert_layout(
                    gl.reshape(quantized.to(gl.float32) * step[:, :, None], (BATCH, D)),
                    layout,
                ),
                is_kv[:, None],
            )
        if EMIT_PACKED:
            gl.store(
                values + token[:, None] * values_stride + dim[None, :],
                payload,
                is_kv[:, None],
            )
            gl.store(
                scales + scale_token[:, None] * scales_stride + group[None, :],
                code,
                scale_kv[:, None],
            )
        if WRITE_RING:
            if PACKED_RING:
                gl.store(
                    pool + base[:, None] + dim[None, :],
                    payload.to(gl.uint8, bitcast=True),
                    write[:, None],
                )
                gl.store(
                    pool
                    + gl.convert_layout(base, scale_rows)[:, None]
                    + D
                    + group[None, :],
                    code,
                    gl.convert_layout(write, scale_rows)[:, None],
                )
            else:
                gl.store(
                    pool + base[:, None] + dim[None, :],
                    gl.convert_layout(
                        gl.reshape(
                            quantized.to(gl.float32) * step[:, :, None], (BATCH, D)
                        ),
                        layout,
                    ),
                    write[:, None],
                )


@gluon.jit
def _rope_window_fp8_kernel(
    query,
    kv,
    cos,
    sin,
    positions,
    qat,
    values,
    scales,
    values_stride,
    scales_stride,
    query_values,
    query_scales,
    pool,
    batches,
    cu,
    slots,
    ring_start,
    kv_stride,
    HEADS: gl.constexpr,
    D: gl.constexpr,
    ROPE: gl.constexpr,
    RING_SLOTS: gl.constexpr,
    SLOT_ROWS: gl.constexpr,
    RING_STRIDE: gl.constexpr,
    RUN_ROWS: gl.constexpr,
    ROW_UNITS: gl.constexpr,
    EMIT_QAT: gl.constexpr,
    EMIT_PACKED: gl.constexpr,
    EMIT_QUERY_FP8: gl.constexpr,
    WRITE_RING: gl.constexpr,
    PACKED_RING: gl.constexpr,
    HEAD_BLOCK: gl.constexpr,
    TOKENS: gl.constexpr,
    SLOTS: gl.constexpr,
):
    # C16: 192 query tiles + 24 KV tiles. The physical launch stays at 256
    # TG x 128 threads; 40 TG have no task. Larger shapes revisit tile+256.
    # Traverse y first within the fixed 16x16 grid; logical KV tiles precede Q.
    # This mapping keeps 256 workers and the same persistent-loop stride.
    # Each tile has a single row type; query-only calls retain the same mapping.
    gl.static_assert(HEADS == 16 and D == 512 and ROPE == 64)
    gl.static_assert(EMIT_QUERY_FP8 and not HEAD_BLOCK)
    q_tiles: gl.constexpr = triton.cdiv(TOKENS * HEADS, 8)
    kv_tiles: gl.constexpr = triton.cdiv(TOKENS, 4) if SLOTS > HEADS else 0
    for iteration in range(triton.cdiv(q_tiles + kv_tiles, 256)):
        tile = (gl.program_id(1) + 16 * gl.program_id(0)) + iteration * 256
        if tile < q_tiles + kv_tiles:
            if tile >= kv_tiles:
                _rope_window_fp8_rows(
                    query,
                    kv,
                    cos,
                    sin,
                    positions,
                    qat,
                    values,
                    scales,
                    values_stride,
                    scales_stride,
                    query_values,
                    query_scales,
                    pool,
                    batches,
                    cu,
                    slots,
                    ring_start,
                    kv_stride,
                    HEADS,
                    D,
                    ROPE,
                    RING_SLOTS,
                    SLOT_ROWS,
                    RING_STRIDE,
                    RUN_ROWS,
                    ROW_UNITS,
                    EMIT_QAT,
                    EMIT_PACKED,
                    EMIT_QUERY_FP8,
                    WRITE_RING,
                    PACKED_RING,
                    HEAD_BLOCK,
                    TOKENS,
                    SLOTS,
                    (tile - kv_tiles) * 8,
                    True,
                    8,
                )
            else:
                _rope_window_fp8_rows(
                    query,
                    kv,
                    cos,
                    sin,
                    positions,
                    qat,
                    values,
                    scales,
                    values_stride,
                    scales_stride,
                    query_values,
                    query_scales,
                    pool,
                    batches,
                    cu,
                    slots,
                    ring_start,
                    kv_stride,
                    HEADS,
                    D,
                    ROPE,
                    RING_SLOTS,
                    SLOT_ROWS,
                    RING_STRIDE,
                    RUN_ROWS,
                    ROW_UNITS,
                    EMIT_QAT,
                    EMIT_PACKED,
                    EMIT_QUERY_FP8,
                    WRITE_RING,
                    PACKED_RING,
                    HEAD_BLOCK,
                    TOKENS,
                    SLOTS,
                    tile * 4,
                    False,
                    4,
                )


def rope_quant_window(
    query,
    kv,
    cos,
    sin,
    positions,
    *,
    rope_dim,
    qat=None,
    values=None,
    scales=None,
    ring=None,
    query_fp8=None,
):
    """Rotate `query` in place and run the KV row to whichever sinks are given.

    `qat` takes the dequantized BF16 row an extend pass attends to, `values`
    and `scales` the FP8 pair (rows of any stride) a later window write or an
    FP8 attention consumes, and `ring` the `(pool, step, window)` a decode
    stores straight into. With none of them there is no KV half, and `kv` may
    be None. `query_fp8`: `(values, scales)` the rotated query is quantized
    into instead of being rotated in place.

    One address serves both ring layouts: `window_row` returns a byte offset
    for a byte pool (FP8 values then codes, `V41PoolGeometry.window_bytes`)
    and a row index for a `[rows, head_dim]` BF16 one.
    """
    tokens, heads, dim = query.shape
    kv_half = ring is not None or qat is not None or values is not None
    pool, step, window = ring or (None, None, None)
    byte_ring = pool is not None and pool.dtype == torch.uint8
    query_values, query_scales = query_fp8 or (None, None)
    # The tile's shape must be powers of two; anything else, a query-less KV
    # launch (no heads: slot 0 must stay the KV), and the MXFP8 query keep a
    # program a head.
    head_block = (
        _HEAD_BLOCK
        and query_fp8 is None
        and heads > 0
        and (heads & (heads - 1)) == 0
        and rope_dim >= 2
        and (rope_dim & (rope_dim - 1)) == 0
    )
    query_programs = 1 if head_block else heads
    fp8_rows = (
        query_fp8 is not None
        and heads == 16
        and dim == 512
        and rope_dim == 64
        and query.dtype == torch.bfloat16
        and (not kv_half or kv.dtype == torch.bfloat16)
    )
    kernel = _rope_window_fp8_kernel if fp8_rows else _rope_window_kernel
    kernel[_PERSISTENT_GRID](
        query,
        kv if kv_half else query,
        cos,
        sin,
        positions,
        qat,
        values,
        None if scales is None else scales.view(torch.uint8),
        0 if values is None else values.stride(-2),
        0 if scales is None else scales.stride(-2),
        query_values,
        None if query_scales is None else query_scales.view(torch.uint8),
        pool,
        None if step is None else step.batch_ids,
        None if step is None else step.cu_seqlens_q,
        None if step is None else step.slots,
        0 if window is None else window.ring_start,
        kv.stride(0) if kv_half else 0,
        HEADS=heads,
        D=dim,
        ROPE=rope_dim,
        RING_SLOTS=1 if window is None else window.ring_slots,
        SLOT_ROWS=0 if window is None else window.slot_rows,
        RING_STRIDE=1 if window is None else window.ring_stride,
        RUN_ROWS=0 if window is None else window.run_rows,
        ROW_UNITS=1 if pool is None or byte_ring else dim,
        PACKED_RING=byte_ring,
        EMIT_QAT=qat is not None,
        EMIT_PACKED=values is not None,
        EMIT_QUERY_FP8=query_fp8 is not None,
        WRITE_RING=ring is not None,
        HEAD_BLOCK=head_block,
        TOKENS=tokens,
        SLOTS=query_programs + kv_half,
        num_warps=_NUM_WARPS,
        llvm_fn_attrs=_LLVM_FN_ATTRS,
    )

# Activate only this operator's explicit MONO compiler stage.
rope_quant_window = compiled_a05(rope_quant_window)
