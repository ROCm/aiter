# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Device codecs of the pre-attention monokernel: one workgroup's share of a role,
quantized into this rank's staging for every destination in the receive layout
(``pre_attention_monokernel_layout``). Scale rules, rounding and packing are the Attention
A2A transport's (``attention_a2a_codecs``), so the received bytes match its packed output.

Row codecs walk a workgroup's token rows (bid, bid + workgroups, ...) in units of 8
channels: consecutive lanes take consecutive units of a row, so every aligned four-lane
group holds one 32-channel MX scale block of one (token, head).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath

from aiter.ops.mha_v4 import AttentionPack

from .attention_a2a_codecs import (
    fp6p_source_token,
    pack_int8_pair,
    pack_transport_pair,
    pack_transport_words,
    pack_v4_v_pair,
    transport_scale,
    v4_fp6_scale,
    v4_fp8_scale,
)
from .quant_utils import emit_f32_to_e2m3_native
from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

THREADS = 256
UNROLL = 8  # 16-byte loads each thread has in flight
HEAD_DIM = 128


def make_row_quantize(
    layout,
    rank,
    world,
    rows,
    heads_local,
    workgroups,
    factors=True,
    first=0,
    count=None,
):
    """quantize(values, factors, staging, scratch, scales) for one role.

    Token-major roles: per-tensor E4M3 or INT8 (``scales``: the per-destination scales),
    per-channel E4M3 (``scales``: address of the [world][heads_local * 128] FP32 scales) or
    an MX row codec (``scales`` unused); tiled codecs have their own functions below.
    ``factors`` (the argument): address of the RMS factors of the role's rows, read for Q/K
    when ``factors`` (the option) is set. ``staging`` points at this role's staging, one
    ``layout.stage_bytes`` block per destination. With ``first``/``count``, only a
    workgroup's rows bid + i * workgroups for i in [first, first + count).
    """
    codec, kind = layout.codec, layout.scale
    has_factor = layout.role < 2 and factors
    if codec == "mxfp6" and layout.role < 2:
        return _make_fp6_quantize(
            layout, rank, world, rows, heads_local, workgroups, has_factor, first, count
        )
    if codec == "mxfp6_p" and layout.role == 2:
        return _make_fp6p_v_quantize(layout, rank, world, rows, heads_local, workgroups)
    if codec == "mxfp4" and layout.role == 2:
        return _make_fp4_v_quantize(layout, rank, world, rows, heads_local, workgroups)
    if kind == "channel":
        return _make_channel_quantize(
            layout, rank, world, rows, heads_local, workgroups, first, count
        )
    if codec not in ("e4m3", "int8", "e4m3_pc", "mxfp8", "mxfp4"):
        raise NotImplementedError(f"row codec {codec} for {'QKV'[layout.role]}")
    # MXFP4 K: per (head, tile) records of 2048 words, word token * 4 + chunk % 4 + chunk // 4 * 512.
    tiled = codec == "mxfp4" and layout.role == 1
    scale_stage = (
        next(part.stage for part in layout.ranges if part.region == "scale")
        if kind == "block"
        else 0
    )
    unit_bytes = 4 if codec == "mxfp4" else 8  # one unit = 8 values
    channels = heads_local * HEAD_DIM  # one destination's share of a token row
    dest_units = channels // 8
    chunks = world * dest_units
    if count is None:
        count = (rows + workgroups - 1) // workgroups - first
    units = count * chunks
    trips = (units + THREADS * UNROLL - 1) // (THREADS * UNROLL)
    row_bytes = dest_units * unit_bytes
    scale_row = heads_local * HEAD_DIM // 32
    # Local rows at or past it are masked (trimmed K/V).
    sequence_end = layout.sequence - rank * rows

    @flyc.jit
    def quantize(values, factors, staging, scratch, scales):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % 64
        source = ptr_buf_tensor(values, fx.BFloat16, unit_elems=8, unit_stride=1)
        row_factor = ptr_buf_tensor(factors, fx.Float32)
        target = ptr_buf_tensor(
            staging, fx.Int32, unit_elems=unit_bytes // 4, unit_stride=1
        )
        target_bytes = ptr_buf_tensor(staging, fx.Int8)

        def load(trip, u):
            unit = (trip * UNROLL + u) * THREADS + tid
            local = unit // chunks
            # Units past the range get a row past the end, so one test covers both.
            row = (local < count).select(
                bid + (local + first) * workgroups, fx.Int32(rows)
            )
            chunk = unit % chunks
            valid = row < min(rows, sequence_end)
            row = valid.select(row, fx.Int32(0))
            # unit_stride=1 views take element indices.
            data = fx.Vector(
                buf_copy_load(source, (row * chunks + chunk) * 8, fx.BFloat16, 8)
            )
            factor = None
            if const_expr(has_factor):
                factor = fx.Float32(buf_copy_load(row_factor, row, fx.Float32))
            return valid, row, chunk, data, factor

        def store(valid, row, chunk, data, factor):
            data = data.to(fx.Float32)
            dest = chunk // dest_units
            within = chunk % dest_units
            base = fx.Int64(dest) * layout.stage_bytes
            if const_expr(tiled):
                token = rank * rows + row
                tok, part = token % 128, within % 16
                offset = (
                    base
                    + fx.Int64(within // 16) * layout.head_stage
                    + fx.Int64(token // 128 - layout.first_tile) * 8192
                    + fx.Int64((tok * 4 + part % 4 + part // 4 * 512) * 4)
                )
            else:
                offset = (
                    base + fx.Int64(row) * row_bytes + fx.Int64(within * unit_bytes)
                )
            offset = valid.select(offset, fx.Int64(scratch))
            if const_expr(kind == "tensor"):
                # The A2A op's per-tensor push: the row factor first; E4M3 multiplies by
                # 1 / scale, INT8 divides by the scale.
                if const_expr(has_factor):
                    data = data * fx.Vector.filled(8, factor, fx.Float32)
                scale = scales[0]
                for d in range_constexpr(1, world):
                    scale = (dest == d).select(scales[d], scale)
                if const_expr(codec == "int8"):
                    pairs = [
                        pack_int8_pair(data[2 * i] / scale, data[2 * i + 1] / scale)
                        for i in range(4)
                    ]
                else:
                    reciprocal = fx.Float32(1.0) / scale
                    pairs = [
                        pack_transport_pair(
                            data[2 * i] * reciprocal,
                            data[2 * i + 1] * reciprocal,
                            "e4m3",
                        )
                        for i in range(4)
                    ]
                buf_copy_store(
                    target,
                    fx.Int32(offset // 4),
                    pack_transport_words(pairs, codec),
                    fx.Int32,
                    2,
                )
            else:
                # MX Q/K as the op's generic transport: its input is BF16 after the row
                # factor, and the Q multiplier is applied in BF16 too.
                if const_expr(has_factor):
                    data = (
                        (data * fx.Vector.filled(8, factor, fx.Float32))
                        .to(fx.BFloat16)
                        .to(fx.Float32)
                    )
                if const_expr(layout.multiplier != 1.0):
                    data = (data * layout.multiplier).to(fx.BFloat16).to(fx.Float32)
                amax = fx.Float32(0.0)
                for i in range_constexpr(8):
                    amax = amax.maximumf(fmath.absf(data[i]))
                amax = valid.select(amax, fx.Float32(0.0))
                for shift in (1, 2):
                    amax = amax.maximumf(amax.shuffle_xor(shift, 64))
                if const_expr(codec == "mxfp8"):
                    scale = v4_fp8_scale(amax, False)
                else:
                    scale = transport_scale(amax, codec, False)
                reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                pairs = [
                    pack_transport_pair(
                        data[2 * i] * reciprocal, data[2 * i + 1] * reciprocal, codec
                    )
                    for i in range(4)
                ]
                words = pack_transport_words(pairs, codec)
                if const_expr(codec == "mxfp4"):
                    buf_copy_store(target, fx.Int32(offset // 4), words, fx.Int32, 1)
                else:
                    buf_copy_store(target, fx.Int32(offset // 4), words, fx.Int32, 2)
                if lane % 4 == 0:
                    scale_offset = (
                        base
                        + scale_stage
                        + fx.Int64(row) * scale_row
                        + fx.Int64(within // 4)
                    )
                    scale_offset = valid.select(scale_offset, fx.Int64(scratch))
                    buf_copy_store(
                        target_bytes, fx.Int32(scale_offset), scale.to(fx.Int8), fx.Int8
                    )

        # Each trip issues all its loads before converting any.
        for trip in range(trips):
            batch = [load(fx.Int32(trip), u) for u in range(UNROLL)]
            for u in range_constexpr(UNROLL):
                store(*batch[u])

    return quantize


def _make_channel_quantize(
    layout, rank, world, rows, heads_local, workgroups, first=0, count=None
):
    """Per-channel E4M3 V as the op's push: each value divided by its destination's
    descale of the channel (``scales``: address of the [world][heads_local * 128] FP32
    descales). Thread t keeps chunks t, t + THREADS, ... of 8 channels and their descales,
    and walks the workgroup's rows, so each descale is loaded once."""
    channels = heads_local * HEAD_DIM  # one destination's share of a token row
    dest_chunks = channels // 8
    chunks = world * dest_chunks
    per_thread = (chunks + THREADS - 1) // THREADS
    if count is None:
        count = (rows + workgroups - 1) // workgroups - first
    trips = (count + UNROLL - 1) // UNROLL
    sequence_end = layout.sequence - rank * rows

    @flyc.jit
    def quantize(values, factors, staging, scratch, scales):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        source = ptr_buf_tensor(values, fx.BFloat16, unit_elems=8, unit_stride=1)
        target = ptr_buf_tensor(staging, fx.Int32, unit_elems=2, unit_stride=1)
        channel_scales = ptr_buf_tensor(scales, fx.Float32, unit_elems=4, unit_stride=1)

        def load(chunk, live, trip, u):
            local = trip * UNROLL + u
            row = (local < count).select(
                bid + (local + first) * workgroups, fx.Int32(rows)
            )
            valid = live & (row < min(rows, sequence_end))
            row = valid.select(row, fx.Int32(0))
            return (
                valid,
                row,
                fx.Vector(
                    buf_copy_load(source, (row * chunks + chunk) * 8, fx.BFloat16, 8)
                ),
            )

        for k in range_constexpr(per_thread):
            chunk = tid + k * THREADS
            live = chunk < chunks
            chunk = live.select(chunk, fx.Int32(0))
            lo = fx.Vector(buf_copy_load(channel_scales, chunk * 8, fx.Float32, 4))
            hi = fx.Vector(buf_copy_load(channel_scales, chunk * 8 + 4, fx.Float32, 4))
            per = [lo[i] for i in range(4)] + [hi[i] for i in range(4)]
            base = fx.Int64(chunk // dest_chunks) * layout.stage_bytes + fx.Int64(
                chunk % dest_chunks * 8
            )
            for trip in range(trips):
                batch = [load(chunk, live, fx.Int32(trip), u) for u in range(UNROLL)]
                for u in range_constexpr(UNROLL):
                    valid, row, data = batch[u]
                    data = data.to(fx.Float32)
                    pairs = [
                        pack_transport_pair(
                            data[2 * i] / per[2 * i],
                            data[2 * i + 1] / per[2 * i + 1],
                            "e4m3",
                        )
                        for i in range(4)
                    ]
                    offset = valid.select(
                        base + fx.Int64(row) * channels, fx.Int64(scratch)
                    )
                    buf_copy_store(
                        target,
                        fx.Int32(offset // 4),
                        pack_transport_words(pairs, "e4m3"),
                        fx.Int32,
                        2,
                    )

    return quantize


FP6_RECORD = (
    17408  # bytes per (head, 128-token tile) of FP6 K: bands a and b, scale images A, B
)


def _make_fp6_quantize(
    layout, rank, world, rows, heads_local, workgroups, has_factor, first=0, count=None
):
    """MXFP6 Q (dense) or K (tiled) as the op's native gfx950 transport: one lane per
    32-value group (token, head, group of 32 channels), packed by
    v_cvt_scalef32_2xpk16_fp6_f32.

    Q: six words per group at (token * heads + head) * 4 + group, E8M0 scales token-major.
    K: per (head, tile) records of four-word band a, two-word band b, and two 512-byte
    scale images: A holds each group's scale, B the scale of the next group in
    (token, group) order."""
    role = layout.role
    dest_groups = heads_local * HEAD_DIM // 32
    groups = world * dest_groups  # per token row
    if count is None:
        count = (rows + workgroups - 1) // workgroups - first
    per_trip = UNROLL // 4  # groups per thread per trip: four 16-byte loads each
    trips = (count * groups + THREADS * per_trip - 1) // (THREADS * per_trip)
    sequence_end = layout.sequence - rank * rows
    first_token = rank * rows
    row_bytes = dest_groups * 24
    # Q and K both keep token-major E8M0 scales; K's images in its records repeat them.
    scale_stage = next(part.stage for part in layout.ranges if part.region == "scale")

    def scale_slot(token):
        # A's and B's field of a token: (16, 2, 4):(16, 256, 4) over the tile's 128 tokens.
        return (token % 16) * 16 + (token // 16 % 2) * 256 + (token // 32) * 4

    @flyc.jit
    def quantize(values, factors, staging, scratch, scales):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        source = ptr_buf_tensor(values, fx.BFloat16, unit_elems=8, unit_stride=1)
        row_factor = ptr_buf_tensor(factors, fx.Float32)
        wide = ptr_buf_tensor(staging, fx.Int32, unit_elems=4, unit_stride=1)
        narrow = ptr_buf_tensor(staging, fx.Int32, unit_elems=2, unit_stride=1)
        target_bytes = ptr_buf_tensor(staging, fx.Int8)

        def load(trip, u):
            unit = (trip * per_trip + u) * THREADS + tid
            local = unit // groups
            row = (local < count).select(
                bid + (local + first) * workgroups, fx.Int32(rows)
            )
            chunk = unit % groups
            valid = row < min(rows, sequence_end)
            row = valid.select(row, fx.Int32(0))
            parts = [
                fx.Vector(
                    buf_copy_load(
                        source, (row * groups + chunk) * 32 + 8 * p, fx.BFloat16, 8
                    )
                )
                for p in range(4)
            ]
            factor = None
            if const_expr(has_factor):
                factor = fx.Float32(buf_copy_load(row_factor, row, fx.Float32))
            return valid, row, chunk, parts, factor

        def store(valid, row, chunk, parts, factor):
            # The op's input is BF16 after the row factor; Q's multiplier is applied in
            # BF16 too.
            vals = []
            for p in range_constexpr(4):
                data = parts[p]
                if const_expr(has_factor):
                    data = (
                        data.to(fx.Float32) * fx.Vector.filled(8, factor, fx.Float32)
                    ).to(fx.BFloat16)
                data = data.to(fx.Float32)
                if const_expr(layout.multiplier != 1.0):
                    data = (data * layout.multiplier).to(fx.BFloat16).to(fx.Float32)
                vals = vals + [data[i] for i in range(8)]
            amax = fx.Float32(0.0)
            for value in vals:
                amax = amax.maximumf(fmath.absf(value))
            scale = v4_fp6_scale(amax)
            reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
            vals = [value * reciprocal for value in vals]
            words = emit_f32_to_e2m3_native(
                fx.Vector.from_elements(vals[:16], fx.Float32),
                fx.Vector.from_elements(vals[16:], fx.Float32),
                fx.Float32(1.0),
            )
            dest = chunk // dest_groups
            within = chunk % dest_groups
            base = fx.Int64(dest) * layout.stage_bytes
            spare = fx.Int64(scratch)
            token_scale = valid.select(
                base + scale_stage + fx.Int64(row) * dest_groups + fx.Int64(within),
                spare,
            )
            if const_expr(role == 0):
                first_word = base + fx.Int64(row) * row_bytes + fx.Int64(within * 24)
                last_word = first_word + 16
                scale_at = token_scale
                image_b = spare
            else:
                token = first_token + row
                tok = token % 128
                head = within // 4
                group = within % 4
                record = (
                    base
                    + fx.Int64(head) * layout.head_stage
                    + fx.Int64(token // 128 - layout.first_tile + 1) * FP6_RECORD
                )
                first_word = record + fx.Int64(
                    ((tok // 32) * 512 + group * 128 + (tok % 32) * 4) * 4
                )
                last_word = record + fx.Int64(
                    (2048 + (tok // 32) * 256 + group * 64 + (tok % 32) * 2) * 4
                )
                scale_at = record + fx.Int64(16384 + scale_slot(tok) + group)
                # B one group back in (token, group) order; a tile's first group writes
                # the last byte of the previous record.
                flat = tok * 4 + group
                previous = (flat > 0).select(flat - 1, fx.Int32(0))
                image_b = (flat > 0).select(
                    record + fx.Int64(16896 + scale_slot(previous // 4) + previous % 4),
                    record - 1,
                )
                image_b = (token * 4 + group > 0).select(image_b, spare)
            first_word = valid.select(first_word, spare)
            last_word = valid.select(last_word, spare)
            scale_at = valid.select(scale_at, spare)
            image_b = valid.select(image_b, spare)
            buf_copy_store(
                wide,
                fx.Int32(first_word // 4),
                fx.Vector.from_elements([words[i] for i in range(4)], fx.Int32),
                fx.Int32,
                4,
            )
            buf_copy_store(
                narrow,
                fx.Int32(last_word // 4),
                fx.Vector.from_elements([words[4], words[5]], fx.Int32),
                fx.Int32,
                2,
            )
            buf_copy_store(target_bytes, fx.Int32(scale_at), scale.to(fx.Int8), fx.Int8)
            if const_expr(role == 1):
                buf_copy_store(
                    target_bytes, fx.Int32(image_b), scale.to(fx.Int8), fx.Int8
                )
                buf_copy_store(
                    target_bytes, fx.Int32(token_scale), scale.to(fx.Int8), fx.Int8
                )

        for trip in range(trips):
            batch = [load(fx.Int32(trip), u) for u in range(per_trip)]
            for u in range_constexpr(per_trip):
                store(*batch[u])

    return quantize


FP4V_RECORD, FP4V_SCALE_RECORD = (
    8192,
    512,
)  # bytes per (head, tile) of MXFP4 V and its scales


def _make_fp4_v_quantize(layout, rank, world, rows, heads_local, workgroups):
    """MXFP4 V as the op's transport_v4_v: work items are (head, tile) of every
    destination, in 32-token quarters. Each channel's E8M0 scale covers a quarter's 32
    source tokens (with the FP6-P pack, position p holds token fp6p_source_token(p)),
    rounded up from the mantissa's half; tokens at or past the sequence count as zero.
    Payload words (chunk, quarter, token) in 2048-word tiles, scale bytes (quarter,
    channel) in 512-byte tiles.

    Wave w takes chunks 4w .. 4w + 3 and lane l token lane l // 4 of chunk 4w + l % 4, two
    tokens per quarter, so a quarter's reduction stays inside the wave (the op's spans
    waves)."""
    if rows % 128:
        raise NotImplementedError("MXFP4 V with tiles across ranks")
    channels = world * heads_local * HEAD_DIM  # a V row
    tiles_here = layout.head_stage // FP4V_RECORD
    items = world * heads_local * tiles_here
    scale_stage = next(part.stage for part in layout.ranges if part.region == "scale")
    first_token = rank * rows
    permuted = layout.v_pack == AttentionPack.V_FOR_FP6_P

    @flyc.jit
    def quantize(values, factors, staging, scratch, scales):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        lane = tid % 64
        token_lane = lane // 4
        chunk = tid // 64 * 4 + lane % 4
        source = ptr_buf_tensor(values, fx.BFloat16, unit_elems=8, unit_stride=1)
        target = ptr_buf_tensor(staging, fx.Int32)
        target_bytes = ptr_buf_tensor(staging, fx.Int8)
        for item in range(bid, fx.Int32(items), fx.Int32(workgroups)):
            dest = item // (heads_local * tiles_here)
            head = item // tiles_here % heads_local
            tile = item % tiles_here + layout.first_tile
            column = (dest * heads_local + head) * HEAD_DIM + chunk * 8
            quarters = []
            for quarter in range_constexpr(4):
                halves = []
                for half in range_constexpr(2):
                    token = tile * 128 + quarter * 32 + token_lane + half * 16
                    if const_expr(permuted):
                        token = fp6p_source_token(token)
                    valid = token < layout.sequence
                    row = valid.select(token - first_token, fx.Int32(0))
                    data = fx.Vector(
                        buf_copy_load(source, row * channels + column, fx.BFloat16, 8)
                    ).to(fx.Float32)
                    halves.append(
                        [
                            fx.Float32(valid.select(data[i], fx.Float32(0.0)))
                            for i in range(8)
                        ]
                    )
                quarters.append(halves)
            base = (
                fx.Int64(dest) * layout.stage_bytes
                + fx.Int64(head) * layout.head_stage
                + fx.Int64(tile - layout.first_tile) * FP4V_RECORD
            )
            scale_base = (
                fx.Int64(dest) * layout.stage_bytes
                + scale_stage
                + fx.Int64(head * tiles_here + tile - layout.first_tile)
                * FP4V_SCALE_RECORD
            )
            for quarter in range_constexpr(4):
                halves = quarters[quarter]
                reciprocals = []
                for i in range_constexpr(8):
                    amax = fmath.absf(halves[0][i]).maximumf(fmath.absf(halves[1][i]))
                    for shift in (4, 8, 16, 32):
                        amax = amax.maximumf(amax.shuffle_xor(shift, 64))
                    bits = amax.maximumf(fx.Float32(1.0e-12)).bitcast(fx.Int32)
                    scale = (
                        ((bits >> 23) & 255)
                        - 2
                        + ((bits & 0x7FFFFF) > 0x400000).to(fx.Int32)
                    )
                    scale = (scale < 0).select(fx.Int32(0), scale)
                    scale = (scale > 255).select(fx.Int32(255), scale)
                    reciprocals.append(
                        ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                    )
                    if token_lane == 0:
                        channel = chunk * 8 + i
                        at = (
                            quarter * 128
                            + channel % 2 * 4
                            + channel // 2 % 16 * 8
                            + channel // 32
                        )
                        buf_copy_store(
                            target_bytes,
                            fx.Int32(scale_base + fx.Int64(at)),
                            scale.to(fx.Int8),
                            fx.Int8,
                        )
                for half in range_constexpr(2):
                    pairs = [
                        pack_v4_v_pair(
                            halves[half][2 * p] * reciprocals[2 * p],
                            halves[half][2 * p + 1] * reciprocals[2 * p + 1],
                        )
                        for p in range(4)
                    ]
                    token = token_lane + half * 16
                    word = (
                        chunk % 4
                        + chunk // 4 * 512
                        + quarter % 2 * 128
                        + quarter // 2 * 256
                        + token % 4 * 4
                        + token // 4 % 2 * 64
                        + token // 8 * 16
                    )
                    buf_copy_store(
                        target,
                        fx.Int32((base + fx.Int64(word * 4)) // 4),
                        pack_transport_words(pairs, "mxfp4"),
                        fx.Int32,
                    )

    return quantize


FP6P_RECORD, FP6P_SCALE_RECORD = (
    12288,
    512,
)  # bytes per (head, tile) of packed FP6 V and its scales


def _make_fp6p_v_quantize(layout, rank, world, rows, heads_local, workgroups):
    """Packed MXFP6 V (V_FOR_FP6_P) as the op's transport_v4_fp6_p_v: work items are (head,
    tile) of every destination; thread (warp, lane) takes channel warp * 32 + lane % 32
    and, per 64-token frame, the 32 tokens of half lane // 32. Tokens past the sequence
    repeat its last one. Payload: six words at frame row warp * 2 + frame, lane * 6;
    scales byte [frame][lane][warp] of the tile's 512."""
    if rows % 64:
        raise NotImplementedError(
            "packed FP6 V with frames across ranks (the op's split V exchange)"
        )
    channels = world * heads_local * HEAD_DIM  # a V row
    tiles_here = layout.head_stage // FP6P_RECORD
    items = world * heads_local * tiles_here
    scale_stage = next(part.stage for part in layout.ranges if part.region == "scale")
    first_token = rank * rows
    last_token = layout.sequence - 1

    @flyc.jit
    def quantize(values, factors, staging, scratch, scales):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        warp = tid // 64
        lane = tid % 64
        half = lane // 32
        channel = warp * 32 + lane % 32
        source = ptr_buf_tensor(values, fx.BFloat16)
        wide = ptr_buf_tensor(staging, fx.Int32, unit_elems=4, unit_stride=1)
        narrow = ptr_buf_tensor(staging, fx.Int32, unit_elems=2, unit_stride=1)
        target_bytes = ptr_buf_tensor(staging, fx.Int8)
        for item in range(bid, fx.Int32(items), fx.Int32(workgroups)):
            dest = item // (heads_local * tiles_here)
            head = item // tiles_here % heads_local
            tile = item % tiles_here + layout.first_tile
            column = (dest * heads_local + head) * HEAD_DIM + channel
            frames = []
            for frame in range_constexpr(2):
                vals = []
                for field in range_constexpr(32):
                    offset = (field % 4) + (field // 4 % 4) * 8 + (field // 16) * 32
                    token = tile * 128 + frame * 64 + half * 4 + offset
                    token = (token < last_token).select(token, fx.Int32(last_token))
                    row = token - first_token
                    vals.append(
                        fx.Float32(
                            buf_copy_load(source, row * channels + column, fx.BFloat16)
                        )
                    )
                frames.append(vals)
            base = (
                fx.Int64(dest) * layout.stage_bytes
                + fx.Int64(head) * layout.head_stage
                + fx.Int64(tile - layout.first_tile) * FP6P_RECORD
            )
            scale_base = (
                fx.Int64(dest) * layout.stage_bytes
                + scale_stage
                + fx.Int64(head * tiles_here + tile - layout.first_tile)
                * FP6P_SCALE_RECORD
            )
            for frame in range_constexpr(2):
                vals = frames[frame]
                amax = fx.Float32(0.0)
                for value in vals:
                    amax = amax.maximumf(fmath.absf(value))
                scale = v4_fp6_scale(amax)
                reciprocal = ((fx.Int32(254) - scale) << 23).bitcast(fx.Float32)
                vals = [value * reciprocal for value in vals]
                words = emit_f32_to_e2m3_native(
                    fx.Vector.from_elements(vals[::2], fx.Float32),
                    fx.Vector.from_elements(vals[1::2], fx.Float32),
                    fx.Float32(1.0),
                )
                first_word = base + fx.Int64(((warp * 2 + frame) * 384 + lane * 6) * 4)
                buf_copy_store(
                    wide,
                    fx.Int32(first_word // 4),
                    fx.Vector.from_elements([words[i] for i in range(4)], fx.Int32),
                    fx.Int32,
                    4,
                )
                buf_copy_store(
                    narrow,
                    fx.Int32(first_word // 4 + 4),
                    fx.Vector.from_elements([words[4], words[5]], fx.Int32),
                    fx.Int32,
                    2,
                )
                buf_copy_store(
                    target_bytes,
                    fx.Int32(scale_base + fx.Int64(frame * 256 + lane * 4 + warp)),
                    scale.to(fx.Int8),
                    fx.Int8,
                )

    return quantize
