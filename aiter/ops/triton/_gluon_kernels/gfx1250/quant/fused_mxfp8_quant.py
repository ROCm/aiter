# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# ---------------------------------------------------------------------------
# DSv4 decode producer, fused -- Gluon port for gfx1250
# ---------------------------------------------------------------------------
# Same contract, grid and record layout as the Triton reference,
# _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_kernel in
# _triton_kernels/quant/fused_mxfp8_quant.py; read that one for the format.
#
# What the port changes is the shape of the work, not the arithmetic. ATT on
# the Triton kernel (T=16, 4 warps) puts ~77% of a live Q wave inside the
# 7-group pack loop, and almost none of it is math:
#
#   - each 64-element group is re-loaded from global, and each load is waited
#     on before the next is issued: 7 serialized round trips
#   - each group's amax crosses warps, so it goes through LDS behind an
#     s_barrier_wait: 7 serialized barriers
#   - 14 one-byte scale stores sit between them
#
# Here the whole 512-element head is ONE register tile, [8, 32, 2]:
#
#   dim 0  the quant group. Rows 0..6 are the 7 NoPE groups, row 7 is the
#          64-element RoPE half -- 8 * 64 = 512 = HEAD, so nothing is padded
#   dim 1  the pair index within the row
#   dim 2  even / odd of the pair, held by the same thread
#
# and one program is one warp. So the head is loaded once (two 16 B loads per
# lane, one wait), every reduction is a warp shuffle with no LDS and no
# barrier, the RoPE partner is already in the lane (gl.split, no gather), and
# the 14 scale bytes leave in a single store after the loop rather than
# interleaved with it.
#
# The RoPE arithmetic is _v4_rope_pair's, term for term, so this can be held
# to the Triton kernel bit for bit. The one place it cannot be is the RMSNorm
# sum: one warp reduces in a different order from four, so with the norm ON the
# Q side may differ in its last bit. The KV record has no norm and must match
# exactly either way.


@gluon.jit
def _v4_rope_rows(x, cs_ptr, pos, half, is_rope, j3, e3):
    """GPT-J RoPE on the row(s) where ``is_rope``; every other value passes.

    cos/sin are loaded as a [1, 32, 2] tile with both pair slots reading the
    same address, so after the split they already carry the layout the split
    halves of ``x`` have and broadcast over the rows without a convert.
    """
    cs_off = j3 + e3 * 0
    cos3 = gl.load(cs_ptr + pos * (2 * half) + cs_off)
    sin3 = gl.load(cs_ptr + pos * (2 * half) + half + cs_off)
    cos, _ = gl.split(cos3)
    sin, _ = gl.split(sin3)
    xe, xo = gl.split(x)
    # _v4_rope_pair, unchanged
    ye = xe * cos - xo * sin
    yo = xe * sin + xo * cos
    y = gl.convert_layout(gl.join(ye, yo), x.type.layout)
    return gl.where(is_rope, y, x)


@gluon.jit
def _group_amax(x):
    """[8, 32, 2] -> [8, 1, 1], each row's absmax. A warp shuffle, no LDS."""
    a = gl.max(gl.max(gl.abs(x), axis=2), axis=1)
    return gl.expand_dims(gl.expand_dims(a, 1), 2)


@gluon.jit
def _gluon_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_kernel(
    q_in_ptr,  # [T, num_heads_q, HEAD] bf16
    q_out_ptr,  # [T, padded_heads, HEAD] bf16
    q_packed_ptr,  # [T, padded_heads, HEAD] uint8 (the 2buff row)
    q_rope_ptr,  # [T, padded_heads, ROPE] bf16
    kv_in_ptr,  # [T, HEAD] bf16
    kv_cache_ptr,  # [nb, block, REC] uint8
    kv_slot_ptr,  # [T] int, -1 to skip
    pos_ptr,  # [T]
    cs_ptr,  # [max_pos, ROPE] fp32, cos || sin
    kv_block_stride,
    kv_cache_block_size,
    eps,
    fp8_max,
    num_heads_q: gl.constexpr,
    padded_heads: gl.constexpr,
    HEAD: gl.constexpr,
    NOPE: gl.constexpr,
    ROPE: gl.constexpr,
    GROUP: gl.constexpr,
    NUM_TILES: gl.constexpr,
    REC: gl.constexpr,
    SC_OFF: gl.constexpr,
    ROPE_OFF: gl.constexpr,
    APPLY_NORM: gl.constexpr,
    PACK_Q: gl.constexpr,
    WRITE_Q: gl.constexpr,
    USE_FNUZ: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """One program per (token, slot); slot == padded_heads is the KV row.

    Padding slots zero-fill every Q output, exactly as the Triton kernel does:
    an unwritten scale byte of 0xFF is E8M0 NaN.
    """
    gl.static_assert(GROUP * (NUM_TILES + 1) == HEAD, "tile is 7 NoPE groups + RoPE")
    gl.static_assert(ROPE == GROUP, "row NUM_TILES of the tile is the RoPE half")
    gl.static_assert(NUM_WARPS == 1 or NUM_WARPS == 2, "a warp spans 4 of the 8 rows")

    ROWS: gl.constexpr = NUM_TILES + 1
    PAIRS: gl.constexpr = GROUP // 2
    # 4 rows x 8 lanes x 4 pairs per lane: each lane holds 8 contiguous
    # elements of a row (16 B of bf16), and a row's 64 elements live in 8
    # lanes of ONE warp -- which is what keeps every row reduction a shuffle.
    L3: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 4, 2],
        threads_per_warp=[4, 8, 1],
        warps_per_cta=[NUM_WARPS, 1, 1],
        order=[2, 1, 0],
    )
    g = gl.arange(0, ROWS, layout=gl.SliceLayout(1, gl.SliceLayout(2, L3)))
    j = gl.arange(0, PAIRS, layout=gl.SliceLayout(0, gl.SliceLayout(2, L3)))
    e = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    g3 = gl.expand_dims(gl.expand_dims(g, 1), 2)  # [8, 1, 1]
    j3 = gl.expand_dims(gl.expand_dims(j, 0), 2)  # [1, 32, 1]
    e3 = gl.expand_dims(gl.expand_dims(e, 0), 1)  # [1, 1, 2]
    off = g3 * GROUP + j3 * 2 + e3  # [8, 32, 2], element index in the head
    row = off // GROUP
    col = off % GROUP
    is_rope = row == NUM_TILES
    is_nope = row < NUM_TILES

    tok = gl.program_id(0)
    h = gl.program_id(1)
    half: gl.constexpr = ROPE // 2

    # ---- KV: RoPE, quantize, write one aligned record ---------------------
    if h == padded_heads:
        # the row load does not depend on pos or slot, so it goes out first
        x = gl.load(kv_in_ptr + tok * HEAD + off).to(gl.float32)
        pos = gl.load(pos_ptr + tok)
        slot = gl.load(kv_slot_ptr + tok)
        if slot != -1:
            blk = slot // kv_cache_block_size
            pos_in_blk = slot % kv_cache_block_size
            # int64: blk * block_stride exceeds 2^31 on a production-sized pool
            rec = kv_cache_ptr + blk.to(gl.int64) * kv_block_stride + pos_in_blk * REC

            amax = gl.maximum(_group_amax(x), 1e-4)
            exponent = gl.ceil(gl.log2(amax / fp8_max))
            scale = gl.exp2(exponent)
            xs = gl.clamp(x / scale, -fp8_max, fp8_max)
            if USE_FNUZ:
                f8 = xs.to(gl.float8e4b8)
            else:
                f8 = xs.to(gl.float8e4nv)
            # [0, 448): NoPE fp8. Row NUM_TILES is not fp8 and the record's
            # [462, 512) pad is left as the Triton kernel leaves it, unwritten.
            gl.store(rec + off, f8.to(gl.uint8, bitcast=True), mask=is_nope)

            # [448, 462): each group's scale twice, in one store
            enc = gl.maximum(gl.minimum(exponent + 127.0, 255.0), 0.0).to(gl.uint8)
            enc = enc + gl.zeros_like(off).to(gl.uint8)
            gl.store(rec + SC_OFF + 2 * row + col, enc, mask=is_nope & (col < 2))

            # [512, 640): RoPE bf16
            y = _v4_rope_rows(x, cs_ptr, pos, half, is_rope, j3, e3)
            rope_out = (rec + ROPE_OFF).to(gl.pointer_type(gl.bfloat16))
            gl.store(rope_out + col, y.to(gl.bfloat16), mask=is_rope)
        return

    # ---- Q: padding slot --------------------------------------------------
    qrow = tok * padded_heads + h
    dst = q_out_ptr + qrow * HEAD
    if h >= num_heads_q:
        if WRITE_Q:
            gl.store(dst + off, gl.zeros_like(off).to(gl.bfloat16))
        if PACK_Q:
            gl.store(q_packed_ptr + qrow * HEAD + off, gl.zeros_like(off).to(gl.uint8))
            gl.store(
                q_rope_ptr + qrow * ROPE + col,
                gl.zeros_like(off).to(gl.bfloat16),
                mask=is_rope,
            )
        return

    # ---- Q: live head -----------------------------------------------------
    x = gl.load(q_in_ptr + (tok * num_heads_q + h) * HEAD + off).to(gl.float32)
    pos = gl.load(pos_ptr + tok)
    # RMSNorm over the whole head, no weight -- one scale for both halves
    if APPLY_NORM:
        ss = gl.sum(gl.sum(gl.sum(x * x, axis=2), axis=1), axis=0)
        x = x * gl.rsqrt(ss / HEAD + eps)
    y = _v4_rope_rows(x, cs_ptr, pos, half, is_rope, j3, e3)

    # bf16 Q out: NoPE normed, RoPE normed and rotated, in one store
    if WRITE_Q:
        gl.store(dst + off, y.to(gl.bfloat16))

    if PACK_Q:
        pdst = q_packed_ptr + qrow * HEAD
        # the NoPE half is normed but NOT rotated; y is x on those rows
        amax = _group_amax(y)
        # clamp the RATIO, which is what the reference packing does
        exponent = gl.ceil(gl.log2(gl.maximum(amax / fp8_max, 1e-4)))
        xs = y / gl.exp2(exponent)
        if USE_FNUZ:
            f8 = xs.to(gl.float8e4b8)
        else:
            f8 = xs.to(gl.float8e4nv)
        # [0, 448) fp8, and [462, 512) of the RoPE row zero, so no stale byte
        # reaches the MMA. [448, 462) is the scale store below; the masks are
        # disjoint, so the two stores need no ordering.
        zero8 = gl.zeros_like(off).to(gl.uint8)
        bits = gl.where(is_nope, f8.to(gl.uint8, bitcast=True), zero8)
        gl.store(pdst + off, bits, mask=is_nope | (col >= 2 * NUM_TILES))
        enc = gl.minimum(gl.maximum(exponent + 127.0, 0.0), 254.0).to(gl.uint8)
        enc = enc + gl.zeros_like(off).to(gl.uint8)
        gl.store(pdst + SC_OFF + 2 * row + col, enc, mask=is_nope & (col < 2))
        # the RoPE plane is never quantized
        gl.store(q_rope_ptr + qrow * ROPE + col, y.to(gl.bfloat16), mask=is_rope)
