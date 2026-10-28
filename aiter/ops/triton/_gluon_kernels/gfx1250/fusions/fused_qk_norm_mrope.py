# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Q/K Gemma RMSNorm and interleaved MRoPE of Qwen3.5 full attention, and the K/V write into the paged KV cache

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)  # rotary_dim 64: two Neox halves


@functools.cache
def get_fused_qk_norm_mrope_layouts(num_warps):
    # Each wave holds one row: 8 values per lane for the norm, 1 per lane for each rope half.
    return {
        "row_layout": gl.BlockedLayout(
            [1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0]
        ),
        "rope_layout": gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0]),
    }


_fused_qk_norm_mrope_repr = make_kernel_repr(
    "fused_qk_norm_mrope_gfx1250", ["Q_HEADS", "K_HEADS", "ROWS", "CACHE"]
)


@gluon.jit(repr=_fused_qk_norm_mrope_repr)
def fused_qk_norm_mrope_gfx1250(
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_ptr,
    num_rows,
    num_slots,
    q_stride,
    k_stride,
    v_stride,
    positions_stride,
    cos_stride,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    ROWS: gl.constexpr,
    CACHE: gl.constexpr,
    BLOCK: gl.constexpr,
    X: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    HEADS: gl.constexpr = Q_HEADS + K_HEADS
    rows_of_row: gl.constexpr = gl.SliceLayout(1, row_layout)
    rows_of_rope: gl.constexpr = gl.SliceLayout(1, rope_layout)

    # Row r is head r % HEADS of token r // HEADS, Q heads first.
    r = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=rows_of_row)
    t = r // HEADS
    h = r % HEADS
    is_q = h < Q_HEADS
    valid = r < num_rows
    in_off = gl.where(
        is_q, t * q_stride + h * HEAD_DIM, t * k_stride + (h - Q_HEADS) * HEAD_DIM
    )
    out_off = gl.where(
        is_q,
        t * (Q_HEADS * HEAD_DIM) + h * HEAD_DIM,
        t * (K_HEADS * HEAD_DIM) + (h - Q_HEADS) * HEAD_DIM,
    )

    # Issue every load before the reduction: the row, its two rope halves, the positions, cos and sin.
    d = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    in_row = gl.where(gl.expand_dims(is_q, 1), q_ptr, k_ptr) + gl.expand_dims(in_off, 1)
    x = gl.load(in_row + gl.expand_dims(d, 0), mask=gl.expand_dims(valid, 1), other=0.0)
    w = gl.where(
        gl.expand_dims(is_q, 1),
        gl.expand_dims(gl.load(q_weight_ptr + d), 0),
        gl.expand_dims(gl.load(k_weight_ptr + d), 0),
    )

    i = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    rq = gl.expand_dims(gl.convert_layout(is_q, rows_of_rope), 1)
    rt = gl.convert_layout(t, rows_of_rope)
    rv = gl.convert_layout(valid, rows_of_rope)
    rope_in = gl.where(rq, q_ptr, k_ptr) + gl.expand_dims(
        gl.convert_layout(in_off, rows_of_rope), 1
    )
    x1 = gl.load(rope_in + gl.expand_dims(i, 0), mask=gl.expand_dims(rv, 1), other=0.0)
    x2 = gl.load(
        rope_in + ROPE_HALF + gl.expand_dims(i, 0),
        mask=gl.expand_dims(rv, 1),
        other=0.0,
    )
    if CACHE:
        # The cache slot and the V row of each K row. Lane l holds V values l, l + 32, ...:
        # the cache keeps the V values of a token 16 bytes apart, so the lanes store side by side.
        rh = gl.convert_layout(h, rows_of_rope)
        # Tokens past the slot table are padding.
        rk = rv & (rh >= Q_HEADS) & (rt < num_slots)
        slot = gl.load(slot_ptr + rt, mask=rk, other=-1).to(gl.int64)
        dv = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, rope_layout))
        v = gl.load(
            v_ptr
            + gl.expand_dims(rt * v_stride + (rh - Q_HEADS) * HEAD_DIM, 1)
            + gl.expand_dims(dv, 0),
            mask=gl.expand_dims(rk, 1),
            other=0.0,
        )
    w1 = gl.where(
        rq,
        gl.expand_dims(gl.load(q_weight_ptr + i), 0),
        gl.expand_dims(gl.load(k_weight_ptr + i), 0),
    )
    w2 = gl.where(
        rq,
        gl.expand_dims(gl.load(q_weight_ptr + ROPE_HALF + i), 0),
        gl.expand_dims(gl.load(k_weight_ptr + ROPE_HALF + i), 0),
    )

    # Frequency i takes the H position, the W position or the T position (interleaved sections).
    pos_t = gl.expand_dims(gl.load(positions_ptr + rt, mask=rv, other=0), 1)
    pos_h = gl.expand_dims(
        gl.load(positions_ptr + positions_stride + rt, mask=rv, other=0), 1
    )
    pos_w = gl.expand_dims(
        gl.load(positions_ptr + 2 * positions_stride + rt, mask=rv, other=0), 1
    )
    use_h = gl.expand_dims((i % 3 == 1) & (i < 3 * SECTION_H), 0)
    use_w = gl.expand_dims((i % 3 == 2) & (i < 3 * SECTION_W), 0)
    pos = gl.where(use_h, pos_h, gl.where(use_w, pos_w, pos_t))
    cos = gl.load(cos_ptr + pos * cos_stride + gl.expand_dims(i, 0))
    sin = gl.load(sin_ptr + pos * cos_stride + gl.expand_dims(i, 0))

    # Gemma RMSNorm: x / rms(x) * (1 + w). Values [64, 256) are done.
    x = x.to(gl.float32)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    y = x * gl.expand_dims(rstd, 1) * (w.to(gl.float32) + 1.0)
    out_row = gl.where(gl.expand_dims(is_q, 1), q_out_ptr, k_out_ptr) + gl.expand_dims(
        out_off, 1
    )
    gl.store(
        out_row + gl.expand_dims(d, 0),
        y.to(q_out_ptr.dtype.element_ty),
        mask=gl.expand_dims(valid, 1) & gl.expand_dims(d >= 2 * ROPE_HALF, 0),
    )

    # Neox rotation of values [0, 64), from the same normalized values.
    s = gl.expand_dims(gl.convert_layout(rstd, rows_of_rope), 1)
    y1 = x1.to(gl.float32) * s * (w1.to(gl.float32) + 1.0)
    y2 = x2.to(gl.float32) * s * (w2.to(gl.float32) + 1.0)
    out1 = (y1 * cos - y2 * sin).to(q_out_ptr.dtype.element_ty)
    out2 = (y2 * cos + y1 * sin).to(q_out_ptr.dtype.element_ty)
    rope_out = gl.where(rq, q_out_ptr, k_out_ptr) + gl.expand_dims(
        gl.convert_layout(out_off, rows_of_rope), 1
    )
    gl.store(rope_out + gl.expand_dims(i, 0), out1, mask=gl.expand_dims(rv, 1))
    gl.store(
        rope_out + ROPE_HALF + gl.expand_dims(i, 0), out2, mask=gl.expand_dims(rv, 1)
    )

    # K and V rows into the paged cache: K[block, h, d // X, offset, d % X] and
    # V[block, h, offset // X, d, offset % X]. A negative slot is a pad token.
    if CACHE:
        write = rk & (slot >= 0)
        blk = slot // BLOCK
        off = slot % BLOCK
        cache_row = (blk * K_HEADS + rh - Q_HEADS) * (HEAD_DIM * BLOCK)
        rw = gl.expand_dims(write, 1)
        v_row = v_cache_ptr + gl.expand_dims(
            cache_row + (off // X) * (HEAD_DIM * X) + off % X, 1
        )
        gl.store(v_row + gl.expand_dims(dv * X, 0), v, mask=rw)

        # The rotated values [0, 64) of the K row, then values [64, 256) at 16 bytes a lane.
        k_row = k_cache_ptr + gl.expand_dims(cache_row + off * X, 1)
        j = ROPE_HALF + i
        gl.store(
            k_row + gl.expand_dims((i // X) * (BLOCK * X) + i % X, 0), out1, mask=rw
        )
        gl.store(
            k_row + gl.expand_dims((j // X) * (BLOCK * X) + j % X, 0), out2, mask=rw
        )
        k_row = k_cache_ptr + gl.expand_dims(
            gl.convert_layout(cache_row + off * X, rows_of_row), 1
        )
        gl.store(
            k_row + gl.expand_dims((d // X) * (BLOCK * X) + d % X, 0),
            y.to(k_cache_ptr.dtype.element_ty),
            mask=gl.expand_dims(gl.convert_layout(write, rows_of_row), 1)
            & gl.expand_dims(d >= 2 * ROPE_HALF, 0),
        )
