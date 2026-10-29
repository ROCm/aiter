# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Q/K Gemma RMSNorm and interleaved MRoPE of Qwen3.5 full attention

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)  # rotary_dim 64: two Neox halves


@functools.cache
def get_fused_qk_norm_mrope_layouts(num_warps):
    # One row per wave: 8 values per lane for the norm, 1 value per lane for the RoPE halves.
    return {
        "row_layout": gl.BlockedLayout(
            [1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0]
        ),
        "rope_layout": gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0]),
    }


_fused_qk_norm_mrope_repr = make_kernel_repr(
    "fused_qk_norm_mrope_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M"]
)


@gluon.jit(repr=_fused_qk_norm_mrope_repr)
def fused_qk_norm_mrope_gfx1250(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    num_rows,
    stride_q,
    stride_k,
    stride_positions,
    stride_cos,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    # Row m of the T * HEADS rows is head m % HEADS of token m // HEADS, Q heads first.
    HEADS: gl.constexpr = Q_HEADS + K_HEADS
    pid_m = gl.program_id(0)
    offs_m = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, row_layout)
    )
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    token = offs_m // HEADS
    head = offs_m % HEADS
    is_q = head < Q_HEADS
    mask_m = offs_m < num_rows
    offs_x = gl.where(
        is_q,
        token * stride_q + head * HEAD_DIM,
        token * stride_k + (head - Q_HEADS) * HEAD_DIM,
    )
    offs_out = gl.where(
        is_q,
        token * (Q_HEADS * HEAD_DIM) + head * HEAD_DIM,
        token * (K_HEADS * HEAD_DIM) + (head - Q_HEADS) * HEAD_DIM,
    )

    # Every load comes before the first wait: the row and its weight, the RoPE halves, the positions.
    x_row = gl.where(is_q, q_ptr, k_ptr) + offs_x
    x = gl.load(x_row[:, None] + offs_n[None, :], mask=mask_m[:, None], other=0.0)
    w = gl.where(
        is_q[:, None],
        gl.load(q_weight_ptr + offs_n)[None, :],
        gl.load(k_weight_ptr + offs_n)[None, :],
    )

    # The same rows in the RoPE layout: columns [0, 32) and [32, 64).
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    token_r = gl.convert_layout(token, gl.SliceLayout(1, rope_layout))
    is_q_r = gl.convert_layout(is_q, gl.SliceLayout(1, rope_layout))
    mask_r = gl.convert_layout(mask_m, gl.SliceLayout(1, rope_layout))
    x_row_r = gl.convert_layout(x_row, gl.SliceLayout(1, rope_layout))
    x_lo = gl.load(x_row_r[:, None] + offs_r[None, :], mask=mask_r[:, None], other=0.0)
    x_hi = gl.load(
        x_row_r[:, None] + ROPE_HALF + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    w_lo = gl.where(
        is_q_r[:, None],
        gl.load(q_weight_ptr + offs_r)[None, :],
        gl.load(k_weight_ptr + offs_r)[None, :],
    )
    w_hi = gl.where(
        is_q_r[:, None],
        gl.load(q_weight_ptr + ROPE_HALF + offs_r)[None, :],
        gl.load(k_weight_ptr + ROPE_HALF + offs_r)[None, :],
    )

    # Frequency i uses the H position, the W position or the T position (interleaved sections).
    pos_t = gl.load(positions_ptr + token_r, mask=mask_r, other=0)
    pos_h = gl.load(positions_ptr + stride_positions + token_r, mask=mask_r, other=0)
    pos_w = gl.load(
        positions_ptr + 2 * stride_positions + token_r, mask=mask_r, other=0
    )
    use_h = (offs_r % 3 == 1) & (offs_r < 3 * SECTION_H)
    use_w = (offs_r % 3 == 2) & (offs_r < 3 * SECTION_W)
    pos = gl.where(
        use_h[None, :],
        pos_h[:, None],
        gl.where(use_w[None, :], pos_w[:, None], pos_t[:, None]),
    )
    cos = gl.load(cos_ptr + pos * stride_cos + offs_r[None, :])
    sin = gl.load(sin_ptr + pos * stride_cos + offs_r[None, :])

    # Gemma RMSNorm: x / rms(x) * (1 + w). Columns [64, 256) are done.
    x = x.to(gl.float32)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    y = (x * rstd[:, None] * (w.to(gl.float32) + 1.0)).to(q_out_ptr.dtype.element_ty)
    out_row = gl.where(is_q, q_out_ptr, k_out_ptr) + offs_out
    gl.store(
        out_row[:, None] + offs_n[None, :],
        y,
        mask=mask_m[:, None] & (offs_n >= 2 * ROPE_HALF)[None, :],
    )

    # Neox RoPE of columns [0, 64), from the same normalized values.
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo.to(gl.float32) * rstd_r * (w_lo.to(gl.float32) + 1.0)
    y_hi = x_hi.to(gl.float32) * rstd_r * (w_hi.to(gl.float32) + 1.0)
    rope_lo = (y_lo * cos - y_hi * sin).to(q_out_ptr.dtype.element_ty)
    rope_hi = (y_hi * cos + y_lo * sin).to(q_out_ptr.dtype.element_ty)
    out_row_r = gl.convert_layout(out_row, gl.SliceLayout(1, rope_layout))
    gl.store(out_row_r[:, None] + offs_r[None, :], rope_lo, mask=mask_r[:, None])
    gl.store(
        out_row_r[:, None] + ROPE_HALF + offs_r[None, :], rope_hi, mask=mask_r[:, None]
    )
