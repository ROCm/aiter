# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Q/K Gemma RMSNorm and interleaved MRoPE

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)  # rotary_dim 64: two Neox halves


@functools.cache
def get_fused_qk_norm_mrope_layouts(num_warps):
    # Rows over the waves: 8 values per lane for the norm, 1 value per lane for the RoPE halves.
    return {
        "row_layout": gl.BlockedLayout(
            [1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0]
        ),
        "rope_layout": gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0]),
    }


_fused_qk_norm_mrope_repr = make_kernel_repr(
    "fused_qk_norm_mrope_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M"]
)
_fused_qk_norm_mrope_tdm_repr = make_kernel_repr(
    "fused_qk_norm_mrope_tdm_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M"]
)


@gluon.jit
def _tdm_tile(ptr, num_tokens, stride, BLOCK_M: gl.constexpr, COLS: gl.constexpr):
    # A [BLOCK_M, COLS] TDM tile of a [num_tokens, COLS] tensor, and its LDS buffer.
    layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0])
    desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        ptr, [num_tokens, COLS], [stride, 1], [BLOCK_M, COLS], layout
    )
    smem = gl.allocate_shared_memory(ptr.dtype.element_ty, [BLOCK_M, COLS], layout)
    return desc, smem


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

    # The positions load first: the cos/sin loads wait for them. Then the row, its weight and the RoPE
    # halves. The values with an _r suffix hold the same rows in the RoPE layout.
    x_row = gl.where(is_q, q_ptr, k_ptr) + offs_x
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    token_r = gl.convert_layout(token, gl.SliceLayout(1, rope_layout))
    is_q_r = gl.convert_layout(is_q, gl.SliceLayout(1, rope_layout))
    mask_r = gl.convert_layout(mask_m, gl.SliceLayout(1, rope_layout))
    x_row_r = gl.convert_layout(x_row, gl.SliceLayout(1, rope_layout))
    pos_t = gl.load(positions_ptr + token_r, mask=mask_r, other=0)
    pos_h = gl.load(positions_ptr + stride_positions + token_r, mask=mask_r, other=0)
    pos_w = gl.load(
        positions_ptr + 2 * stride_positions + token_r, mask=mask_r, other=0
    )
    x = gl.load(x_row[:, None] + offs_n[None, :], mask=mask_m[:, None], other=0.0)
    w = gl.where(
        is_q[:, None],
        gl.load(q_weight_ptr + offs_n)[None, :],
        gl.load(k_weight_ptr + offs_n)[None, :],
    )

    # The RoPE halves: columns [0, 32) and [32, 64).
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
    use_h = (offs_r % 3 == 1) & (offs_r < 3 * SECTION_H)
    use_w = (offs_r % 3 == 2) & (offs_r < 3 * SECTION_W)
    pos = gl.where(
        use_h[None, :],
        pos_h[:, None],
        gl.where(use_w[None, :], pos_w[:, None], pos_t[:, None]),
    )
    # The mask keeps these loads before the norm: the compiler moves an unmasked load next to its use.
    cos = gl.load(
        cos_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    sin = gl.load(
        sin_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )

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


@gluon.jit
def _norm_rope_tile(
    x_s,
    out_s,
    w_ptr,
    t0,
    num_tokens,
    positions_ptr,
    stride_positions,
    cos_ptr,
    sin_ptr,
    stride_cos,
    EPS: gl.constexpr,
    HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    WAIT: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    # Row m of the tile is head m % HEADS of token t0 + m // HEADS.
    offs_m = gl.arange(0, x_s.shape[0], layout=gl.SliceLayout(1, row_layout))
    offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
    mask_m = t0 + offs_m // HEADS < num_tokens

    # The positions, cos/sin and weights load while the tile is in flight.
    offs_r = gl.arange(0, ROPE_HALF, layout=gl.SliceLayout(0, rope_layout))
    token_r = gl.convert_layout(t0 + offs_m // HEADS, gl.SliceLayout(1, rope_layout))
    mask_r = gl.convert_layout(mask_m, gl.SliceLayout(1, rope_layout))
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
    cos = gl.load(
        cos_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    sin = gl.load(
        sin_ptr + pos * stride_cos + offs_r[None, :], mask=mask_r[:, None], other=0.0
    )
    w = gl.load(w_ptr + offs_n)[None, :].to(gl.float32)
    w_lo = gl.load(w_ptr + offs_r)[None, :].to(gl.float32)
    w_hi = gl.load(w_ptr + ROPE_HALF + offs_r)[None, :].to(gl.float32)

    # Gemma RMSNorm: x / rms(x) * (1 + w). All columns go to LDS, then the RoPE values
    # replace columns [0, 64) after the barrier.
    gl.amd.gfx1250.tdm.async_wait(WAIT)
    x = x_s.load(row_layout).to(gl.float32)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    out_s.store((x * rstd[:, None] * (w + 1.0)).to(out_s.dtype))
    gl.barrier()

    # Neox RoPE of columns [0, 64), from the same normalized values.
    x_lo = x_s.slice(0, ROPE_HALF, dim=1).load(rope_layout).to(gl.float32)
    x_hi = x_s.slice(ROPE_HALF, ROPE_HALF, dim=1).load(rope_layout).to(gl.float32)
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo * rstd_r * (w_lo + 1.0)
    y_hi = x_hi * rstd_r * (w_hi + 1.0)
    out_s.slice(0, ROPE_HALF, dim=1).store((y_lo * cos - y_hi * sin).to(out_s.dtype))
    out_s.slice(ROPE_HALF, ROPE_HALF, dim=1).store(
        (y_hi * cos + y_lo * sin).to(out_s.dtype)
    )


@gluon.jit(repr=_fused_qk_norm_mrope_tdm_repr)
def fused_qk_norm_mrope_tdm_gfx1250(
    q_ptr,
    k_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    k_out_ptr,
    num_tokens,
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
    # A program does BLOCK_M tokens. Their Q heads and their K heads move between memory and
    # LDS as one TDM tile each.
    Q_COLS: gl.constexpr = Q_HEADS * HEAD_DIM
    K_COLS: gl.constexpr = K_HEADS * HEAD_DIM
    t0 = gl.program_id(0) * BLOCK_M
    q_desc, q_s = _tdm_tile(q_ptr, num_tokens, stride_q, BLOCK_M, Q_COLS)
    k_desc, k_s = _tdm_tile(k_ptr, num_tokens, stride_k, BLOCK_M, K_COLS)
    q_out_desc, q_out_s = _tdm_tile(q_out_ptr, num_tokens, Q_COLS, BLOCK_M, Q_COLS)
    k_out_desc, k_out_s = _tdm_tile(k_out_ptr, num_tokens, K_COLS, BLOCK_M, K_COLS)
    gl.amd.gfx1250.tdm.async_load(q_desc, [t0, 0], q_s)
    gl.amd.gfx1250.tdm.async_load(k_desc, [t0, 0], k_s)
    _norm_rope_tile(
        q_s.reshape([BLOCK_M * Q_HEADS, HEAD_DIM]),
        q_out_s.reshape([BLOCK_M * Q_HEADS, HEAD_DIM]),
        q_weight_ptr,
        t0,
        num_tokens,
        positions_ptr,
        stride_positions,
        cos_ptr,
        sin_ptr,
        stride_cos,
        EPS,
        Q_HEADS,
        SECTION_H,
        SECTION_W,
        1,
        row_layout,
        rope_layout,
    )
    _norm_rope_tile(
        k_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]),
        k_out_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]),
        k_weight_ptr,
        t0,
        num_tokens,
        positions_ptr,
        stride_positions,
        cos_ptr,
        sin_ptr,
        stride_cos,
        EPS,
        K_HEADS,
        SECTION_H,
        SECTION_W,
        0,
        row_layout,
        rope_layout,
    )
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(q_out_desc, [t0, 0], q_out_s)
    gl.amd.gfx1250.tdm.async_store(k_out_desc, [t0, 0], k_out_s)
    gl.amd.gfx1250.tdm.async_wait(0)
