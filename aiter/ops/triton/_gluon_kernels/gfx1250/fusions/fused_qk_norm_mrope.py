# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Q/K Gemma RMSNorm and interleaved MRoPE of Qwen3.5 full attention, and the K/V write into the paged KV cache
# (bf16, or fp8 with one per-tensor scale and Q emitted in fp8)

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)  # rotary_dim 64: two Neox halves


@functools.cache
def get_fused_qk_norm_mrope_layouts(num_warps):
    # One row per wave: 8 values per lane for the norm, 1 value per lane for the RoPE halves and V.
    return {
        "row_layout": gl.BlockedLayout(
            [1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0]
        ),
        "rope_layout": gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0]),
    }


_fused_qk_norm_mrope_repr = make_kernel_repr(
    "fused_qk_norm_mrope_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M", "CACHE", "FP8"]
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
    # The first 120 bytes of arguments arrive in registers. Later ones load at their
    # first use, so the arguments of the cache write come last.
    v_ptr,
    slot_mapping_ptr,
    stride_v,
    num_slots,
    k_cache_ptr,
    v_cache_ptr,
    kv_inv_scale,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    CACHE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    X: gl.constexpr,
    FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
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
    head_r = gl.convert_layout(head, gl.SliceLayout(1, rope_layout))
    if CACHE:
        # The slot and the V row of each K row. Tokens past the slot table are padding.
        # Lane l holds V values l, l + 32, ...: the cache keeps the V values of a token 16 bytes apart.
        is_k_r = mask_r & (head_r >= Q_HEADS) & (token_r < num_slots)
        slot = gl.load(slot_mapping_ptr + token_r, mask=is_k_r, other=-1).to(gl.int64)
        offs_v = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, rope_layout))
        v_row = v_ptr + token_r * stride_v + (head_r - Q_HEADS) * HEAD_DIM
        v = gl.load(v_row[:, None] + offs_v[None, :], mask=is_k_r[:, None], other=0.0)
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
    y32 = x * rstd[:, None] * (w.to(gl.float32) + 1.0)
    tail = (offs_n >= 2 * ROPE_HALF)[None, :]

    # Neox RoPE of columns [0, 64), from the same normalized values.
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo.to(gl.float32) * rstd_r * (w_lo.to(gl.float32) + 1.0)
    y_hi = x_hi.to(gl.float32) * rstd_r * (w_hi.to(gl.float32) + 1.0)
    rope_lo32 = y_lo * cos - y_hi * sin
    rope_hi32 = y_hi * cos + y_lo * sin

    if FP8:
        # Q goes out in fp8 and K stays bf16, so the two land in separate stores. Values are
        # rounded to bf16 first, as the unfused path did (bf16 q/k, then clamp and cast), and
        # clamped: the fp8 cast does not saturate.
        offs_out_r = gl.convert_layout(offs_out, gl.SliceLayout(1, rope_layout))
        y = y32.to(k_out_ptr.dtype.element_ty)
        rope_lo = rope_lo32.to(k_out_ptr.dtype.element_ty)
        rope_hi = rope_hi32.to(k_out_ptr.dtype.element_ty)
        fp8_ty = q_out_ptr.dtype.element_ty
        q_rows = mask_m & is_q
        k_rows = mask_m & (head >= Q_HEADS)
        q_rows_r = mask_r & is_q_r
        k_rows_r = mask_r & (head_r >= Q_HEADS)
        out_n = offs_out[:, None] + offs_n[None, :]
        gl.store(
            q_out_ptr + out_n,
            gl.minimum(gl.maximum(y.to(gl.float32), -FP8_MAX), FP8_MAX).to(fp8_ty),
            mask=q_rows[:, None] & tail,
        )
        gl.store(k_out_ptr + out_n, y, mask=k_rows[:, None] & tail)
        out_r = offs_out_r[:, None] + offs_r[None, :]
        q_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32), -FP8_MAX), FP8_MAX)
        q_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32), -FP8_MAX), FP8_MAX)
        gl.store(q_out_ptr + out_r, q_lo.to(fp8_ty), mask=q_rows_r[:, None])
        gl.store(q_out_ptr + ROPE_HALF + out_r, q_hi.to(fp8_ty), mask=q_rows_r[:, None])
        gl.store(k_out_ptr + out_r, rope_lo, mask=k_rows_r[:, None])
        gl.store(k_out_ptr + ROPE_HALF + out_r, rope_hi, mask=k_rows_r[:, None])
    else:
        y = y32.to(q_out_ptr.dtype.element_ty)
        out_row = gl.where(is_q, q_out_ptr, k_out_ptr) + offs_out
        gl.store(out_row[:, None] + offs_n[None, :], y, mask=mask_m[:, None] & tail)
        rope_lo = rope_lo32.to(q_out_ptr.dtype.element_ty)
        rope_hi = rope_hi32.to(q_out_ptr.dtype.element_ty)
        out_row_r = gl.convert_layout(out_row, gl.SliceLayout(1, rope_layout))
        gl.store(out_row_r[:, None] + offs_r[None, :], rope_lo, mask=mask_r[:, None])
        gl.store(
            out_row_r[:, None] + ROPE_HALF + offs_r[None, :],
            rope_hi,
            mask=mask_r[:, None],
        )

    if CACHE:
        # K[block, head, d // X, offset, d % X] and V[block, head, offset // X, d, offset % X].
        # A negative slot is a pad token.
        write = is_k_r & (slot >= 0)
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        cache_row = (block * K_HEADS + head_r - Q_HEADS) * (HEAD_DIM * BLOCK_SIZE)
        v_dst = cache_row + (offset // X) * (HEAD_DIM * X) + offset % X
        k_dst = cache_row + offset * X
        offs_hi = ROPE_HALF + offs_r
        k_dst_n = gl.convert_layout(k_dst, gl.SliceLayout(1, row_layout))
        write_n = gl.convert_layout(write, gl.SliceLayout(1, row_layout))
        if FP8:
            # One per-tensor scale, as reshape_and_cache(kv_cache_dtype="fp8") applies it.
            cache_ty = k_cache_ptr.dtype.element_ty
            v_c = gl.minimum(gl.maximum(v.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX).to(cache_ty)
            k_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX).to(cache_ty)
            k_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX).to(cache_ty)
            k_n = gl.minimum(gl.maximum(y.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX).to(cache_ty)
        else:
            v_c = v
            k_lo = rope_lo
            k_hi = rope_hi
            k_n = y
        gl.store(
            v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_c, mask=write[:, None]
        )
        gl.store(
            k_cache_ptr
            + k_dst[:, None]
            + ((offs_r // X) * (BLOCK_SIZE * X) + offs_r % X)[None, :],
            k_lo,
            mask=write[:, None],
        )
        gl.store(
            k_cache_ptr
            + k_dst[:, None]
            + ((offs_hi // X) * (BLOCK_SIZE * X) + offs_hi % X)[None, :],
            k_hi,
            mask=write[:, None],
        )
        # Columns [64, 256) from the norm layout, 16 bytes per lane.
        gl.store(
            k_cache_ptr
            + k_dst_n[:, None]
            + ((offs_n // X) * (BLOCK_SIZE * X) + offs_n % X)[None, :],
            k_n,
            mask=write_n[:, None] & tail,
        )
