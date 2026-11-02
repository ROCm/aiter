# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Q/K Gemma RMSNorm and interleaved MRoPE, and the K/V write into the paged KV cache
# (bf16, or fp8 with one per-tensor scale and Q emitted in fp8: row kernel only)

import functools

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

HEAD_DIM = gl.constexpr(256)
ROPE_HALF = gl.constexpr(32)  # rotary_dim 64: two Neox halves


@functools.cache
def get_fused_qk_norm_mrope_layouts(num_warps):
    # Rows over the waves: 8 values per lane for the norm, 1 value per lane for the RoPE halves and V.
    return {
        "row_layout": gl.BlockedLayout(
            [1, HEAD_DIM // 32], [1, 32], [num_warps, 1], [1, 0]
        ),
        "rope_layout": gl.BlockedLayout([1, 1], [1, 32], [num_warps, 1], [1, 0]),
    }


_fused_qk_norm_mrope_repr = make_kernel_repr(
    "fused_qk_norm_mrope_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M", "CACHE", "FP8"]
)
_fused_qk_norm_mrope_tdm_repr = make_kernel_repr(
    "fused_qk_norm_mrope_tdm_gfx1250", ["Q_HEADS", "K_HEADS", "BLOCK_M", "CACHE"]
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
    FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    X: gl.constexpr = 128 // k_cache_ptr.dtype.element_ty.primitive_bitwidth
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

    # Gemma RMSNorm: x / rms(x) * (1 + w). Columns [64, 256) are done. K keeps the input
    # dtype; so does Q unless it goes out in fp8.
    x = x.to(gl.float32)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / HEAD_DIM + EPS)
    y = (x * rstd[:, None] * (w.to(gl.float32) + 1.0)).to(k_out_ptr.dtype.element_ty)
    tail = (offs_n >= 2 * ROPE_HALF)[None, :]

    # Neox RoPE of columns [0, 64), from the same normalized values.
    rstd_r = gl.convert_layout(rstd, gl.SliceLayout(1, rope_layout))[:, None]
    y_lo = x_lo.to(gl.float32) * rstd_r * (w_lo.to(gl.float32) + 1.0)
    y_hi = x_hi.to(gl.float32) * rstd_r * (w_hi.to(gl.float32) + 1.0)
    rope_lo = (y_lo * cos - y_hi * sin).to(k_out_ptr.dtype.element_ty)
    rope_hi = (y_hi * cos + y_lo * sin).to(k_out_ptr.dtype.element_ty)

    if FP8:
        # Q goes out in fp8 and K stays bf16, so they need separate stores. Q is the bf16
        # value clamped and cast, as on the unfused path: the fp8 cast does not saturate.
        fp8_ty = q_out_ptr.dtype.element_ty
        q_rows = mask_m & is_q
        k_rows = mask_m & (head >= Q_HEADS)
        q_rows_r = mask_r & is_q_r
        k_rows_r = mask_r & (head_r >= Q_HEADS)
        out_n = offs_out[:, None] + offs_n[None, :]
        q_n = gl.minimum(gl.maximum(y.to(gl.float32), -FP8_MAX), FP8_MAX)
        gl.store(q_out_ptr + out_n, q_n.to(fp8_ty), mask=q_rows[:, None] & tail)
        gl.store(k_out_ptr + out_n, y, mask=k_rows[:, None] & tail)
        offs_out_r = gl.convert_layout(offs_out, gl.SliceLayout(1, rope_layout))
        out_r = offs_out_r[:, None] + offs_r[None, :]
        q_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32), -FP8_MAX), FP8_MAX)
        q_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32), -FP8_MAX), FP8_MAX)
        gl.store(q_out_ptr + out_r, q_lo.to(fp8_ty), mask=q_rows_r[:, None])
        gl.store(q_out_ptr + ROPE_HALF + out_r, q_hi.to(fp8_ty), mask=q_rows_r[:, None])
        gl.store(k_out_ptr + out_r, rope_lo, mask=k_rows_r[:, None])
        gl.store(k_out_ptr + ROPE_HALF + out_r, rope_hi, mask=k_rows_r[:, None])
    else:
        out_row = gl.where(is_q, q_out_ptr, k_out_ptr) + offs_out
        gl.store(out_row[:, None] + offs_n[None, :], y, mask=mask_m[:, None] & tail)
        out_row_r = gl.convert_layout(out_row, gl.SliceLayout(1, rope_layout))
        gl.store(out_row_r[:, None] + offs_r[None, :], rope_lo, mask=mask_r[:, None])
        gl.store(
            out_row_r[:, None] + ROPE_HALF + offs_r[None, :],
            rope_hi,
            mask=mask_r[:, None],
        )

    if CACHE:
        # K[block, head, d // X, offset, d % X] and V[block, head, offset // X, d, offset % X].
        # A masked slot load gives -1, a pad slot.
        write = (slot >= 0)[:, None]
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        cache_row = (block * K_HEADS + head_r - Q_HEADS) * (HEAD_DIM * BLOCK_SIZE)
        v_dst = cache_row + (offset // X) * (HEAD_DIM * X) + offset % X
        if FP8:
            # One per-tensor scale, as reshape_and_cache(kv_cache_dtype="fp8") applies it.
            cache_ty = k_cache_ptr.dtype.element_ty
            v_c = gl.minimum(gl.maximum(v.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
            kc_lo = gl.minimum(gl.maximum(rope_lo.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
            kc_hi = gl.minimum(gl.maximum(rope_hi.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
            kc_n = gl.minimum(gl.maximum(y.to(gl.float32) * kv_inv_scale, -FP8_MAX), FP8_MAX)
            v_c, kc_lo, kc_hi, kc_n = (
                v_c.to(cache_ty),
                kc_lo.to(cache_ty),
                kc_hi.to(cache_ty),
                kc_n.to(cache_ty),
            )
        else:
            v_c, kc_lo, kc_hi, kc_n = v, rope_lo, rope_hi, y
        gl.store(v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_c, mask=write)
        # ROPE_HALF is a multiple of X: the high RoPE half is a constant step after the low one.
        k_dst = cache_row + offset * X
        k_lo = k_dst[:, None] + (offs_r // X * BLOCK_SIZE * X + offs_r % X)[None, :]
        gl.store(k_cache_ptr + k_lo, kc_lo, mask=write)
        gl.store(k_cache_ptr + k_lo + ROPE_HALF * BLOCK_SIZE, kc_hi, mask=write)
        # Columns [64, 256) from the norm layout, 16 bytes per lane.
        k_dst_n = gl.convert_layout(k_dst, gl.SliceLayout(1, row_layout))
        write_n = gl.convert_layout(slot >= 0, gl.SliceLayout(1, row_layout))
        k_n = k_dst_n[:, None] + (offs_n // X * BLOCK_SIZE * X + offs_n % X)[None, :]
        mask_n = write_n[:, None] & tail
        gl.store(k_cache_ptr + k_n, kc_n, mask=mask_n)


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
    # The arguments of the cache write come last, as in the row kernel.
    v_ptr,
    slot_mapping_ptr,
    stride_v,
    num_slots,
    k_cache_ptr,
    v_cache_ptr,
    EPS: gl.constexpr,
    Q_HEADS: gl.constexpr,
    K_HEADS: gl.constexpr,
    SECTION_H: gl.constexpr,
    SECTION_W: gl.constexpr,
    BLOCK_M: gl.constexpr,
    CACHE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    row_layout: gl.constexpr,
    rope_layout: gl.constexpr,
):
    # A program does BLOCK_M tokens. Their Q heads and their K heads move between memory and
    # LDS as one TDM tile each.
    Q_COLS: gl.constexpr = Q_HEADS * HEAD_DIM
    K_COLS: gl.constexpr = K_HEADS * HEAD_DIM
    X: gl.constexpr = 128 // k_cache_ptr.dtype.element_ty.primitive_bitwidth
    t0 = gl.program_id(0) * BLOCK_M
    q_desc, q_s = _tdm_tile(q_ptr, num_tokens, stride_q, BLOCK_M, Q_COLS)
    k_desc, k_s = _tdm_tile(k_ptr, num_tokens, stride_k, BLOCK_M, K_COLS)
    q_out_desc, q_out_s = _tdm_tile(q_out_ptr, num_tokens, Q_COLS, BLOCK_M, Q_COLS)
    k_out_desc, k_out_s = _tdm_tile(k_out_ptr, num_tokens, K_COLS, BLOCK_M, K_COLS)
    gl.amd.gfx1250.tdm.async_load(q_desc, [t0, 0], q_s)
    gl.amd.gfx1250.tdm.async_load(k_desc, [t0, 0], k_s)
    if CACHE:
        # The V tile, and the slot of each K row. Tokens past the slot table are padding.
        v_desc, v_s = _tdm_tile(v_ptr, num_tokens, stride_v, BLOCK_M, K_COLS)
        gl.amd.gfx1250.tdm.async_load(v_desc, [t0, 0], v_s)
        offs_k = gl.arange(0, BLOCK_M * K_HEADS, layout=gl.SliceLayout(1, row_layout))
        token_k = t0 + offs_k // K_HEADS
        in_table = (token_k < num_tokens) & (token_k < num_slots)
        slot = gl.load(slot_mapping_ptr + token_k, mask=in_table, other=-1).to(gl.int64)
    # Each tile waits for its own TDM load only: the V tile can still be in flight.
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
        1 + CACHE,
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
        CACHE * 1,
        row_layout,
        rope_layout,
    )
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(q_out_desc, [t0, 0], q_out_s)
    gl.amd.gfx1250.tdm.async_store(k_out_desc, [t0, 0], k_out_s)
    if CACHE:
        # K from the K tile in LDS, 16 bytes a lane, and V one value a lane, after the V tile is in.
        gl.amd.gfx1250.tdm.async_wait(2)
        offs_n = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, row_layout))
        offs_v = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, rope_layout))
        block = slot // BLOCK_SIZE
        offset = slot % BLOCK_SIZE
        cache_row = (block * K_HEADS + offs_k % K_HEADS) * (HEAD_DIM * BLOCK_SIZE)
        k_dst = cache_row + offset * X
        k_n = k_dst[:, None] + (offs_n // X * BLOCK_SIZE * X + offs_n % X)[None, :]
        k_rows = k_out_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]).load(row_layout)
        gl.store(k_cache_ptr + k_n, k_rows, mask=(slot >= 0)[:, None])
        v_dst = cache_row + (offset // X) * (HEAD_DIM * X) + offset % X
        v_dst = gl.convert_layout(v_dst, gl.SliceLayout(1, rope_layout))
        write_v = gl.convert_layout(slot >= 0, gl.SliceLayout(1, rope_layout))[:, None]
        v_rows = v_s.reshape([BLOCK_M * K_HEADS, HEAD_DIM]).load(rope_layout)
        gl.store(
            v_cache_ptr + v_dst[:, None] + offs_v[None, :] * X, v_rows, mask=write_v
        )
    gl.amd.gfx1250.tdm.async_wait(0)
