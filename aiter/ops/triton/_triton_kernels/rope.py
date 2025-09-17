# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _get_neox_rotated_x_1D(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 1)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    x_rotated = tl.flip(x_rotated, 0)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x_1D(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    x_rotated = tl.where(x_rotated_mask, x, -x)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 1)
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    return x_rotated


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    x_rotated = tl.flip(x_rotated, 1)
    return x_rotated


@triton.jit
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    IS_BWD: tl.constexpr = False,
):
    if IS_BWD:
        x_rotated = tl.where(x_rotated_mask, -x, x)
    else:
        x_rotated = tl.where(x_rotated_mask, x, -x)

    x_rotated = tl.reshape(x_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
    x_rotated = tl.flip(x_rotated, 2)
    x_rotated = tl.reshape(
        x_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    return x_rotated


@triton.jit
def _rope_kernel_sbhd_fwd(
    x_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_s,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    stride_freqs_s,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_s,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    S,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_freqs_offs = tl.where(
                (d_offs >= BLOCK_D_HALF) & (d_offs < BLOCK_D),
                d_offs - BLOCK_D_HALF,
                d_offs,
            ).to(d_offs.dtype)
            d_freqs_mask = d_freqs_offs < BLOCK_D
        else:
            d_freqs_offs = d_offs // 2
            d_freqs_mask = d_freqs_offs < BLOCK_D_HALF
    else:
        d_freqs_offs = d_offs
        d_freqs_mask = d_freqs_offs < BLOCK_D

    freqs_mask = s_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        s_offs[:, None] * stride_freqs_s + d_freqs_offs[None, :] * stride_freqs_d
    )

    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + (d_offs + nope_offs)[None, :] * stride_x_d
    )
    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        b * stride_out_b
        + s_offs[:, None] * stride_out_s
        + h * stride_out_h
        + (d_offs + nope_offs)[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


@triton.jit
def _rope_kernel_sbhd_bwd(
    x_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_s,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    stride_freqs_s,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_s,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    S,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # Parallelize over batch and head. Handle 1 sequence per program
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_freqs_offs = tl.where(
                (d_offs >= BLOCK_D_HALF) & (d_offs < BLOCK_D),
                d_offs - BLOCK_D_HALF,
                d_offs,
            ).to(d_offs.dtype)
            d_freqs_mask = d_freqs_offs < BLOCK_D
        else:
            d_freqs_offs = d_offs // 2
            d_freqs_mask = d_freqs_offs < BLOCK_D_HALF
    else:
        d_freqs_offs = d_offs
        d_freqs_mask = d_freqs_offs < BLOCK_D

    freqs_mask = s_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        s_offs[:, None] * stride_freqs_s + d_freqs_offs[None, :] * stride_freqs_d
    )

    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x * sin, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF, True
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x * sin, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF, True
        )

    out_x = x * cos + x_rotated
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        b * stride_out_b
        + s_offs[:, None] * stride_out_s
        + h * stride_out_h
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


@triton.jit
def _rope_kernel_thd_fwd(
    x_ptr,
    cu_seqlens_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_freqs_t,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_start = tl.load(cu_seqlens_ptr + b)
    t_end = tl.load(cu_seqlens_ptr + b + 1)
    T = t_end - t_start
    if pid_t * BLOCK_T >= T:
        return

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_freqs_offs = tl.where(
                (d_offs >= BLOCK_D_HALF) & (d_offs < BLOCK_D),
                d_offs - BLOCK_D_HALF,
                d_offs,
            ).to(d_offs.dtype)
            d_freqs_mask = d_freqs_offs < BLOCK_D
        else:
            d_freqs_offs = d_offs // 2
            d_freqs_mask = d_freqs_offs < BLOCK_D_HALF
    else:
        d_freqs_offs = d_offs
        d_freqs_mask = d_freqs_offs < BLOCK_D

    freqs_mask = t_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        t_offs[:, None] * stride_freqs_t + d_freqs_offs[None, :] * stride_freqs_d
    )
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        (t_start + t_offs)[:, None] * stride_x_t
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        (t_start + t_offs)[:, None] * stride_out_t
        + h * stride_out_h
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


@triton.jit
def _rope_kernel_thd_bwd(
    x_ptr,
    cu_seqlens_ptr,
    freqs_ptr,
    out_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_freqs_t,
    stride_freqs_b,
    stride_freqs_h,
    stride_freqs_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_start = tl.load(cu_seqlens_ptr + b)
    t_end = tl.load(cu_seqlens_ptr + b + 1)
    T = t_end - t_start
    if pid_t * BLOCK_T >= T:
        return

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_freqs_offs = tl.where(
                (d_offs >= BLOCK_D_HALF) & (d_offs < BLOCK_D),
                d_offs - BLOCK_D_HALF,
                d_offs,
            ).to(d_offs.dtype)
            d_freqs_mask = d_freqs_offs < BLOCK_D
        else:
            d_freqs_offs = d_offs // 2
            d_freqs_mask = d_freqs_offs < BLOCK_D_HALF
    else:
        d_freqs_offs = d_offs
        d_freqs_mask = d_freqs_offs < BLOCK_D

    freqs_mask = t_mask[:, None] & d_freqs_mask[None, :]
    freqs_offs = (
        t_offs[:, None] * stride_freqs_t + d_freqs_offs[None, :] * stride_freqs_d
    )
    freqs = tl.load(freqs_ptr + freqs_offs, mask=freqs_mask)
    cos = tl.cos(freqs.to(tl.float32))
    sin = tl.sin(freqs.to(tl.float32))

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        (t_start + t_offs)[:, None] * stride_x_t
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
        )

    out_x = x * cos + x_rotated
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        (t_start + t_offs)[:, None] * stride_out_t
        + h * stride_out_h
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


@triton.jit
def _rope_kernel_sbhd_cached_fwd(
    x_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_ptr,
    stride_x_s,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    stride_cos_s,
    stride_cos_b,
    stride_cos_h,
    stride_cos_d,
    stride_pos_s,
    stride_pos_b,
    stride_out_s,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    S,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

    if HAVE_POS:
        pos_offs = s_offs * stride_pos_s + b * stride_pos_b
        pos = tl.load(pos_ptr + pos_offs, mask=s_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=s_mask)
            s_cos_offs = pos + offset
        else:
            s_cos_offs = pos
    else:
        s_cos_offs = s_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs >= BLOCK_D_HALF) & (d_cos_offs < BLOCK_D),
                d_cos_offs - BLOCK_D_HALF,
                d_cos_offs,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D
        else:
            d_cos_offs = d_offs // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = s_mask[:, None] & d_cos_mask[None, :]
    cos_offs = s_cos_offs[:, None] * stride_cos_s + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF
        )

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        b * stride_out_b
        + s_offs[:, None] * stride_out_s
        + h * stride_out_h
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


@triton.jit
def _rope_kernel_sbhd_cached_bwd(
    x_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_ptr,
    stride_x_s,
    stride_x_b,
    stride_x_h,
    stride_x_d,
    stride_cos_s,
    stride_cos_b,
    stride_cos_h,
    stride_cos_d,
    stride_pos_s,
    stride_pos_b,
    stride_out_s,
    stride_out_b,
    stride_out_h,
    stride_out_d,
    S,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offs = tl.arange(0, BLOCK_D)
    s_mask = s_offs < S

    if HAVE_POS:
        pos_offs = s_offs * stride_pos_s + b * stride_pos_b
        pos = tl.load(pos_ptr + pos_offs, mask=s_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=s_mask)
            s_cos_offs = pos + offset
        else:
            s_cos_offs = pos
    else:
        s_cos_offs = s_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs >= BLOCK_D_HALF) & (d_cos_offs < BLOCK_D),
                d_cos_offs - BLOCK_D_HALF,
                d_cos_offs,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D
        else:
            d_cos_offs = d_offs // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = s_mask[:, None] & d_cos_mask[None, :]
    cos_offs = s_cos_offs[:, None] * stride_cos_s + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = s_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x * sin, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF, True
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x * sin, x_rotated_mask, BLOCK_S, BLOCK_D, BLOCK_D_HALF, True
        )

    out_x = x * cos + x_rotated
    out_x = out_x.to(x_ptr.dtype.element_ty)
    x_out_offs = (
        b * stride_out_b
        + s_offs[:, None] * stride_out_s
        + h * stride_out_h
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs - BLOCK_D * stride_out_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_ptr + x_out_offs + BLOCK_D * stride_out_d, x, mask=x_mask)


@triton.jit
def _rope_kernel_thd_cached_2c_fwd(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_x_ptr,
    out_y_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_x_t,
    stride_out_x_h,
    stride_out_x_d,
    stride_out_y_t,
    stride_out_y_h,
    stride_out_y_d,
    T,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLIT_H_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    num_stages: tl.constexpr,
):
    h_s = tl.program_id(0)
    pid_t = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_x_t > 0)
    tl.assume(stride_out_x_h > 0)
    tl.assume(stride_out_x_d > 0)
    tl.assume(stride_out_y_t > 0)
    tl.assume(stride_out_y_h > 0)
    tl.assume(stride_out_y_d > 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    h_start_idx = h_s * SPLIT_H_SIZE

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    for h in tl.range(0, SPLIT_H_SIZE, 1, num_stages=num_stages):
        x_offs = (
            t_offs[:, None] * stride_x_t
            + d_offs[None, :] * stride_x_d
            + (h_start_idx + h) * stride_x_h
        )
        y_offs = (
            t_offs[:, None] * stride_y_t
            + d_offs[None, :] * stride_y_d
            + (h_start_idx + h) * stride_y_h
        )

        x = tl.load(x_ptr + x_offs, mask=x_mask)
        y = tl.load(y_ptr + y_offs, mask=x_mask)

        if IS_NEOX:
            x_rotated = _get_neox_rotated_x(
                x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
            y_rotated = _get_neox_rotated_x(
                y, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
        else:
            x_rotated = _get_gptj_rotated_x(
                x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
            y_rotated = _get_gptj_rotated_x(
                y, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )

        out_x = x * cos + x_rotated * sin
        out_x = out_x.to(x_ptr.dtype.element_ty)
        out_y = y * cos + y_rotated * sin
        out_y = out_y.to(y_ptr.dtype.element_ty)

        out_x_offs = (
            t_offs[:, None] * stride_out_x_t
            + d_offs[None, :] * stride_out_x_d
            + (h_start_idx + h) * stride_out_x_h
        )
        out_y_offs = (
            t_offs[:, None] * stride_out_y_t
            + d_offs[None, :] * stride_out_y_d
            + (h_start_idx + h) * stride_out_y_h
        )
        tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)
        tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)

        if HAVE_NOPE and not INPLACE:
            if NOPE_FIRST:
                x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs - BLOCK_D * stride_out_x_d, x, mask=x_mask
                )
                y = tl.load(y_ptr + y_offs - BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs - BLOCK_D * stride_out_y_d, y, mask=x_mask
                )
            else:
                x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs + BLOCK_D * stride_out_x_d, x, mask=x_mask
                )
                y = tl.load(y_ptr + y_offs + BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs + BLOCK_D * stride_out_y_d, y, mask=x_mask
                )


@triton.jit
def _rope_kernel_thd_cached_2c_bwd(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_x_ptr,
    out_y_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_x_t,
    stride_out_x_h,
    stride_out_x_d,
    stride_out_y_t,
    stride_out_y_h,
    stride_out_y_d,
    T,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SPLIT_H_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    num_stages: tl.constexpr,
):
    h_s = tl.program_id(0)
    pid_t = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_x_t > 0)
    tl.assume(stride_out_x_h > 0)
    tl.assume(stride_out_x_d > 0)
    tl.assume(stride_out_y_t > 0)
    tl.assume(stride_out_y_h > 0)
    tl.assume(stride_out_y_d > 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    h_start_idx = h_s * SPLIT_H_SIZE

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    for h in tl.range(0, SPLIT_H_SIZE, 1, num_stages=num_stages):
        x_offs = (
            t_offs[:, None] * stride_x_t
            + d_offs[None, :] * stride_x_d
            + (h_start_idx + h) * stride_x_h
        )
        y_offs = (
            t_offs[:, None] * stride_y_t
            + d_offs[None, :] * stride_y_d
            + (h_start_idx + h) * stride_y_h
        )

        x = tl.load(x_ptr + x_offs, mask=x_mask)
        y = tl.load(y_ptr + y_offs, mask=x_mask)

        if IS_NEOX:
            x_rotated = _get_neox_rotated_x(
                x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )
            y_rotated = _get_neox_rotated_x(
                y * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )
        else:
            x_rotated = _get_gptj_rotated_x(
                x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )
            y_rotated = _get_gptj_rotated_x(
                y * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )

        out_x = x * cos + x_rotated
        out_x = out_x.to(x_ptr.dtype.element_ty)
        out_y = y * cos + y_rotated
        out_y = out_y.to(y_ptr.dtype.element_ty)

        out_x_offs = (
            t_offs[:, None] * stride_out_x_t
            + d_offs[None, :] * stride_out_x_d
            + (h_start_idx + h) * stride_out_x_h
        )
        out_y_offs = (
            t_offs[:, None] * stride_out_y_t
            + d_offs[None, :] * stride_out_y_d
            + (h_start_idx + h) * stride_out_y_h
        )
        tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)
        tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)

        if HAVE_NOPE and not INPLACE:
            # TODO check
            if NOPE_FIRST:
                x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs - BLOCK_D * stride_out_x_d, x, mask=x_mask
                )
                y = tl.load(y_ptr + y_offs - BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs - BLOCK_D * stride_out_y_d, y, mask=x_mask
                )
            else:
                x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs + BLOCK_D * stride_out_x_d, x, mask=x_mask
                )
                y = tl.load(y_ptr + y_offs + BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs + BLOCK_D * stride_out_y_d, y, mask=x_mask
                )


@triton.jit
def _rope_kernel_cached_thd_2c_gqa_fwd(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_x_ptr,
    out_y_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_x_t,
    stride_out_x_h,
    stride_out_x_d,
    stride_out_y_t,
    stride_out_y_h,
    stride_out_y_d,
    T,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QH_per_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    num_stages: tl.constexpr,
):
    h_s = tl.program_id(0)
    pid_t = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_x_t > 0)
    tl.assume(stride_out_x_h > 0)
    tl.assume(stride_out_x_d > 0)
    tl.assume(stride_out_y_t > 0)
    tl.assume(stride_out_y_h > 0)
    tl.assume(stride_out_y_d > 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    h_start_idx = h_s * QH_per_G

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    y_offs = (
        t_offs[:, None] * stride_y_t + d_offs[None, :] * stride_y_d + h_s * stride_y_h
    )
    y = tl.load(y_ptr + y_offs, mask=x_mask)

    if IS_NEOX:
        y_rotated = _get_neox_rotated_x(
            y, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        y_rotated = _get_gptj_rotated_x(
            y, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    out_y_offs = (
        t_offs[:, None] * stride_out_y_t
        + d_offs[None, :] * stride_out_y_d
        + h_s * stride_out_y_h
    )
    out_y = y * cos + y_rotated * sin
    out_y = out_y.to(y_ptr.dtype.element_ty)
    tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            y = tl.load(y_ptr + y_offs - BLOCK_D * stride_y_d, mask=x_mask)
            tl.store(out_y_ptr + out_y_offs - BLOCK_D * stride_out_y_d, y, mask=x_mask)
        else:
            y = tl.load(y_ptr + y_offs + BLOCK_D * stride_y_d, mask=x_mask)
            tl.store(out_y_ptr + out_y_offs + BLOCK_D * stride_out_y_d, y, mask=x_mask)

    for h in tl.range(0, QH_per_G, 1, num_stages=num_stages):
        x_offs = (
            t_offs[:, None] * stride_x_t
            + d_offs[None, :] * stride_x_d
            + (h_start_idx + h) * stride_x_h
        )

        x = tl.load(x_ptr + x_offs, mask=x_mask)

        if IS_NEOX:
            x_rotated = _get_neox_rotated_x(
                x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
        else:
            x_rotated = _get_gptj_rotated_x(
                x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )

        out_x_offs = (
            t_offs[:, None] * stride_out_x_t
            + d_offs[None, :] * stride_out_x_d
            + (h_start_idx + h) * stride_out_x_h
        )
        out_x = x * cos + x_rotated * sin
        out_x = out_x.to(x_ptr.dtype.element_ty)

        tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)

        if HAVE_NOPE and not INPLACE:
            if NOPE_FIRST:
                x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs - BLOCK_D * stride_out_x_d, x, mask=x_mask
                )
            else:
                x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs + BLOCK_D * stride_out_x_d, x, mask=x_mask
                )


@triton.jit
def _rope_kernel_cached_thd_2c_gqa_onehead_fwd(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_x_ptr,
    out_y_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_x_t,
    stride_out_x_h,
    stride_out_x_d,
    stride_out_y_t,
    stride_out_y_h,
    stride_out_y_d,
    T,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    pid_t = tl.program_id(0)
    hq = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_x_t > 0)
    tl.assume(stride_out_x_h > 0)
    tl.assume(stride_out_x_d > 0)
    tl.assume(stride_out_y_t > 0)
    tl.assume(stride_out_y_h > 0)
    tl.assume(stride_out_y_d > 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        t_offs[:, None] * stride_x_t + d_offs[None, :] * stride_x_d + hq * stride_x_h
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )

    out_x_offs = (
        t_offs[:, None] * stride_out_x_t
        + d_offs[None, :] * stride_out_x_d
        + hq * stride_out_x_h
    )
    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_x_ptr + out_x_offs - BLOCK_D * stride_out_x_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_x_ptr + out_x_offs + BLOCK_D * stride_out_x_d, x, mask=x_mask)

    if hq < G:
        y_offs = (
            t_offs[:, None] * stride_y_t
            + d_offs[None, :] * stride_x_d
            + hq * stride_y_h
        )
        y = tl.load(y_ptr + y_offs, mask=x_mask)

        if IS_NEOX:
            y_rotated = _get_neox_rotated_x(
                y, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )
        else:
            y_rotated = _get_gptj_rotated_x(
                y, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
            )

        out_y_offs = (
            t_offs[:, None] * stride_out_y_t
            + d_offs[None, :] * stride_out_y_d
            + hq * stride_out_y_h
        )
        out_y = y * cos + y_rotated * sin
        out_y = out_y.to(y_ptr.dtype.element_ty)
        tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)

        if HAVE_NOPE and not INPLACE:
            if NOPE_FIRST:
                y = tl.load(y_ptr + y_offs - BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs - BLOCK_D * stride_out_y_d, y, mask=x_mask
                )
            else:
                y = tl.load(y_ptr + y_offs + BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs + BLOCK_D * stride_out_y_d, y, mask=x_mask
                )


@triton.jit
def _rope_kernel_cached_thd_2c_gqa_bwd(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_x_ptr,
    out_y_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_x_t,
    stride_out_x_h,
    stride_out_x_d,
    stride_out_y_t,
    stride_out_y_h,
    stride_out_y_d,
    T,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QH_per_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    num_stages: tl.constexpr,
):
    h_s = tl.program_id(0)
    pid_t = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_x_t > 0)
    tl.assume(stride_out_x_h > 0)
    tl.assume(stride_out_x_d > 0)
    tl.assume(stride_out_y_t > 0)
    tl.assume(stride_out_y_h > 0)
    tl.assume(stride_out_y_d > 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    h_start_idx = h_s * QH_per_G

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    y_offs = (
        t_offs[:, None] * stride_y_t + d_offs[None, :] * stride_y_d + h_s * stride_y_h
    )
    y = tl.load(y_ptr + y_offs, mask=x_mask)

    if IS_NEOX:
        y_rotated = _get_neox_rotated_x(
            y * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
        )
    else:
        y_rotated = _get_gptj_rotated_x(
            y * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
        )

    out_y_offs = (
        t_offs[:, None] * stride_out_y_t
        + d_offs[None, :] * stride_out_y_d
        + h_s * stride_out_y_h
    )
    out_y = y * cos + y_rotated
    out_y = out_y.to(y_ptr.dtype.element_ty)
    tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            y = tl.load(y_ptr + y_offs - BLOCK_D * stride_y_d, mask=x_mask)
            tl.store(out_y_ptr + out_y_offs - BLOCK_D * stride_out_y_d, y, mask=x_mask)
        else:
            y = tl.load(y_ptr + y_offs + BLOCK_D * stride_y_d, mask=x_mask)
            tl.store(out_y_ptr + out_y_offs + BLOCK_D * stride_out_y_d, y, mask=x_mask)

    for h in tl.range(0, QH_per_G, 1, num_stages=num_stages):
        x_offs = (
            t_offs[:, None] * stride_x_t
            + d_offs[None, :] * stride_x_d
            + (h_start_idx + h) * stride_x_h
        )

        x = tl.load(x_ptr + x_offs, mask=x_mask)

        if IS_NEOX:
            x_rotated = _get_neox_rotated_x(
                x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )
        else:
            x_rotated = _get_gptj_rotated_x(
                x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )

        out_x_offs = (
            t_offs[:, None] * stride_out_x_t
            + d_offs[None, :] * stride_out_x_d
            + (h_start_idx + h) * stride_out_x_h
        )
        out_x = x * cos + x_rotated
        out_x = out_x.to(x_ptr.dtype.element_ty)

        tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)

        if HAVE_NOPE and not INPLACE:
            if NOPE_FIRST:
                x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs - BLOCK_D * stride_out_x_d, x, mask=x_mask
                )
            else:
                x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
                tl.store(
                    out_x_ptr + out_x_offs + BLOCK_D * stride_out_x_d, x, mask=x_mask
                )


@triton.jit
def _rope_kernel_cached_thd_2c_gqa_onehead_bwd(
    x_ptr,
    y_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    off_ptr,
    out_x_ptr,
    out_y_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_x_t,
    stride_out_x_h,
    stride_out_x_d,
    stride_out_y_t,
    stride_out_y_h,
    stride_out_y_d,
    T,
    HAVE_NOPE: tl.constexpr,
    NOPE_FIRST: tl.constexpr,
    INPLACE: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HAVE_POS: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    pid_t = tl.program_id(0)
    hq = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_x_t > 0)
    tl.assume(stride_out_x_h > 0)
    tl.assume(stride_out_x_d > 0)
    tl.assume(stride_out_y_t > 0)
    tl.assume(stride_out_y_h > 0)
    tl.assume(stride_out_y_d > 0)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, BLOCK_D)
    t_mask = t_offs < T

    if HAVE_POS:
        pos_offs = t_offs * stride_pos_t
        pos = tl.load(pos_ptr + pos_offs, mask=t_mask)
        if HAVE_OFFS:
            offset = tl.load(off_ptr + pos_offs, mask=t_mask)
            t_cos_offs = pos + offset
        else:
            t_cos_offs = pos
    else:
        t_cos_offs = t_offs

    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_offs
            d_cos_offs = tl.where(
                (d_cos_offs < BLOCK_D_HALF),
                d_cos_offs,
                d_cos_offs - BLOCK_D_HALF,
            ).to(d_cos_offs.dtype)
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
        else:
            d_cos_offs = tl.arange(0, BLOCK_D) // 2
            d_cos_mask = d_cos_offs < BLOCK_D_HALF
    else:
        d_cos_offs = d_offs
        d_cos_mask = d_cos_offs < BLOCK_D

    cos_mask = t_mask[:, None] & d_cos_mask[None, :]
    cos_offs = t_cos_offs[:, None] * stride_cos_t + d_cos_offs[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    nope_offs = 0
    if HAVE_NOPE and NOPE_FIRST:
        nope_offs = BLOCK_D

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    d_offs += nope_offs
    x_offs = (
        t_offs[:, None] * stride_x_t + d_offs[None, :] * stride_x_d + hq * stride_x_h
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated = _get_neox_rotated_x(
            x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
        )
    else:
        x_rotated = _get_gptj_rotated_x(
            x * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
        )

    out_x_offs = (
        t_offs[:, None] * stride_out_x_t
        + d_offs[None, :] * stride_out_x_d
        + hq * stride_out_x_h
    )
    out_x = x * cos + x_rotated
    out_x = out_x.to(x_ptr.dtype.element_ty)
    tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)

    if HAVE_NOPE and not INPLACE:
        if NOPE_FIRST:
            x = tl.load(x_ptr + x_offs - BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_x_ptr + out_x_offs - BLOCK_D * stride_out_x_d, x, mask=x_mask)
        else:
            x = tl.load(x_ptr + x_offs + BLOCK_D * stride_x_d, mask=x_mask)
            tl.store(out_x_ptr + out_x_offs + BLOCK_D * stride_out_x_d, x, mask=x_mask)

    if hq < G:
        y_offs = (
            t_offs[:, None] * stride_y_t
            + d_offs[None, :] * stride_x_d
            + hq * stride_y_h
        )
        y = tl.load(y_ptr + y_offs, mask=x_mask)

        if IS_NEOX:
            y_rotated = _get_neox_rotated_x(
                y * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )
        else:
            y_rotated = _get_gptj_rotated_x(
                y * sin, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF, True
            )

        out_y_offs = (
            t_offs[:, None] * stride_out_y_t
            + d_offs[None, :] * stride_out_y_d
            + hq * stride_out_y_h
        )
        out_y = y * cos + y_rotated
        out_y = out_y.to(y_ptr.dtype.element_ty)
        tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)

        if HAVE_NOPE and not INPLACE:
            if NOPE_FIRST:
                y = tl.load(y_ptr + y_offs - BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs - BLOCK_D * stride_out_y_d, y, mask=x_mask
                )
            else:
                y = tl.load(y_ptr + y_offs + BLOCK_D * stride_y_d, mask=x_mask)
                tl.store(
                    out_y_ptr + out_y_offs + BLOCK_D * stride_out_y_d, y, mask=x_mask
                )


@triton.jit
def _rope_fwd_2d_kernel_neox(
    x_ptr,
    cos_h_ptr,
    sin_h_ptr,
    cos_w_ptr,
    sin_w_ptr,
    out_ptr,
    stride_x_b,
    stride_x_wh,
    stride_x_h,
    stride_x_d,
    stride_cos_h_b,
    stride_cos_h_ht,
    stride_cos_h_h,
    stride_cos_h_d,
    stride_cos_w_b,
    stride_cos_w_w,
    stride_cos_w_h,
    stride_cos_w_d,
    WH: tl.constexpr,
    HEIGHT: tl.constexpr,
    WEIGHT: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # load cos_h [HT, BLOCK_D]
    offs_wh = tl.arange(0, WH)
    offs_cos_h_h = offs_wh // WEIGHT
    offs_d = tl.arange(0, BLOCK_D)
    offs_cos_h = (
        stride_cos_h_h * offs_cos_h_h[:, None] + stride_cos_h_d * offs_d[None, :]
    )
    mask_cos_h = offs_d < BLOCK_D // 2
    cos_h = tl.load(cos_h_ptr + offs_cos_h, mask=mask_cos_h[None, :])

    # load sin_h
    sin_h = tl.load(sin_h_ptr + offs_cos_h, mask=mask_cos_h[None, :])

    # load cos_w
    offs_cos_w_w = offs_wh % WEIGHT
    offs_cos_w_d = offs_d - BLOCK_D // 2
    offs_cos_w = (
        stride_cos_w_w * offs_cos_w_w[:, None] + stride_cos_w_d * offs_cos_w_d[None, :]
    )
    mask_cos_w = (offs_cos_w_d >= 0) & (offs_cos_w_d < BLOCK_D // 2)
    cos_w = tl.load(cos_w_ptr + offs_cos_w, mask=mask_cos_w[None, :])

    # load sin_w
    sin_w = tl.load(sin_w_ptr + offs_cos_w, mask=mask_cos_w[None, :])

    # load x
    offs_wh = tl.arange(0, WH)
    offs_x = (
        stride_x_b * b
        + stride_x_wh * offs_wh[:, None]
        + stride_x_h * h
        + stride_x_d * offs_d[None, :]
    )
    x = tl.load(x_ptr + offs_x)

    # load x_rotated
    offs_wh = tl.arange(0, WH)
    offs_d_rotated = tl.where(offs_d < BLOCK_D // 4, offs_d + BLOCK_D // 4, offs_d)
    offs_d_rotated = tl.where(
        (offs_d >= BLOCK_D // 4) & (offs_d < BLOCK_D // 2),
        offs_d_rotated - BLOCK_D // 4,
        offs_d_rotated,
    )
    offs_d_rotated = tl.where(
        (offs_d >= BLOCK_D // 2) & (offs_d < 3 * BLOCK_D // 4),
        offs_d_rotated + BLOCK_D // 4,
        offs_d_rotated,
    )
    offs_d_rotated = tl.where(
        (offs_d >= 3 * BLOCK_D // 4) & (offs_d < BLOCK_D),
        offs_d_rotated - BLOCK_D // 4,
        offs_d_rotated,
    )
    offs_x_rotated = (
        stride_x_b * b
        + stride_x_wh * offs_wh[:, None]
        + stride_x_h * h
        + stride_x_d * offs_d_rotated[None, :]
    )
    x_rotated = tl.load(x_ptr + offs_x_rotated)
    neg_x_rotated = tl.where((offs_d >= BLOCK_D // 4) & (offs_d < BLOCK_D // 2), 1, 0)
    neg_x_rotated = tl.where(
        (offs_d >= 3 * BLOCK_D // 4) & (offs_d < BLOCK_D), 1, neg_x_rotated
    )
    x_rotated = tl.where(neg_x_rotated, x_rotated, -x_rotated)

    # compute x1
    x1 = x * cos_h + x_rotated * sin_h

    # compute x2
    x2 = x * cos_w + x_rotated * sin_w

    # compute output
    out = x1 + x2

    # store output
    tl.store(out_ptr + offs_x, out)
