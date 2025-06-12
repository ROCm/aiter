# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from torch import autograd
from enum import IntEnum


class RotateStyle(IntEnum):
    NEOX = (0,)
    GPTJ = 1


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
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
):
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
def _rope_fwd_kernel_sbhd(
    x_ptr: torch.Tensor,
    freqs_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
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

    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
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
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)


@triton.jit
def _rope_fwd_kernel_thd(
    x_ptr: torch.Tensor,
    cu_seqlens_ptr: torch.Tensor,
    freqs_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
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

    x_offs = (
        (t_start + t_offs)[:, None] * stride_x_t
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
    )
    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask)

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
        x_rotated = _get_neox_rotated_x(
            x, x_rotated_mask, BLOCK_T, BLOCK_D, BLOCK_D_HALF
        )
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]
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


@triton.jit
def _rope_fwd_kernel_sbhd_cached(
    x_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    off_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
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

    x_offs = (
        b * stride_x_b
        + s_offs[:, None] * stride_x_s
        + h * stride_x_h
        + d_offs[None, :] * stride_x_d
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
        + d_offs[None, :] * stride_out_d
    )

    tl.store(out_ptr + x_out_offs, out_x, mask=x_mask)


@triton.jit
def _rope_2c_fwd_kernel_thd_cached(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    off_ptr: torch.Tensor,
    out_x_ptr: torch.Tensor,
    out_y_ptr: torch.Tensor,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    T,
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
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_t > 0)
    tl.assume(stride_out_h > 0)
    tl.assume(stride_out_d > 0)

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

    h_start_idx = h_s * SPLIT_H_SIZE

    x_mask = t_mask[:, None] & (d_offs < BLOCK_D)[None, :]
    x_offs_base = t_offs[:, None] * stride_x_t + d_offs[None, :] * stride_x_d

    if IS_NEOX:
        x_rotated_mask = (d_offs < BLOCK_D_HALF)[None, :]
    else:
        x_rotated_mask = (d_offs % 2 == 0)[None, :]

    for h in tl.range(0, SPLIT_H_SIZE, 1, num_stages=num_stages):
        x_offs = x_offs_base + (h_start_idx + h) * stride_x_h

        x = tl.load(x_ptr + x_offs, mask=x_mask)
        y = tl.load(y_ptr + x_offs, mask=x_mask)

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

        tl.store(out_x_ptr + x_offs, out_x, mask=x_mask)
        tl.store(out_y_ptr + x_offs, out_y, mask=x_mask)


@triton.jit
def _rope_fwd_kernel_neox_cached_thd_position_offsets_2c_gqa(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    off_ptr: torch.Tensor,
    out_x_ptr: torch.Tensor,
    out_y_ptr: torch.Tensor,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    T,
    reuse_freqs_front_part: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QH_per_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    num_stages: tl.constexpr,
):

    h_s = tl.program_id(0)
    s = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_t > 0)
    tl.assume(stride_out_h > 0)
    tl.assume(stride_out_d > 0)

    pos_offs = s * BLOCK_T + tl.arange(0, BLOCK_T)
    pos_mask = pos_offs < T
    pos = tl.load(pos_ptr + pos_offs, mask=pos_mask)
    if HAVE_OFFS:
        offset = tl.load(off_ptr + pos_offs, mask=pos_mask)
        cos_offs_t = pos + offset
    else:
        cos_offs_t = pos

    if reuse_freqs_front_part:
        cos_offs_d = tl.arange(0, BLOCK_D)
        cos_offs_d = tl.where(
            (cos_offs_d < BLOCK_D_HALF),
            cos_offs_d,
            cos_offs_d - BLOCK_D_HALF,
        ).to(cos_offs_d.dtype)
        cos_mask_d = cos_offs_d < BLOCK_D_HALF
    else:
        cos_offs_d = tl.arange(0, BLOCK_D)
        cos_mask_d = cos_offs_d < BLOCK_D

    cos_mask = (pos_offs)[:, None] & (cos_mask_d)[None, :]
    cos_offs = cos_offs_t[:, None] * stride_cos_t + cos_offs_d[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    h_start_idx = h_s * QH_per_G

    x_offs_t = pos_offs
    x_offs_d = tl.arange(0, BLOCK_D)
    x_mask = (x_offs_t < T)[:, None] & (x_offs_d < BLOCK_D)[None, :]
    x_offs_base = x_offs_t[:, None] * stride_x_t + x_offs_d[None, :] * stride_x_d

    x_rotated_mask = (x_offs_d < BLOCK_D_HALF)[None, :]

    y_offs = (
        x_offs_t[:, None] * stride_y_t
        + x_offs_d[None, :] * stride_y_d
        + h_s * stride_y_h
    )
    y = tl.load(y_ptr + y_offs, mask=x_mask)
    y_rotated = tl.where(x_rotated_mask, y, -y)
    y_rotated = tl.reshape(y_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
    y_rotated = tl.flip(y_rotated, 2)
    y_rotated = tl.reshape(
        y_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )
    y_rotated = tl.flip(y_rotated, 1)

    out_y = y * cos + y_rotated * sin
    out_y = out_y.to(y_ptr.dtype.element_ty)
    tl.store(out_y_ptr + y_offs, out_y, mask=x_mask)

    for h in tl.range(0, QH_per_G, 1, num_stages=num_stages):
        x_offs = x_offs_base + (h_start_idx + h) * stride_x_h

        x = tl.load(x_ptr + x_offs, mask=x_mask)
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

        out_x = x * cos + x_rotated * sin
        out_x = out_x.to(x_ptr.dtype.element_ty)

        tl.store(out_x_ptr + x_offs, out_x, mask=x_mask)


@triton.jit
def _rope_fwd_kernel_gptj_cached_thd_position_offsets_2c_gqa(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    off_ptr: torch.Tensor,
    out_x_ptr: torch.Tensor,
    out_y_ptr: torch.Tensor,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    T,
    reuse_freqs_front_part: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QH_per_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
    num_stages: tl.constexpr,
):

    h_s = tl.program_id(0)
    s = tl.program_id(1)

    tl.assume(stride_x_t > 0)
    tl.assume(stride_x_h > 0)
    tl.assume(stride_x_d > 0)
    tl.assume(stride_y_t > 0)
    tl.assume(stride_y_h > 0)
    tl.assume(stride_y_d > 0)
    tl.assume(stride_cos_t > 0)
    tl.assume(stride_cos_d > 0)
    tl.assume(stride_pos_t > 0)
    tl.assume(stride_out_t > 0)
    tl.assume(stride_out_h > 0)
    tl.assume(stride_out_d > 0)

    pos_offs = s * BLOCK_T + tl.arange(0, BLOCK_T)
    pos_mask = pos_offs < T
    pos = tl.load(pos_ptr + pos_offs, mask=pos_mask)
    if HAVE_OFFS:
        offset = tl.load(off_ptr + pos_offs, mask=pos_mask)
        cos_offs_t = pos + offset
    else:
        cos_offs_t = pos

    if reuse_freqs_front_part:
        cos_offs_d = tl.arange(0, BLOCK_D) // 2
        cos_mask_d = cos_offs_d < BLOCK_D_HALF
    else:
        cos_offs_d = tl.arange(0, BLOCK_D)
        cos_mask_d = cos_offs_d < BLOCK_D

    cos_mask = (pos_mask)[:, None] & (cos_mask_d)[None, :]
    cos_offs = cos_offs_t[:, None] * stride_cos_t + cos_offs_d[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    h_start_idx = h_s * QH_per_G

    x_offs_t = pos_offs
    x_offs_d = tl.arange(0, BLOCK_D)
    x_mask = (x_offs_t < T)[:, None] & (x_offs_d < BLOCK_D)[None, :]
    x_offs_base = x_offs_t[:, None] * stride_x_t + x_offs_d[None, :] * stride_x_d

    x_rotated_mask = (x_offs_d % 2 == 0)[None, :]

    y_offs = (
        x_offs_t[:, None] * stride_y_t
        + x_offs_d[None, :] * stride_y_d
        + h_s * stride_y_h
    )
    y = tl.load(y_ptr + y_offs, mask=x_mask)
    y_rotated = tl.where(x_rotated_mask, y, -y)
    y_rotated = tl.reshape(y_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
    y_rotated = tl.flip(y_rotated, 2)
    y_rotated = tl.reshape(
        y_rotated,
        (
            BLOCK_T,
            BLOCK_D,
        ),
    )

    out_y = y * cos + y_rotated * sin
    out_y = out_y.to(y_ptr.dtype.element_ty)
    tl.store(out_y_ptr + y_offs, out_y, mask=x_mask)

    for h in tl.range(0, QH_per_G, 1, num_stages=num_stages):
        x_offs = x_offs_base + (h_start_idx + h) * stride_x_h

        x = tl.load(x_ptr + x_offs, mask=x_mask)
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

        out_x = x * cos + x_rotated * sin
        out_x = out_x.to(x_ptr.dtype.element_ty)

        tl.store(out_x_ptr + x_offs, out_x, mask=x_mask)


@triton.jit
def _rope_fwd_kernel_neox_cached_thd_position_offsets_2c_gqa_one_head(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    off_ptr: torch.Tensor,
    out_x_ptr: torch.Tensor,
    out_y_ptr: torch.Tensor,
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
    reuse_freqs_front_part: tl.constexpr,
    BLOCK_T: tl.constexpr,
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
):

    s = tl.program_id(0)
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

    pos_offs = s * BLOCK_T + tl.arange(0, BLOCK_T)
    pos_mask = pos_offs < T
    pos = tl.load(pos_ptr + pos_offs, mask=pos_mask)
    if HAVE_OFFS:
        offset = tl.load(off_ptr + pos_offs, mask=pos_mask)
        cos_offs_t = pos + offset
    else:
        cos_offs_t = pos

    if reuse_freqs_front_part:
        cos_offs_d = tl.arange(0, BLOCK_D)
        cos_offs_d = tl.where(
            (cos_offs_d < BLOCK_D_HALF),
            cos_offs_d,
            cos_offs_d - BLOCK_D_HALF,
        ).to(cos_offs_d.dtype)
        cos_mask_d = cos_offs_d < BLOCK_D_HALF
    else:
        cos_offs_d = tl.arange(0, BLOCK_D)
        cos_mask_d = cos_offs_d < BLOCK_D

    cos_mask = (pos_mask)[:, None] & (cos_mask_d)[None, :]
    cos_offs = cos_offs_t[:, None] * stride_cos_t + cos_offs_d[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    x_offs_t = pos_offs
    x_offs_d = tl.arange(0, BLOCK_D)
    x_mask = (x_offs_t < T)[:, None] & (x_offs_d < BLOCK_D)[None, :]
    x_rotated_mask = (x_offs_d < BLOCK_D_HALF)[None, :]

    x_offs = (
        x_offs_t[:, None] * stride_x_t
        + x_offs_d[None, :] * stride_x_d
        + hq * stride_x_h
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)
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

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)

    out_x_offs = (
        x_offs_t[:, None] * stride_out_x_t
        + x_offs_d[None, :] * stride_out_x_d
        + hq * stride_out_x_h
    )
    tl.store(out_x_ptr + out_x_offs, out_x, mask=x_mask)

    if hq < G:
        y_offs = (
            x_offs_t[:, None] * stride_y_t
            + x_offs_d[None, :] * stride_y_d
            + hq * stride_y_h
        )
        y = tl.load(y_ptr + y_offs, mask=x_mask)
        y_rotated = tl.where(x_rotated_mask, y, -y)
        y_rotated = tl.reshape(y_rotated, (BLOCK_T, 2, BLOCK_D_HALF))
        y_rotated = tl.flip(y_rotated, 2)
        y_rotated = tl.reshape(
            y_rotated,
            (
                BLOCK_T,
                BLOCK_D,
            ),
        )
        y_rotated = tl.flip(y_rotated, 1)

        out_y = y * cos + y_rotated * sin
        out_y = out_y.to(y_ptr.dtype.element_ty)

        out_y_offs = (
            x_offs_t[:, None] * stride_out_y_t
            + x_offs_d[None, :] * stride_out_y_d
            + hq * stride_out_y_h
        )
        tl.store(out_y_ptr + out_y_offs, out_y, mask=x_mask)


@triton.jit
def _rope_fwd_kernel_gptj_cached_thd_position_offsets_2c_gqa_one_head(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    cos_ptr: torch.Tensor,
    sin_ptr: torch.Tensor,
    pos_ptr: torch.Tensor,
    off_ptr: torch.Tensor,
    out_x_ptr: torch.Tensor,
    out_y_ptr: torch.Tensor,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_y_t,
    stride_y_h,
    stride_y_d,
    stride_cos_t,
    stride_cos_d,
    stride_pos_t,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    T,
    reuse_freqs_front_part: tl.constexpr,
    BLOCK_T: tl.constexpr,
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
    HAVE_OFFS: tl.constexpr,
):

    s = tl.program_id(0)
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
    tl.assume(stride_out_t > 0)
    tl.assume(stride_out_h > 0)
    tl.assume(stride_out_d > 0)

    pos_offs = s * BLOCK_T + tl.arange(0, BLOCK_T)
    pos_mask = pos_offs < T
    pos = tl.load(pos_ptr + pos_offs, mask=pos_mask)
    if HAVE_OFFS:
        offset = tl.load(off_ptr + pos_offs, mask=pos_mask)
        cos_offs_t = pos + offset
    else:
        cos_offs_t = pos

    if reuse_freqs_front_part:
        cos_offs_d = tl.arange(0, BLOCK_D) // 2
        cos_mask_d = cos_offs_d < BLOCK_D_HALF
    else:
        cos_offs_d = tl.arange(0, BLOCK_D)
        cos_mask_d = cos_offs_d < BLOCK_D

    cos_mask = (pos_mask)[:, None] & (cos_mask_d)[None, :]
    cos_offs = cos_offs_t[:, None] * stride_cos_t + cos_offs_d[None, :] * stride_cos_d
    cos = tl.load(cos_ptr + cos_offs, mask=cos_mask)
    sin = tl.load(sin_ptr + cos_offs, mask=cos_mask)

    x_offs_t = pos_offs
    x_offs_d = tl.arange(0, BLOCK_D)
    x_mask = (x_offs_t < T)[:, None] & (x_offs_d < BLOCK_D)[None, :]
    x_rotated_mask = (x_offs_d % 2 == 0)[None, :]

    x_offs = (
        x_offs_t[:, None] * stride_x_t
        + x_offs_d[None, :] * stride_x_d
        + hq * stride_x_h
    )
    x = tl.load(x_ptr + x_offs, mask=x_mask)
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

    out_x = x * cos + x_rotated * sin
    out_x = out_x.to(x_ptr.dtype.element_ty)
    tl.store(out_x_ptr + x_offs, out_x, mask=x_mask)

    if hq < G:
        y_offs = (
            x_offs_t[:, None] * stride_y_t
            + x_offs_d[None, :] * stride_y_d
            + hq * stride_y_h
        )
        y = tl.load(y_ptr + y_offs, mask=x_mask)
        y_rotated = tl.where(x_rotated_mask, y, -y)
        y_rotated = tl.reshape(y_rotated, (BLOCK_T, BLOCK_D_HALF, 2))
        y_rotated = tl.flip(y_rotated, 2)
        y_rotated = tl.reshape(
            y_rotated,
            (
                BLOCK_T,
                BLOCK_D,
            ),
        )

        out_y = y * cos + y_rotated * sin
        out_y = out_y.to(y_ptr.dtype.element_ty)
        tl.store(out_y_ptr + y_offs, out_y, mask=x_mask)


@triton.jit
def _rope_fwd_2d_kernel_neox(
    x_ptr: torch.Tensor,
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


# TODO: For now BLOCK_D is assumed to be power of 2. Expand to handle other value of D.
def _rope_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    inplace: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape

    if freqs.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif freqs.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
        if nope_first:
            x_ = x[..., BLOCK_D:]
            out_ = out[..., BLOCK_D:]
            if not inplace:
                out[..., :BLOCK_D] = x[..., :BLOCK_D]
        else:
            x_ = x[..., :BLOCK_D]
            out_ = out[..., :BLOCK_D]
            if not inplace:
                out[..., BLOCK_D:] = x[..., BLOCK_D:]
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2
        x_ = x
        out_ = out

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    _rope_fwd_kernel_sbhd[grid](
        x_,
        freqs,
        out_,
        *x_.stride(),
        *freqs.stride(),
        *out_.stride(),
        s,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_fwd(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_fwd(
        x,
        out,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        False,
        transpose_output,
    )

    return out


def rope_fwd_inplace(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    out = x

    _rope_fwd(
        x,
        out,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        transpose_output,
    )

    return out


def _rope_fwd_thd(
    x: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    inplace: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    b = torch.numel(cu_seqlens) - 1
    t, h, d = x.shape

    if freqs.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif freqs.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
        if nope_first:
            x_ = x[..., BLOCK_D:]
            out_ = out[..., BLOCK_D:]
            if not inplace:
                out[..., :BLOCK_D] = x[..., :BLOCK_D]
        else:
            x_ = x[..., :BLOCK_D]
            out_ = out[..., :BLOCK_D]
            if not inplace:
                out[..., BLOCK_D:] = x[..., BLOCK_D:]
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2
        x_ = x
        out_ = out

    # TODO: performance optimization
    BLOCK_T = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(t, BLOCK_T))

    _rope_fwd_kernel_thd[grid](
        x_,
        cu_seqlens,
        freqs,
        out_,
        *x_.stride(),
        *freqs.stride(),
        *out_.stride(),
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_fwd_thd(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    t, h, d = x.shape
    out = torch.empty((t, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_fwd_thd(
        x,
        out,
        cu_seqlens,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        False,
        transpose_output,
    )

    return out


def rope_fwd_thd_inplace(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    out = x

    _rope_fwd_thd(
        x,
        out,
        cu_seqlens,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        transpose_output,
    )

    return out


# TODO: For now BLOCK_D is assumed to be power of 2. Expand to handle other value of D.
def _rope_cached_fwd(
    x: torch.Tensor,
    out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    inplace: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape

    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    if have_nope:
        BLOCK_D = d // 2
        BLOCK_D_HALF = d // 4
        if nope_first:
            x_ = x[..., BLOCK_D:]
            out_ = out[..., BLOCK_D:]
            if not inplace:
                out[..., :BLOCK_D] = x[..., :BLOCK_D]
        else:
            x_ = x[..., :BLOCK_D]
            out_ = out[..., :BLOCK_D]
            if not inplace:
                out[..., BLOCK_D:] = x[..., BLOCK_D:]
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2
        x_ = x
        out_ = out

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    pos_stride = positions.stride() if positions is not None else (1, 1)
    _rope_fwd_kernel_sbhd_cached[grid](
        x_,
        cos,
        sin,
        positions,
        offsets,
        out_,
        *x_.stride(),
        *cos.stride(),
        *pos_stride,
        *out_.stride(),
        s,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_cached_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_fwd(
        x,
        out,
        cos,
        sin,
        None,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        False,
        transpose_output,
    )

    return out


def rope_cached_fwd_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    out = x

    _rope_cached_fwd(
        x,
        out,
        cos,
        sin,
        None,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        transpose_output,
    )

    return out


def rope_cached_positions_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_fwd(
        x,
        out,
        cos,
        sin,
        positions,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        False,
        transpose_output,
    )

    return out


def rope_cached_positions_fwd_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    out = x

    _rope_cached_fwd(
        x,
        out,
        cos,
        sin,
        positions,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        transpose_output,
    )

    return out


def rope_cached_positions_offsets_fwd(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_fwd(
        x,
        out,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        False,
        transpose_output,
    )

    return out


def rope_cached_positions_offsets_fwd_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    out = x

    _rope_cached_fwd(
        x,
        out,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        transpose_output,
    )

    return out


def _rope_cached_thd_positions_offsets_2c_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    if nope_first:
        raise NotImplementedError(
            "nope style has not been implemented in RoPE Triton backend."
        )

    t, h, d = x.shape
    # TODO: enable nope
    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    BLOCK_D = d
    BLOCK_D_HALF = d // 2

    BLOCK_T = 32
    SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T

    if t >= 8192:
        MIN_NUM_WG = 2048
    elif t >= 1024:
        MIN_NUM_WG = 1024
    else:
        MIN_NUM_WG = 512

    if SPLIT_T < MIN_NUM_WG:
        SPLIT_H_SIZE = h
        SPLIT_H = (triton.next_power_of_2(h) + SPLIT_H_SIZE - 1) // SPLIT_H_SIZE
        while SPLIT_H * SPLIT_T < MIN_NUM_WG and SPLIT_H_SIZE > 1:
            SPLIT_H_SIZE = SPLIT_H_SIZE // 2
            SPLIT_H = (triton.next_power_of_2(h) + SPLIT_H_SIZE - 1) // SPLIT_H_SIZE
    else:
        SPLIT_H_SIZE = h

    SPLIT_H = (triton.next_power_of_2(h) + SPLIT_H_SIZE - 1) // SPLIT_H_SIZE

    grid = (SPLIT_H, SPLIT_T, 1)

    num_warps = 4
    waves_per_eu = 0
    num_stages = 2 if SPLIT_H_SIZE > 1 else 1

    _rope_2c_fwd_kernel_thd_cached[grid](
        x,
        y,
        cos,
        sin,
        positions,
        offsets,
        out_x,
        out_y,
        *x.stride(),
        *cos.stride(),
        *positions.stride(),
        *out_x.stride(),
        t,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        HAVE_POS=(positions is not None),
        HAVE_OFFS=(offsets is not None),
        BLOCK_T=BLOCK_T,
        SPLIT_H_SIZE=SPLIT_H_SIZE,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        num_stages=num_stages,
    )

    return out_x, out_y


def rope_cached_thd_positions_2c_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    t, h, d = x.shape
    out_x = torch.empty((t, h, d), dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty((t, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_offsets_2c_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_2c_fwd_inplace(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    out_x = x
    out_y = y

    _rope_cached_thd_positions_offsets_2c_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_offsets_2c_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    t, h, d = x.shape
    out_x = torch.empty((t, h, d), dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty((t, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_offsets_2c_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_offsets_2c_fwd_inplace(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    out_x = x
    out_y = y

    _rope_cached_thd_positions_offsets_2c_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def _rope_cached_thd_positions_offsets_2c_gqa_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    out_x: torch.Tensor,
    out_y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    if nope_first:
        raise NotImplementedError(
            "nope style has not been implemented in RoPE Triton backend."
        )

    t, hx, d = x.shape
    _, hy, _ = y.shape

    if hx == hy:
        raise ValueError(
            "Q heads is the same as K heads, please use the following API for MHA implementation:\n\trope_cached_thd_positions_offsets_2c_fwd_inplace,\n\trope_cached_thd_positions_offsets_2c_fwd,\n\trope_cached_thd_positions_2c_fwd_inplace,\n\trope_cached_thd_positions_2c_fwd"
        )

    if hx % hy != 0:
        raise ValueError("Q heads should be divisible by K heads.")

    if cos.shape[-1] == d // 2:
        if reuse_freqs_front_part:
            have_nope = False
        else:
            have_nope = True
    elif cos.shape[-1] == d // 4:
        have_nope = True
    else:
        have_nope = False

    BLOCK_D = d
    BLOCK_D_HALF = d // 2

    BLOCK_T = min(max(triton.next_power_of_2(t), 16), 32)
    SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
    grid = (SPLIT_T, hx, 1)
    num_warps = 4
    waves_per_eu = 0

    if rotate_style == RotateStyle.GPTJ:
        if have_nope:
            # TODO: add a new kernel for nope
            raise NotImplementedError(
                "nope style has not been implemented in RoPE Triton backend."
            )
        else:
            if t >= 1024:
                BLOCK_T = 32
                SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
                G = hy
                QH_per_G = hx // hy
                grid = (G, SPLIT_T, 1)
                num_warps = 4
                waves_per_eu = 0
                num_stages = 2 if QH_per_G > 1 else 1

                _rope_fwd_kernel_gptj_cached_thd_position_offsets_2c_gqa[grid](
                    x,
                    y,
                    cos,
                    sin,
                    positions,
                    offsets,
                    out_x,
                    out_y,
                    *x.stride(),
                    *y.stride(),
                    *cos.stride(),
                    *positions.stride(),
                    *out_x.stride(),
                    T=t,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    BLOCK_T=BLOCK_T,
                    QH_per_G=QH_per_G,
                    BLOCK_D=BLOCK_D,
                    BLOCK_D_HALF=BLOCK_D_HALF,
                    HAVE_OFFS=(offsets is not None),
                    num_warps=num_warps,
                    waves_per_eu=waves_per_eu,
                    num_stages=num_stages,
                )
            else:
                _rope_fwd_kernel_gptj_cached_thd_position_offsets_2c_gqa_one_head[grid](
                    x,
                    y,
                    cos,
                    sin,
                    positions,
                    offsets,
                    out_x,
                    out_y,
                    *x.stride(),
                    *y.stride(),
                    *cos.stride(),
                    *positions.stride(),
                    *out_x.stride(),
                    T=t,
                    reuse_freqs_front_part=reuse_freqs_front_part,
                    BLOCK_T=BLOCK_T,
                    G=hy,
                    BLOCK_D=BLOCK_D,
                    BLOCK_D_HALF=BLOCK_D_HALF,
                    HAVE_OFFS=(offsets is not None),
                    num_warps=num_warps,
                    waves_per_eu=waves_per_eu,
                )

    elif rotate_style == RotateStyle.NEOX:
        if have_nope:
            # TODO: add a new kernel for nope
            raise NotImplementedError(
                "nope style has not been implemented in RoPE Triton backend."
            )
        else:
            # TODO check boundary
            # BLOCK_T = 32
            # SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
            # G = hy
            # QH_per_G = hx // hy
            # grid = (G, SPLIT_T, 1)
            # num_warps = 4
            # waves_per_eu = 0
            # num_stages = 2 if QH_per_G > 1 else 1
            # _rope_fwd_kernel_neox_cached_thd_position_offsets_2c_gqa[grid](
            #         x,
            #         y,
            #         cos,
            #         sin,
            #         positions,
            #         offsets,
            #         out_x,
            #         out_y,
            #         *x.stride(),
            #         *y.stride(),
            #         *cos.stride(),
            #         *positions.stride(),
            #         *out_x.stride(),
            #         T=t,
            #         reuse_freqs_front_part=reuse_freqs_front_part,
            #         BLOCK_T=BLOCK_T,
            #         QH_per_G=QH_per_G,
            #         BLOCK_D=BLOCK_D,
            #         BLOCK_D_HALF=BLOCK_D_HALF,
            #         HAVE_OFFS=(offsets is not None),
            #         num_warps=num_warps,
            #         waves_per_eu=waves_per_eu,
            #         num_stages=num_stages
            #     )

            _rope_fwd_kernel_neox_cached_thd_position_offsets_2c_gqa_one_head[grid](
                x,
                y,
                cos,
                sin,
                positions,
                offsets,
                out_x,
                out_y,
                *x.stride(),
                *y.stride(),
                *cos.stride(),
                *positions.stride(),
                *out_x.stride(),
                *out_y.stride(),
                T=t,
                reuse_freqs_front_part=reuse_freqs_front_part,
                BLOCK_T=BLOCK_T,
                G=hy,
                BLOCK_D=BLOCK_D,
                BLOCK_D_HALF=BLOCK_D_HALF,
                HAVE_OFFS=(offsets is not None),
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )

    return out_x, out_y


def rope_cached_thd_positions_2c_gqa_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    t, hx, d = x.shape
    _, hy, _ = y.shape
    out_x = torch.empty((t, hx, d), dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty((t, hy, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_offsets_2c_gqa_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_2c_gqa_fwd_inplace(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    out_x = x
    out_y = y

    _rope_cached_thd_positions_offsets_2c_gqa_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        None,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_offsets_2c_gqa_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    t, hx, d = x.shape
    _, hy, _ = y.shape
    out_x = torch.empty((t, hx, d), dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty((t, hy, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_offsets_2c_gqa_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_offsets_2c_gqa_fwd_inplace(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    out_x = x
    out_y = y

    _rope_cached_thd_positions_offsets_2c_gqa_fwd(
        x,
        y,
        out_x,
        out_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out_x, out_y


def _rope_fwd_2d(
    x: torch.Tensor,
    out: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: torch.Tensor,
    img_width: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    b, wh, h, d = x.shape
    # out = torch.empty((b,wh,h,d), dtype=x.dtype, device=x.device, requires_grad=False)

    grid = (b, h, 1)
    _rope_fwd_2d_kernel_neox[grid](
        x,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        out,
        *x.stride(),
        *cos_h.stride(),
        *cos_w.stride(),
        wh,
        img_height,
        img_width,
        BLOCK_D=d,
    )

    return out


def rope_fwd_2d(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: torch.Tensor,
    img_width: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    b, wh, h, d = x.shape
    out = torch.empty(
        (b, wh, h, d), dtype=x.dtype, device=x.device, requires_grad=False
    )

    # grid = (b,h,1)
    # _rope_fwd_2d_kernel_neox[grid](x, cos_h, sin_h, cos_w, sin_w, out, *x.stride(), *cos_h.stride(), *cos_w.stride(), wh, img_height, img_width, BLOCK_D=d)

    _rope_fwd_2d(
        x,
        out,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        img_height,
        img_width,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out


def rope_fwd_2d_inplace(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
    img_height: torch.Tensor,
    img_width: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
):
    out = x
    _rope_fwd_2d(
        x,
        out,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        img_height,
        img_width,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    return out


class RoPE(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        freqs: torch.Tensor,
        rotate_style: int,
        reuse_freqs_front_part: bool,
        nope_first: bool,
        transpose_output: bool = False,
    ) -> torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(freqs)
        return rope_fwd(
            x, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output
        )


class RoPETHD(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        freqs: torch.Tensor,
        rotate_style: int,
        reuse_freqs_front_part: bool,
        nope_first: bool,
    ):
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.save_for_backward(cu_seqlens, freqs)
        return rope_fwd_thd(
            x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
        )


class RoPECached(autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        rotate_style: int,
        reuse_freqs_front_part: bool,
        nope_first: bool,
        transpose_output: bool = False,
    ) -> torch.Tensor:
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.transpose_output = transpose_output
        ctx.save_for_backward(cos, sin)
        return rope_cached_fwd(
            x,
            cos,
            sin,
            rotate_style,
            reuse_freqs_front_part,
            nope_first,
            transpose_output,
        )


class RoPE2D(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos_height: torch.Tensor,
        sin_height: torch.Tensor,
        cos_width: torch.Tensor,
        sin_width: torch.Tensor,
        img_height: int,
        img_width: int,
        rotate_style: int,
        reuse_freqs_front_part: bool,
        nope_first: bool,
    ) -> torch.Tensor:
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.rotate_style = rotate_style
        ctx.reuse_freqs_front_part = reuse_freqs_front_part
        ctx.nope_first = nope_first
        ctx.save_for_backward(cos_height, sin_height, cos_width, sin_width)
        return rope_fwd_2d(
            x,
            cos_height,
            sin_height,
            cos_width,
            sin_width,
            img_height,
            img_width,
            rotate_style,
            reuse_freqs_front_part,
            nope_first,
        )
