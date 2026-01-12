# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from torch import autograd
from enum import IntEnum
from typing import Tuple, Union
from aiter.ops.triton._triton_kernels.rope import (
    _get_neox_rotated_x_1D,
    _get_gptj_rotated_x_1D,
    _get_neox_rotated_x,
    _get_gptj_rotated_x,
    _rope_kernel_sbhd_fwd,
    _rope_kernel_sbhd_bwd,
    _rope_kernel_thd_fwd,
    _rope_kernel_thd_bwd,
    _rope_kernel_sbhd_cached_fwd,
    _rope_kernel_sbhd_cached_bwd,
    _rope_kernel_thd_cached_2c_fwd,
    _rope_kernel_thd_cached_2c_bwd,
    _rope_kernel_cached_thd_2c_gqa_fwd,
    _rope_kernel_cached_thd_2c_gqa_onehead_fwd,
    _rope_kernel_cached_thd_2c_gqa_bwd,
    _rope_kernel_cached_thd_2c_gqa_onehead_bwd,
    _rope_fwd_2d_kernel_neox,
    _rope_fwd_3d,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


class RotateStyle(IntEnum):
    NEOX = (0,)
    GPTJ = 1


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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    _rope_kernel_sbhd_fwd[grid](
        x,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        s,
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
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


def _rope_bwd(
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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    _rope_kernel_sbhd_bwd[grid](
        x,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        s,
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_bwd(
    x: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    s, b, h, d = x.shape
    out = torch.empty((s, b, h, d), dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_bwd(
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


def _rope_thd_fwd(
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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_T = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(t, BLOCK_T))

    _rope_kernel_thd_fwd[grid](
        x,
        cu_seqlens,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_thd_fwd(
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

    _rope_thd_fwd(
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


def rope_thd_fwd_inplace(
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    nope_first: bool,
    transpose_output: bool = False,
) -> torch.Tensor:
    out = x

    _rope_thd_fwd(
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


def _rope_thd_bwd(
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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_T = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(t, BLOCK_T))

    _rope_kernel_thd_bwd[grid](
        x,
        cu_seqlens,
        freqs,
        out,
        *x.stride(),
        *freqs.stride(),
        *out.stride(),
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=(rotate_style == RotateStyle.NEOX),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_D_HALF=BLOCK_D_HALF,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    return out


def rope_thd_bwd(
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

    _rope_thd_bwd(
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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    pos_stride = positions.stride() if positions is not None else (1, 1)
    _rope_kernel_sbhd_cached_fwd[grid](
        x,
        cos,
        sin,
        positions,
        offsets,
        out,
        *x.stride(),
        *cos.stride(),
        *pos_stride,
        *out.stride(),
        s,
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
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


def _rope_cached_bwd(
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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    # TODO: performance optimization
    BLOCK_S = 32
    num_warps = 4
    waves_per_eu = 0
    grid = (b, h, triton.cdiv(s, BLOCK_S))

    pos_stride = positions.stride() if positions is not None else (1, 1)
    _rope_kernel_sbhd_cached_bwd[grid](
        x,
        cos,
        sin,
        positions,
        offsets,
        out,
        *x.stride(),
        *cos.stride(),
        *pos_stride,
        *out.stride(),
        s,
        HAVE_NOPE=have_nope,
        NOPE_FIRST=nope_first,
        INPLACE=inplace,
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


def rope_cached_bwd(
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

    _rope_cached_bwd(
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


def rope_cached_positions_bwd(
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

    _rope_cached_bwd(
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


def rope_cached_positions_offsets_bwd(
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

    _rope_cached_bwd(
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


def _rope_cached_thd_2c_fwd(
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
    inplace: bool,
    transpose_output: bool = False,
):
    t, h, d = x.shape
    ty, kh, dy = y.shape

    assert (
        t == ty
    ), f"The number of tokens should be the same for the two inputs, but got {t} and {ty}"
    assert (
        d == dy
    ), f"The head dimension should be the same for the two inputs, but got {d} and {dy}"
    assert h % kh == 0, f"QH should be multiple of KH, but got QH={h} and KH={kh}"

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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    if h == kh:
        BLOCK_T = 32
        SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T

        if t >= 8192:
            MIN_NUM_WG = 4096
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

        _rope_kernel_thd_cached_2c_fwd[grid](
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
            t,
            HAVE_NOPE=have_nope,
            NOPE_FIRST=nope_first,
            INPLACE=inplace,
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
    else:
        # TODO check boundary
        if rotate_style == RotateStyle.GPTJ and t >= 1024:
            BLOCK_T = 32
            SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
            QH_per_G = h // kh
            grid = (kh, SPLIT_T, 1)
            num_warps = 4
            waves_per_eu = 0
            num_stages = 2 if QH_per_G > 1 else 1

            _rope_kernel_cached_thd_2c_gqa_fwd[grid](
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
                t,
                HAVE_NOPE=have_nope,
                NOPE_FIRST=nope_first,
                INPLACE=inplace,
                REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
                IS_NEOX=(rotate_style == RotateStyle.NEOX),
                HAVE_POS=(positions is not None),
                HAVE_OFFS=(offsets is not None),
                BLOCK_T=BLOCK_T,
                QH_per_G=QH_per_G,
                BLOCK_D=BLOCK_D,
                BLOCK_D_HALF=BLOCK_D_HALF,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
                num_stages=num_stages,
            )
        else:
            BLOCK_T = min(max(triton.next_power_of_2(t), 16), 32)
            SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
            grid = (SPLIT_T, h, 1)
            num_warps = 4
            waves_per_eu = 0
            _rope_kernel_cached_thd_2c_gqa_onehead_fwd[grid](
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
                t,
                HAVE_NOPE=have_nope,
                NOPE_FIRST=nope_first,
                INPLACE=inplace,
                REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
                IS_NEOX=(rotate_style == RotateStyle.NEOX),
                HAVE_POS=(positions is not None),
                HAVE_OFFS=(offsets is not None),
                BLOCK_T=BLOCK_T,
                G=kh,
                BLOCK_D=BLOCK_D,
                BLOCK_D_HALF=BLOCK_D_HALF,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
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
    out_x = torch.empty(*x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty(*y.shape, dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_2c_fwd(
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
        False,
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

    _rope_cached_thd_2c_fwd(
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
        True,
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
    out_x = torch.empty(*x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty(*y.shape, dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_2c_fwd(
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
        False,
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

    _rope_cached_thd_2c_fwd(
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
        True,
        transpose_output,
    )

    return out_x, out_y


def _rope_cached_thd_positions_offsets_2c_bwd(
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
    inplace: bool,
    transpose_output: bool = False,
):
    t, h, d = x.shape
    ty, kh, dy = y.shape

    assert (
        t == ty
    ), f"The number of tokens should be the same for the two inputs, but got {t} and {ty}"
    assert (
        d == dy
    ), f"The head dimension should be the same for the two inputs, but got {d} and {dy}"
    assert h % kh == 0, f"QH should be multiple of KH, but got QH={h} and KH={kh}"

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
    else:
        BLOCK_D = d
        BLOCK_D_HALF = d // 2

    if h == kh:
        BLOCK_T = 32
        SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T

        if t >= 8192:
            MIN_NUM_WG = 4096
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

        _rope_kernel_thd_cached_2c_bwd[grid](
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
            t,
            HAVE_NOPE=have_nope,
            NOPE_FIRST=nope_first,
            INPLACE=inplace,
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
    else:
        # TODO check boundary
        if rotate_style == RotateStyle.GPTJ and t >= 1024:
            BLOCK_T = 32
            SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
            QH_per_G = h // kh
            grid = (kh, SPLIT_T, 1)
            num_warps = 4
            waves_per_eu = 0
            num_stages = 2 if QH_per_G > 1 else 1

            _rope_kernel_cached_thd_2c_gqa_bwd[grid](
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
                t,
                HAVE_NOPE=have_nope,
                NOPE_FIRST=nope_first,
                INPLACE=inplace,
                REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
                IS_NEOX=(rotate_style == RotateStyle.NEOX),
                HAVE_POS=(positions is not None),
                HAVE_OFFS=(offsets is not None),
                BLOCK_T=BLOCK_T,
                QH_per_G=QH_per_G,
                BLOCK_D=BLOCK_D,
                BLOCK_D_HALF=BLOCK_D_HALF,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
                num_stages=num_stages,
            )
        else:
            BLOCK_T = min(max(triton.next_power_of_2(t), 16), 32)
            SPLIT_T = (triton.next_power_of_2(t) + BLOCK_T - 1) // BLOCK_T
            grid = (SPLIT_T, h, 1)
            num_warps = 4
            waves_per_eu = 0
            _rope_kernel_cached_thd_2c_gqa_onehead_bwd[grid](
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
                t,
                HAVE_NOPE=have_nope,
                NOPE_FIRST=nope_first,
                INPLACE=inplace,
                REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
                IS_NEOX=(rotate_style == RotateStyle.NEOX),
                HAVE_POS=(positions is not None),
                HAVE_OFFS=(offsets is not None),
                BLOCK_T=BLOCK_T,
                G=kh,
                BLOCK_D=BLOCK_D,
                BLOCK_D_HALF=BLOCK_D_HALF,
                num_warps=num_warps,
                waves_per_eu=waves_per_eu,
            )

    return out_x, out_y


def rope_cached_thd_positions_2c_bwd(
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
    out_x = torch.empty(*x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty(*y.shape, dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_offsets_2c_bwd(
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
        False,
        transpose_output,
    )

    return out_x, out_y


def rope_cached_thd_positions_offsets_2c_bwd(
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
    out_x = torch.empty(*x.shape, dtype=x.dtype, device=x.device, requires_grad=False)
    out_y = torch.empty(*y.shape, dtype=x.dtype, device=x.device, requires_grad=False)

    _rope_cached_thd_positions_offsets_2c_bwd(
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
        False,
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


def rope_fwd_3d(
    x,
    grid_sizes: tl.constexpr,
    freqs: tl.constexpr,
    sp_size: tl.constexpr,
    sp_rank: tl.constexpr,
):
    B, s, n_heads, C = x.shape
    c_total = C // 2  # 64
    c1 = c_total - 2 * (c_total // 3)  # 22
    c2 = c_total // 3  # 21
    c3 = c_total // 3  # 21
    device = x.device

    grid_sizes = grid_sizes.to(device=device, dtype=torch.int32).contiguous()

    freqs_real = freqs.real.to(dtype=torch.float32, device=device).contiguous()
    freqs_imag = freqs.imag.to(dtype=torch.float32, device=device).contiguous()
    out = torch.empty_like(x, dtype=torch.float32, device=device)

    BLOCK_L, BLOCK_N, BLOCK_C = 32, 4, 64

    grid = (B, n_heads, triton.cdiv(s, BLOCK_L))

    num_warps = 4
    waves_per_eu = 1

    _rope_fwd_3d[grid](
        x,
        freqs_real,
        freqs_imag,
        grid_sizes,
        out,
        *x.stride(),
        freqs_real.stride(0),
        freqs_real.stride(1),
        *grid_sizes.stride(),
        *out.stride(),
        s,
        n_heads,
        C,
        c_total,
        sp_size,
        sp_rank,
        freqs.shape[0],
        s,
        1.0,
        0.0,
        BLOCK_L=BLOCK_L,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        C1=c1,
        C2=c2,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
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

    @staticmethod
    def backward(
        ctx, output_grads: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        (freqs,) = ctx.saved_tensors
        return (
            rope_bwd(
                output_grads,
                freqs,
                ctx.rotate_style,
                ctx.reuse_freqs_front_part,
                ctx.nope_first,
                ctx.transpose_output,
            ),
            None,
            None,
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
        return rope_thd_fwd(
            x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
        )

    @staticmethod
    def backward(ctx, output_grads) -> Tuple[Union[torch.Tensor, None], ...]:
        cu_seqlens, freqs = ctx.saved_tensors
        return (
            rope_thd_bwd(
                output_grads,
                cu_seqlens,
                freqs,
                ctx.rotate_style,
                ctx.reuse_freqs_front_part,
                ctx.nope_first,
            ),
            None,
            None,
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

    @staticmethod
    def backward(ctx, output_grads) -> Tuple[Union[torch.Tensor, None], ...]:
        cos, sin = ctx.saved_tensors
        return (
            rope_cached_bwd(
                output_grads,
                cos,
                sin,
                ctx.rotate_style,
                ctx.reuse_freqs_front_part,
                ctx.nope_first,
                ctx.transpose_output,
            ),
            None,
            None,
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
