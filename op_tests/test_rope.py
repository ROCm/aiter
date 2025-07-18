# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, perftest
import itertools
from enum import IntEnum
import argparse
from aiter import dtypes


@perftest()
def hip_rope_fwd(
    input, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output
):
    return aiter.rope_fwd(
        input, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output
    )


@perftest()
def hip_rope_bwd(
    output_grads,
    freqs,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_bwd(
        output_grads,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_2c_fwd(
    input_x,
    input_y,
    freqs,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_2c_fwd(
        input_x,
        input_y,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_2c_bwd(
    output_grads_x,
    output_grads_y,
    freqs,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_2c_bwd(
        output_grads_x,
        output_grads_y,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_fwd(
    input, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output
):
    return aiter.rope_cached_fwd(
        input,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_bwd(
    output_grads,
    cos,
    sin,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_bwd(
        output_grads,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_2c_fwd(
    input_x,
    input_y,
    cos,
    sin,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_2c_fwd(
        input_x,
        input_y,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_2c_bwd(
    output_grads_x,
    output_grads_y,
    cos,
    sin,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_2c_bwd(
        output_grads_x,
        output_grads_y,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_positions_fwd(
    input,
    cos,
    sin,
    positions,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_positions_fwd(
        input,
        cos,
        sin,
        positions,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_positions_offsets_fwd(
    input,
    cos,
    sin,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_positions_offsets_fwd(
        input,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_positions_fwd_inplace(
    input, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first
):
    return aiter.rope_cached_positions_fwd_inplace(
        input, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first
    )


@perftest()
def hip_rope_cached_positions_offsets_fwd_inplace(
    input,
    cos,
    sin,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
):
    return aiter.rope_cached_positions_offsets_fwd_inplace(
        input,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )


@perftest()
def hip_rope_cached_positions_2d_fwd(
    input_x,
    input_y,
    cos,
    sin,
    positions,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_positions_2c_fwd(
        input_x,
        input_y,
        cos,
        sin,
        positions,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_positions_offsets_2d_fwd(
    input_x,
    input_y,
    cos,
    sin,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    return aiter.rope_cached_positions_offsets_2c_fwd(
        input_x,
        input_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )


@perftest()
def hip_rope_cached_positions_2d_fwd_inplace(
    input_x,
    input_y,
    cos,
    sin,
    positions,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
):
    return aiter.rope_cached_positions_2c_fwd_inplace(
        input_x,
        input_y,
        cos,
        sin,
        positions,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )


@perftest()
def hip_rope_cached_positions_offsets_2d_fwd_inplace(
    input_x,
    input_y,
    cos,
    sin,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
):
    return aiter.rope_cached_positions_offsets_2c_fwd_inplace(
        input_x,
        input_y,
        cos,
        sin,
        positions,
        offsets,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )


@perftest()
def hip_rope_thd_fwd(
    input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
):
    return aiter.rope_thd_fwd(
        input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
    )


@perftest()
def hip_rope_thd_bwd(
    output_grads, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
):
    return aiter.rope_thd_bwd(
        output_grads,
        cu_seqlens,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )


@perftest()
def hip_rope_2d_fwd(
    input,
    height,
    width,
    cos_h,
    sin_h,
    cos_w,
    sin_w,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
):
    return aiter.rope_2d_fwd(
        input,
        height,
        width,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )


@perftest()
def hip_rope_2d_bwd(
    output_grads,
    height,
    width,
    cos_h,
    sin_h,
    cos_w,
    sin_w,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
):
    return aiter.rope_2d_bwd(
        output_grads,
        height,
        width,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
    )


@perftest()
def legacy_rope_cached_positions_2d_fwd(
    input_x, input_y, cos, sin, positions, rotate_style, nope_first
):
    aiter.rotary_embedding_fwd(
        positions,
        input_x,
        input_y,
        d,
        cos,
        sin,
        rotate_style is RotateStyle.NEOX,
        nope_first,
    )
    return input_x, input_y


@perftest()
def legacy_rope_cached_positions_offsets_2d_fwd(
    input_x, input_y, cos, sin, positions, offsets, rotate_style, nope_first
):
    rotate_dim = sin.size(-1) * 2
    aiter.batched_rotary_embedding(
        positions,
        input_x,
        input_y,
        d,
        cos,
        sin,
        rotate_style is RotateStyle.NEOX,
        nope_first,
        rotate_dim,
        offsets.view(-1),
    )
    return input_x, input_y


class RotateStyle(IntEnum):
    NEOX = (0,)
    GPTJ = 1


def rotate_half_neox(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_gptj(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def ref_rope_sbhd_fwd(
    x_,
    freqs_,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    simulate_cached=False,
    comp_with_fp32=False,
):
    x = x_.to(dtype=torch.float32) if comp_with_fp32 else x_
    freqs = freqs_.to(dtype=torch.float32) if comp_with_fp32 else freqs_
    rotate_half = (
        rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    )
    rotate_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    if nope_first:
        d = x.shape[-1]
        x, x_forward = x[..., d - rotate_dim :], x[..., : d - rotate_dim]
    else:
        x, x_forward = x[..., :rotate_dim], x[..., rotate_dim:]
    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.NEOX:
            freqs = freqs.repeat([1] * (freqs.dim() - 1) + [2])
        elif rotate_style == RotateStyle.GPTJ:
            freqs = freqs.repeat_interleave(2, dim=-1)
    cos = (
        torch.cos(freqs).to(dtype=freqs_.dtype).to(dtype=torch.float32)
        if simulate_cached and comp_with_fp32
        else torch.cos(freqs)
    )
    sin = (
        torch.sin(freqs).to(dtype=freqs_.dtype).to(dtype=torch.float32)
        if simulate_cached and comp_with_fp32
        else torch.sin(freqs)
    )
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return (
        torch.cat((x_forward, x_embed.to(dtype=x.dtype)), dim=-1).to(dtype=x_.dtype)
        if nope_first
        else torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1).to(
            dtype=x_.dtype
        )
    )


def ref_rope_thd_fwd(
    x,
    cu_seqlens,
    freqs,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    simulate_cached=False,
    comp_with_fp32=False,
):
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    x_embed = torch.cat(
        [
            ref_rope_sbhd_fwd(
                xi.unsqueeze(1),
                freqs[: xi.size(0)],
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                simulate_cached,
                comp_with_fp32,
            )
            for xi in torch.split(x, seqlens)
        ]
    )
    return x_embed.squeeze(1)


def ref_rope_2d_fwd(x_, size_h, size_w, cos_h_, sin_h_, cos_w_, sin_w_, rotate_style):
    x = x_.to(dtype=torch.float32)
    cos_h = cos_h_.to(dtype=torch.float32)
    sin_h = sin_h_.to(dtype=torch.float32)
    cos_w = cos_w_.to(dtype=torch.float32)
    sin_w = sin_w_.to(dtype=torch.float32)
    rotate_half = (
        rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    )
    s, b, h, d = x.shape
    x = x.view(s, size_h, size_w, h, d)
    x1, x2 = x.chunk(2, dim=-1)
    cos_h = cos_h[:, :size_h].unsqueeze(2)  # [1, H, 1, 1, D//2]
    sin_h = sin_h[:, :size_h].unsqueeze(2)  # [1, H, 1, 1, D//2]
    x1 = (x1 * cos_h) + (rotate_half(x1) * sin_h)
    cos_w = cos_w[:, :size_w].unsqueeze(1)  # [1, 1, W, 1, D//2]
    sin_w = sin_w[:, :size_w].unsqueeze(1)  # [1, 1, W, 1, D//2]
    x2 = (x2 * cos_w) + (rotate_half(x2) * sin_w)
    return torch.cat([x1, x2], dim=-1).view(s, b, h, d).to(dtype=x_.dtype)


def test_rope_sbhd(
    input,
    freqs,
    grad,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
nope_first: {nope_first}, \
transpose_output: {transpose_output}
"""

    input_cached = input.clone().detach().requires_grad_(True)

    ref = ref_rope_sbhd_fwd(
        input, freqs, rotate_style, reuse_freqs_front_part, nope_first, False, True
    )
    ref_cached = ref_rope_sbhd_fwd(
        input_cached,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        True,
    )
    ref.backward(grad)
    ref_cached.backward(grad)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    hip_fwd, hip_fwd_avg = hip_rope_fwd(
        input, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output
    )
    hip_bwd, hip_bwd_avg = hip_rope_bwd(
        grad, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output
    )
    hip_cached_fwd, hip_cached_fwd_avg = hip_rope_cached_fwd(
        input_cached,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )
    hip_cached_bwd, hip_cached_bwd_avg = hip_rope_cached_bwd(
        grad,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    checkAllclose(
        ref,
        hip_fwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input.grad,
        hip_bwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        ref_cached,
        hip_cached_fwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_fwd - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input_cached.grad,
        hip_cached_bwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_bwd - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n",
    )


def test_rope_sbhd_2c(
    input_x,
    input_y,
    freqs,
    grad_x,
    grad_y,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    assert (
        input_x.shape[0:2] == input_y.shape[0:2]
        and input_x.shape[3] == input_y.shape[3]
    )
    assert input_x.dtype == input_y.dtype

    input_msg = f"""
dtype: {input_x.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input_x.shape):<20} - {str(input_y.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
nope_first: {nope_first}, \
transpose_output: {transpose_output}
"""

    input_x_cached = input_x.clone().detach().requires_grad_(True)
    input_y_cached = input_y.clone().detach().requires_grad_(True)

    ref_x = ref_rope_sbhd_fwd(
        input_x, freqs, rotate_style, reuse_freqs_front_part, nope_first, False, True
    )
    ref_y = ref_rope_sbhd_fwd(
        input_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, False, True
    )
    ref_x_cached = ref_rope_sbhd_fwd(
        input_x_cached,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        True,
    )
    ref_y_cached = ref_rope_sbhd_fwd(
        input_y_cached,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        True,
    )
    ref_x.backward(grad_x)
    ref_y.backward(grad_y)
    ref_x_cached.backward(grad_x)
    ref_y_cached.backward(grad_y)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    (hip_fwd_x, hip_fwd_y), hip_fwd_avg = hip_rope_2c_fwd(
        input_x,
        input_y,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )
    (hip_bwd_x, hip_bwd_y), hip_bwd_avg = hip_rope_2c_bwd(
        grad_x,
        grad_y,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )
    (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = hip_rope_cached_2c_fwd(
        input_x_cached,
        input_y_cached,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )
    (hip_cached_bwd_x, hip_cached_bwd_y), hip_cached_bwd_avg = hip_rope_cached_2c_bwd(
        grad_x,
        grad_y,
        cos,
        sin,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        transpose_output,
    )

    checkAllclose(
        ref_x,
        hip_fwd_x,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_2c_fwd_x - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        ref_y,
        hip_fwd_y,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_2c_fwd_y - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input_x.grad,
        hip_bwd_x,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_2c_bwd_x - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input_y.grad,
        hip_bwd_y,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_2c_bwd_y - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        ref_x_cached,
        hip_cached_fwd_x,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_2c_fwd_x - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        ref_y_cached,
        hip_cached_fwd_y,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_2c_fwd_y - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input_x_cached.grad,
        hip_cached_bwd_x,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_2c_bwd_x - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input_y_cached.grad,
        hip_cached_bwd_y,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_2c_bwd_y - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n",
    )


def test_rope_sbhd_1c_positions(
    input,
    freqs,
    grad,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
dim_positions: {str(positions.shape):<20}, \
dim_offsets: {str(offsets.shape) if offsets is not None else 'None'}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
nope_first: {nope_first}, \
transpose_output: {transpose_output}
"""

    ref = ref_rope_sbhd_fwd(
        input,
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        ),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        True,
    )

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    if offsets is None:
        (hip_cached_fwd), hip_cached_fwd_avg = hip_rope_cached_positions_fwd(
            input,
            cos,
            sin,
            positions,
            rotate_style,
            reuse_freqs_front_part,
            nope_first,
            transpose_output,
        )
    else:
        (hip_cached_fwd), hip_cached_fwd_avg = hip_rope_cached_positions_offsets_fwd(
            input,
            cos,
            sin,
            positions,
            offsets,
            rotate_style,
            reuse_freqs_front_part,
            nope_first,
            transpose_output,
        )

    checkAllclose(
        ref,
        hip_cached_fwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_position_fwd - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n",
    )


def test_rope_sbhd_2c_positions(
    input_x,
    input_y,
    freqs,
    grad_x,
    grad_y,
    positions,
    offsets,
    rotate_style,
    reuse_freqs_front_part,
    nope_first,
    transpose_output,
):
    input_msg = f"""
dtype: {input_x.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input_x.shape):<20} - {str(input_y.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
dim_positions: {str(positions.shape):<20}, \
dim_offsets: {str(offsets.shape) if offsets is not None else 'None'}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
nope_first: {nope_first}, \
transpose_output: {transpose_output}
"""

    ref_x = ref_rope_sbhd_fwd(
        input_x,
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        ),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        True,
    )
    ref_y = ref_rope_sbhd_fwd(
        input_y,
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        ),
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        True,
        True,
    )

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    if offsets is None:
        (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = (
            hip_rope_cached_positions_2d_fwd(
                input_x,
                input_y,
                cos,
                sin,
                positions,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output,
            )
        )
    else:
        (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = (
            hip_rope_cached_positions_offsets_2d_fwd(
                input_x,
                input_y,
                cos,
                sin,
                positions,
                offsets,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output,
            )
        )

    checkAllclose(
        ref_x,
        hip_cached_fwd_x,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_position_2d_fwd_x - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        ref_y,
        hip_cached_fwd_y,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_cached_position_2d_fwd_y - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n",
    )


def compare_rope_sbhd_2c_positions_with_legacy(
    input_x,
    input_y,
    freqs,
    positions,
    offsets,
    rotate_style,
    nope_first,
    check_correction=False,
):
    input_msg = f"""dtype: {input_x.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input_x.shape):<20} - {str(input_y.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
dim_positions: {str(positions.shape):<20}, \
dim_offsets: {str(offsets.shape) if offsets is not None else 'None'}, \
rotate_style: {rotate_style.value}, \
nope_first: {nope_first}
"""

    s, b, h_x, d = input_x.shape

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # perftest cannot test correction of inplace operators
    if check_correction:
        ref_x = ref_rope_sbhd_fwd(
            input_x,
            freqs[
                positions if offsets is None else torch.add(positions, offsets)
            ].squeeze(-2),
            rotate_style,
            True,
            nope_first,
            True,
            True,
        )
        ref_y = ref_rope_sbhd_fwd(
            input_y,
            freqs[
                positions if offsets is None else torch.add(positions, offsets)
            ].squeeze(-2),
            rotate_style,
            True,
            nope_first,
            True,
            True,
        )
        h_y = input_y.shape[2]
        hip_input_x, hip_input_y = input_x, input_y
        leg_input_x, leg_input_y = input_x.clone().view(
            s * b, -1
        ), input_y.clone().view(s * b, -1)
        if offsets is None:
            aiter.rope_cached_positions_2c_fwd_inplace(
                hip_input_x,
                hip_input_y,
                cos,
                sin,
                positions,
                rotate_style,
                True,
                nope_first,
            )
            aiter.rotary_embedding_fwd(
                positions,
                leg_input_x,
                leg_input_y,
                d,
                cos,
                sin,
                rotate_style is RotateStyle.NEOX,
                nope_first,
            )
        else:
            aiter.rope_cached_positions_offsets_2c_fwd_inplace(
                hip_input_x,
                hip_input_y,
                cos,
                sin,
                positions,
                offsets,
                rotate_style,
                True,
                nope_first,
            )
            aiter.batched_rotary_embedding(
                positions,
                leg_input_x,
                leg_input_y,
                d,
                cos,
                sin,
                rotate_style is RotateStyle.NEOX,
                nope_first,
                cos.size(-1) * 2,
                offsets.view(-1),
            )

        checkAllclose(
            ref_x,
            hip_input_x,
            rtol=1e-3,
            atol=1e-3,
            msg=f"correction: hip_fwd_x - {input_msg}\n",
        )
        checkAllclose(
            ref_y,
            hip_input_y,
            rtol=1e-3,
            atol=1e-3,
            msg=f"correction: hip_fwd_y - {input_msg}\n",
        )
        checkAllclose(
            ref_x,
            leg_input_x.view(s, b, h_x, d),
            rtol=1e-3,
            atol=1e-3,
            msg=f"correction: leg_fwd_x - {input_msg}\n",
        )
        checkAllclose(
            ref_y,
            leg_input_y.view(s, b, h_y, d),
            rtol=1e-3,
            atol=1e-3,
            msg=f"correction: leg_fwd_y - {input_msg}\n",
        )

    leg_cached_fwd_avg = 0.0001
    hip_cached_fwd_avg = 0.0001
    if offsets is None:
        _, leg_cached_fwd_avg = legacy_rope_cached_positions_2d_fwd(
            input_x.view(s * b, -1),
            input_y.view(s * b, -1),
            cos,
            sin,
            positions,
            rotate_style,
            nope_first,
        )
        _, hip_cached_fwd_avg = hip_rope_cached_positions_2d_fwd_inplace(
            input_x, input_y, cos, sin, positions, rotate_style, True, nope_first
        )
    else:
        _, leg_cached_fwd_avg = legacy_rope_cached_positions_offsets_2d_fwd(
            input_x.view(s * b, -1),
            input_y.view(s * b, -1),
            cos,
            sin,
            positions,
            offsets,
            rotate_style,
            nope_first,
        )
        _, hip_cached_fwd_avg = hip_rope_cached_positions_offsets_2d_fwd_inplace(
            input_x,
            input_y,
            cos,
            sin,
            positions,
            offsets,
            rotate_style,
            True,
            nope_first,
        )

    color = (
        "\033[91m"
        if hip_cached_fwd_avg > leg_cached_fwd_avg
        else (
            "\033[92m" if hip_cached_fwd_avg < leg_cached_fwd_avg * 0.75 else "\033[93m"
        )
    )
    endc = "\033[0m"
    print(
        f"{color}{input_msg}hip: {hip_cached_fwd_avg:<8.2f} us. leg: {leg_cached_fwd_avg:<8.2f} us. diff: {100*hip_cached_fwd_avg/leg_cached_fwd_avg}%.\n{endc}"
    )


def test_rope_thd(
    input, cu_seqlens, freqs, grad, rotate_style, reuse_freqs_front_part, nope_first
):
    torch.set_printoptions(profile="full")
    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
nope_first: {nope_first}, \
cu_seqlens: {cu_seqlens}
"""
    torch.set_printoptions(profile="default")

    ref = ref_rope_thd_fwd(
        input,
        cu_seqlens,
        freqs,
        rotate_style,
        reuse_freqs_front_part,
        nope_first,
        False,
        True,
    )
    ref.backward(grad)

    hip_fwd, hip_fwd_avg = hip_rope_thd_fwd(
        input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
    )
    hip_bwd, hip_bwd_avg = hip_rope_thd_bwd(
        grad, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first
    )

    checkAllclose(
        ref,
        hip_fwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_thd_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input.grad,
        hip_bwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_thd_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n",
    )


def test_rope_2d(input, height, width, freqs_h, freqs_w, grad):
    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs_h.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs_h.shape):<20}
"""

    cos_h = freqs_h.cos()
    sin_h = freqs_h.sin()
    cos_w = freqs_w.cos()
    sin_w = freqs_w.sin()

    ref = ref_rope_2d_fwd(
        input, height, width, cos_h, sin_h, cos_w, sin_w, RotateStyle.NEOX
    )
    ref.backward(grad)

    hip_fwd, hip_fwd_avg = hip_rope_2d_fwd(
        input, cos_h, sin_h, cos_w, sin_w, height, width, RotateStyle.NEOX, False, False
    )
    hip_bwd, hip_bwd_avg = hip_rope_2d_bwd(
        grad, cos_h, sin_h, cos_w, sin_w, height, width, RotateStyle.NEOX, False, False
    )

    checkAllclose(
        ref,
        hip_fwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_2d_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n",
    )
    checkAllclose(
        input.grad,
        hip_bwd,
        rtol=1e-3,
        atol=1e-3,
        msg=f"rope_2d_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n",
    )


if __name__ == "__main__":
    l_dtype = ("fp16", "bf16")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--no_check",
        action="store_true",
        help="Do not check correctness of ops. Default: False.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with legacy implementation. Default: False",
    )
    parser.add_argument(
        "--compare_check",
        action="store_true",
        help="Check correctness when compare with legacy implementation. Default: False",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=l_dtype,
        nargs="?",
        const=None,
        default=None,
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-t",
        "--transpose_output",
        default=(False, True),
        nargs="*",
        type=dtypes.str2bool,
        help="""Transpose output. Default: (False, True).
    e.g.: -t f   # for False
    e.g.: -t t   # for True""",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=[4],
        nargs="*",
        help="""Batch sizes for testing. The default is 4, but you can choose from: 1, 2, 4.
    e.g.: -b 1""",
    )
    parser.add_argument(
        "-s",
        "--seq_size",
        type=int,
        default=[2048],
        nargs="*",
        help="""Sequence sizes to test. Default: 2048, but you can choose from: 1024, 2048, 4096.
    e.g.: -s 1024""",
    )
    parser.add_argument(
        "-hs",
        "--head_size",
        type=int,
        default=[64],
        nargs="*",
        help="""Head sizes to test. Default is 64, but you can choose from: 32, 64.
    e.g.: -hs 32""",
    )
    parser.add_argument(
        "-hd",
        "--hidden_dim",
        type=int,
        default=[256],
        nargs="*",
        help="""Hidden dimensions to test. Default is 256, bui you can choose from: 128, 256.
    e.g.: -hd 128""",
    )
    parser.add_argument(
        "-ht",
        "--height",
        default=[64],
        nargs="*",
        type=int,
        help="""Height sizes to test. Default is 64, but you can choose from: 32, 64.
    e.g.: -ht 32""",
    )
    parser.add_argument(
        "-wd",
        "--width",
        default=[64],
        nargs="*",
        type=int,
        help="""Width sizes to test. Default is 64, but you can choose from: 32, 64.
    e.g.: -wd 32""",
    )
    parser.add_argument(
        "-m",
        "--margin",
        default=[0, 3],
        nargs="*",
        type=int,
        help="""Margin sizes to test. Default is [0,3].
    e.g.: -m 0""",
    )
    d_rs = {"neox": RotateStyle.NEOX, "gptj": RotateStyle.GPTJ}
    parser.add_argument(
        "-rs",
        "--rotate_style",
        default=list(d_rs.keys()),
        type=str,
        choices=list(d_rs.keys()),
        nargs="*",
        help="""Rotate style. Default is all combinations of neox and gptj.
    e.g.: -rs neox""",
    )
    d_rr = {
        # [0]: rotary percentage, [1]: reuse front part, [2]: nope first
        0: (1.0, True, False),
        1: (1.0, False, False),
        2: (0.5, False, False),
        3: (0.5, True, False),
        4: (0.5, True, True),
        5: (0.5, False, True),
    }
    parser.add_argument(
        "-rr",
        "--rotary_percent_and_reuse",
        default=list(d_rr.keys()),
        type=int,
        nargs="*",
        choices=list(d_rr.keys()),
        help="""Rotary percentage and reuse front part. Default is all combinations of:
(1.0, True, False), (1.0, False, False), (0.5, False, False), (0.5, True, False), (0.5, True, True), (0.5, False, True).
    e.g.: -rr 0  # for (1.0, True, False)""",
    )

    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]

    args.rotate_style = [d_rs[rs] for rs in args.rotate_style]
    args.rotary_percent_and_reuse = [d_rr[rr] for rr in args.rotary_percent_and_reuse]

    # Test sbhd format for both cached and uncached
    if not args.no_check:
        for (
            dtype,
            fdtype,
            transpose_output,
            rotate_style,
            rotary_percent_and_reuse,
            b,
            s,
            h,
            d,
        ) in itertools.product(
            l_dtype,
            l_dtype,
            args.transpose_output,
            args.rotate_style,
            args.rotary_percent_and_reuse,
            args.batch_size,
            args.seq_size,
            args.head_size,
            args.hidden_dim,
        ):
            rotary_percent = rotary_percent_and_reuse[0]
            reuse_freqs_front_part = rotary_percent_and_reuse[1]
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            input = torch.randn(
                (s, b, h, d), dtype=dtype, device="cuda", requires_grad=True
            )
            freqs = torch.randn(
                (s, 1, 1, int(d * rotary_percent) // freqs_ratio),
                dtype=fdtype,
                device="cuda",
            )
            grad = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            test_rope_sbhd(
                input,
                freqs,
                grad,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output,
            )
            input_x = torch.randn(
                (s, b, h, d), dtype=dtype, device="cuda", requires_grad=True
            )
            input_y = torch.randn(
                (s, b, h, d), dtype=dtype, device="cuda", requires_grad=True
            )
            grad_y = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            test_rope_sbhd_2c(
                input_x,
                input_y,
                freqs,
                grad,
                grad_y,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output,
            )

    # Test sbhd format for cached with position (and offsets)
    if not args.no_check:
        for (
            dtype,
            fdtype,
            transpose_output,
            rotate_style,
            rotary_percent_and_reuse,
            has_offsets,
            b,
            s,
            h_x,
            h_y,
            d,
        ) in itertools.product(
            l_dtype,
            l_dtype,
            args.transpose_output,
            args.rotate_style,
            args.rotary_percent_and_reuse,
            (False, True),
            args.batch_size,
            args.seq_size,
            args.head_size,
            args.head_size,
            args.hidden_dim,
        ):
            rotary_percent = rotary_percent_and_reuse[0]
            reuse_freqs_front_part = rotary_percent_and_reuse[1]
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            freqs = torch.randn(
                (s * 2, 1, 1, int(d * rotary_percent) // freqs_ratio),
                dtype=fdtype,
                device="cuda",
            )
            positions = torch.randint(
                int(s * 0.25) if has_offsets else 0,
                int(s * 0.75) if has_offsets else s,
                (
                    s,
                    b,
                ),
                device="cuda",
            )
            offsets = (
                torch.randint(
                    int(s * -0.25),
                    int(s * 0.25),
                    (
                        s,
                        b,
                    ),
                    device="cuda",
                )
                if has_offsets
                else None
            )
            input_x = torch.randn((s, b, h_x, d), dtype=dtype, device="cuda")
            input_y = torch.randn((s, b, h_y, d), dtype=dtype, device="cuda")
            grad_x = torch.randn((s, b, h_x, d), dtype=dtype, device="cuda")
            grad_y = torch.randn((s, b, h_y, d), dtype=dtype, device="cuda")
            # Note that the below tests cannot run together if backward is enabled due to grad info in inputs are not reset.
            test_rope_sbhd_1c_positions(
                input_x,
                freqs,
                grad_x,
                positions,
                offsets,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output,
            )
            test_rope_sbhd_2c_positions(
                input_x,
                input_y,
                freqs,
                grad_x,
                grad_y,
                positions,
                offsets,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
                transpose_output,
            )

    # Compare new with legacy
    if args.compare:
        # [0]: rotary percentage, [1]: reuse front part, [2]: nope first
        # reuse front part should always be True here since legacy implementation doesn't support the opposite setting.
        rotary_percent_and_reuse_compare_ = (
            (1.0, True, False),
            (0.5, True, False),
        )
        for (
            dtype,
            rotate_style,
            rotary_percent_and_reuse,
            has_offsets,
            b,
            s,
            h_x,
            h_y,
            d,
        ) in itertools.product(
            l_dtype,  # legacy implementation doesn't support different scalar type between input/output and freqs/sin/cos
            args.rotate_style,
            rotary_percent_and_reuse_compare_,
            (False, True),
            args.batch_size,
            args.seq_size,
            args.head_size,
            args.head_size,
            args.hidden_dim,
        ):
            color, endc = "\033[95m", "\033[0m"
            print(
                f"{color}dtype: {dtype}, rotate_style: {rotate_style}, rpar: {rotary_percent_and_reuse}, (s,b,hx,hy,d): {s, b, h_x, h_y, d}, has_offsets: {has_offsets}{endc}"
            )
            rotary_percent = rotary_percent_and_reuse[0]
            reuse_freqs_front_part = rotary_percent_and_reuse[1]
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            freqs = torch.randn(
                (s * 2, 1, 1, int(d * rotary_percent) // freqs_ratio),
                dtype=dtype,
                device="cuda",
            )
            positions = torch.randint(
                int(s * 0.25) if has_offsets else 0,
                int(s * 0.75) if has_offsets else s,
                (
                    s,
                    b,
                ),
                device="cuda",
            )
            offsets = (
                torch.randint(
                    int(s * -0.25),
                    int(s * 0.25),
                    (
                        s,
                        b,
                    ),
                    device="cuda",
                )
                if has_offsets
                else None
            )
            input_x = torch.randn((s, b, h_x, d), dtype=dtype, device="cuda")
            input_y = torch.randn((s, b, h_y, d), dtype=dtype, device="cuda")
            compare_rope_sbhd_2c_positions_with_legacy(
                input_x,
                input_y,
                freqs,
                positions,
                offsets,
                rotate_style,
                nope_first,
                args.compare_check,
            )

    # Test thd format for uncached
    if not args.no_check:
        cu_seqlens = torch.tensor(
            [
                0,
                100,
                102,
                128,
                233,
                456,
                460,
                711,
                1024,
                1536,
                1739,
                1888,
                2000,
                2001,
                2048,
            ],
            dtype=dtypes.i32,
            device="cuda",
        )
        for (
            dtype,
            fdtype,
            rotate_style,
            rotary_percent_and_reuse,
            h,
            d,
        ) in itertools.product(
            l_dtype,
            l_dtype,
            args.rotate_style,
            args.rotary_percent_and_reuse,
            args.head_size,
            args.hidden_dim,
        ):
            rotary_percent = rotary_percent_and_reuse[0]
            reuse_freqs_front_part = rotary_percent_and_reuse[1]
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            input = torch.randn(
                (cu_seqlens[-1], h, d), dtype=dtype, device="cuda", requires_grad=True
            )
            freqs = torch.randn(
                (cu_seqlens[-1], 1, 1, int(d * rotary_percent) // freqs_ratio),
                dtype=fdtype,
                device="cuda",
            )
            grad = torch.randn((cu_seqlens[-1], h, d), dtype=dtype, device="cuda")
            test_rope_thd(
                input,
                cu_seqlens,
                freqs,
                grad,
                rotate_style,
                reuse_freqs_front_part,
                nope_first,
            )

    # Test 2d image format for cached
    if not args.no_check:
        for dtype, fdtype, b, h, d, height, width, margin in itertools.product(
            l_dtype,
            l_dtype,
            args.batch_size,
            args.head_size,
            args.hidden_dim,
            args.height[-1:],
            args.width[-1:],
            args.margin,
        ):
            input = torch.randn(
                (b, height * width, h, d),
                dtype=dtype,
                device="cuda",
                requires_grad=True,
            )
            freqs_h = torch.randn(
                (1, height + margin, 1, d // 2), dtype=fdtype, device="cuda"
            )
            freqs_w = torch.randn(
                (1, width + margin, 1, d // 2), dtype=fdtype, device="cuda"
            )
            grad = torch.randn((b, height * width, h, d), dtype=dtype, device="cuda")
            test_rope_2d(input, height, width, freqs_h, freqs_w, grad)
