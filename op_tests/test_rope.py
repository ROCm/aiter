# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest
import itertools
from enum import IntEnum


@perftest()
def hip_rope_fwd(input, freqs, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_fwd(input, freqs, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_bwd(output_grads, freqs, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_bwd(output_grads, freqs, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_2c_fwd(input_x, input_y, freqs, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_2c_fwd(input_x, input_y, freqs, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_2c_bwd(output_grads_x, output_grads_y, freqs, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_2c_bwd(output_grads_x, output_grads_y, freqs, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_cached_fwd(input, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_cached_fwd(input, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_cached_bwd(output_grads, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_cached_bwd(output_grads, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_cached_2c_fwd(input_x, input_y, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_cached_2c_fwd(input_x, input_y, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_cached_2c_bwd(output_grads_x, output_grads_y, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output):
    return aiter.rope_cached_2c_bwd(output_grads_x, output_grads_y, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)

@perftest()
def hip_rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part):
    return aiter.rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part)

@perftest()
def hip_rope_thd_bwd(output_grads, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part):
    return aiter.rope_thd_bwd(output_grads, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part)

@perftest()
def hip_rope_2d_fwd(input, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part):
    return aiter.rope_2d_fwd(input, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part)

@perftest()
def hip_rope_2d_bwd(output_grads, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part):
    return aiter.rope_2d_bwd(output_grads, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part)


class RotateStyle(IntEnum):
    NEOX = 0,
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


def ref_rope_sbhd_fwd(x, freqs, rotate_style, reuse_freqs_front_part):
    rotate_half = rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    freqs_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    x, x_forward = x[..., :freqs_dim], x[..., freqs_dim:]
    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.NEOX:
            freqs = freqs.repeat([1] * (freqs.dim()-1) + [2])
        elif rotate_style == RotateStyle.GPTJ:
            freqs = freqs.repeat_interleave(2, dim=-1)
    x_embed = (x * torch.cos(freqs)) + (rotate_half(x) * torch.sin(freqs))
    return torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1)


def ref_rope_thd_fwd(x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part):
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    x_embed = torch.cat([
        ref_rope_sbhd_fwd(xi.unsqueeze(1), freqs[: xi.size(0)], rotate_style, reuse_freqs_front_part)
        for xi in torch.split(x, seqlens)
    ])
    return x_embed.squeeze(1)


def ref_rope_2d_fwd(x, size_h, size_w, cos_h, sin_h, cos_w, sin_w, rotate_style):
    rotate_half = rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    s, b, h, d = x.shape
    x = x.view(s, size_h, size_w, h, d)
    x1, x2 = x.chunk(2, dim=-1)
    cos_h = cos_h[:, :size_h].unsqueeze(2)  # [1, H, 1, 1, D//2]
    sin_h = sin_h[:, :size_h].unsqueeze(2)  # [1, H, 1, 1, D//2]
    x1 = (x1 * cos_h) + (rotate_half(x1) * sin_h)
    cos_w = cos_w[:, :size_w].unsqueeze(1)  # [1, 1, W, 1, D//2]
    sin_w = sin_w[:, :size_w].unsqueeze(1)  # [1, 1, W, 1, D//2]
    x2 = (x2 * cos_w) + (rotate_half(x2) * sin_w)
    return torch.cat([x1, x2], dim=-1).view(s, b, h, d).to(dtype=x.dtype)



def test_rope_sbhd(input, freqs, grad, rotate_style, reuse_freqs_front_part, transpose_output):
    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
transpose_output: {transpose_output}
"""

    ref = ref_rope_sbhd_fwd(input, freqs, rotate_style, reuse_freqs_front_part)
    if rotate_style == RotateStyle.NEOX:
        ref.backward(grad)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    hip_fwd,        hip_fwd_avg        = hip_rope_fwd(input, freqs, rotate_style, reuse_freqs_front_part, transpose_output)
    if rotate_style == RotateStyle.NEOX:
        hip_bwd,        hip_bwd_avg        = hip_rope_bwd(grad, freqs, rotate_style, reuse_freqs_front_part, transpose_output)
    hip_cached_fwd, hip_cached_fwd_avg = hip_rope_cached_fwd(input, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)
    if rotate_style == RotateStyle.NEOX:
        hip_cached_bwd, hip_cached_bwd_avg = hip_rope_cached_bwd(grad, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)

    checkAllclose(ref,        hip_fwd,        msg=f"rope_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    if rotate_style == RotateStyle.NEOX:
        checkAllclose(input.grad, hip_bwd,        msg=f"rope_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref,        hip_cached_fwd, msg=f"rope_cached_fwd - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    if rotate_style == RotateStyle.NEOX:
        checkAllclose(input.grad, hip_cached_bwd, msg=f"rope_cached_bwd - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n")


def test_rope_sbhd_2c(input_x, input_y, freqs, grad_x, grad_y, rotate_style, reuse_freqs_front_part, transpose_output):
    assert(input_x.shape == input_y.shape)
    assert(input_x.dtype == input_y.dtype)

    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
transpose_output: {transpose_output}
"""

    ref_x = ref_rope_sbhd_fwd(input_x, freqs, rotate_style, reuse_freqs_front_part)
    ref_y = ref_rope_sbhd_fwd(input_y, freqs, rotate_style, reuse_freqs_front_part)
    if rotate_style == RotateStyle.NEOX:
        ref_x.backward(grad_x)
        ref_y.backward(grad_y)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    (hip_fwd_x, hip_fwd_y), hip_fwd_avg = hip_rope_2c_fwd(input_x, input_y, freqs, rotate_style, reuse_freqs_front_part, transpose_output)
    if rotate_style == RotateStyle.NEOX:
        (hip_bwd_x, hip_bwd_y), hip_bwd_avg = hip_rope_2c_bwd(grad_x, grad_y, freqs, rotate_style, reuse_freqs_front_part, transpose_output)
    (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = hip_rope_cached_2c_fwd(input_x, input_y, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)
    if rotate_style == RotateStyle.NEOX:
        (hip_cached_bwd_x, hip_cached_bwd_y), hip_cached_bwd_avg = hip_rope_cached_2c_bwd(grad_x, grad_y, cos, sin, rotate_style, reuse_freqs_front_part, transpose_output)

    checkAllclose(ref_x,        hip_fwd_x,        msg=f"rope_2c_fwd_x - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_y,        hip_fwd_y,        msg=f"rope_2c_fwd_y - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    if rotate_style == RotateStyle.NEOX:
        checkAllclose(input_x.grad, hip_bwd_x,        msg=f"rope_2c_bwd_x - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")
        checkAllclose(input_y.grad, hip_bwd_y,        msg=f"rope_2c_bwd_y - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_x,        hip_cached_fwd_x, msg=f"rope_cached_2c_fwd_x - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_y,        hip_cached_fwd_y, msg=f"rope_cached_2c_fwd_y - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    if rotate_style == RotateStyle.NEOX:
        checkAllclose(input_x.grad, hip_cached_bwd_x, msg=f"rope_cached_2c_bwd_x - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n")
        checkAllclose(input_y.grad, hip_cached_bwd_y, msg=f"rope_cached_2c_bwd_y - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n")


def test_rope_thd(input, cu_seqlens, freqs, grad, rotate_style, reuse_freqs_front_part):
    torch.set_printoptions(profile="full")
    input_msg = f"""
dtype: {input.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
cu_seqlens: {cu_seqlens}
"""
    torch.set_printoptions(profile="default")

    ref = ref_rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part)
    if rotate_style == RotateStyle.NEOX:
        ref.backward(grad)

    hip_fwd, hip_fwd_avg = hip_rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part)
    if rotate_style == RotateStyle.NEOX:
        hip_bwd, hip_bwd_avg = hip_rope_thd_bwd(grad, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part)

    checkAllclose(ref,        hip_fwd, msg=f"rope_thd_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    if rotate_style == RotateStyle.NEOX:
        checkAllclose(input.grad, hip_bwd, msg=f"rope_thd_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")


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

    ref = ref_rope_2d_fwd(input, height, width, cos_h, sin_h, cos_w, sin_w, RotateStyle.NEOX)
    ref.backward(grad)

    hip_fwd, hip_fwd_avg = hip_rope_2d_fwd(input, cos_h, sin_h, cos_w, sin_w, height, width, RotateStyle.NEOX, False)
    hip_bwd, hip_bwd_avg = hip_rope_2d_bwd(grad, cos_h, sin_h, cos_w, sin_w, height, width, RotateStyle.NEOX, False)

    checkAllclose(ref,        hip_fwd, msg=f"rope_2d_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input.grad, hip_bwd, msg=f"rope_2d_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")



if __name__ == "__main__":
    dtype_ = (torch.float, torch.float16, torch.bfloat16)
    transpose_output_ = (False, True)
    batch_size_ = (1, 2, 4)
    seq_size_ = (1024, 2048, 4096)
    head_size_ = (32, 64)
    hidden_dim_ = (128, 256)
    rotary_percent_and_reuse_ = ((1.0, True), (0.5, False), (1.0, False))
    height_ = (32, 64)
    width_ = (32, 64)
    margin_ = (0, 1, 3)
    rotate_style_ = (RotateStyle.NEOX, RotateStyle.GPTJ)

    # Test sbhd format for both cached and uncached
    for (dtype, fdtype,
         transpose_output,
         rotate_style,
         rotary_percent_and_reuse,
         b, s, h, d
    ) in itertools.product(
        dtype_, dtype_,
        transpose_output_,
        rotate_style_,
        rotary_percent_and_reuse_,
        batch_size_, seq_size_, head_size_, hidden_dim_
    ):
        rotary_percent = rotary_percent_and_reuse[0]
        reuse_freqs_front_part = rotary_percent_and_reuse[1]
        input = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
        freqs = torch.randn((s, 1, 1, int(d * rotary_percent) // 2), dtype=fdtype, device="cuda")
        if not reuse_freqs_front_part:
            if rotate_style == RotateStyle.NEOX:
                freqs = freqs.repeat([1] * (freqs.dim()-1) + [2])
            elif rotate_style == RotateStyle.GPTJ:
                freqs = freqs.repeat_interleave(2, dim=-1)
        grad  = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
        test_rope_sbhd(input, freqs, grad, rotate_style, reuse_freqs_front_part, transpose_output)
        input_x = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
        input_y = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
        grad_y  = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
        test_rope_sbhd_2c(input_x, input_y, freqs, grad, grad_y, rotate_style, reuse_freqs_front_part, transpose_output)

    # Test thd format for uncached
    cu_seqlens = torch.tensor([0, 100, 102, 128, 233, 456, 460, 711, 1024, 1536, 1739, 1888, 2000, 2001, 2048],
                              dtype=torch.int32, device="cuda")
    for (dtype, fdtype,
         rotate_style,
         rotary_percent_and_reuse,
         h, d
    ) in itertools.product(
        dtype_, dtype_,
        rotate_style_,
        rotary_percent_and_reuse_,
        head_size_, hidden_dim_
    ):
        rotary_percent = rotary_percent_and_reuse[0]
        reuse_freqs_front_part = rotary_percent_and_reuse[1]
        input = torch.randn((cu_seqlens[-1], h, d), dtype=dtype, device="cuda", requires_grad=True)
        freqs = torch.randn((cu_seqlens[-1], 1, 1, int(d * rotary_percent) // freqs_ratio), dtype=fdtype, device="cuda")
        if not reuse_freqs_front_part:
            if rotate_style == RotateStyle.NEOX:
                freqs = freqs.repeat([1] * (freqs.dim()-1) + [2])
            elif rotate_style == RotateStyle.GPTJ:
                freqs = freqs.repeat_interleave(2, dim=-1)
        grad  = torch.randn((cu_seqlens[-1], h, d), dtype=dtype, device="cuda")
        test_rope_thd(input, cu_seqlens, freqs, grad, rotate_style, reuse_freqs_front_part)

    # Test 2d image format for cached
    for (dtype, fdtype,
         b, h, d,
         height, width, margin
    ) in itertools.product(
        dtype_, dtype_,
        batch_size_, head_size_, hidden_dim_,
        height_, width_, margin_
    ):
        input   = torch.randn((b, height * width, h, d), dtype=dtype, device="cuda", requires_grad=True)
        freqs_h = torch.randn((1, height + margin, 1, d // 2), dtype=fdtype, device="cuda")
        freqs_w = torch.randn((1, width + margin, 1, d // 2), dtype=fdtype, device="cuda")
        grad    = torch.randn((b, height * width, h, d), dtype=dtype, device="cuda")
        test_rope_2d(input, height, width, freqs_h, freqs_w, grad)
