# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, perftest
import itertools
from enum import IntEnum
import argparse


@perftest()
def hip_rope_fwd(input, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_fwd(input, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_bwd(output_grads, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_bwd(output_grads, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_2c_fwd(input_x, input_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_2c_fwd(input_x, input_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_2c_bwd(output_grads_x, output_grads_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_2c_bwd(output_grads_x, output_grads_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_fwd(input, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_cached_fwd(input, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_bwd(output_grads, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_cached_bwd(output_grads, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_2c_fwd(input_x, input_y, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_cached_2c_fwd(input_x, input_y, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_2c_bwd(output_grads_x, output_grads_y, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_cached_2c_bwd(output_grads_x, output_grads_y, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_positions_2d_fwd(input_x, input_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_cached_positions_2c_fwd(input_x, input_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_positions_offsets_2d_fwd(input_x, input_y, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    return aiter.rope_cached_positions_offsets_2c_fwd(input_x, input_y, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

@perftest()
def hip_rope_cached_positions_2d_fwd_inplace(input_x, input_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first):
    return aiter.rope_cached_positions_2c_fwd_inplace(input_x, input_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first)

@perftest()
def hip_rope_cached_positions_offsets_2d_fwd_inplace(input_x, input_y, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first):
    return aiter.rope_cached_positions_offsets_2c_fwd_inplace(input_x, input_y, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first)

@perftest()
def hip_rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first):
    return aiter.rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first)

@perftest()
def hip_rope_thd_bwd(output_grads, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first):
    return aiter.rope_thd_bwd(output_grads, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first)

@perftest()
def hip_rope_2d_fwd(input, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part, nope_first):
    return aiter.rope_2d_fwd(input, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part, nope_first)

@perftest()
def hip_rope_2d_bwd(output_grads, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part, nope_first):
    return aiter.rope_2d_bwd(output_grads, height, width, cos_h, sin_h, cos_w, sin_w, rotate_style, reuse_freqs_front_part, nope_first)

@perftest()
def legacy_rope_cached_positions_2d_fwd(input_x, input_y, cos_sin, positions, rotate_style, nope_first):
    s, b, h, d = input_x.shape
    aiter.rotary_embedding_fwd(positions, input_x.view(s * b, -1), input_y.view(s * b, -1), d, cos_sin, rotate_style is RotateStyle.NEOX)
    return input_x, input_y

@perftest()
def legacy_rope_cached_positions_offsets_2d_fwd(input_x, input_y, cos_sin, positions, offsets, rotate_style, nope_first):
    s, b, h, d = input_x.shape
    rotate_dim =  cos_sin.size(1)
    aiter.batched_rotary_embedding(positions, input_x.view(s * b, -1), input_y.view(s * b, -1), d, cos_sin, rotate_style is RotateStyle.NEOX, rotate_dim, offsets.view(-1))
    return input_x, input_y


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


def ref_rope_sbhd_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first):
    rotate_half = rotate_half_neox if rotate_style == RotateStyle.NEOX else rotate_half_gptj
    rotate_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    if nope_first:
        d = x.shape[-1]
        x, x_forward = x[..., d - rotate_dim:], x[..., :d - rotate_dim]
    else:
        x, x_forward = x[..., :rotate_dim], x[..., rotate_dim:]
    if reuse_freqs_front_part:
        if rotate_style == RotateStyle.NEOX:
            freqs = freqs.repeat([1] * (freqs.dim()-1) + [2])
        elif rotate_style == RotateStyle.GPTJ:
            freqs = freqs.repeat_interleave(2, dim=-1)
    x_embed = (x * torch.cos(freqs)) + (rotate_half(x) * torch.sin(freqs))
    return torch.cat((x_forward, x_embed.to(dtype=x.dtype)), dim=-1) if nope_first else torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1)


def ref_rope_thd_fwd(x, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first):
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    x_embed = torch.cat([
        ref_rope_sbhd_fwd(xi.unsqueeze(1), freqs[: xi.size(0)], rotate_style, reuse_freqs_front_part, nope_first)
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



def test_rope_sbhd(input, freqs, grad, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
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

    ref = ref_rope_sbhd_fwd(input, freqs, rotate_style, reuse_freqs_front_part, nope_first)
    ref.backward(grad)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    hip_fwd,        hip_fwd_avg        = hip_rope_fwd(input, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    hip_bwd,        hip_bwd_avg        = hip_rope_bwd(grad, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    hip_cached_fwd, hip_cached_fwd_avg = hip_rope_cached_fwd(input, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    hip_cached_bwd, hip_cached_bwd_avg = hip_rope_cached_bwd(grad, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    checkAllclose(ref,        hip_fwd,        msg=f"rope_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input.grad, hip_bwd,        msg=f"rope_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref,        hip_cached_fwd, msg=f"rope_cached_fwd - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input.grad, hip_cached_bwd, msg=f"rope_cached_bwd - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n")


def test_rope_sbhd_2c(input_x, input_y, freqs, grad_x, grad_y, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    assert(input_x.shape == input_y.shape)
    assert(input_x.dtype == input_y.dtype)

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

    ref_x = ref_rope_sbhd_fwd(input_x, freqs, rotate_style, reuse_freqs_front_part, nope_first)
    ref_y = ref_rope_sbhd_fwd(input_y, freqs, rotate_style, reuse_freqs_front_part, nope_first)
    ref_x.backward(grad_x)
    ref_y.backward(grad_y)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    (hip_fwd_x, hip_fwd_y), hip_fwd_avg = hip_rope_2c_fwd(input_x, input_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    (hip_bwd_x, hip_bwd_y), hip_bwd_avg = hip_rope_2c_bwd(grad_x, grad_y, freqs, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = hip_rope_cached_2c_fwd(input_x, input_y, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    (hip_cached_bwd_x, hip_cached_bwd_y), hip_cached_bwd_avg = hip_rope_cached_2c_bwd(grad_x, grad_y, cos, sin, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    checkAllclose(ref_x,        hip_fwd_x,        msg=f"rope_2c_fwd_x - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_y,        hip_fwd_y,        msg=f"rope_2c_fwd_y - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input_x.grad, hip_bwd_x,        msg=f"rope_2c_bwd_x - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input_y.grad, hip_bwd_y,        msg=f"rope_2c_bwd_y - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_x,        hip_cached_fwd_x, msg=f"rope_cached_2c_fwd_x - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_y,        hip_cached_fwd_y, msg=f"rope_cached_2c_fwd_y - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input_x.grad, hip_cached_bwd_x, msg=f"rope_cached_2c_bwd_x - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input_y.grad, hip_cached_bwd_y, msg=f"rope_cached_2c_bwd_y - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg}\n")


def test_rope_sbhd_2c_positions(input_x, input_y, freqs, grad_x, grad_y, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output):
    input_msg = f"""
dtype: {input_x.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input_x.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
dim_positions: {str(positions.shape):<20}, \
dim_offsets: {str(offsets.shape) if offsets is not None else 'None'}, \
rotate_style: {rotate_style.value}, \
reuse_freqs_front_part: {reuse_freqs_front_part}, \
nope_first: {nope_first}, \
transpose_output: {transpose_output}
"""

    ref_x = ref_rope_sbhd_fwd(input_x, freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(-2), rotate_style, reuse_freqs_front_part, nope_first)
    ref_y = ref_rope_sbhd_fwd(input_y, freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(-2), rotate_style, reuse_freqs_front_part, nope_first)

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    if offsets is None:
        (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = hip_rope_cached_positions_2d_fwd(
            input_x, input_y, cos, sin, positions, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
    else:
        (hip_cached_fwd_x, hip_cached_fwd_y), hip_cached_fwd_avg = hip_rope_cached_positions_offsets_2d_fwd(
            input_x, input_y, cos, sin, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    checkAllclose(ref_x, hip_cached_fwd_x, msg=f"rope_cached_position_2d_fwd_x - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(ref_y, hip_cached_fwd_y, msg=f"rope_cached_position_2d_fwd_y - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg}\n")


def compare_rope_sbhd_2c_positions_with_legacy(input_x, input_y, freqs, positions, offsets, rotate_style, nope_first, check_correction=False):
    input_msg = f"""dtype: {input_x.dtype}, \
freq_dtype: {freqs.dtype}, \
dim_input: {str(input_x.shape):<20}, \
dim_freqs: {str(freqs.shape):<20}, \
dim_positions: {str(positions.shape):<20}, \
dim_offsets: {str(offsets.shape) if offsets is not None else 'None'}, \
rotate_style: {rotate_style.value}, \
nope_first: {nope_first}
"""

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos_sin = torch.cat((cos, sin), dim=-1).squeeze(1,2)

    # perftest cannot test correction of inplace operators
    if check_correction:
        ref_x = ref_rope_sbhd_fwd(input_x, freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(-2), rotate_style, True, nope_first)
        ref_y = ref_rope_sbhd_fwd(input_y, freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(-2), rotate_style, True, nope_first)
        s, b, h, d = input_x.shape
        hip_input_x, hip_input_y = input_x.clone(), input_y.clone()
        leg_input_x, leg_input_y = input_x.clone().view(s * b, -1), input_y.clone().view(s * b, -1)
        if offsets is None:
            aiter.rope_cached_positions_2c_fwd_inplace(hip_input_x, hip_input_y, cos, sin, positions, rotate_style, True, nope_first)
            aiter.rotary_embedding_fwd(positions, leg_input_x, leg_input_y, d, cos_sin, rotate_style is RotateStyle.NEOX)
        else:
            aiter.rope_cached_positions_offsets_2c_fwd_inplace(hip_input_x, hip_input_y, cos, sin, positions, offsets, rotate_style, True, nope_first)
            aiter.batched_rotary_embedding(positions, leg_input_x, leg_input_y, d, cos_sin, rotate_style is RotateStyle.NEOX, cos_sin.size(1), offsets.view(-1))

        checkAllclose(ref_x, hip_input_x, msg=f"correction: hip_fwd_x - {input_msg}\n")
        checkAllclose(ref_y, hip_input_y, msg=f"correction: hip_fwd_y - {input_msg}\n")
        checkAllclose(ref_x, leg_input_x.view(s, b, h, d), msg=f"correction: leg_fwd_x - {input_msg}\n")
        checkAllclose(ref_y, leg_input_y.view(s, b, h, d), msg=f"correction: leg_fwd_y - {input_msg}\n")

    if offsets is None:
        _, hip_cached_fwd_avg = hip_rope_cached_positions_2d_fwd_inplace(
            input_x, input_y, cos, sin, positions, rotate_style, True, nope_first)
        _, leg_cached_fwd_avg = legacy_rope_cached_positions_2d_fwd(input_x, input_y, cos_sin, positions, rotate_style, nope_first)
    else:
        _, hip_cached_fwd_avg = hip_rope_cached_positions_offsets_2d_fwd_inplace(
            input_x, input_y, cos, sin, positions, offsets, rotate_style, True, nope_first)
        _, leg_cached_fwd_avg = legacy_rope_cached_positions_offsets_2d_fwd(input_x, input_y, cos_sin, positions, offsets, rotate_style, nope_first)

    color = '\033[91m' if hip_cached_fwd_avg > leg_cached_fwd_avg else '\033[92m' if hip_cached_fwd_avg < leg_cached_fwd_avg * 0.75 else '\033[93m'
    print(f"{color}{input_msg}hip: {hip_cached_fwd_avg:<8.2f} us. leg: {leg_cached_fwd_avg:<8.2f} us.\n{color}")



def test_rope_thd(input, cu_seqlens, freqs, grad, rotate_style, reuse_freqs_front_part, nope_first):
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

    ref = ref_rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first)
    ref.backward(grad)

    hip_fwd, hip_fwd_avg = hip_rope_thd_fwd(input, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first)
    hip_bwd, hip_bwd_avg = hip_rope_thd_bwd(grad, cu_seqlens, freqs, rotate_style, reuse_freqs_front_part, nope_first)

    checkAllclose(ref,        hip_fwd, msg=f"rope_thd_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
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

    hip_fwd, hip_fwd_avg = hip_rope_2d_fwd(input, cos_h, sin_h, cos_w, sin_w, height, width, RotateStyle.NEOX, False, False)
    hip_bwd, hip_bwd_avg = hip_rope_2d_bwd(grad, cos_h, sin_h, cos_w, sin_w, height, width, RotateStyle.NEOX, False, False)

    checkAllclose(ref,        hip_fwd, msg=f"rope_2d_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg}\n")
    checkAllclose(input.grad, hip_bwd, msg=f"rope_2d_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_check', action='store_true', help="Do not check correctness of ops. Default: False.")
    parser.add_argument('--compare', action='store_true', help="Compare with legacy implementation. Default: False")
    parser.add_argument('--compare_check', action='store_true', help="Check correctness when compare with legacy implementation. Default: False")
    args = parser.parse_args()

    dtype_ = (torch.float, torch.float16, torch.bfloat16)
    transpose_output_ = (False, True)
    batch_size_ = (1, 2, 4)
    seq_size_ = (1024, 2048, 4096)
    head_size_ = (32, 64)
    hidden_dim_ = (128, 256)
    # [0]: rotary percentage, [1]: reuse front part, [2]: nope first
    rotary_percent_and_reuse_ = ((1.0, True, False),
                                 (1.0, False, False),
                                 (0.5, False, False),
                                 (0.5, True, False),
                                 (0.5, True, True),
                                 (0.5, False, True))
    height_ = (32, 64)
    width_ = (32, 64)
    margin_ = (0, 1, 3)
    rotate_style_ = (RotateStyle.NEOX, RotateStyle.GPTJ)

    # Test sbhd format for both cached and uncached
    if not args.no_check:
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
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            input = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
            freqs = torch.randn((s, 1, 1, int(d * rotary_percent) // freqs_ratio), dtype=fdtype, device="cuda")
            grad  = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            test_rope_sbhd(input, freqs, grad, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)
            input_x = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
            input_y = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
            grad_y  = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            test_rope_sbhd_2c(input_x, input_y, freqs, grad, grad_y, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    # Test sbhd format for cached with position (and offsets)
    if not args.no_check:
        for (dtype, fdtype,
            transpose_output,
            rotate_style,
            rotary_percent_and_reuse,
            has_offsets,
            b, s, h, d
        ) in itertools.product(
            dtype_, dtype_,
            transpose_output_,
            rotate_style_,
            rotary_percent_and_reuse_,
            (False, True),
            batch_size_, seq_size_, head_size_, hidden_dim_
        ):
            rotary_percent = rotary_percent_and_reuse[0]
            reuse_freqs_front_part = rotary_percent_and_reuse[1]
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            freqs   = torch.randn((s * 2, 1, 1, int(d * rotary_percent) // freqs_ratio), dtype=fdtype, device="cuda")
            positions = torch.randint(int(s * 0.25) if has_offsets else 0, int(s * 0.75) if has_offsets else s, (s,b,), device="cuda")
            offsets   = torch.randint(int(s * -0.25), int(s * 0.25), (s,b,), device="cuda") if has_offsets else None
            input_x = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
            input_y = torch.randn((s, b, h, d), dtype=dtype, device="cuda", requires_grad=True)
            grad_x  = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            grad_y  = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            test_rope_sbhd_2c_positions(input_x, input_y, freqs, grad_x, grad_y, positions, offsets, rotate_style, reuse_freqs_front_part, nope_first, transpose_output)

    # Compare new with legacy
    if args.compare:
        # [0]: rotary percentage, [1]: reuse front part, [2]: nope first
        # reuse front part should always be True here since legacy implementation doesn't support the opposite setting.
        rotary_percent_and_reuse_compare_ = (
            (1.0, True, False),
            (0.5, True, False),)
        for (dtype,
            transpose_output,
            rotate_style,
            rotary_percent_and_reuse,
            has_offsets,
            b, s, h, d
        ) in itertools.product(
            dtype_, # legacy implementation doesn't support different scalar type between input/output and freqs/sin/cos
            transpose_output_,
            rotate_style_,
            rotary_percent_and_reuse_compare_,
            (False, True),
            batch_size_[-1:], seq_size_[-1:], head_size_[-1:], hidden_dim_[-1:]
        ):
            rotary_percent = rotary_percent_and_reuse[0]
            reuse_freqs_front_part = rotary_percent_and_reuse[1]
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            freqs   = torch.randn((s * 2, 1, 1, int(d * rotary_percent) // freqs_ratio), dtype=dtype, device="cuda")
            positions = torch.randint(int(s * 0.25) if has_offsets else 0, int(s * 0.75) if has_offsets else s, (s,b,), device="cuda")
            offsets   = torch.randint(int(s * -0.25), int(s * 0.25), (s,b,), device="cuda") if has_offsets else None
            input_x = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            input_y = torch.randn((s, b, h, d), dtype=dtype, device="cuda")
            compare_rope_sbhd_2c_positions_with_legacy(input_x, input_y, freqs, positions, offsets, rotate_style, nope_first, args.compare_check)

    # Test thd format for uncached
    if not args.no_check:
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
            nope_first = (rotary_percent >= 1.0) and rotary_percent_and_reuse[2]
            freqs_ratio = 2 if reuse_freqs_front_part else 1
            input = torch.randn((cu_seqlens[-1], h, d), dtype=dtype, device="cuda", requires_grad=True)
            freqs = torch.randn((cu_seqlens[-1], 1, 1, int(d * rotary_percent) // freqs_ratio), dtype=fdtype, device="cuda")
            grad  = torch.randn((cu_seqlens[-1], h, d), dtype=dtype, device="cuda")
            test_rope_thd(input, cu_seqlens, freqs, grad, rotate_style, reuse_freqs_front_part, nope_first)

    # Test 2d image format for cached
    if not args.no_check:
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
