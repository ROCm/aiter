# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest


@perftest()
def hip_rope_fwd(input, freqs, transpose_output):
    return aiter.rope_fwd(input, freqs, transpose_output)

@perftest()
def hip_rope_bwd(output_grads, freqs, transpose_output):
    return aiter.rope_bwd(output_grads, freqs, transpose_output)

@perftest()
def hip_rope_cached_fwd(input, cos, sin, transpose_output):
    return aiter.rope_cached_fwd(input, cos, sin, transpose_output)

@perftest()
def hip_rope_cached_bwd(output_grads, cos, sin, transpose_output):
    return aiter.rope_cached_bwd(output_grads, cos, sin, transpose_output)

# @perftest()
def hip_rope_thd_fwd(input, cu_seqlens, freqs):
    return aiter.rope_thd_fwd(input, cu_seqlens, freqs)

# @perftest()
def hip_rope_thd_bwd(output_grads, cu_seqlens, freqs):
    return aiter.rope_thd_bwd(output_grads, cu_seqlens, freqs)

# @perftest()
def hip_rope_2d_fwd(input, cos_height, sin_height, cos_width, sin_width):
    return aiter.rope_2d_fwd(input, cos_height, sin_height, cos_width, sin_width)

# @perftest()
def hip_rope_2d_bwd(output_grads, cos_height, sin_height, cos_width, sin_width):
    return aiter.rope_2d_bwd(output_grads, cos_height, sin_height, cos_width, sin_width)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def ref_rope_fwd(x, freqs):
    x_embed = (x * torch.cos(freqs)) + (rotate_half(x) * torch.sin(freqs))
    return x_embed.to(dtype=x.dtype)


def test_rope_sbhd(dtype, fdtype, dim_i, dim_freqs, transpose_output):
    input_msg = f"dtype: {dtype}, freq_dtype: {fdtype}, dim_input: {str(dim_i):<20}, dim_freqs: {str(dim_freqs):<20}, transpose_output: {transpose_output}"

    input = torch.randn(dim_i, dtype=dtype, device="cuda", requires_grad=True)
    freqs = torch.randn(dim_freqs, dtype=fdtype, device="cuda")
    cos   = torch.cos(freqs)
    sin   = torch.sin(freqs)
    grad  = torch.randn(dim_i, dtype=dtype, device="cuda")

    ref = ref_rope_fwd(input, freqs)
    ref.backward(grad)

    hip_fwd,        hip_fwd_avg        = hip_rope_fwd(input, freqs, transpose_output)
    hip_bwd,        hip_bwd_avg        = hip_rope_bwd(grad, freqs, transpose_output)
    hip_cached_fwd, hip_cached_fwd_avg = hip_rope_cached_fwd(input, cos, sin, transpose_output)
    hip_cached_bwd, hip_cached_bwd_avg = hip_rope_cached_bwd(grad, cos, sin, transpose_output)

    checkAllclose(ref,        hip_fwd,        msg=f"rope_fwd - avg: {hip_fwd_avg:<8.2f} us - {input_msg} ")
    checkAllclose(input.grad, hip_bwd,        msg=f"rope_bwd - avg: {hip_bwd_avg:<8.2f} us - {input_msg} ")
    checkAllclose(ref,        hip_cached_fwd, msg=f"rope_cached_fwd - avg: {hip_cached_fwd_avg:<8.2f} us - {input_msg} ")
    checkAllclose(input.grad, hip_cached_bwd, msg=f"rope_cached_bwd - avg: {hip_cached_bwd_avg:<8.2f} us - {input_msg} ")


if __name__ == "__main__":
    for dtype in [torch.float, torch.float16, torch.bfloat16]:
        for fdtype in [torch.float, torch.float16, torch.bfloat16]:
            for transpose_output in (False, True):
                for b in (1,2,4,8,16):
                    for s in (16,32,64,128):
                        for h in (1,2,4,8,16):
                            for input_d in (128,160):
                                # for freq_d in (64, 128, 160, 192):
                                    test_rope_sbhd(dtype, fdtype, (b, s, h, input_d), (b, 1, 1, input_d), transpose_output)
