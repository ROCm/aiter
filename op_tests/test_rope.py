# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, perftest


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def ref_rope_fwd(x, freqs):
    x_embed = (x * torch.cos(freqs)) + (rotate_half(x) * torch.sin(freqs))
    return x_embed.to(dtype=x.dtype)

def ref_rope_cached_fwd(x, cos, sin):
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.to(dtype=x.dtype)

@perftest()
def hip_rope_fwd(input, freqs, transpose_output):
    if transpose_output:
        b, s, h, d = input.shape
        output = torch.empty((s, b, h, d)).transpose(0, 1)
    else:
        output = torch.empty_like(input)
    aiter.rope_fwd(output, input, freqs)
    return output

@perftest()
def hip_rope_cached_fwd(input, cos, sin, transpose_output):
    if transpose_output: 
        b, s, h, d = input.shape
        output = torch.empty((s, b, h, d), dtype=input.dtype, device=input.device).transpose(0, 1)
    else:
        output = torch.empty_like(input)
    aiter.rope_cached_fwd(output, input, cos, sin)
    return output

def test_rope(dtype, dim_i, dim_freq, transpose_output):
    input = torch.randn(dim_i, dtype=dtype, device="cuda")
    freqs = torch.randn(dim_freq, dtype=torch.float, device="cuda")
    ref = ref_rope_fwd(input, freqs)
    ref_cached = ref_rope_cached_fwd(input, freqs.cos(), freqs.sin())
    hip, hip_avg = hip_rope_fwd(input, freqs, transpose_output)
    hip_cached, hip_cached_avg = hip_rope_cached_fwd(input, freqs.cos(), freqs.sin(), transpose_output)
    msg = f"[perf] dim_input: {str(dim_i):<20}, dtype_input: {dtype}, dim_freqs: {str(dim_freq):<20}, dtype_freqs: {torch.float}, transpose_output: {transpose_output}, avg: {hip_avg:<8.2f}, cached_avg: {hip_cached_avg:<8.2f}"
    checkAllclose(ref, ref_cached)
    checkAllclose(ref, hip, msg=msg)
    checkAllclose(ref, hip_cached)

if __name__ == "__main__":
    for dtype in [torch.float16, torch.bfloat16]:
        for transpose_output in (False, True):
            for b in (1,2,4,8,16):
                for s in (16,32,64,128):
                    for h in (1,2,4,8,16):
                        for input_d in (128,160):
                            # for freq_d in (64, 128, 160, 192):
                                test_rope(dtype, (b, s, h, input_d), (b, 1, 1, input_d), transpose_output)
