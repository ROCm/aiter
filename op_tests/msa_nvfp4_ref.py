# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Torch reference for the nvfp4 MSA index cache pages.

A page holds [block_size][head_dim / 2] e2m1 pairs (low nibble = even dim),
then [block_size][head_dim / 16] e4m3 scales.
"""

import torch

FP8 = torch.float8_e4m3fn
MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoints between adjacent e2m1 magnitudes.
MIDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]


def ref_nvfp4_pages(keys, block_size):
    """Keys [P * block_size, head_dim] -> nvfp4 byte pages [P, page bytes].

    scale = E4M3(clamp(amax / 6, 2^-9, 448)) per 16 dims; each element takes
    the e2m1 magnitude nearest |x| / scale, ties to the smaller one.
    """
    head_dim = keys.shape[1]
    p = keys.shape[0] // block_size
    g = keys.float().view(p, block_size, head_dim // 16, 16)
    s8 = (g.abs().amax(-1, keepdim=True) / 6.0).clamp(2.0**-9, 448.0).to(FP8)
    s = s8.float()
    code = torch.zeros_like(g, dtype=torch.uint8)
    for mid in MIDS:
        code += g.abs() > mid * s
    nib = (code | ((g < 0).to(torch.uint8) << 3)).view(p, block_size, head_dim)
    data = nib[..., 0::2] | (nib[..., 1::2] << 4)
    return torch.cat([data.view(p, -1), s8.view(torch.uint8).view(p, -1)], 1)


def ref_nvfp4_dequant(cache, block_size, head_dim):
    """Byte pages [P, page bytes] -> [P, block_size, head_dim] fp32, as the
    kernels see them: E4M3(min(scale * magnitude, 448)) with the element's sign."""
    p = cache.shape[0]
    data_bytes = block_size * head_dim // 2
    data = cache[:, :data_bytes].reshape(p, block_size, head_dim // 2)
    nib = torch.stack([data & 15, data >> 4], -1).reshape(p, block_size, head_dim)
    sb = cache[:, data_bytes : data_bytes + block_size * head_dim // 16]
    s = sb.view(FP8).float().reshape(p, block_size, head_dim // 16, 1)
    mags = torch.tensor(MAGS, device=cache.device)
    m = mags[(nib & 7).long()].view(p, block_size, head_dim // 16, 16)
    v = (s * m).clamp(max=448.0).to(FP8).float()
    v = torch.where((nib >= 8).view_as(v), -v, v)
    return v.view(p, block_size, head_dim)
