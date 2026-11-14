# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(query, key, cosine, sine, query_out, key_out,
          COUNT: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    channel = offsets % 128
    token = offsets // (28 * 128)
    rotated = channel < 96
    partner = offsets + tl.where(channel < 48, 48, -48)
    source = tl.where(tl.program_id(1) == 0, query, key)
    target = tl.where(tl.program_id(1) == 0, query_out, key_out)
    original = tl.load(source + offsets, offsets < COUNT, other=0).to(tl.float32)
    paired = tl.load(source + partner, (offsets < COUNT) & rotated, other=0).to(tl.float32)
    cos_value = tl.load(cosine + token * 96 + channel, (offsets < COUNT) & rotated, other=0)
    sin_value = tl.load(sine + token * 96 + channel, (offsets < COUNT) & rotated, other=0)
    rotated_value = tl.where(channel < 48, -paired, paired)
    result = original * cos_value + rotated_value * sin_value
    result = tl.where(rotated, result, original)
    tl.store(target + offsets, result.to(tl.bfloat16), offsets < COUNT)


def apply(query, key, cosine, sine):
    if query.ndim != 3 or query.shape[1:] != (28, 128) or key.shape != query.shape:
        raise ValueError("Expected local TP2 Q/K [S,28,128]")
    rows = query.shape[0]
    if rows < 1 or query.numel() >= 2**31:
        raise ValueError("Positive rows and int32 indexing required")
    for tensor in (query, key, cosine, sine):
        if tensor.device != query.device or not tensor.is_cuda or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("Same-device contiguous inference tensors required")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        raise ValueError("BF16 Q/K required")
    if any(tensor.shape != (rows, 96) or tensor.dtype != torch.float32 for tensor in (cosine, sine)):
        raise ValueError("Expected full FP32 cosine/sine [S,96]")
    query_out, key_out = torch.empty_like(query), torch.empty_like(key)
    _rope[(triton.cdiv(query.numel(), 1024), 2)](
        query, key, cosine, sine, query_out, key_out,
        COUNT=query.numel(), BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return query_out, key_out
