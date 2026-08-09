# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Memory-efficient dynamic per-head FP8 quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_head_absmax_partial(
    x,
    partial,
    num_tokens,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_t,
    stride_h,
    stride_d,
    elements_per_head,
    partial_stride,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    block = tl.program_id(1)
    element = block * BLOCK + tl.arange(0, BLOCK)
    token = element // head_dim
    dim = element % head_dim
    offset = token * stride_t + head * stride_h + dim * stride_d
    value = tl.load(
        x + offset,
        mask=(element < elements_per_head) & (token < num_tokens),
        other=0.0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(value), axis=0)
    tl.store(partial + head * partial_stride + block, maximum)


@triton.jit
def _per_head_absmax_reduce(
    partial,
    descale,
    num_partials,
    partial_stride,
    fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    head = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    value = tl.load(
        partial + head * partial_stride + offset,
        mask=offset < num_partials,
        other=0.0,
    )
    maximum = tl.max(value, axis=0)
    scale = tl.maximum(maximum / fp8_max, 1.0e-12)
    tl.store(descale + head, scale)


@triton.jit
def _per_head_quantize(
    x,
    output,
    descale,
    total_elements,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_t,
    stride_h,
    stride_d,
    fp8_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = element // (num_heads * head_dim)
    head_dim_offset = element % (num_heads * head_dim)
    head = head_dim_offset // head_dim
    dim = head_dim_offset % head_dim
    input_offset = token * stride_t + head * stride_h + dim * stride_d
    mask = element < total_elements
    value = tl.load(x + input_offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(descale + head, mask=mask, other=1.0)
    quantized = tl.maximum(tl.minimum(value / scale, fp8_max), -fp8_max)
    tl.store(output + element, quantized, mask=mask)


def dynamic_per_head_quant_fp8(
    value: torch.Tensor,
    fp8_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize contiguous ``[tokens, heads, dim]`` data with one scale per head."""
    if value.ndim != 3:
        raise ValueError(f"expected rank-3 input, got {value.shape}")
    if value.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported input dtype {value.dtype}")

    num_tokens, num_heads, head_dim = value.shape
    elements_per_head = num_tokens * head_dim
    partial_block = 4096
    num_partials = triton.cdiv(elements_per_head, partial_block)
    partial = torch.empty(
        (num_heads, num_partials), dtype=torch.float32, device=value.device
    )
    descale = torch.empty((1, num_heads), dtype=torch.float32, device=value.device)
    output = torch.empty(value.shape, dtype=fp8_dtype, device=value.device)
    fp8_max = torch.finfo(fp8_dtype).max

    _per_head_absmax_partial[(num_heads, num_partials)](
        value,
        partial,
        num_tokens,
        num_heads,
        head_dim,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        elements_per_head,
        partial.stride(0),
        BLOCK=partial_block,
        num_warps=8,
    )
    reduce_block = triton.next_power_of_2(num_partials)
    _per_head_absmax_reduce[(num_heads,)](
        partial,
        descale,
        num_partials,
        partial.stride(0),
        fp8_max,
        BLOCK=reduce_block,
        num_warps=min(8, max(1, reduce_block // 256)),
    )
    quant_block = 1024
    total_elements = value.numel()
    _per_head_quantize[(triton.cdiv(total_elements, quant_block),)](
        value,
        output,
        descale,
        total_elements,
        num_heads,
        head_dim,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        fp8_max,
        BLOCK=quant_block,
        num_warps=4,
    )
    return output, descale
