# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Ported from
# https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_q_rope_mxfp4_quant_repr = make_kernel_repr(
    "_q_rope_mxfp4_quant_kernel",
    ["HEAD_SIZE", "ROPE_DIM", "BLOCK_H", "HAS_WEIGHTS", "PROJECT"],
)


@triton.jit(repr=_q_rope_mxfp4_quant_repr)
def _q_rope_mxfp4_quant_kernel(
    q_ptr,  # [T, H, HEAD_SIZE] bf16/fp16
    q_stride_t,
    q_stride_h,
    positions_ptr,  # [T]
    cos_sin_ptr,  # [max_pos, ROPE_DIM]: cos half, then sin half
    cos_sin_stride,
    q_fp4_ptr,  # u8 [T, H, HEAD_SIZE // 2]
    q_fp4_stride_t,
    q_fp4_stride_h,
    q_scale_ptr,  # u8 e8m0 [T, H, HEAD_SIZE // 32]
    q_scale_stride_t,
    q_scale_stride_h,
    weights_ptr,  # [T, H]
    weights_stride,
    weights_scale,
    weights_out_ptr,  # f32/bf16/f16 [T, H]
    weights_out_stride,
    hidden_ptr,  # [T, PROJ_IN] bf16/fp16, PROJECT input
    hidden_stride,
    wproj_ptr,  # [H, PROJ_IN] bf16/fp16, head-major
    wproj_stride_h,
    num_heads,
    HEAD_SIZE: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    PROJECT: tl.constexpr,
    PROJ_IN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (token, BLOCK_H heads)."""
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2

    tok = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < num_heads
    dims = tl.arange(0, HEAD_SIZE)

    q = tl.load(
        q_ptr + tok * q_stride_t + heads[:, None] * q_stride_h + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    pos = tl.load(positions_ptr + tok)
    rope_pair = tl.arange(0, NUM_PAIRS) - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cs_base = cos_sin_ptr + pos * cos_sin_stride
    cos = tl.load(cs_base + cs_idx, mask=is_rope, other=1.0).to(tl.float32)
    sin = tl.load(cs_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0).to(tl.float32)
    even, odd = tl.split(tl.reshape(q, (BLOCK_H, NUM_PAIRS, 2)))
    # Rotated dims round through bf16 to match the reference; NoPE dims don't.
    rot_even = (even * cos - odd * sin).to(tl.bfloat16).to(tl.float32)
    rot_odd = (odd * cos + even * sin).to(tl.bfloat16).to(tl.float32)
    even = tl.where(is_rope, rot_even, even)
    odd = tl.where(is_rope, rot_odd, odd)
    q = tl.reshape(tl.join(even, odd), (BLOCK_H, HEAD_SIZE))

    q_fp4, q_scale = _mxfp4_quant_op(
        q, HEAD_SIZE, BLOCK_H, 32, SCALING_MODE=1, USE_ASM=True
    )
    fp4_dims = tl.arange(0, HEAD_SIZE // 2)
    tl.store(
        q_fp4_ptr
        + tok * q_fp4_stride_t
        + heads[:, None] * q_fp4_stride_h
        + fp4_dims[None, :],
        q_fp4,
        mask=head_mask[:, None],
    )
    scale_dims = tl.arange(0, HEAD_SIZE // 32)
    tl.store(
        q_scale_ptr
        + tok * q_scale_stride_t
        + heads[:, None] * q_scale_stride_h
        + scale_dims[None, :],
        q_scale,
        mask=head_mask[:, None],
    )

    if PROJECT:
        # Head weights projected here (hidden @ wproj.T) so the per-layer
        # weights_proj GEMM is not its own launch. Summed in fp32, then rounded
        # through bf16 like a standalone bf16 GEMM's accumulation.
        acc = tl.zeros((BLOCK_H,), tl.float32)
        for k0 in range(0, PROJ_IN, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            hvec = tl.load(hidden_ptr + tok * hidden_stride + ks).to(tl.float32)
            wmat = tl.load(
                wproj_ptr + heads[:, None] * wproj_stride_h + ks[None, :],
                mask=head_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wmat * hvec[None, :], axis=1)
        w = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(
            weights_out_ptr + tok * weights_out_stride + heads,
            (w * weights_scale).to(weights_out_ptr.dtype.element_ty),
            mask=head_mask,
        )
    elif HAS_WEIGHTS:
        w = tl.load(weights_ptr + tok * weights_stride + heads, mask=head_mask)
        tl.store(
            weights_out_ptr + tok * weights_out_stride + heads,
            (w.to(tl.float32) * weights_scale).to(weights_out_ptr.dtype.element_ty),
            mask=head_mask,
        )
