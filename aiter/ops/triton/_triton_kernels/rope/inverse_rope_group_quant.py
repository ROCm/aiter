# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Ported from the DeepSeek-V4.1 ROCm attention output path in vLLM
# (vllm/v1/attention/ops/rocm_aiter_mla_sparse.py).

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_inverse_rope_group_quant_repr = make_kernel_repr(
    "_inverse_rope_group_quant_kernel",
    ["HEAD_DIM", "ROPE_DIM", "QUANT_GROUP_SIZE", "BLOCK_H"],
)


@triton.jit(repr=_inverse_rope_group_quant_repr)
def _inverse_rope_group_quant_kernel(
    o_ptr,  # [S, H, HEAD_DIM] bf16/fp16
    o_stride_s,
    o_stride_h,
    positions_ptr,  # [S]
    cos_ptr,  # [max_pos, ROPE_DIM // 2]
    cos_stride,
    sin_ptr,  # [max_pos, ROPE_DIM // 2]
    sin_stride,
    x_ptr,  # fp8 [S, H * HEAD_DIM]
    x_stride_s,
    scale_ptr,  # u8 e8m0 [S, H * HEAD_DIM // QUANT_GROUP_SIZE]
    scale_stride_s,
    num_heads,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One program per (row, BLOCK_H heads)."""
    NUM_PAIRS: tl.constexpr = HEAD_DIM // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_DIM - ROPE_DIM) // 2
    GROUPS: tl.constexpr = HEAD_DIM // QUANT_GROUP_SIZE

    row = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = heads < num_heads
    dims = tl.arange(0, HEAD_DIM)
    x = tl.load(
        o_ptr + row * o_stride_s + heads[:, None] * o_stride_h + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Inverse GPT-J RoPE on the last ROPE_DIM dims. NoPE pairs get cos=1, sin=0,
    # so one expression handles the whole row.
    pos = tl.load(positions_ptr + row).to(tl.int64)
    rope_pair = tl.arange(0, NUM_PAIRS) - NOPE_PAIRS
    is_rope = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    cos = tl.load(cos_ptr + pos * cos_stride + cs_idx, mask=is_rope, other=1.0).to(
        tl.float32
    )
    sin = tl.load(sin_ptr + pos * sin_stride + cs_idx, mask=is_rope, other=0.0).to(
        tl.float32
    )
    even, odd = tl.split(tl.reshape(x, (BLOCK_H, NUM_PAIRS, 2)))
    x = tl.reshape(
        tl.join(
            even * cos[None, :] + odd * sin[None, :],
            odd * cos[None, :] - even * sin[None, :],
        ),
        (BLOCK_H, HEAD_DIM),
    )

    # E8M0 scale 2^ceil(log2(amax / FP8_MAX)) for each group, so no value saturates.
    groups = tl.reshape(x, (BLOCK_H, GROUPS, QUANT_GROUP_SIZE))
    amax = tl.maximum(tl.max(tl.abs(groups), axis=2), 1.1754943508222875e-38)
    bits = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
    bits = tl.minimum(tl.maximum(bits, 0.0), 254.0)
    # Multiply by the reciprocal: a divisor of 2^-127 would flush to zero.
    xq = tl.reshape(groups * tl.exp2(127.0 - bits)[:, :, None], (BLOCK_H, HEAD_DIM))

    tl.store(
        x_ptr + row * x_stride_s + heads[:, None] * HEAD_DIM + dims[None, :],
        xq.to(x_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )
    group_offs = tl.arange(0, GROUPS)
    tl.store(
        scale_ptr
        + row * scale_stride_s
        + heads[:, None] * GROUPS
        + group_offs[None, :],
        bits.to(tl.uint8),
        mask=head_mask[:, None],
    )
