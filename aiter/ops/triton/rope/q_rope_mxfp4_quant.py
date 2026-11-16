# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.rope.q_rope_mxfp4_quant import (
    _q_rope_mxfp4_quant_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_SCALE_GROUP = 32


def _launch_config(num_tokens: int, num_heads: int, head_size: int) -> tuple[int, int]:
    # From a gfx950 sweep of 32 and 64 heads: ~32 values per lane.
    rows = num_tokens * num_heads
    if rows <= 8192:
        block_h = 4
    elif rows <= 262144:
        block_h = 16
    else:
        block_h = 64
    block_h = min(triton.next_power_of_2(num_heads), block_h)
    return block_h, max(1, block_h * head_size // 2048)


def q_rope_mxfp4_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor | None = None,
    weights_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """GPT-J RoPE on the last rope_dim dims -> MXFP4 query, plus scaled weights.

    MXFP4 is e2m1 values, two per byte (low nibble first), with one e8m0 scale
    per 32 values chosen as 2^ceil(log2(amax / 6)), so no value saturates.

    Args:
        q: [T, H, head_size] bf16/fp16 (strided is fine, the last dim must be
            unit-stride).
        positions: [T] token positions.
        cos_sin_cache: [max_pos, rope_dim], the cos half then the sin half.
        weights: optional [T, H] per-head weights.
        weights_scale: multiplied into the weights.

    Returns:
        (q_fp4 u8 [T, H, head_size // 2], q_scale u8 e8m0
        [T, H, head_size // 32], weights f32 [T, H] or None).
    """
    _LOGGER.info(
        "Q_ROPE_MXFP4_QUANT: q=%s rope_dim=%d weights=%s",
        tuple(q.shape),
        cos_sin_cache.shape[-1],
        None if weights is None else tuple(weights.shape),
    )
    if q.ndim != 3 or q.stride(2) != 1:
        raise ValueError(
            f"q must be [T, H, D] with unit last stride, got "
            f"{tuple(q.shape)} {q.stride()}"
        )
    num_tokens, num_heads, head_size = q.shape
    rope_dim = cos_sin_cache.shape[-1]
    if head_size % _SCALE_GROUP or rope_dim % 2 or rope_dim > head_size:
        raise ValueError(
            f"head_size {head_size} must be a multiple of "
            f"{_SCALE_GROUP} and hold the {rope_dim}-dim rope"
        )
    if positions.numel() != num_tokens:
        raise ValueError(
            f"positions {tuple(positions.shape)} must cover q's T={num_tokens}"
        )
    if weights is not None and (
        tuple(weights.shape) != (num_tokens, num_heads) or weights.stride(1) != 1
    ):
        raise ValueError(
            f"weights {tuple(weights.shape)} must be q's [T={num_tokens}, "
            f"H={num_heads}] with unit head stride"
        )
    q_fp4 = torch.empty(
        (num_tokens, num_heads, head_size // 2), dtype=torch.uint8, device=q.device
    )
    q_scale = torch.empty(
        (num_tokens, num_heads, head_size // _SCALE_GROUP),
        dtype=torch.uint8,
        device=q.device,
    )
    weights_out = None
    if weights is not None:
        weights_out = torch.empty(
            (num_tokens, num_heads), dtype=torch.float32, device=q.device
        )
    if num_tokens == 0:
        return q_fp4, q_scale, weights_out

    block_h, num_warps = _launch_config(num_tokens, num_heads, head_size)
    _q_rope_mxfp4_quant_kernel[(num_tokens, triton.cdiv(num_heads, block_h))](
        q,
        q.stride(0),
        q.stride(1),
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp4,
        q_fp4.stride(0),
        q_fp4.stride(1),
        q_scale,
        q_scale.stride(0),
        q_scale.stride(1),
        weights,
        0 if weights is None else weights.stride(0),
        weights_scale,
        weights_out,
        0 if weights_out is None else weights_out.stride(0),
        num_heads,
        HEAD_SIZE=head_size,
        ROPE_DIM=rope_dim,
        BLOCK_H=block_h,
        HAS_WEIGHTS=weights is not None,
        num_warps=num_warps,
    )
    return q_fp4, q_scale, weights_out
