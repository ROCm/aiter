# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.rope.inverse_rope_group_quant import (
    _inverse_rope_group_quant_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

_LOGGER = AiterTritonLogger()

# One head per program launches on every arch. Measured archs pin a wider block.
_FALLBACK_CONFIG = triton.Config({"BLOCK_H": 1}, num_warps=4)


def _launch_config(num_heads: int, head_dim: int) -> triton.Config:
    return get_tuned_kernel_config(
        "fusions",
        "INVERSE_ROPE_GROUP_QUANT",
        f"_inverse_rope_group_quant_kernel_H{num_heads}_D{head_dim}",
        fallback=_FALLBACK_CONFIG,
        warn_on_miss=False,
    )


def inverse_rope_group_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    num_groups: int,
    quant_group_size: int = 32,
    scale_layout: str = "row",
    x_fp8: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undo the GPT-J RoPE of attention output rows and quantize them to FP8.

    This is the Triton version of ``aiter.ops.inverse_rope_group_quant`` for the
    "row" scale layout. It makes the input of a DeepSeek-V4 wo_a GEMM. Each head
    keeps its first head_dim - rope_dim values and rotates its last rope_dim
    values by -theta. Each quant_group_size values of a head get one E8M0 scale,
    2^ceil(log2(amax / fp8_max)), so no value saturates.

    Args:
        o: (S, H, head_dim) bf16 or fp16 attention output, before the inverse
            RoPE. The last dim must have unit stride. head_dim must be a power
            of 2.
        positions: (S,) token positions.
        cos_cache: (max_pos, rope_dim // 2) cos table. Rows can have any stride.
        sin_cache: (max_pos, rope_dim // 2) sin table. Rows can have any stride.
        num_groups: Number of output groups G. Each output row is G groups of
            H * head_dim // G values.
        quant_group_size: Values for each scale: 32 (MXFP8) or 128.
        scale_layout: Only "row" is supported.
        x_fp8: Optional (S, G, H * head_dim // G) output, e4m3fn or e4m3fnuz.
        x_scale: Optional (S, G, H * head_dim // G // quant_group_size) E8M0 or
            uint8 output.

    Returns:
        (x_fp8, x_scale).
    """
    _LOGGER.info(
        "INVERSE_ROPE_GROUP_QUANT: o=%s rope_dim=%d num_groups=%d group=%d",
        tuple(o.shape),
        2 * cos_cache.shape[-1],
        num_groups,
        quant_group_size,
    )
    if scale_layout != "row":
        raise NotImplementedError(
            f"scale_layout={scale_layout!r}: the Triton kernel writes only 'row'"
        )
    if o.ndim != 3 or o.stride(2) != 1:
        raise ValueError(
            f"o must be [S, H, head_dim] with unit last stride, got "
            f"{tuple(o.shape)} {o.stride()}"
        )
    num_rows, num_heads, head_dim = o.shape
    rope_dim = 2 * cos_cache.shape[-1]
    if head_dim & (head_dim - 1) or rope_dim > head_dim:
        raise ValueError(
            f"head_dim {head_dim} must be a power of 2 and hold the "
            f"{rope_dim}-dim rope"
        )
    if quant_group_size not in (32, 128) or head_dim % quant_group_size:
        raise ValueError(
            f"quant_group_size must be 32 or 128 and divide head_dim, got "
            f"{quant_group_size} for head_dim {head_dim}"
        )
    if (num_heads * head_dim) % num_groups:
        raise ValueError(
            f"H * head_dim = {num_heads * head_dim} must be divisible by "
            f"num_groups={num_groups}"
        )
    if cos_cache.shape != sin_cache.shape or cos_cache.stride(1) != 1:
        raise ValueError(
            "cos_cache and sin_cache must have the same [max_pos, rope_dim // 2] "
            "shape and unit last stride"
        )
    if positions.numel() != num_rows:
        raise ValueError(f"positions must have S={num_rows} values")

    group_dim = num_heads * head_dim // num_groups
    if group_dim % quant_group_size:
        raise ValueError(
            f"group width {group_dim} must be divisible by "
            f"quant_group_size={quant_group_size}"
        )
    scale_groups = group_dim // quant_group_size
    if x_fp8 is None:
        x_fp8 = torch.empty(
            (num_rows, num_groups, group_dim),
            dtype=get_fp8_e4m3_dtype(),
            device=o.device,
        )
    if x_scale is None:
        x_scale = torch.empty(
            (num_rows, num_groups, scale_groups),
            dtype=torch.float8_e8m0fnu,
            device=o.device,
        )
    if tuple(x_fp8.shape) != (num_rows, num_groups, group_dim) or x_fp8.dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ):
        raise ValueError(
            f"x_fp8 must be e4m3 ({num_rows}, {num_groups}, {group_dim}), got "
            f"{x_fp8.dtype} {tuple(x_fp8.shape)}"
        )
    if tuple(x_scale.shape) != (num_rows, num_groups, scale_groups):
        raise ValueError(
            f"x_scale must be ({num_rows}, {num_groups}, {scale_groups}), got "
            f"{tuple(x_scale.shape)}"
        )
    # The kernel writes each row as H * head_dim contiguous values.
    if x_fp8.stride(2) != 1 or x_fp8.stride(1) != group_dim:
        raise ValueError("x_fp8 rows must be contiguous")
    if x_scale.stride(2) != 1 or x_scale.stride(1) != scale_groups:
        raise ValueError("x_scale rows must be contiguous")
    if num_rows == 0:
        return x_fp8, x_scale

    config = _launch_config(num_heads, head_dim)
    block_h = config.kwargs["BLOCK_H"]
    _inverse_rope_group_quant_kernel[(num_rows, triton.cdiv(num_heads, block_h))](
        o,
        o.stride(0),
        o.stride(1),
        positions,
        cos_cache,
        cos_cache.stride(0),
        sin_cache,
        sin_cache.stride(0),
        x_fp8,
        x_fp8.stride(0),
        x_scale.view(torch.uint8),
        x_scale.stride(0),
        num_heads,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        QUANT_GROUP_SIZE=quant_group_size,
        FP8_MAX=torch.finfo(x_fp8.dtype).max,
        BLOCK_H=block_h,
        num_warps=config.num_warps,
    )
    return x_fp8, x_scale
