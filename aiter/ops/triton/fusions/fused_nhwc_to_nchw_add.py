# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_nhwc_to_nchw_add import (
    _fused_nhwc_to_nchw_add_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["fused_nhwc_to_nchw_add"]


def fused_nhwc_to_nchw_add(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Channels-last to NCHW, plus an optional NCHW residual, in one pass.

    Computes ``x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous() + residual``.
    This is the exit of a UNet attention block (diffusers ``Transformer2DModel``
    with ``use_linear_projection``): the token-major ``proj_out`` output goes back
    to NCHW and gets the block input added.

    Args:
        x (torch.Tensor): (B, H * W, C) or (B, H, W, C), fp32, fp16 or bf16.
            The channel dim must be contiguous; rows can have a stride.
        residual (Optional[torch.Tensor]): (B, C, H, W) with a contiguous H * W
            plane, same dtype as x. Without it, only the layout changes.
        height (int): H.
        width (int): W.

    Returns:
        torch.Tensor: Contiguous (B, C, H, W) output, dtype of x.

    Limits: the add is in fp32 and rounds once, as in eager PyTorch, so the
    result is bit-identical to eager. One batch of x, residual and the output
    must each fit in int32 element offsets.
    """
    assert x.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), f"unsupported dtype: {x.dtype}"
    B, P = x.shape[0], height * width
    C = x.shape[-1]
    x3 = x.reshape(B, P, C) if x.dim() == 4 else x
    assert (
        x3.dim() == 3 and x3.shape[1] == P
    ), f"x must be (B, H * W, C) or (B, H, W, C) with H * W = {P}, got {tuple(x.shape)}"
    assert x3.stride(-1) == 1, "x must have a contiguous channel dim"
    if residual is not None:
        assert residual.shape == (
            B,
            C,
            height,
            width,
        ), f"residual must be ({B}, {C}, {height}, {width}), got {tuple(residual.shape)}"
        assert (
            residual.dtype == x.dtype
        ), f"dtype mismatch: {residual.dtype} vs {x.dtype}"
        assert residual.device == x.device, "residual must be on the device of x"
        assert (
            residual.stride(-1) == 1 and residual.stride(-2) == width
        ), "residual must have a contiguous H * W plane"
    for size in (x3.stride(0), P * C):
        assert size < 2**31, "one batch must fit in int32 element offsets"

    _LOGGER.info(
        "FUSED_NHWC_TO_NCHW_ADD: x=%s residual=%s",
        tuple(x.shape),
        None if residual is None else tuple(residual.shape),
    )

    out = torch.empty((B, C, height, width), dtype=x.dtype, device=x.device)
    if out.numel() == 0:
        return out

    config = _get_config(C)
    BLOCK_P = config.pop("BLOCK_P")
    BLOCK_C = config.pop("BLOCK_C")
    res = residual if residual is not None else out
    grid = (triton.cdiv(P, BLOCK_P) * triton.cdiv(C, BLOCK_C), B)
    _fused_nhwc_to_nchw_add_kernel[grid](
        x3,
        res,
        out,
        P,
        C,
        x3.stride(0),
        x3.stride(1),
        res.stride(0),
        res.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_P=BLOCK_P,
        BLOCK_C=BLOCK_C,
        HAS_RESIDUAL=residual is not None,
        **config,
    )
    return out
