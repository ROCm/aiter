# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.conv.conv2d_nearest_up2x_3x3 import (
    _conv2d_nearest_up2x_3x3_kernel,
)
from aiter.ops.triton.conv._prepack import _get_or_make_pack, _LRUPackCache
from aiter.ops.triton.utils.conv_config_utils import format_shape_key, get_conv_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["conv2d_nearest_up2x_3x3"]

_PACK_CACHE_UP2X = _LRUPackCache()


def _fold_up2x_weight(weight: torch.Tensor, _block: int = 0) -> torch.Tensor:
    """(K, C, 3, 3) -> (4, K, 4, C): per output phase (a, b), the 2x2 weights
    w[phase, k, 2 * dr + dc, c] on the input. Summed in fp32, rounded once."""
    w = weight.float()
    # Rows of phase a = 0 read input rows (p - 1, p), of a = 1 rows (p, p + 1).
    rows = (
        torch.stack((w[:, :, 0], w[:, :, 1] + w[:, :, 2]), dim=2),
        torch.stack((w[:, :, 0] + w[:, :, 1], w[:, :, 2]), dim=2),
    )
    phases = []
    for r in rows:
        for b in range(2):
            if b == 0:
                cols = (r[..., 0], r[..., 1] + r[..., 2])
            else:
                cols = (r[..., 0] + r[..., 1], r[..., 2])
            phases.append(torch.stack(cols, dim=-1))  # (K, C, 2, 2)
    folded = torch.stack(phases)  # (4, K, C, 2, 2)
    K, C = weight.shape[:2]
    return folded.reshape(4, K, C, 4).permute(0, 1, 3, 2).to(weight.dtype).contiguous()


def conv2d_nearest_up2x_3x3(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Nearest-neighbour x2 upsample followed by a 3x3 conv (stride 1, padding 1),
    without materializing the upsampled tensor.

    Computes ``F.conv2d(F.interpolate(x, scale_factor=2, mode="nearest"),
    weight, bias, padding=1)``; for a scale of 2, "nearest" and "nearest-exact"
    pick the same source pixel. This is the upsample block of the Wan VAE
    (``WanResample``) and of UNet / 2D VAE up blocks (``Upsample2D`` with a conv).

    Every output pixel (2p + a, 2q + b) reads a 2x2 window of x, so the op runs
    four 2x2 convs on x (one per output phase) with weights summed from the
    3x3 kernel: 4/9 of the multiply-adds and a quarter of the input reads of
    the conv on the upsampled tensor.

    Args:
        x (torch.Tensor): Input (N, C, H, W), NCHW with a contiguous last dim.
            fp16 or bf16.
        weight (torch.Tensor): (K, C, 3, 3), same dtype as x.
        bias (Optional[torch.Tensor]): K elements, same dtype as x.

    Returns:
        torch.Tensor: Contiguous output (N, K, 2H, 2W), dtype of x.

    Limits: the folded weights are summed in fp32 and rounded once to the input
    dtype, so the result differs from the conv on the upsampled tensor by the
    rounding of the weights and the summation order. The folded weights are
    cached per weight tensor (as in the other aiter conv ops), except for
    inference tensors. x, the weights and the output must each fit in int32
    element offsets.
    """
    assert x.dim() == 4, f"x must be (N, C, H, W), got {tuple(x.shape)}"
    assert x.stride(-1) == 1, "x must have a contiguous last dim (NCHW)"
    assert x.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"unsupported dtype: {x.dtype} (fp16 and bf16 only)"
    N, C, H, W = x.shape
    assert weight.dim() == 4 and weight.shape[1:] == (
        C,
        3,
        3,
    ), f"weight must be (K, {C}, 3, 3), got {tuple(weight.shape)}"
    assert weight.dtype == x.dtype, f"dtype mismatch: {weight.dtype} vs {x.dtype}"
    assert weight.device == x.device, "weight must be on the device of x"
    K = weight.shape[0]
    if bias is not None:
        assert bias.numel() == K, f"bias must have {K} elements, got {bias.numel()}"
        assert bias.dtype == x.dtype, f"dtype mismatch: {bias.dtype} vs {x.dtype}"
        assert bias.device == x.device, "bias must be on the device of x"
        bias = bias.contiguous()
    for size in (N * x.stride(0), 16 * K * C, N * K * 4 * H * W):
        assert size < 2**31, "tensors must fit in int32 element offsets"

    _LOGGER.info(
        "CONV2D_NEAREST_UP2X_3X3: x=%s K=%d bias=%s",
        tuple(x.shape),
        K,
        bias is not None,
    )

    y = torch.empty((N, K, 2 * H, 2 * W), dtype=x.dtype, device=x.device)
    if y.numel() == 0:
        return y

    w_folded = _get_or_make_pack(_PACK_CACHE_UP2X, weight, 0, _fold_up2x_weight)
    shape_key = format_shape_key(N, C, H, W, K, 3, 3, 1, 1, 1, 1, 1, 1)
    config = get_conv_config(
        "CONV2D-NEAREST-UP2X-3X3", shape_key=shape_key, M=N * H * W
    )
    BLOCK_M = config.pop("BLOCK_M")
    BLOCK_N = config.pop("BLOCK_N")
    grid = (triton.cdiv(N * H * W, BLOCK_M) * triton.cdiv(K, BLOCK_N), 4)
    _conv2d_nearest_up2x_3x3_kernel[grid](
        x,
        w_folded,
        bias,
        y,
        N,
        C,
        H,
        W,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_BIAS=bias is not None,
        **config,
    )
    return y
