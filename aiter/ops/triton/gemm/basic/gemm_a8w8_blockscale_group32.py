# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import gemm_afp8wfp8


def gemm_a8w8_blockscale_group32(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    y: torch.Tensor | None = None,
    weight_group_rows: int = 32,
    split_k: int | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Compatibility entry point for E4M3 GEMM with 1x32 or 32x32 E8M0 scales.

    Execution and tuning use the shared AFP8WFP8 implementation. The original
    contiguous-input contract and argument names remain supported.
    """
    assert x.ndim == w.ndim == 2, "Expected two matrix operands"
    M, K = x.shape
    N, weight_k = w.shape
    assert (
        K > 0 and K % 32 == 0 and weight_k == K and N > 0
    ), "Expected matching positive K divisible by 32 and positive N"
    assert weight_group_rows in (1, 32), "Weight scales must be 1x32 or 32x32"
    assert x.dtype in (torch.float8_e4m3fn, torch.uint8), "x must be E4M3"
    assert w.dtype in (torch.float8_e4m3fn, torch.uint8), "w must be E4M3"
    assert x_scale.dtype in (torch.float8_e8m0fnu, torch.uint8), "x_scale must be E8M0"
    assert w_scale.dtype in (torch.float8_e8m0fnu, torch.uint8), "w_scale must be E8M0"
    assert x_scale.shape == (M, K // 32), "Invalid activation scale shape"
    assert w_scale.shape == (
        -(-N // weight_group_rows),
        K // 32,
    ), "Invalid weight scale shape"
    assert all(
        t.is_cuda and t.device == x.device and t.is_contiguous()
        for t in (x, w, x_scale, w_scale)
    ), "Operands must be contiguous on the same GPU"
    assert dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), "Output must be BF16, FP16 or FP32"
    assert split_k is None or (
        type(split_k) is int and split_k > 0
    ), "split_k must be a positive integer or None"
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    else:
        assert (
            y.shape == (M, N) and y.dtype == dtype and y.device == x.device
        ), "Invalid output shape, dtype or device"
        assert y.is_contiguous(), "Output must be contiguous"
    if M == 0:
        return y
    assert get_gfx_runtime() == "gfx950", "Group32 FP8 GEMM requires gfx950"

    return gemm_afp8wfp8(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        y=y,
        config=config,
        x_scale_group_size=32,
        w_scale_group_size=(weight_group_rows, 32),
        split_k=split_k,
    )
