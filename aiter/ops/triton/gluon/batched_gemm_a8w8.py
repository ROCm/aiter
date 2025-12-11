# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import torch
import triton
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.gluon._gluon_kernels.batched_gemm_a8w8 import (
    _batched_gemm_a8w8_kernel,
)
from triton import language as tl

_LOGGER = AiterTritonLogger()


@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-BATCHED_GEMM-A8W8.json"
        print(f"fpath={fpath}")
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    if M + N >= 4096:
        return _get_config._config_dict["large"]
    else:
        return _get_config._config_dict["small"]


def batched_gemm_a8w8(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    splitK: Optional[int] = None,
    YQ: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    use_async_copy: bool = True,
):
    """
    Computes the matmul YQ[i] = XQ[i] x WQ[i]T and applies a conversion scale for every i in a given batch.
    Optionally, adds a bias to each result.

    The conversion scale for each matmul is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - XQ: Batch tensor XQ with shape (B, M, K).
    - WQ: Batch tensor WQ with shape (B, N, K).
    - X_scale: First scale batch tensor with shape (B, M, 1).
    - W_scale: Second scale batch tensor with shape (B, 1, N).
    - Bias: Bias batch tensor with shape (B, 1, N).
    - YQ: Output Matrix Y with shape (B, M, N). If this is none, then it's created by this API and returned as output

    Returns:
    - YQ: The output batch tensor with shape (B, M, N).
    """
    _LOGGER.info(
        f"BATCHED_GEMM_A8W8: x={tuple(XQ.shape)} w={tuple(WQ.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    # Make sure XQ and WQ are contiguous in memory
    XQ = XQ.contiguous()
    WQ = WQ.contiguous()

    # Check constraints.
    assert XQ.shape[0] == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert XQ.shape[2] == WQ.shape[2], "Incompatible K dimensions!!!"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    # Transpose N and K dimensions of WQ: (B, N, K) -> (B, K, N)
    WQ = WQ.transpose(1, 2)

    B = XQ.shape[0]
    M = XQ.shape[1]
    K = XQ.shape[2]
    N = WQ.shape[2]

    has_bias = bias is not None
    if YQ is None:
        YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)

    if config is None:
        config = _get_config(M, N, K)

    grid = lambda META: (  # noqa: E731
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _batched_gemm_a8w8_kernel[grid](
        XQ,
        WQ,
        YQ,
        x_scale,
        w_scale,
        bias,
        M,
        N,
        K,
        XQ.stride(0),
        XQ.stride(1),
        XQ.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0),
        YQ.stride(1),
        YQ.stride(2),
        x_scale.stride(0),
        w_scale.stride(0),
        bias.stride(0) if has_bias else 0,
        has_bias,
        **config,
    )

    return YQ
