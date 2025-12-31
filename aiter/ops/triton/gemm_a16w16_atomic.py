# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton._triton_kernels.gemm_a16w16_atomic import (
    _gemm_a16_w16_atomic_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard

_LOGGER = AiterTritonLogger()


def gemm_a16w16_atomic_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
) -> torch.Tensor:
    if y is None:
        M, _ = x.shape
        _, N = w.shape
        return torch.zeros((M, N), dtype=dtype, device=x.device)
    return y


@torch_compile_guard(gen_fake=gemm_a16w16_atomic_fake_tensor)
def gemm_a16w16_atomic_(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
) -> torch.Tensor:
    _LOGGER.info(
        f"GEMM_A16W16_ATOMIC: x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)} "
    )

    w = w.T

    M, K = x.shape
    K, N = w.shape

    if config is None:
        config, _ = _get_config(M, N, K)
    else:
        config = deserialize_str(config)

    # For compatability reasons, these keys may not exist in the config
    # TODO: This needs to be embedded in the configs later
    if "NUM_KSPLIT" not in config:
        config["NUM_KSPLIT"] = 1
    if "cache_modifier" not in config:
        config["cache_modifier"] = ""

    if y is None:
        # atomic add requires 0 tensor
        if config["NUM_KSPLIT"] == 1:
            y = torch.empty((M, N), dtype=dtype, device=x.device)
        else:
            y = torch.zeros((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"])
        * META["NUM_KSPLIT"],
    )
    # NOTE: if k split doesnt divide K evenly, this will waste compute
    SPLITK_BLOCK_SIZE = triton.cdiv(K, config["NUM_KSPLIT"])
    config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
    _gemm_a16_w16_atomic_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        **config,
    )

    return y


def gemm_a16w16_atomic(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Computes 16 bit matrix multiplication Y = X @ W^T using atomic operations for split-K reduction.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
            Note: BF16 atomic aggregation may have slight precision loss.
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
            Must be zero-initialized for split-K (NUM_KSPLIT > 1).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, cache_modifier).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    config_hashable = serialize_dict(config) if config else None
    return gemm_a16w16_atomic_(x, w, dtype, y, config_hashable)
