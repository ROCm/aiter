# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Union
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.fused_gemm_a8w8_blockscale_mul_add import (
    _fused_gemm_a8w8_blockscale_mul_add_kernel,
    _fused_gemm_a8w8_blockscale_mul_add_reduce_kernel,
    _get_config,
)

_LOGGER = AiterTritonLogger()

global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False


def set_use_gemm_splitk_bf16(value: bool):
    global _USE_GEMM_SPLITK_BF16
    _USE_GEMM_SPLITK_BF16 = value


def get_splitk(K: int, BLOCK_SIZE_K: int, NUM_KSPLIT: int):
    # heuristics for make "EVEN_K == True" as much as possible
    NUM_KSPLIT_STEP = 2
    BLOCK_SIZE_K_STEP = 2
    SPLITK_BLOCK_SIZE = (
        triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
    )
    while NUM_KSPLIT > 1 and BLOCK_SIZE_K > 16:
        if (
            K % (SPLITK_BLOCK_SIZE // 2) == 0
            and SPLITK_BLOCK_SIZE % BLOCK_SIZE_K == 0
            and K % (BLOCK_SIZE_K // 2) == 0
        ):
            break
        elif K % (SPLITK_BLOCK_SIZE // 2) != 0 and NUM_KSPLIT > 1:
            NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
        elif SPLITK_BLOCK_SIZE % BLOCK_SIZE_K != 0:
            if NUM_KSPLIT > 1:
                NUM_KSPLIT = NUM_KSPLIT // NUM_KSPLIT_STEP
            elif BLOCK_SIZE_K > 16:
                BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        elif K % (BLOCK_SIZE_K // 2) != 0 and BLOCK_SIZE_K > 16:
            BLOCK_SIZE_K = BLOCK_SIZE_K // BLOCK_SIZE_K_STEP
        else:
            break

        SPLITK_BLOCK_SIZE = (
            triton.cdiv((2 * triton.cdiv(K, NUM_KSPLIT)), BLOCK_SIZE_K) * BLOCK_SIZE_K
        )

    return SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT


def fused_gemm_a8w8_blockscale_mul_add(
    x,
    w,
    x_scales,
    w_scales,
    a: Union[torch.Tensor, float, int],
    b: Union[torch.Tensor, float, int],
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    fuse_type: Optional[int] = 0,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP8 activations and FP8 weights.
    if fuse_type == 0:
        the final output = a * Y + b
    elif fuse_type == 1
        the final output = a * b + Y

    Args:
        x (torch.Tensor): FP8 input matrix with shape (M, K).
        w (torch.Tensor): FP8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Block-wise scale for x with shape (M, scale_k).
            scale_k = ceil(K / scale_block_size_k).
        w_scale (torch.Tensor): Block-wise scale for w with shape (scale_n, scale_k).
            scale_n = ceil(N / scale_block_size_n).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    _LOGGER.info(
        f"FUSED_GEMM_A8W8_BLOCKSCALE_MUL_ADD: x.shape={tuple(x.shape)} w.shape={tuple(w.shape)} x_scale={tuple(x_scales.shape)} w_scale={tuple(w_scales.shape)} "
    )

    if isinstance(a, float) or isinstance(a, int):
        IS_A_SCALAR = True
        IS_A_TENSOR = False
    elif isinstance(a, torch.Tensor) and a.is_contiguous():
        IS_A_TENSOR = True
        if a.numel() == 1:
            IS_A_SCALAR = True
        else:
            IS_A_SCALAR = False
    if isinstance(b, float) or isinstance(b, int):
        IS_B_SCALAR = True
        IS_B_TENSOR = False
    elif isinstance(b, torch.Tensor) and b.is_contiguous():
        IS_B_TENSOR = True
        if b.numel() == 1:
            IS_B_SCALAR = True
        else:
            IS_B_SCALAR = False

    M, K = x.shape
    N, K = w.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w
    w = w.T
    w_scales = w_scales.T  # (scale_k, scale_n)

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        y_pp = None

    # If block size is greater than split k size, shrink the block size
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(
        config["BLOCK_SIZE_K"], 16
    )  # minimum block size is 16 for perf

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(
        triton.cdiv(K, w_scales.shape[0])
    )  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(
        triton.cdiv(N, w_scales.shape[1])
    )  # scale_block_size_n

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _fused_gemm_a8w8_blockscale_mul_add_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        a,
        b,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        0 if IS_A_SCALAR else a.stride(0),
        0 if IS_A_SCALAR else a.stride(1),
        0 if IS_B_SCALAR else b.stride(0),
        0 if IS_B_SCALAR else b.stride(1),
        IS_A_SCALAR=IS_A_SCALAR,
        IS_B_SCALAR=IS_B_SCALAR,
        IS_A_TENSOR=IS_A_TENSOR,
        IS_B_TENSOR=IS_B_TENSOR,
        FUSE_TYPE=fuse_type,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_a8w8_blockscale_mul_add_reduce_kernel[grid_reduce](
            y_pp,
            y,
            a,
            b,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            0 if IS_A_SCALAR else a.stride(0),
            0 if IS_A_SCALAR else a.stride(1),
            0 if IS_B_SCALAR else b.stride(0),
            0 if IS_B_SCALAR else b.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            IS_A_SCALAR=IS_A_SCALAR,
            IS_B_SCALAR=IS_B_SCALAR,
            IS_A_TENSOR=IS_A_TENSOR,
            IS_B_TENSOR=IS_B_TENSOR,
            FUSE_TYPE=fuse_type,
        )

    return y
