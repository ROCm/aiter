# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_kernel,
    _gemm_afp4wfp4_kernel_preshuffle_scales,
    _gemm_afp4wfp4_preshuffle_kernel,
    _gemm_afp4wfp4_reduce_kernel,
    _get_config,
)
from .utils.core import AITER_TRITON_CONFIGS_PATH

import os
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext

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


def gemm_afp4wfp4(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    skip_reduce: Optional[bool] = False,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K).
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K), internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M, K//32).
            One scale per 32 elements in K dimension.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_AFPWFP4: x.shape={tuple(x.shape)} w.shape={tuple(w.shape)} x_scale={tuple(x_scales.shape)} w_scale={tuple(w_scales.shape)} "
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # Transpose w
    w = w.T

    if config is None:
        config = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        config["NUM_KSPLIT"] = 1
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 128)

    if config["NUM_KSPLIT"] > 1:
        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=x.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _gemm_afp4wfp4_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
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
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y


def gemm_afp4wfp4_preshuffled_scales(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights using preshuffled scales.
    Scales are arranged with M/N dimension grouped by 32 instead of K dimension.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K). M >= 32 required.
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N, K), internally transposed.
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M//32, K).
            Groups of 32 rows in M dimension share K scales.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N//32, K).
            Groups of 32 rows in N dimension share K scales.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # Transpose w
    w = w.T

    assert M >= 32, f"M >= 32 is required, but got {M=}"

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _gemm_afp4wfp4_kernel_preshuffled_scales[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
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
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y


def gemm_afp4wfp4_preshuffle(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    use_aot: Optional[bool] = True,
):
    """
    Computes matrix multiplication Y = X @ W^T with FP4 activations and FP4 weights using preshuffled weight scales.
    Weight matrix and scales are stored in optimized layout for improved performance.

    Args:
        x (torch.Tensor): FP4 E2M1 input matrix with shape (M, K).
        w (torch.Tensor): FP4 E2M1 weight matrix with shape (N//16, K*16), internally transposed.
            Preshuffled layout: logical shape after unpacking is (N, K).
        x_scales (torch.Tensor): E8M0 per-group scale for x with shape (M//32, K) if M >= 32,
            or (M, K//32) if M < 32.
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N//32, K).
            Groups of 32 rows in N dimension share K scales.
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        use_aot (Optional[bool]): Enable ahead-of-time compilation metadata.

    Returns:
        torch.Tensor: Output with shape (M, N).
    """

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K, True)

    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        if _USE_GEMM_SPLITK_BF16:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=y.dtype, device=y.device
            )
        else:
            y_pp = torch.empty(
                (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
            )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)
    if M < 32:
        assert (
            config["BLOCK_SIZE_M"] <= 16
        ), "for M < 32, BLOCK_SIZE_M must be 16 or less as x_scale are assumed to be un-shuffled"
    else:
        assert (
            config["BLOCK_SIZE_M"] >= 32
        ), "for M >= 32, BLOCK_SIZE_M must be 32 or more as x_scale are assumed to be preshuffled"

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    M_POW2 = triton.next_power_of_2(M)
    if M < 32 and M_POW2 > 16:
        M_POW2 = 16
    metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/gemm/aot/{_gemm_afp4wfp4_preshuffle_kernel.fn.__name__}_M={M_POW2}-N={N}-K={K*2}"
    if use_aot and os.path.exists(metadata_pth):
        with AOTMetadataContext(
            _gemm_afp4wfp4_preshuffle_kernel.fn.__name__,
            f"{metadata_pth}",
        ):
            _gemm_afp4wfp4_preshuffle_kernel[grid](
                x,
                w,
                y if config["NUM_KSPLIT"] == 1 else y_pp,
                x_scales,
                w_scales,
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
                **config,
            )
    else:
        _gemm_afp4wfp4_preshuffle_kernel[grid](
            x,
            w,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scales,
            w_scales,
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
            **config,
        )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 16
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
        REDUCE_BLOCK_SIZE_N = 128 if _USE_GEMM_SPLITK_BF16 else 64
        ACTUAL_KSPLIT = triton.cdiv(K, (config["SPLITK_BLOCK_SIZE"] // 2))

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_afp4wfp4_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y


def gemm_afp4wfp4_preshuffled_weight_scales(
    x,
    w,
    x_scales,
    w_scales,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    use_aot: Optional[bool] = True,
):
    _LOGGER.info(
        "gemm_afp4wfp4_preshuffled_weight_scales will be deprecated in future AITER release, please switch to gemm_afp4wfp4_preshuffle"
    )
    return gemm_afp4wfp4_preshuffle(x, w, x_scales, w_scales, dtype, y, config, use_aot)
