# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._triton_kernels.fused_gemm_afp4wfp4_a16w16 import (
    _fused_gemm_afp4wfp4_a16w16_kernel,
    _fused_gemm_afp4wfp4_preshuffle_a16w16_kernel,
    _fused_gemm_afp4wfp4_a16w16_reduce_kernel,
    _get_config,
)
from aiter.ops.triton.gemm_afp4wfp4 import get_splitk
from .utils.core import AITER_TRITON_CONFIGS_PATH

import os
from aiter.utility.triton.triton_metadata_redirect import AOTMetadataContext

_LOGGER = AiterTritonLogger()


def fused_gemm_afp4wfp4_a16w16(
    x_fp4: torch.Tensor,
    w_fp4: torch.Tensor,
    x_fp4_scale: torch.Tensor,
    w_fp4_scale: torch.Tensor,
    x_bf16: torch.Tensor,
    w_bf16: torch.Tensor,
    is_fp4_preshuffled: bool = True,
    bias_fp4: Optional[torch.Tensor] = None,
    bias_bf16: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y_fp4: Optional[torch.Tensor] = None,
    y_bf16: Optional[torch.Tensor] = None,
    skip_reduce: Optional[bool] = False,
    config: Optional[dict] = None,
    use_aot: Optional[bool] = True,
):
    """
    Computes the 8 bit matmul Y = X x WT + B using the block-scale quantization approach for x_fp4 and w_fp4.
    Computes the 16 bit matmul Y = X x WT + B for x_bf16 and w_bf16

    This fusion is primarily aiming for fusing the gate up-projections and MOE gating:
        gate up-projections: (M, K) x (2N, K) = (M, 2N)
        MOE gating: (M, K) x (N, K) + (N, ) = (M, N)

    Key parameters:
    - x_fp4: Matrix X with shape (M, K).
    - w_fp4: Matrix W with shape (N_fp4, K).
    - x_fp4_scale: Scale tensor for X with shape (M, K // 32)
    - w_fp4_scale: Scale tensor for W with shape (N, K // 32)
    - x_bf16: Matrix X with shape (M, K).
    - w_bf16: Matrix W with shape (N_bf16, K).

    Note: M, N, K must be identical for x_fp4 and x_bf16, but the N-dim fow w_fp4 and w_bf16 can be different

    Returns:
    - y_fp4: The output matrix with shape (M, N_fp4).
    - y_bf16: The output matrix with shape (M, N_bf16).

    """
    _LOGGER.info(
        f"FUSED_GEMM_A8W8_BLOCKSCALE_A16W16: x_fp4={tuple(x_fp4.shape)} w_fp4={tuple(w_fp4.shape)} x_fp4_scale={tuple(x_fp4_scale.shape)} w_fp4_scale={tuple(w_fp4_scale.shape)} x_bf16={tuple(x_bf16.shape)} w_bf16={tuple(w_bf16.shape)}"
    )

    M, K = x_fp4.shape
    N_fp4, K = w_fp4.shape
    if is_fp4_preshuffled:
        N_fp4 = N_fp4 * 16
        K = K // 16
    M, _ = x_bf16.shape
    N_bf16, _ = w_bf16.shape

    # Check constraints.
    assert (
        x_fp4.shape[0] == x_bf16.shape[0]
    ), "M-dim should be identical for x_fp4 and x_bf16"
    assert (
        x_fp4.shape[1] * 2 == x_bf16.shape[1]
    ), "K-dim should be identical for x_fp4 and x_bf16"
    if is_fp4_preshuffled:
        assert x_fp4.shape[1] == w_fp4.shape[1] // 16, "Incompatible dimensions!!!"
    else:
        assert x_fp4.shape[1] == w_fp4.shape[1], "Incompatible dimensions!!!"
    assert x_bf16.shape[1] == w_bf16.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    if not is_fp4_preshuffled:
        w_fp4 = w_fp4.T
    w_bf16 = w_bf16.T

    if config is None:
        config = _get_config(M, N_fp4, N_bf16, K, is_fp4_preshuffled)

    if y_fp4 is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y_fp4 = torch.empty((M, N_fp4), dtype=dtype, device=x_fp4.device)

    if y_bf16 is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y_bf16 = torch.empty((M, N_bf16), dtype=dtype, device=x_bf16.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(K, config["NUM_KSPLIT"])
    if config["NUM_KSPLIT"] > 1:
        SPLITK_BLOCK_SIZE, BLOCK_SIZE_K, NUM_KSPLIT = get_splitk(
            K, config["BLOCK_SIZE_K"], config["NUM_KSPLIT"]
        )

        config["SPLITK_BLOCK_SIZE"] = SPLITK_BLOCK_SIZE
        config["BLOCK_SIZE_K"] = BLOCK_SIZE_K
        config["NUM_KSPLIT"] = NUM_KSPLIT

        y_fp4_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_fp4),
            dtype=torch.float32,
            device=y_fp4.device if y_fp4 is not None else x_fp4.device,
        )
        y_bf16_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N_bf16),
            dtype=torch.float32,
            device=y_bf16.device if y_bf16 is not None else x_bf16.device,
        )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_fp4_pp = None
        y_bf16_pp = None

    if config["BLOCK_SIZE_K"] >= 2 * K:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
        config["SPLITK_BLOCK_SIZE"] = 2 * K

    config["BLOCK_SIZE_N"] = max(config["BLOCK_SIZE_N"], 32)
    if is_fp4_preshuffled:
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
            * (
                triton.cdiv(N_fp4, META["BLOCK_SIZE_N"])
                + triton.cdiv(N_bf16, META["BLOCK_SIZE_N"])
            )
        ),
    )
    selected_kernel = (
        _fused_gemm_afp4wfp4_preshuffle_a16w16_kernel
        if is_fp4_preshuffled
        else _fused_gemm_afp4wfp4_a16w16_kernel
    )

    def selected_kernel_wrapper():
        selected_kernel[grid](
            x_fp4,
            w_fp4,
            bias_fp4,
            x_fp4_scale,
            w_fp4_scale,
            y_fp4 if config["NUM_KSPLIT"] == 1 else y_fp4_pp,
            x_bf16,
            w_bf16,
            bias_bf16,
            y_bf16 if config["NUM_KSPLIT"] == 1 else y_bf16_pp,
            M,
            N_fp4,
            N_bf16,
            K,
            x_fp4.stride(0),
            x_fp4.stride(1),
            w_fp4.stride(0),
            w_fp4.stride(1),
            x_fp4_scale.stride(0),
            x_fp4_scale.stride(1),
            w_fp4_scale.stride(0),
            w_fp4_scale.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_fp4_pp.stride(0),
            y_fp4.stride(0) if config["NUM_KSPLIT"] == 1 else y_fp4_pp.stride(1),
            y_fp4.stride(1) if config["NUM_KSPLIT"] == 1 else y_fp4_pp.stride(2),
            x_bf16.stride(0),
            x_bf16.stride(1),
            w_bf16.stride(0),
            w_bf16.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(0),
            y_bf16.stride(0) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(1),
            y_bf16.stride(1) if config["NUM_KSPLIT"] == 1 else y_bf16_pp.stride(2),
            ADD_BIAS_FP4=(bias_fp4 is not None),
            ADD_BIAS_BF16=(bias_bf16 is not None),
            SKIP_REDUCE=skip_reduce,
            **config,
        )

    M_POW2 = triton.next_power_of_2(M)
    if M < 32 and M_POW2 > 16:
        M_POW2 = 16
    metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/gemm/aot/{selected_kernel.fn.__name__}_M={M_POW2}-N4={N_fp4}-N16={N_bf16}-K={K*2}"
    if use_aot and os.path.exists(metadata_pth):
        with AOTMetadataContext(
            selected_kernel.fn.__name__,
            f"{metadata_pth}",
        ):
            selected_kernel_wrapper()
    else:
        selected_kernel_wrapper()

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_fp4_pp, y_bf16_pp
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N_fp4, REDUCE_BLOCK_SIZE_N)
            + triton.cdiv(N_bf16, REDUCE_BLOCK_SIZE_N),
        )
        _fused_gemm_afp4wfp4_a16w16_reduce_kernel[grid_reduce](
            bias_fp4,
            y_fp4_pp,
            y_fp4,
            bias_bf16,
            y_bf16_pp,
            y_bf16,
            M,
            N_fp4,
            N_bf16,
            y_fp4_pp.stride(0),
            y_fp4_pp.stride(1),
            y_fp4_pp.stride(2),
            y_fp4.stride(0),
            y_fp4.stride(1),
            y_bf16_pp.stride(0),
            y_bf16_pp.stride(1),
            y_bf16_pp.stride(2),
            y_bf16.stride(0),
            y_bf16.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS_fp4=(bias_fp4 is not None),
            ADD_BIAS_BF16=(bias_bf16 is not None),
        )

    return y_fp4, y_bf16
