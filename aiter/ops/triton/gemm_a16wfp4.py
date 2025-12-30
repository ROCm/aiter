# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.ops.triton._triton_kernels.gemm_a16wfp4 import (
    _gemm_a16wfp4_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.gemm_afp4wfp4 import (
    _gemm_afp4wfp4_reduce_kernel,
)
from aiter.ops.triton.gemm_afp4wfp4 import (
    get_splitk,
)
from aiter.jit.utils.torch_guard import torch_compile_guard


_LOGGER = AiterTritonLogger()


def gemm_a16wfp4_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    atomic_add: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
) -> torch.Tensor:
    if y is None:
        M, _ = x.shape
        N, _ = w.shape
        return torch.zeros((M, N), dtype=dtype, device=x.device)
    return y


@torch_compile_guard(gen_fake=gemm_a16wfp4_fake_tensor)
def gemm_a16wfp4_(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    atomic_add: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[str] = None,
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with BF16 activations and FP4 weights.

    Key parameters:
        x (torch.Tensor): BF16/FP16 input matrix X with shape (M, K).
            Quantized to MXFP4 on-the-fly during GEMM.
        w (torch.Tensor): FP4 E2M1 weight matrix W with shape (N, K//2).
        w_scales (torch.Tensor): E8M0 per-group scale for w with shape (N, K//32).
            One scale per 32 elements in K dimension.
        atomic_add (Optional[bool]): use atomic_add for reduction
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[str]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).

    Returns:
        y (torch.Tensor): Output with shape (M, N).
    """

    _LOGGER.info(
        f"GEMM_A16WFP4: x={tuple(x.shape)} w={tuple(w.shape)} w_scale={tuple(w_scales.shape)} "
    )

    assert arch_info.is_fp4_avail(), "MXFP4 is not available on your device"

    M, K = x.shape
    N, K = w.shape

    # inner kernel expects (K, N)
    w = w.T

    if config is None:
        config = _get_config(M, N, K)
    elif isinstance(config, str):
        config = deserialize_str(config)

    if y is None:
        if atomic_add:
            y = torch.zeros((M, N), dtype=dtype, device=x.device)
        else:
            y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config["NUM_KSPLIT"] > 1 and not atomic_add:
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
    config["BLOCK_SIZE_K"] = max(config["BLOCK_SIZE_K"], 64)

    if config["NUM_KSPLIT"] > 1 and not atomic_add:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        config["SPLITK_BLOCK_SIZE"] = 2 * K
        y_pp = None

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16wfp4_kernel[grid](
        x,
        w,
        y if y_pp is None else y_pp,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if y_pp is None else y_pp.stride(0),
        y.stride(0) if y_pp is None else y_pp.stride(1),
        y.stride(1) if y_pp is None else y_pp.stride(2),
        w_scales.stride(0),
        w_scales.stride(1),
        ATOMIC_ADD=atomic_add,
        **config,
    )

    if config["NUM_KSPLIT"] > 1 and not atomic_add:
        REDUCE_BLOCK_SIZE_M = 16
        REDUCE_BLOCK_SIZE_N = 64
        # TODO: Need to debug - REDUCE_BLOCK_SIZE_N=128 with fp32 partials fails
        # NOTE: REDUCE_BLOCK_SIZE_N=16 gives best perf with fp32 partials and
        # REDUCE_BLOCK_SIZE_N=128 gives best perf with bf16 partials
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


def gemm_a16wfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scales: torch.Tensor,
    atomic_add: Optional[bool] = False,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    config_hashable = serialize_dict(config) if config else None
    return gemm_a16wfp4_(x, w, w_scales, atomic_add, dtype, y, config_hashable)
