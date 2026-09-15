# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import (
    _gemm_a8w8_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import (
    get_scaled_dot_format_string,
    torch_to_triton_dtype,
)

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx950",)


def gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    backend: str = "triton",
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, SPLITK_BLOCK_SIZE).
        skip_reduce (Optional[bool]): Skip reduction of split-K partial results.
            Enables kernel fusion with downstream operations (FP8/FP4 quantization,
            RMSNorm). Returns shape (NUM_KSPLIT, M, N) instead of (M, N).

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    M, K = x.shape
    N, K = w.shape

    w = w.T

    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            get_arch() in _GLUON_SUPPORTED_ARCHS
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        assert x.dtype == w.dtype, "Input types must be the same"

    if config is None:
        if backend == "gluon":
            config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")
        else:
            config, _ = _get_config(M, N, K)

    if y is None and (config.get("NUM_KSPLIT", 1) == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if backend == "gluon":
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8 import (
            _gemm_a8w8_kernel as _gluon_gemm_a8w8_kernel,
        )

        _LOGGER.info(
            f"GEMM_A8W8 [gluon/{get_arch()}]: x={tuple(x.shape)} w={tuple(w.shape)}"
        )

        fp8_format = (
            None
            if x.dtype == torch.int8
            else get_scaled_dot_format_string(torch_to_triton_dtype[x.dtype])
        )
        grid = (
            triton.cdiv(M, config["BLOCK_SIZE_M"])
            * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        )
        _gluon_gemm_a8w8_kernel[grid](
            x,
            w,
            x_scale,
            w_scale,
            bias,
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
            bias is not None,
            NUM_XCDS=get_num_xcds(),
            NUM_WARPS=config["num_warps"],
            **config,
            FP8_FORMAT=fp8_format,
        )
        return y

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
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
        (bias is not None) and (config["NUM_KSPLIT"] == 1),
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            bias,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=bias is not None,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_reduce_kernel",
        )

    return y


def gemm_a8w8_preshuffle(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias,
    taking weights in a pre-shuffled layout for better memory access.

    Args:
        x (torch.Tensor): INT8/FP8 input matrix with shape (M, K).
        w (torch.Tensor): INT8/FP8 weight matrix pre-shuffled to (N*16, K//16),
            internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output with shape (M, N) in higher precision format.
    """
    assert (
        get_arch() in _GLUON_SUPPORTED_ARCHS
    ), f"gemm_a8w8_preshuffle requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8 import (
        _gemm_a8w8_preshuffled_kernel as _gluon_gemm_a8w8_preshuffled_kernel,
    )

    _LOGGER.info(
        f"GEMM_A8W8 PRESHUFFLE [gluon/{get_arch()}]: x={tuple(x.shape)} w={tuple(w.shape)}"
    )

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    if config is None:
        config, _ = get_gemm_config("GEMM-A8W8", M, N, K, backend="gluon")

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    assert (
        K % config["BLOCK_SIZE_K"] == 0
    ), "K must be multiple of BLOCK_SIZE_K for preshuffling"

    fp8_format = (
        None
        if x.dtype == torch.int8
        else get_scaled_dot_format_string(torch_to_triton_dtype[x.dtype])
    )
    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gluon_gemm_a8w8_preshuffled_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
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
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        NUM_WARPS=config["num_warps"],
        **config,
        FP8_FORMAT=fp8_format,
    )

    return y
