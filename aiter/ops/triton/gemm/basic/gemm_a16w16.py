# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _gemm_a16_w16_kernel,
    _gemm_a16w16_reduce_kernel,
    _get_config as _get_triton_config,
)
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.arch_info import get_arch

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """Check if gluon backend is available for the current GPU architecture."""
    try:
        arch = get_arch()
        return any(supported in arch for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:
        return False


def _resolve_backend(backend: Optional[str]) -> str:
    """Resolve backend selection: None -> auto-detect, else validate."""
    if backend is None:
        return "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"
    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
    return backend


def _gemm_a16w16_triton(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
):
    """Triton backend implementation of A16W16 GEMM."""
    _LOGGER.info(f"GEMM_A16W16 [triton]: x={tuple(x.shape)} w={tuple(w.shape)}")

    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, K = w.shape
    w = w.T

    if config is None:
        config, _ = _get_triton_config(M, N, K)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=y.device if y is not None else x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a16_w16_kernel[grid](
        x,
        w,
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
        activation=_get_activation_from_str(activation) if activation else "",
        use_activation=activation is not None,
        ADD_BIAS=(bias is not None),
        SKIP_REDUCE=skip_reduce,
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
        _gemm_a16w16_reduce_kernel[grid_reduce](
            bias,
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
            activation=_get_activation_from_str(activation) if activation else "",
            use_activation=activation is not None,
            ADD_BIAS=(bias is not None),
        )

    return y


def _gemm_a16w16_gluon(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    kernel_type: str = "basic",
):
    """Gluon backend implementation of A16W16 GEMM (gfx1250)."""
    from aiter.ops.triton._gluon_kernels.gemm.basic.gemm_a16w16_gfx1250 import (
        gemm_a16w16_gfx1250,
    )

    return gemm_a16w16_gfx1250(
        x,
        w,
        bias=bias,
        dtype=dtype,
        y=y,
        config=config,
        activation=activation,
        kernel_type=kernel_type,
    )


def gemm_a16w16(
    x,
    w,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    activation: Optional[str] = None,
    skip_reduce: Optional[bool] = False,
    kernel_type: str = "basic",
    backend: Optional[str] = None,
):
    """
    Computes 16 bit matrix multiplication Y = X @ W^T

    Dispatches to the triton or gluon backend based on the ``backend`` argument.
    When ``backend`` is ``None`` (default), gluon is used automatically on
    supported architectures (gfx1250) and triton everywhere else.

    Args:
        x (torch.Tensor): Input matrix with shape (M, K).
        w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters.
        activation (Optional[str]): Activation function ("gelu", "gelu_tanh", "silu",
            "silu_exp2", "relu").
        skip_reduce (Optional[bool]): [triton only] Skip reduction of split-K partial
            results. Returns shape (NUM_KSPLIT, M, N) instead of (M, N).
        kernel_type (str): [gluon only] Kernel variant ("basic", "warp_priority",
            "k_subtiling").
        backend (Optional[str]): "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (M, N) or (NUM_KSPLIT, M, N) if skip_reduce=True.
    """
    resolved = _resolve_backend(backend)

    if resolved == "gluon":
        return _gemm_a16w16_gluon(
            x,
            w,
            bias=bias,
            dtype=dtype,
            y=y,
            config=config,
            activation=activation,
            kernel_type=kernel_type,
        )
    else:
        return _gemm_a16w16_triton(
            x,
            w,
            bias=bias,
            dtype=dtype,
            y=y,
            config=config,
            activation=activation,
            skip_reduce=skip_reduce,
        )
