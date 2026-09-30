# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    X: torch.Tensor,
    WQ: torch.Tensor,
    w_scale: torch.Tensor,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype | None = torch.bfloat16,
    splitK: int | None = None,
    YQ: torch.Tensor | None = None,
    transpose_bm: bool | None = False,
    transpose_bm_in: bool | None = False,
    config: dict | None = None,
    backend: str | None = None,
):
    """
    Computes batched 8 bit matrix multiplication Y[i] = X[i] @ W[i]^T with active activation quantization.
    X is quantized to INT8 during computation using per-token grouped quantization.
    W is pre-quantized INT8 with per-batch-element scaling.

    Args:
        X (torch.Tensor): Higher precision input batch with shape (B, M, K) or (M, B, K) if transpose_bm_in=True.
            Quantized to INT8 on-the-fly during GEMM.
        WQ (torch.Tensor): Pre-quantized INT8 weight batch with shape (B, N, K), internally transposed.
        w_scale (torch.Tensor): Per-batch scale for WQ with shape (1,).
        group_size (int): Group size for per-token grouped quantization of X. Must be power of 2.
        bias (Optional[torch.Tensor]): Bias batch with shape (B, 1, N).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        splitK (Optional[int]): Not supported. Must be None.
        YQ (Optional[torch.Tensor]): Pre-allocated output tensor with shape (B, M, N) or (M, B, N) if transpose_bm=True.
        transpose_bm (Optional[bool]): Transpose batch and M dimensions in output.
        transpose_bm_in (Optional[bool]): Transpose batch and M dimensions in input.
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M).
        backend (Optional[str]): "triton", "gluon", or None (auto: gluon on gfx1250
            when supported). The gluon path (gfx1250) keeps X in bf16 and upcasts WQ
            to bf16 in registers instead of quantizing X per token-group, so it is
            at least as accurate; it does not support bias.

    Returns:
        torch.Tensor: Output batch with shape (B, M, N) or (M, B, N) if transpose_bm=True.
    """

    # Check constraints.
    if not transpose_bm_in:
        B = X.shape[0]
        M = X.shape[1]
    else:
        M = X.shape[0]
        B = X.shape[1]
    K = X.shape[2]
    N = WQ.shape[1]

    assert B == WQ.shape[0], "Incompatible Batch dimensions!!!"
    assert K == WQ.shape[2], "Incompatible K dimensions!!!"
    assert (
        triton.next_power_of_2(group_size) == group_size
    ), "group_size mush be power of 2"
    assert dtype in [
        torch.bfloat16,
        torch.float16,
    ], f"Output {dtype=} is currently not supported in batched_gemm_a8w8"
    assert splitK is None, "Currently, there isn't any support for splitK on Triton"

    if backend is None:
        backend = (
            "gluon"
            if _is_gluon_available()
            and bias is None
            and X.stride(2) == 1
            and WQ.stride(2) == 1
            and X.dtype in (torch.bfloat16, torch.float16)
            else "triton"
        )
    if backend == "gluon":
        from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import (
            batched_gemm_a16w8,
        )

        assert bias is None, "gluon backend does not support bias"
        X_bmk = X.transpose(0, 1) if transpose_bm_in else X
        return batched_gemm_a16w8(
            X_bmk, WQ, w_scale, dtype=dtype, YQ=YQ, transpose_bm=transpose_bm
        )
    assert backend == "triton", f"Unknown backend '{backend}'"

    WQ = WQ.transpose(1, 2)

    has_bias = bias is not None
    if YQ is None:
        if transpose_bm:
            YQ = torch.empty((M, B, N), dtype=dtype, device=X.device)
        else:
            YQ = torch.empty((B, M, N), dtype=dtype, device=X.device)
    else:
        if transpose_bm:
            assert (
                YQ.shape[0] == M and YQ.shape[1] == B and YQ.shape[2] == N
            ), "Output dimension error"
        else:
            assert (
                YQ.shape[0] == B and YQ.shape[1] == M and YQ.shape[2] == N
            ), "Output dimension error"

    if config is None:
        config, _ = _get_config(M, N, K)
    config["BLOCK_SIZE_K"] = group_size

    grid = lambda META: (
        B,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    DTYPE_MAX = (
        torch.finfo(WQ.dtype).max
        if torch.is_floating_point(WQ)
        else torch.iinfo(WQ.dtype).max
    )

    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel[
        grid
    ](
        X,
        WQ,
        YQ,
        w_scale,
        bias,
        M,
        N,
        K,
        X.stride(0) if not transpose_bm_in else X.stride(1),
        X.stride(1) if not transpose_bm_in else X.stride(0),
        X.stride(2),
        WQ.stride(0),
        WQ.stride(1),
        WQ.stride(2),
        YQ.stride(0) if not transpose_bm else YQ.stride(1),
        YQ.stride(1) if not transpose_bm else YQ.stride(0),
        YQ.stride(2),
        bias.stride(0) if has_bias else 0,
        has_bias,
        DTYPE_MAX=DTYPE_MAX,
        DTYPE_MIN=-DTYPE_MAX,
        **config,
    )

    return YQ
