# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel,
    _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_splitk_kernel,
    _batched_gemm_a8w8_splitk_reduce_per_token_quant_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

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
    y_scale: torch.Tensor | None = None,
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
        config (Optional[dict]): Kernel tuning parameters for the selected backend
            (triton: BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M, ...; gluon: see
            batched_gemm_a16w8). Passing a config with backend=None keeps the
            triton path, since the config is interpreted as a triton config.
        backend (Optional[str]): "triton", "gluon", or None (auto). Auto selects
            gluon on gfx1250 when bias, config and y_scale are None and the inputs fit
            batched_gemm_a16w8 (bf16/fp16 X, fp8 WQ, both K-contiguous);
            otherwise triton. The gluon path keeps X in bf16 and upcasts
            WQ to bf16 in registers instead of quantizing X per token-group, so
            its numerics differ from the triton path (it is closer to the exact
            bf16 product); group_size does not apply to it and bias and y_scale are unsupported.
        y_scale (Optional[torch.Tensor]): Pre-allocated scale tensor with shape (M, 1) for a per-token quantized FP8 YQ.

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

    X_bmk = X.transpose(0, 1) if transpose_bm_in else X
    if backend is None:
        backend = "triton"
        if (
            _is_gluon_available()
            and bias is None
            and config is None
            and y_scale is None
        ):
            from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import (
                is_batched_gemm_a16w8_supported,
            )

            if is_batched_gemm_a16w8_supported(X_bmk, WQ):
                backend = "gluon"
    _LOGGER.info(
        "BATCHED_GEMM_A8W8_A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT "
        "[%s]: x=%s w=%s",
        backend,
        tuple(X.shape),
        tuple(WQ.shape),
    )
    if backend == "gluon":
        assert _is_gluon_available(), f"gluon backend requires {_GLUON_SUPPORTED_ARCHS}"
        assert bias is None, "gluon backend does not support bias"
        assert y_scale is None, "gluon backend does not support y_scale"
        from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import (
            batched_gemm_a16w8,
        )

        return batched_gemm_a16w8(
            X_bmk,
            WQ,
            w_scale,
            dtype=dtype,
            YQ=YQ,
            transpose_bm=transpose_bm,
            config=config,
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
        if y_scale is None:
            config, _ = _get_config(M, N, K)
        else:
            config, _ = _get_config(M, N, K, B, y_per_token_quant=True)
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

    if y_scale is not None:
        assert (
            transpose_bm and not has_bias
        ), "y_scale (per-token quant of Y) needs transpose_bm=True and no bias"
        assert (
            torch.is_floating_point(YQ) and YQ.element_size() == 1
        ), f"y_scale needs a pre-allocated FP8 YQ, got {YQ.dtype=}"
        assert (
            YQ.stride(2) == 1 and YQ.stride(1) == N
        ), "YQ must hold the B * N values of a token contiguously"
        assert (
            y_scale.shape == (M, 1) and y_scale.dtype == torch.float32
        ), "y_scale must be FP32 with shape (M, 1)"

        # Split K in whole quant groups and launch only non-empty splits.
        num_k_groups = triton.cdiv(K, group_size)
        splitk_block_size = triton.cdiv(num_k_groups, config["NUM_KSPLIT"]) * group_size
        num_ksplit = triton.cdiv(K, splitk_block_size)
        y_pp = torch.empty((num_ksplit, M, B, N), dtype=torch.float32, device=X.device)

        grid_splitk = (
            B,
            num_ksplit
            * triton.cdiv(M, config["BLOCK_SIZE_M"])
            * triton.cdiv(N, config["BLOCK_SIZE_N"]),
        )
        _batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_splitk_kernel[
            grid_splitk
        ](
            X,
            WQ,
            y_pp,
            M,
            N,
            K,
            X.stride(0) if not transpose_bm_in else X.stride(1),
            X.stride(1) if not transpose_bm_in else X.stride(0),
            X.stride(2),
            WQ.stride(0),
            WQ.stride(1),
            WQ.stride(2),
            y_pp.stride(0),
            y_pp.stride(2),
            y_pp.stride(1),
            y_pp.stride(3),
            DTYPE_MAX=DTYPE_MAX,
            DTYPE_MIN=-DTYPE_MAX,
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=group_size,
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            NUM_KSPLIT=num_ksplit,
            SPLITK_BLOCK_SIZE=splitk_block_size,
            cache_modifier=config["cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        )

        Y_DTYPE_MAX = torch.finfo(YQ.dtype).max
        grid_reduce = (triton.cdiv(M, config["REDUCE_BLOCK_SIZE_M"]),)
        _batched_gemm_a8w8_splitk_reduce_per_token_quant_kernel[grid_reduce](
            y_pp,
            YQ,
            y_scale,
            w_scale,
            M,
            B * N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(3),
            YQ.stride(0),
            YQ.stride(2),
            y_scale.stride(0),
            DTYPE_MAX=Y_DTYPE_MAX,
            DTYPE_MIN=-Y_DTYPE_MAX,
            BLOCK_SIZE_M=config["REDUCE_BLOCK_SIZE_M"],
            BLOCK_SIZE_N=triton.next_power_of_2(B * N),
            NUM_KSPLIT=num_ksplit,
            num_warps=config["REDUCE_num_warps"],
        )

        return YQ

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
