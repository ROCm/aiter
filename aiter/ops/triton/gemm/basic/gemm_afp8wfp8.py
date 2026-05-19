# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp8wfp8 import (
    _gemm_afp8wfp8_kernel,
    _gemm_afp8wfp8_preshuffle_kernel,
    _get_config,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
    _gemm_a8w8_blockscale_reduce_kernel,
)

# -----------------------------------------------------------------------------
# Tuned-config lookup for gemm_afp8wfp8_preshuffle.
#
# Configs live under aiter.ops.triton.configs.gemm using the standard aiter
# naming: gfx{arch}-GEMM-MXFP8-PRESHUFFLE-N={N}-K={K}.json, keyed by
# M_LEQ_x / M_GEQ_x / any per STANDARD_M_BOUNDS. A generic fallback
# gfx{arch}-GEMM-MXFP8-PRESHUFFLE.json covers untuned (N, K) shapes.
# -----------------------------------------------------------------------------

# Shapes that have OOM'd at runtime with the tuned config. Future calls
# bypass the tuned lookup for these and use _DEFAULT_CONFIG. The tuned
# benches don't run under HIP-graph capture, so they can pick configs that
# look fine in isolation but blow LDS once captured. This cache survives
# across requests in the same process.
_OOM_SHAPES: set = set()


def _mark_oom(M: int, N: int, K: int):
    _OOM_SHAPES.add((M, N, K))


def gemm_afp8wfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    skip_reduce: Optional[bool] = False,
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with MXFP8 activations and FP8
    weights (1x32 e8m0 act scales, 128x128 e8m0 weight scales).

    Args:
        x: FP8 e4m3 (or uint8 view) input matrix with shape (M, K).
        w: FP8 e4m3 (or uint8 view) weight matrix with shape (N, K) — internally
           transposed to (K, N) before the kernel call.
        x_scales: e8m0 (uint8) per-group scale for x with shape (M, K // 32).
        w_scales: e8m0 (uint8) per-block scale for w with shape (N // 128, K // 128).
        dtype: Output dtype (BF16 or FP16). Default bf16.
        y: Optional pre-allocated output tensor with shape (M, N).
        config: Optional kernel-tuning dict. If None uses defaults.

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: x has K={K}, w has K={K_w}"

    # Transpose w to (K, N) for the kernel.
    w_t = w.T

    # tl.dot_scaled with format "e4m3" expects uint8-typed operands; reinterpret
    # the FP8 buffers as uint8 (bit-identical view).
    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    if w_t.dtype != torch.uint8:
        w_t = w_t.view(torch.uint8)

    if config is None:
        config, _ = _get_config(M, N, K)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

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

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _gemm_afp8wfp8_kernel[grid](
        x,
        w_t,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
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

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
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


def gemm_afp8wfp8_preshuffle(
    x: torch.Tensor,
    w_shuffled: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
    skip_reduce: Optional[bool] = False,
) -> torch.Tensor:
    """
    Preshuffle variant of gemm_afp8wfp8. The weight tensor has already been
    permuted via aiter.ops.shuffle.shuffle_weight(..., layout=(16, 16)). Scales
    are left unshuffled in the compact 128x128 layout.

    Args:
        x: FP8 e4m3 activations with shape (M, K).
        w_shuffled: FP8 e4m3 weights, shuffled in place to (N, K) storage
            (same total bytes; bytes rearranged for the kernel's read pattern).
        x_scales: e8m0 (uint8) per-token scale with shape (M, K // 32).
        w_scales: e8m0 (uint8) per-block weight scale with shape (N // 128, K // 128).
        dtype: Output dtype.
        y: Optional pre-allocated output (M, N).
        config: Optional kernel-tuning dict.

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    M, K = x.shape
    N, K_w = w_shuffled.shape
    assert K == K_w, f"K mismatch: x={K}, w={K_w}"
    assert N % 16 == 0, f"N must be divisible by 16 for preshuffle, got {N}"

    # The kernel expects to address the shuffled tensor as (N//16, K*16).
    w_view = w_shuffled.view(N // 16, K * 16)

    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    if w_view.dtype != torch.uint8:
        w_view = w_view.view(torch.uint8)

    if config is None:
        config, _ = _get_config(M, N, K, shuffle=True)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

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

    grid = lambda META: (  # noqa: E731
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_afp8wfp8_preshuffle_kernel[grid](
        x,
        w_view,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_view.stride(0),
        w_view.stride(1),
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

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
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
