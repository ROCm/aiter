# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.basic.gemm_mxfp8 import (
    _gemm_mxfp8_kernel,
    _gemm_mxfp8_preshuffle_kernel,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config


_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8,
    "NUM_KSPLIT": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": "",
}


def _get_default_config() -> dict:
    return dict(_DEFAULT_CONFIG)


# -----------------------------------------------------------------------------
# Tuned-config lookup for gemm_mxfp8_preshuffle.
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


def _get_config(M: int, N: int, K: int) -> dict:
    """Load the best tuned config for (M, N, K) via the standard aiter JSON
    config mechanism. Falls back to the generic fallback file or to
    _DEFAULT_CONFIG if no JSON is available."""
    try:
        config, _ = get_gemm_config("GEMM-MXFP8-PRESHUFFLE", M, N, K)
    except (AssertionError, KeyError, FileNotFoundError):
        config = _get_default_config()
    # Always pin NUM_KSPLIT=1 (this version of the kernel doesn't split K).
    config["NUM_KSPLIT"] = 1
    # The JSON uses cache_modifier=null which decodes to Python None; the
    # kernel accepts None or "", but we keep the empty-string default from
    # _DEFAULT_CONFIG to match the legacy code path.
    if config.get("cache_modifier") is None:
        config["cache_modifier"] = ""
    return config


def gemm_mxfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
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
        config = _get_default_config()
    else:
        # Merge with defaults so missing keys are filled.
        merged = _get_default_config()
        merged.update(config)
        config = merged

    # First-version: NUM_KSPLIT must be 1
    config["NUM_KSPLIT"] = 1

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    _gemm_mxfp8_kernel[grid](
        x,
        w_t,
        y,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        y.stride(0),
        y.stride(1),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    return y


def gemm_mxfp8_preshuffle(
    x: torch.Tensor,
    w_shuffled: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """
    Preshuffle variant of gemm_mxfp8. The weight tensor has already been
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
        if (M, N, K) in _OOM_SHAPES:
            # We've previously OOM'd here under HIP-graph capture; skip the
            # tuned lookup and use the conservative default.
            config = _get_default_config()
        else:
            config = _get_config(M, N, K)
    else:
        merged = _get_default_config()
        merged.update(config)
        config = merged

    config["NUM_KSPLIT"] = 1

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    def _launch(cfg):
        _gemm_mxfp8_preshuffle_kernel[grid](
            x,
            w_view,
            y,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w_view.stride(0),
            w_view.stride(1),
            y.stride(0),
            y.stride(1),
            x_scales.stride(0),
            x_scales.stride(1),
            w_scales.stride(0),
            w_scales.stride(1),
            **cfg,
        )

    try:
        _launch(config)
    except Exception as e:
        # The tuned config may overflow gfx950 LDS (163840 bytes) under HIP
        # graph capture even when the standalone bench succeeds. Fall back
        # to default and cache the failure to avoid retrying on every call.
        if (
            "OutOfResources" in type(e).__name__
            or "shared memory" in str(e)
            or "Required" in str(e)
        ):
            _mark_oom(M, N, K)
            _launch(_get_default_config() | {"NUM_KSPLIT": 1})
        else:
            raise

    return y
