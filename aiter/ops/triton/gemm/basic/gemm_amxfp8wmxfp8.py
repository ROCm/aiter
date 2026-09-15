# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_amxfp8wmxfp8 import (
    _gemm_amxfp8wmxfp8_kernel,
    _gemm_amxfp8wmxfp8_preshuffle_kernel,
    _get_config,
)
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params


def gemm_amxfp8wmxfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with MXFP8 activations and MXFP8
    weights, i.e. both operands are FP8 e4m3 with 1x32 e8m0 block scales. The
    scaled matmul runs natively on CDNA4 (gfx950) via ``tl.dot_scaled``.

    Inputs are expected to be already quantized. Quantize activations with
    ``aiter.ops.triton.quant.quant.dynamic_mxfp8_quant`` before calling this.

    Args:
        x: FP8 e4m3 (or uint8 view) activations with shape (M, K).
        w: FP8 e4m3 (or uint8 view) weights with shape (N, K) — internally
           transposed to (K, N) before the kernel call.
        x_scales: e8m0 (uint8) per-1x32 scale for x with shape (M, K // 32).
        w_scales: e8m0 (uint8) per-1x32 scale for w with shape (N, K // 32).
        dtype: Output dtype (BF16 or FP16). Default bf16.
        y: Optional pre-allocated output tensor with shape (M, N).
        config: Optional kernel-tuning dict. If None, resolved via
            ``get_gemm_config("GEMM-AMXFP8WMXFP8", M, N, K)``. Keys used:
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps,
            num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier,
            NUM_KSPLIT (split-K factor), SPLITK_BLOCK_SIZE.
        skip_reduce: When split-K is active (NUM_KSPLIT > 1), skip the final
            reduction and return the (NUM_KSPLIT, M, N) fp32 partials instead of
            (M, N). Lets a downstream op fuse the reduction. Ignored when
            NUM_KSPLIT == 1.

    Returns:
        torch.Tensor: Output with shape (M, N), or (NUM_KSPLIT, M, N) fp32 when
        ``skip_reduce`` is set and split-K is active.
    """
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: x has K={K}, w has K={K_w}"
    assert K % 32 == 0, f"K={K} must be a multiple of 32 (1x32 microscaling)"
    assert x_scales.shape == (
        M,
        K // 32,
    ), f"x_scales shape {tuple(x_scales.shape)} != ({M}, {K // 32})"
    assert w_scales.shape == (
        N,
        K // 32,
    ), f"w_scales shape {tuple(w_scales.shape)} != ({N}, {K // 32})"

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
    else:
        # Ensure a caller-provided config has consistent split-K params
        # (SPLITK_BLOCK_SIZE derived from NUM_KSPLIT, BLOCK_SIZE_K aligned).
        config = compute_splitk_params(dict(config), K)

    NUM_KSPLIT = config["NUM_KSPLIT"]

    if y is None and (NUM_KSPLIT == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # Partial-sum buffer, one M x N slab per K-split (summed by the reduce stage).
    if NUM_KSPLIT > 1:
        y_pp = torch.empty(
            (NUM_KSPLIT, M, N),
            dtype=torch.float32,
            device=x.device,
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

    _gemm_amxfp8wmxfp8_kernel[grid](
        x,
        w_t,
        y if NUM_KSPLIT == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        0 if NUM_KSPLIT == 1 else y_pp.stride(0),
        y.stride(0) if NUM_KSPLIT == 1 else y_pp.stride(1),
        y.stride(1) if NUM_KSPLIT == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    if NUM_KSPLIT > 1:
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
            None,
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
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_amxfp8wmxfp8_reduce_kernel",
        )

    return y


def gemm_amxfp8wmxfp8_preshuffle(
    x: torch.Tensor,
    w_shuffled: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
) -> torch.Tensor:
    """
    Preshuffle variant of ``gemm_amxfp8wmxfp8``. Computes Y = X @ W^T with MXFP8
    activations and MXFP8 weights (both FP8 e4m3 with 1x32 e8m0 block scales),
    running natively on CDNA4 (gfx950) via ``tl.dot_scaled``.

    The weight *data* must already be permuted with
    ``shuffle_weight(..., layout=(16, 16))`` (from
    ``aiter.ops.triton.utils.shuffle`` or ``aiter.ops.shuffle``); the kernel
    reads it in the shuffled (N // 16, K * 16) order for a better memory access
    pattern and un-shuffles each tile in-registers before the matmul. The 1x32
    scales are left unshuffled — pass them exactly as for ``gemm_amxfp8wmxfp8``.

    Quantize + shuffle the weight once, offline; quantize activations with
    ``aiter.ops.triton.quant.quant.dynamic_mxfp8_quant`` before calling this.

    Args:
        x: FP8 e4m3 (or uint8 view) activations with shape (M, K).
        w_shuffled: FP8 e4m3 (or uint8 view) weights, pre-shuffled to (N, K)
            storage (same bytes as (N, K), rearranged for the kernel read
            pattern). N must be a multiple of 16.
        x_scales: e8m0 (uint8) per-1x32 scale for x with shape (M, K // 32).
        w_scales: e8m0 (uint8) per-1x32 scale for w with shape (N, K // 32),
            indexed by the logical (un-shuffled) N row.
        dtype: Output dtype (BF16 or FP16). Default bf16.
        y: Optional pre-allocated output tensor with shape (M, N).
        config: Optional kernel-tuning dict. If None, resolved via
            ``get_gemm_config("GEMM-AMXFP8WMXFP8_PRESHUFFLED", M, N, K)``.
        skip_reduce: When split-K is active (NUM_KSPLIT > 1), skip the final
            reduction and return the (NUM_KSPLIT, M, N) fp32 partials instead of
            (M, N). Lets a downstream op fuse the reduction. Ignored when
            NUM_KSPLIT == 1.

    Returns:
        torch.Tensor: Output with shape (M, N), or (NUM_KSPLIT, M, N) fp32 when
        ``skip_reduce`` is set and split-K is active.
    """
    M, K = x.shape
    N, K_w = w_shuffled.shape
    assert K == K_w, f"K mismatch: x has K={K}, w has K={K_w}"
    assert K % 32 == 0, f"K={K} must be a multiple of 32 (1x32 microscaling)"
    assert N % 16 == 0, f"N={N} must be a multiple of 16 for preshuffle"
    assert x_scales.shape == (
        M,
        K // 32,
    ), f"x_scales shape {tuple(x_scales.shape)} != ({M}, {K // 32})"
    assert w_scales.shape == (
        N,
        K // 32,
    ), f"w_scales shape {tuple(w_scales.shape)} != ({N}, {K // 32})"

    # The kernel addresses the shuffled tensor as (N // 16, K * 16).
    w_view = w_shuffled.reshape(N // 16, K * 16)

    # tl.dot_scaled with format "e4m3" expects uint8-typed operands; reinterpret
    # the FP8 buffers as uint8 (bit-identical view).
    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    if w_view.dtype != torch.uint8:
        w_view = w_view.view(torch.uint8)

    if config is None:
        config, _ = _get_config(M, N, K, shuffle=True)
    else:
        # Ensure a caller-provided config has consistent split-K params
        # (SPLITK_BLOCK_SIZE derived from NUM_KSPLIT, BLOCK_SIZE_K aligned).
        config = compute_splitk_params(dict(config), K)

    NUM_KSPLIT = config["NUM_KSPLIT"]

    if y is None and (NUM_KSPLIT == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    # Partial-sum buffer, one M x N slab per K-split (summed by the reduce stage).
    if NUM_KSPLIT > 1:
        y_pp = torch.empty(
            (NUM_KSPLIT, M, N),
            dtype=torch.float32,
            device=x.device,
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

    _gemm_amxfp8wmxfp8_preshuffle_kernel[grid](
        x,
        w_view,
        y if NUM_KSPLIT == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_view.stride(0),
        w_view.stride(1),
        0 if NUM_KSPLIT == 1 else y_pp.stride(0),
        y.stride(0) if NUM_KSPLIT == 1 else y_pp.stride(1),
        y.stride(1) if NUM_KSPLIT == 1 else y_pp.stride(2),
        x_scales.stride(0),
        x_scales.stride(1),
        w_scales.stride(0),
        w_scales.stride(1),
        **config,
    )

    if NUM_KSPLIT > 1:
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
            None,
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
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_amxfp8wmxfp8_preshuffle_reduce_kernel",
        )

    return y
