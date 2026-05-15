# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton

from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _mxfp8_quant_kernel,
    _fp8_legacy_to_mxfp8_kernel,
    _rmsnorm_mxfp8_quant_kernel,
    _dual_rmsnorm_mxfp8_quant_kernel,
)


_QUANT_BLOCK_SIZE = 32


def per_1x32_mxfp8_quant_triton(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-1x32 MXFP8 quantization (e8m0 scale + FP8 e4m3 values).

    Args:
        x: Input tensor (..., K). Typically bf16 or fp16. K % 32 == 0.
        scale: Pre-allocated scale tensor (M, K // 32) uint8. Optional.
        quant_dtype: FP8 dtype to cast quantized values to. On MI3xx
            torch.float8_e4m3fnuz is the canonical FP8 e4m3 type. torch.float8_e4m3fn
            is acceptable on hardware that supports it.

    Returns:
        Tuple of:
            y: FP8 tensor of shape x.shape.
            s: e8m0 (uint8) scale tensor of shape (..., K // 32).
    """
    assert x.dim() >= 2, f"x must be at least 2D, got {x.dim()}"
    orig_shape = x.shape
    K = orig_shape[-1]
    assert (
        K % _QUANT_BLOCK_SIZE == 0
    ), f"last dim K={K} must be a multiple of {_QUANT_BLOCK_SIZE}"

    x2d = x.reshape(-1, K).contiguous()
    M = x2d.shape[0]
    Ns = K // _QUANT_BLOCK_SIZE  # number of scales per row

    y = torch.empty((M, K), dtype=quant_dtype, device=x.device)
    if scale is None:
        scale = torch.empty((M, Ns), dtype=torch.uint8, device=x.device)
    else:
        assert scale.shape == (M, Ns), f"scale shape {scale.shape} != ({M},{Ns})"
        assert scale.dtype == torch.uint8

    BLOCK_SIZE_N = triton.next_power_of_2(K)
    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _mxfp8_quant_kernel[grid](
        x2d,
        y,
        scale,
        M,
        K,
        x2d.stride(0),
        x2d.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )

    y = y.view(*orig_shape[:-1], K)
    s = scale.view(*orig_shape[:-1], Ns)
    return y, s


_LEGACY_BLOCK_SIZE = 128


def fp8_legacy_to_mxfp8(
    x_fnuz: torch.Tensor,
    x_scale_fp32: torch.Tensor,
    y_fn: Optional[torch.Tensor] = None,
    y_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transcode (FP8 e4m3fnuz, fp32 1x128 scale) -> (FP8 e4m3fn, e8m0 1x32 scale)
    in a single Triton launch. Replaces the Python dequant+requant cascade
    used when MXFP8 path receives legacy-formatted (FP8 + fp32 1x128) inputs.

    Args:
        x_fnuz: FP8 e4m3fnuz tensor of shape (M, N), N % 32 == 0.
        x_scale_fp32: fp32 scale of shape (M, N // 128).
        y_fn: optional preallocated output FP8 e4m3fn tensor.
        y_scale: optional preallocated uint8 e8m0 scale tensor.

    Returns:
        y_fn (M, N) fp8 e4m3fn, y_scale (M, N // 32) uint8 e8m0.
    """
    assert x_fnuz.dim() == 2, f"x must be 2D, got {x_fnuz.dim()}"
    M, N = x_fnuz.shape
    assert N % _QUANT_BLOCK_SIZE == 0
    assert N % _LEGACY_BLOCK_SIZE == 0
    assert x_scale_fp32.shape == (M, N // _LEGACY_BLOCK_SIZE), (
        f"x_scale_fp32 shape {x_scale_fp32.shape} != ({M},{N // _LEGACY_BLOCK_SIZE})"
    )

    Ns = N // _QUANT_BLOCK_SIZE
    if y_fn is None:
        y_fn = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x_fnuz.device)
    if y_scale is None:
        y_scale = torch.empty((M, Ns), dtype=torch.uint8, device=x_fnuz.device)

    BLOCK_SIZE_M = 1
    grid = (triton.cdiv(M, BLOCK_SIZE_M), Ns)

    _fp8_legacy_to_mxfp8_kernel[grid](
        x_fnuz,
        x_scale_fp32,
        y_fn,
        y_scale,
        M,
        N,
        x_fnuz.stride(0),
        x_fnuz.stride(1),
        x_scale_fp32.stride(0),
        x_scale_fp32.stride(1),
        y_fn.stride(0),
        y_fn.stride(1),
        y_scale.stride(0),
        y_scale.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        LEGACY_BLOCK_SIZE=_LEGACY_BLOCK_SIZE,
    )

    return y_fn, y_scale


def rmsnorm_mxfp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    y: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + MXFP8 (1x32 e8m0) quant in a single Triton launch.

    Args:
        x: (M, K) bf16 or fp16.
        weight: (K,) bf16 or fp16 RMSNorm weight.
        eps: RMSNorm epsilon.
        y: optional preallocated FP8 e4m3fn output (M, K).
        scale: optional preallocated uint8 e8m0 output (M, K // 32).

    Returns:
        y (M, K) fp8 e4m3fn, scale (M, K // 32) uint8.
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}"
    M, K = x.shape
    assert weight.shape == (K,), f"weight shape {weight.shape} != ({K},)"
    assert K % _QUANT_BLOCK_SIZE == 0
    Ns = K // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    if y is None:
        y = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty((M, Ns), dtype=torch.uint8, device=x.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _rmsnorm_mxfp8_quant_kernel[grid](
        x,
        weight,
        y,
        scale,
        M,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        eps,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
    return y, scale


def dual_rmsnorm_mxfp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps_q: float,
    eps_k: Optional[float] = None,
    yq: Optional[torch.Tensor] = None,
    sq: Optional[torch.Tensor] = None,
    yk: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual RMSNorm in a single Triton launch.

    - Q side: RMSNorm(q, q_weight, eps_q) -> MXFP8 (FP8 e4m3fn + uint8 e8m0 1x32).
    - K side: RMSNorm(k, k_weight, eps_k) -> bf16.

    Replaces the CK `fused_qk_rmsnorm_group_quant` kernel for the MXFP8 GEMM
    path on V4 (Task #77): one launch instead of two (rmsnorm_mxfp8_quant +
    rmsnorm2d_fwd_), eliminating the ~6us/layer launch-overhead regression.

    Args:
        q: (M, KQ) bf16 or fp16 — Q-side input (e.g. q_lora).
        k: (M, KK) bf16 or fp16 — K-side input (e.g. kv_pre).
        q_weight: (KQ,) bf16 or fp16 — Q RMSNorm weight.
        k_weight: (KK,) bf16 or fp16 — K RMSNorm weight.
        eps_q: Q RMSNorm epsilon.
        eps_k: K RMSNorm epsilon; defaults to eps_q.
        yq, sq, yk: optional pre-allocated outputs.

    Returns:
        yq (M, KQ) fp8 e4m3fn, sq (M, KQ // 32) uint8 e8m0, yk (M, KK) bf16.
    """
    assert q.dim() == 2, f"q must be 2D, got {q.dim()}"
    assert k.dim() == 2, f"k must be 2D, got {k.dim()}"
    M, KQ = q.shape
    Mk, KK = k.shape
    assert M == Mk, f"q rows {M} != k rows {Mk}"
    assert q_weight.shape == (KQ,), f"q_weight shape {q_weight.shape} != ({KQ},)"
    assert k_weight.shape == (KK,), f"k_weight shape {k_weight.shape} != ({KK},)"
    assert KQ % _QUANT_BLOCK_SIZE == 0, (
        f"KQ={KQ} must be a multiple of {_QUANT_BLOCK_SIZE}"
    )
    if eps_k is None:
        eps_k = eps_q

    Ns = KQ // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_KQ = triton.next_power_of_2(KQ)
    BLOCK_SIZE_KK = triton.next_power_of_2(KK)

    if yq is None:
        yq = torch.empty((M, KQ), dtype=torch.float8_e4m3fn, device=q.device)
    if sq is None:
        sq = torch.empty((M, Ns), dtype=torch.uint8, device=q.device)
    if yk is None:
        yk = torch.empty((M, KK), dtype=k.dtype, device=k.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _dual_rmsnorm_mxfp8_quant_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        yq,
        sq,
        yk,
        M,
        KQ,
        KK,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        yq.stride(0),
        yq.stride(1),
        sq.stride(0),
        sq.stride(1),
        yk.stride(0),
        yk.stride(1),
        eps_q,
        eps_k,
        BLOCK_SIZE_KQ=BLOCK_SIZE_KQ,
        BLOCK_SIZE_KK=BLOCK_SIZE_KK,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
    return yq, sq, yk
