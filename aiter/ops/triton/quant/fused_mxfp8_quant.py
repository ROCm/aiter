# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (
    _fused_rms_mxfp8_kernel,
    _fused_dual_rmsnorm_mxfp8_quant_kernel,
    _fused_flatten_mxfp8_quant_kernel,
)

__all__ = [
    "fused_rms_mxfp8_quant",
    "fused_dual_rmsnorm_mxfp8_quant",
    "fused_flatten_mxfp8_quant",
]

_QUANT_BLOCK_SIZE = 32


def fused_rms_mxfp8_quant(
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

    _fused_rms_mxfp8_kernel[grid](
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


def fused_dual_rmsnorm_mxfp8_quant(
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
    path on V4 (Task #77): one launch instead of two (fused_rms_mxfp8_quant +
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
    assert (
        KQ % _QUANT_BLOCK_SIZE == 0
    ), f"KQ={KQ} must be a multiple of {_QUANT_BLOCK_SIZE}"
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

    _fused_dual_rmsnorm_mxfp8_quant_kernel[grid](
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


def fused_flatten_mxfp8_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten the last two dimensions of x and apply per-1x32 MXFP8 quant along
    the flattened axis (FP8 e4m3 values + uint8 e8m0 scales).

    Equivalent in shape to `fused_flatten_fp8_group_quant` but emits MXFP8
    1x32 (e8m0) scales using the same recipe as `dynamic_mxfp8_quant`.

    Args:
        x: Input tensor of shape (M, N1, N2). N2 must be a multiple of 32.
        quant_dtype: FP8 dtype to cast quantized values to (defaults to
            torch.float8_e4m3fn).

    Returns:
        Tuple of:
            out: FP8 tensor of shape (M, N1 * N2).
            out_scales: e8m0 (uint8) scale tensor of shape
                (M, (N1 * N2) // 32).
    """
    assert x.dim() == 3, f"x must be 3D, got {x.dim()}"
    M, N1, N2 = x.shape
    assert (
        N2 % _QUANT_BLOCK_SIZE == 0
    ), f"N2={N2} must be a multiple of {_QUANT_BLOCK_SIZE}"

    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), _QUANT_BLOCK_SIZE)
    N = N1 * N2

    out = torch.empty((M, N), dtype=quant_dtype, device=x.device)
    out_scales = torch.empty(
        (M, N // _QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )

    grid = (M, N1)
    _fused_flatten_mxfp8_quant_kernel[grid](
        x,
        out,
        out_scales,
        *x.stride(),
        *out.stride(),
        *out_scales.stride(),
        N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
    )

    return out, out_scales
