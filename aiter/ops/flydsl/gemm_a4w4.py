# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level gfx950 dense, inline-quantized MXFP4 x MXFP4 FlyDSL API."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter.ops.asm.asm_utils import get_gfx_from_device
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

__all__ = [
    "PreshuffledA4W4Weight",
    "flydsl_gemm_a4w4",
    "prepare_gemm_a4w4_weight",
]


@dataclass(frozen=True)
class PreshuffledA4W4Weight:
    """Packed E2M1 weight and E8M0 scales in the gfx950 MFMA load layout."""

    weight: torch.Tensor
    scale: torch.Tensor
    n: int
    k: int


def _require_gfx950(device: torch.device) -> None:
    arch = get_gfx_from_device(device)
    if arch != "gfx950":
        raise RuntimeError(f"flydsl_gemm_a4w4 requires gfx950, got {arch!r}")


def _validate_shape(m: int, n: int, k: int) -> None:
    if n <= 0 or n % 256:
        raise ValueError(f"N must be a positive multiple of 256, got {n}")
    if k <= 0 or k % 128:
        raise ValueError(f"K must be a positive multiple of 128, got {k}")
    tile_k = 256 if k % 256 == 0 else 128
    if k // tile_k < 3 or k > 8192:
        raise ValueError("K requires at least three BK tiles and at most 8192 elements")
    if m < 0 or m * k * 2 >= 2**31:
        raise ValueError("BF16 input requires M >= 0 and M*K*2 < 2**31 bytes")
    if m * n >= 2**31:
        raise ValueError("Output requires M*N < 2**31 elements")
    if n * (k // 2) >= 2**31 or n * ((k + 255) // 256) * 8 >= 2**31:
        raise ValueError("Packed weight and scales must each fit in 2**31 bytes")


def _validate_prepared(prepared: PreshuffledA4W4Weight) -> None:
    _validate_shape(0, prepared.n, prepared.k)
    if tuple(prepared.weight.shape) != (prepared.n, prepared.k // 2) or tuple(
        prepared.scale.shape
    ) != (prepared.n, ((prepared.k + 255) // 256) * 8):
        raise ValueError(
            "Prepared weight/scale shapes do not match N, K and scale padding"
        )
    for tensor in (prepared.weight, prepared.scale):
        if tensor.dtype != torch.uint8 or not tensor.is_contiguous():
            raise ValueError(
                "Prepared weight and scales must be contiguous uint8 tensors"
            )
    if prepared.weight.device != prepared.scale.device:
        raise ValueError("Prepared weight and scales must be on the same device")


def _select_bm(m: int, n: int, k: int) -> int:
    """Select the measured BM16/BM64 crossover for K3 dense projections."""
    thresholds = {
        (8448, 7168): 128,
        (7168, 4224): 512,
        (1536, 7168): 2048,
        (7168, 768): 512,
    }
    threshold = thresholds.get((n, k))
    return 64 if threshold is not None and m >= threshold else 16


def prepare_gemm_a4w4_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> PreshuffledA4W4Weight:
    """Preshuffle packed E2M1 ``[N,K/2]`` and E8M0 ``[N,K/32]``.

    This preparation only rearranges the supplied packed bytes. It neither
    creates nor caches a BF16 weight.
    """
    if weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("weight and weight_scale must both be 2D")
    n, packed_k = weight.shape
    k = packed_k * 2
    _validate_shape(0, n, k)
    if tuple(weight_scale.shape) != (n, k // 32):
        raise ValueError(
            f"weight_scale must have shape {(n, k // 32)}, "
            f"got {tuple(weight_scale.shape)}"
        )
    if weight.device.type != "cuda" or weight_scale.device.type != "cuda":
        raise ValueError("weight and weight_scale must be CUDA/ROCm tensors")
    if weight.device != weight_scale.device:
        raise ValueError("weight and weight_scale must be on the same device")
    if weight.element_size() != 1 or weight_scale.element_size() != 1:
        raise TypeError("weight and weight_scale must contain packed byte payloads")
    _require_gfx950(weight.device)
    with torch.cuda.device(weight.device):
        weight_u8 = weight.contiguous().view(torch.uint8)
        scale_u8 = weight_scale.contiguous().view(torch.uint8)
        return PreshuffledA4W4Weight(
            weight=shuffle_weight_a16w4(
                weight_u8.unsqueeze(0), NLane=16, gate_up=False
            ).squeeze(0),
            scale=shuffle_scale_a16w4(scale_u8, experts_cnt=1, gate_up=False),
            n=n,
            k=k,
        )


def flydsl_gemm_a4w4(
    a: torch.Tensor,
    weight: torch.Tensor | PreshuffledA4W4Weight,
    weight_scale: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
    _bm: int | None = None,
) -> torch.Tensor:
    """Compute dense ``A @ W.T`` with in-kernel per-1x32 MXFP4 A quantization.

    ``weight`` may be a prepared object for repeated calls, or the original
    packed E2M1 ``[N,K/2]`` tensor together with its E8M0 ``[N,K/32]`` scales.
    No global activation quantization/QDQ buffer is allocated.

    Activations use per-32 RoundUp E8M0 scales (ceil_pow2(amax/6)), not the
    Even rounding default of Triton's dynamic_mxfp4_quant.

    ``_bm`` may explicitly force the 16- or 64-row kernel for tuning. By
    default, measured K3 shape-specific crossover thresholds are used.

    Empty batches return BF16 ``[0,N]`` without compiling or launching. K is
    limited to 8192 by the shared weight-scale loader's two 16-chunk bases.
    """
    if a.ndim != 2:
        raise ValueError(f"a must be 2D, got a.ndim={a.ndim}")
    if a.dtype != torch.bfloat16:
        raise TypeError(f"a must be BF16, got {a.dtype}")
    if _bm is not None and _bm not in (16, 64):
        raise ValueError(f"_bm must be None, 16, or 64, got {_bm}")
    if isinstance(weight, PreshuffledA4W4Weight):
        if weight_scale is not None:
            raise ValueError("weight_scale must be omitted for a prepared weight")
        prepared = weight
    else:
        if weight_scale is None:
            raise ValueError("weight_scale is required with an unprepared weight")
        prepared = prepare_gemm_a4w4_weight(weight, weight_scale)

    m, k = a.shape
    _validate_shape(m, prepared.n, k)
    _validate_prepared(prepared)
    if k != prepared.k:
        raise ValueError(f"a K={k} does not match weight K={prepared.k}")
    if a.device != prepared.weight.device:
        raise ValueError("a and weight must be on the same device")
    if a.device.type != "cuda" or not a.is_contiguous():
        raise ValueError("a must be a contiguous CUDA/ROCm tensor")
    _require_gfx950(a.device)
    selected_bm = _select_bm(m, prepared.n, prepared.k) if _bm is None else _bm

    expected = (m, prepared.n)
    if out is None:
        out = torch.empty(expected, dtype=torch.bfloat16, device=a.device)
    elif (
        tuple(out.shape) != expected
        or out.dtype != torch.bfloat16
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous BF16 {expected} on {a.device}")
    elif m:
        output_start = out.data_ptr()
        output_end = output_start + out.numel() * out.element_size()
        for source in (a, prepared.weight, prepared.scale):
            source_start = source.data_ptr()
            source_end = source_start + source.numel() * source.element_size()
            if output_start < source_end and source_start < output_end:
                raise ValueError(
                    "out must not overlap activation, weight or scale storage"
                )

    launch_stream = torch.cuda.current_stream(a.device) if stream is None else stream
    if launch_stream.device != a.device:
        raise ValueError("stream and tensors must be on the same device")
    if m == 0:
        return out

    from .kernels.gemm_a4w4 import compile_gemm_a4w4
    from .kernels.tensor_shim import _run_compiled

    with torch.cuda.device(a.device):
        launcher = compile_gemm_a4w4(N=prepared.n, K=prepared.k, BM=selected_bm)
        _run_compiled(
            launcher,
            a.data_ptr(),
            prepared.weight.data_ptr(),
            prepared.scale.data_ptr(),
            out.data_ptr(),
            int(m),
            launch_stream,
        )
    return out
