# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Thin strided-batched MXFP4/MXFP6/MXFP8 preshuffle GEMM launcher (gfx950): out[b] =
dequant(x[b]) @ dequant(w[b]).T, per-1x32 e8m0 scales folded into a scaled 16x16x128 MFMA.
Operands are preshuffled + laid out by the caller (once, off the launch path). layout 'bmn' =
contiguous [B,M,N], 'mbn' = the deepseek-v4 grouped-output [M,B,N] (returned as a
non-contiguous [B,M,N] view)."""

from __future__ import annotations

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.mxfp4_preshuffle import launch_gemm
from .kernels.tensor_shim import ptr_arg

SCALE_GROUP_SIZE = 32

# a_dtype -> A bytes per code (fp4 = 2 codes/byte; fp6/fp8 = 1 byte/code).
_A_CODES_PER_BYTE = {"fp4": 2, "fp6": 1, "fp8": 1}


def flydsl_batched_gemm_mxfp4(
    a: torch.Tensor,
    w: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    a_dtype: str = "fp4",
    layout: str = "bmn",
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 256,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Thin strided-batched MXFP A x MXFP4 B launcher (gfx950). Operands are ALREADY prepared
    (see preshuffle_operands) -- no shuffle happens here. `a` is plain codes shaped [B,M,arow]
    (bmn) / [M,B,arow] (mbn) with arow = K//2 (fp4) or K (fp8/fp6); `w`, `a_scales`, `w_scales`
    are the flat preshuffled buffers. `layout='mbn'` is the deepseek-v4 grouped-output path
    (returned as a non-contiguous [B,M,N] view of a physical [M,B,N] buffer). Returns (B,M,N).
    """
    if get_gfx() != "gfx950":
        raise RuntimeError(
            f"[FlyDSL] MXFP4 preshuffle GEMM requires gfx950, got {get_gfx()}"
        )
    if a_dtype not in _A_CODES_PER_BYTE:
        raise ValueError(
            f"[FlyDSL] a_dtype must be one of {sorted(_A_CODES_PER_BYTE)}; got {a_dtype!r}"
        )
    if layout not in ("bmn", "mbn"):
        raise ValueError(f"[FlyDSL] layout must be 'bmn' or 'mbn'; got {layout!r}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"[FlyDSL] unsupported out dtype {dtype}")

    B, M = (a.shape[0], a.shape[1]) if layout == "bmn" else (a.shape[1], a.shape[0])
    a_row_bytes = a.shape[-1]
    K = a_row_bytes * _A_CODES_PER_BYTE[a_dtype]

    # tile_m % 16 / tile_n % 64 must be exact or the kernel's chunk counts silently drop work.
    if tile_m % 16 != 0:
        raise RuntimeError(f"[FlyDSL] tile_m ({tile_m}) must be a multiple of 16")
    if tile_n % 64 != 0:
        raise RuntimeError(f"[FlyDSL] tile_n ({tile_n}) must be a multiple of 64")
    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL] N ({N}) must be a multiple of tile_n ({tile_n})")
    if K % 256 != 0:
        raise RuntimeError(f"[FlyDSL] K ({K}) must be a multiple of 256")
    if tile_k not in (128, 256) or K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL] tile_k must be 128/256 dividing K; got {tile_k}, K={K}"
        )

    out_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"
    # mbn: A/scale_a/C are physically [M,B,*]; explicit strides index the interleaved batch dim
    # (each <0 = the contiguous bmn default). scale_chunk_dw = dwords per 32-row e8m0 chunk.
    if layout == "mbn":
        cdw = (K // 32 // 4 // 2) * 64
        strides = (B * a_row_bytes, a_row_bytes, B * cdw, cdw * 4, B * N, N * 2)
        shape = (M, B, N)
    else:
        strides = (-1, -1, -1, -1, -1, -1)
        shape = (B, M, N)
    out_phys = (
        out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    )

    # @flyc.jit caches per Constexpr config internally; M rides i32_m at runtime (not baked).
    # Operands go in as ptr_arg (raw data_ptr) so each launch skips per-tensor DLPack conversion.
    launch_gemm(
        ptr_arg(out_phys.view(-1)),
        ptr_arg(a.reshape(-1)),
        ptr_arg(w),
        ptr_arg(a_scales),
        ptr_arg(w_scales),
        M,
        N,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        a_dtype,
        out_dtype,
        B,
        *strides,
        0,
    )

    # mbn C physical [M,B,N] -> logical [B,M,N] view.
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys
