# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Thin strided-batched MXFP4/MXFP6/MXFP8 preshuffle GEMM launcher: out[b] =
dequant(x[b]) @ dequant(w[b]).T, per-1x32 e8m0 scales folded into a scaled 16x16x128
matrix op. gfx950 uses the wave64 MFMA path (a4w4/a8w4, fp4/fp6/fp8 A); gfx1250 uses the
wave32 WMMA path (a8w4 only: MXFP8 E4M3 A x MXFP4 B). Operands are preshuffled + laid out
by the caller (once, off the launch path) -- see the arch-specific preshuffle in the tests.
layout 'bmn' = contiguous [B,M,N], 'mbn' = the deepseek-v4 grouped-output [M,B,N] (returned
as a non-contiguous [B,M,N] view)."""

from __future__ import annotations

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.tensor_shim import ptr_arg

SCALE_GROUP_SIZE = 32
WMMA_K_GFX1250 = 128

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
    gfx = get_gfx()
    if a_dtype not in _A_CODES_PER_BYTE:
        raise ValueError(
            f"[FlyDSL] a_dtype must be one of {sorted(_A_CODES_PER_BYTE)}; got {a_dtype!r}"
        )
    if layout not in ("bmn", "mbn"):
        raise ValueError(f"[FlyDSL] layout must be 'bmn' or 'mbn'; got {layout!r}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"[FlyDSL] unsupported out dtype {dtype}")

    if gfx == "gfx1250":
        return _run_gfx1250(
            a, w, a_scales, w_scales, N, dtype,
            a_dtype=a_dtype, layout=layout,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, out=out,
        )
    if gfx != "gfx950":
        raise RuntimeError(
            f"[FlyDSL] MXFP preshuffle GEMM requires gfx950/gfx1250, got {gfx}"
        )
    from .kernels.mxfp4_preshuffle import launch_gemm  # gfx950 wave64 MFMA path

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


def _run_gfx1250_bufload(a, w, a_scales, w_scales, N, dtype, *, layout, tile_m, tile_n, out):
    """Direct buffer-load WMMA kernel (best for large-M compute-bound a8w4)."""
    from .kernels.mxfp4_preshuffle_gfx1250 import launch_gemm_a8w4

    B, M = (a.shape[0], a.shape[1]) if layout == "bmn" else (a.shape[1], a.shape[0])
    K = a.shape[-1]
    n_warp = max(1, tile_n // 64)
    warp_tile_n = tile_n // n_warp
    m_warp = (tile_m // 16) if tile_m <= 64 else (tile_m // 32)
    warp_tile_m = tile_m // m_warp
    out_is_f16 = 1 if dtype == torch.float16 else 0
    layout_mbn = 1 if layout == "mbn" else 0
    shape = (M, B, N) if layout == "mbn" else (B, M, N)
    out_phys = out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    launch_gemm_a8w4(
        ptr_arg(out_phys), ptr_arg(a.reshape(-1)), ptr_arg(w),
        ptr_arg(a_scales), ptr_arg(w_scales), M, N, torch.cuda.current_stream(),
        N, K, tile_m, tile_n, m_warp, n_warp, warp_tile_m, warp_tile_n,
        out_is_f16, B, layout_mbn, 0,
    )
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys


def _run_gfx1250(
    a, w, a_scales, w_scales, N, dtype, *, a_dtype, layout, tile_m, tile_n, tile_k, out
):
    """gfx1250 wave32 WMMA path. a8w4 only (MXFP8 E4M3 A x MXFP4 B).

    TDM (tensor-DMA) global->LDS with an ``num_buffers``-stage LDS ring overlaps the
    weight/activation DMA with WMMA compute.
    """
    from .kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm

    if a_dtype != "fp8":
        raise NotImplementedError(
            f"[FlyDSL gfx1250] only a8w4 (a_dtype='fp8') is supported, got {a_dtype!r}"
        )

    B, M = (a.shape[0], a.shape[1]) if layout == "bmn" else (a.shape[1], a.shape[0])
    K = a.shape[-1]  # fp8: 1 byte/code

    # The TDM path streams n32k4 e8m0 scales as whole 32-row/col supers, so tile_m
    # (and tile_n) must be a multiple of 32. Round a smaller/odd caller tile_m up.
    if tile_m % 32 != 0:
        tile_m = ((tile_m + 31) // 32) * 32
    if tile_n % 32 != 0:
        raise RuntimeError(f"[FlyDSL gfx1250] tile_n ({tile_n}) must be a multiple of 32")
    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL gfx1250] N ({N}) must be a multiple of tile_n ({tile_n})")
    if K % WMMA_K_GFX1250 != 0:
        raise RuntimeError(f"[FlyDSL gfx1250] K ({K}) must be a multiple of 128")

    # Pipeline K-tile: tile_k must be a multiple of 128 dividing K. Default 256.
    if tile_k % WMMA_K_GFX1250 != 0 or K % tile_k != 0:
        tile_k = WMMA_K_GFX1250 if K % 256 != 0 else 256
    k_tiles = K // tile_k

    try:
        _num_cu = torch.cuda.get_device_properties(a.device).multi_processor_count
    except Exception:
        _num_cu = 256
    _n_bn = B * ((N + tile_n - 1) // tile_n)
    bw_bound = _n_bn >= 2 * _num_cu

    # Regime dispatch: the TDM pipeline wins the weight-BW-bound / small-M (MoE, decode)
    # cases by overlapping DMA with compute, but for large-M compute-bound GEMMs the
    # fully-unrolled direct-buffer-load kernel schedules WMMAs denser and is faster.
    if M >= 1024 and not bw_bound:
        return _run_gfx1250_bufload(
            a, w, a_scales, w_scales, N, dtype,
            layout=layout, tile_m=tile_m, tile_n=tile_n, out=out,
        )

    # Shrink tile_m to the smallest {16,32,64,128} that still covers M -- but only when the
    # batch x N-tile grid already fills the GPU. Massively-batched MoE shapes (small M per
    # expert) are then weight-bandwidth-bound, so an oversized tile_m just loads+discards
    # padding rows; small grids stay latency-bound and prefer the larger tile_m for
    # occupancy/latency hiding. Never grow past the caller's tile_m.
    if bw_bound:
        _cands = [t for t in (32, 64, 128) if t <= tile_m]
        if _cands:
            tile_m = next((t for t in _cands if t >= M), _cands[-1])
        # Weight-BW-bound MoE regime: tuned TDM shape (256-N tiles maximise B reuse per
        # A/scale load; 128-K tiles with a deep ring keep the DMA queues full). With
        # A/B *and* both e8m0 scale tiles streamed through the ring via TDM, a deeper
        # ring (see _target_nb below) is needed to hide the extra DMA latency -- there
        # is a sharp cliff below ~5 stages on gfx1250 for this shape.
        if N % 256 == 0:
            tile_n = 256
        if K % 128 == 0:
            tile_k = 128
            k_tiles = K // tile_k

    # Warp tiling (tuned on gfx1250): 64-wide N warp-tiles; M warp-tiles of 16 rows for
    # tile_m<=64, 32 rows for tile_m=128. Fewer N-warps cut redundant fp8-A global traffic.
    n_warp = max(1, tile_n // 64)
    warp_tile_n = tile_n // n_warp
    m_warp = (tile_m // 16) if tile_m <= 64 else (tile_m // 32)
    warp_tile_m = tile_m // m_warp
    if m_warp * n_warp * 32 > 1024:
        raise RuntimeError(
            f"[FlyDSL gfx1250] block {m_warp * n_warp * 32} > 1024 for tile "
            f"{tile_m}x{tile_n}"
        )

    # Multi-buffer depth: with 4 TDM streams/tile (A+B+scaleA+scaleB) the BW-bound
    # regime needs a 6-deep ring to stay above the pipeline cliff; else 3 (2 for short K).
    _target_nb = 6 if bw_bound else 3
    num_buffers = min(_target_nb if k_tiles >= _target_nb else k_tiles, k_tiles)

    out_is_f16 = 1 if dtype == torch.float16 else 0
    layout_mbn = 1 if layout == "mbn" else 0
    shape = (M, B, N) if layout == "mbn" else (B, M, N)
    out_phys = (
        out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    )

    launch_gemm_a8w4_tdm(
        out_phys,
        a.contiguous(),
        # 2-D view (each dim < 2^31) so the DLPack arg packing doesn't overflow i32 on
        # multi-GB weights; TDM only needs the base address.
        w.reshape(B * (N // 16), (K // 2) * 16),
        a_scales.view(torch.int32),
        w_scales.view(torch.int32),
        M,
        N,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        warp_tile_m,
        warp_tile_n,
        out_is_f16,
        B,
        layout_mbn,
        num_buffers,
        0,
    )
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys
