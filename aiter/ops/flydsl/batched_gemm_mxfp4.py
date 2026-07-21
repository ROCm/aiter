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


def flydsl_grouped_gemm_a8w4_masked(
    out,
    a,
    w,
    a_scales,
    w_scales,
    m_tile_map,
    *,
    n_experts,
    contiguous_m,
    N,
    K,
    tile_m=64,
    tile_n=256,
    tile_k=256,
    m_warp=1,
    n_warp=4,
    num_buffers=3,
    out_is_f16=0,
    a_is_fp4=0,
    stage1_act=0,
    bias=None,
    swiglu_limit=7.0,
    stream=None,
    stage1_quant_out=0,
    quant_scale=None,
    quant_wmma_rep=1,
):
    """Contiguous-M grouped a8w4 GEMM on the batched TDM kernel.

    Mirrors the MoE grouped-gemm contiguous-M scheduling: a compact grid over the
    (1, contiguous_m, *) buffers, with a per-M-tile expert id (``tile_expert``)
    selecting the per-expert B / B-scale slab. Only valid M tiles launch.
      out          (1, contiguous_m, N)  bf16/f16  (or fp8 payload when stage1_quant_out)
      a            (1, contiguous_m, K)  uint8 (fp8 payload)
      w            (E, N, K//2) uint8 (moe_shuffle_weight == cat_e shuffle_weight_gfx1250)
      a_scales     grouped A-scale (1, contiguous_m//wmma_rep, (K//32)*wmma_rep) viewed int32
      w_scales     n32k4 B-scale (E, N//32, (K//32)*32) viewed int32
      m_tile_map   (n_experts,) int32 psum (per-expert exclusive end-row)
    contiguous_m must be a multiple of tile_m (holds by construction).

    When ``stage1_quant_out=1`` (fp8), the epilogue fuses silu/swiglu + MX fp8
    quantization + e8m0 scale preshuffle into the kernel.  ``out`` receives the
    fp8 payload (uint8, 1 byte/elem) and ``quant_scale`` receives the preshuffled
    e8m0 scale (uint8).  ``quant_wmma_rep`` is gemm2's ``warp_tile_m // 16``,
    controlling the scale preshuffle tile geometry.
    """
    from .kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm
    from .kernels.tensor_shim import _run_compiled

    if stream is None:
        stream = torch.cuda.current_stream()
    nb = min(num_buffers, max(1, K // tile_k))
    has_bias = 1 if bias is not None else 0
    bias_ptr = ptr_arg(bias) if bias is not None else ptr_arg(a)
    # When quant is off, pass a dummy pointer for arg_quant_scale (unused).
    if quant_scale is None:
        quant_scale_ptr = ptr_arg(out)  # dummy, never written
    else:
        quant_scale_ptr = ptr_arg(quant_scale.view(torch.uint8))
    _run_compiled(
        launch_gemm_a8w4_tdm,
        ptr_arg(out),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scales.view(torch.int32)),
        ptr_arg(w_scales.view(torch.int32)),
        contiguous_m,
        stream,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        nb,
        a_is_fp4,
        ptr_arg(m_tile_map),
        n_experts,
        stage1_act,
        has_bias,
        bias_ptr,
        float(swiglu_limit),
        stage1_quant_out,
        quant_wmma_rep,
        quant_scale_ptr,
    )
    return out


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


def _run_gfx1250(
    a, w, a_scales, w_scales, N, dtype, *, a_dtype, layout, tile_m, tile_n, tile_k, out
):
    """gfx1250 wave32 WMMA path. a8w4 (MXFP8 E4M3 A) or a4w4 (MXFP4 A), both x MXFP4 B.

    TDM (tensor-DMA) global->LDS with an ``num_buffers``-stage LDS ring overlaps the
    weight/activation DMA with WMMA compute.
    """
    from .kernels.mxfp4_preshuffle_gfx1250_tdm import launch_gemm_a8w4_tdm
    from .kernels.tensor_shim import _run_compiled

    if a_dtype not in ("fp8", "fp4"):
        raise NotImplementedError(
            f"[FlyDSL gfx1250] only a8w4 (fp8) / a4w4 (fp4) supported, got {a_dtype!r}"
        )
    a_is_fp4 = 1 if a_dtype == "fp4" else 0

    B, M = (a.shape[0], a.shape[1]) if layout == "bmn" else (a.shape[1], a.shape[0])
    K = a.shape[-1] * _A_CODES_PER_BYTE[a_dtype]  # fp8: 1 byte/code, fp4: 2 codes/byte

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

    # Regime dispatch. The TDM kernel serves every gfx1250 batched shape now:
    #  - large-M compute-bound: tuned 256x256x256 w4x2 (nb2) config (~4900 TF; beats
    #    the old direct-buffer-load path and gemm_fp8fp4). Wave-specialized TDM
    #    (A/B/scaleA/scaleB each on a dedicated loader-wave pair) drives it;
    #  - weight-BW-bound / small-M (MoE, decode): cooperative all-wave TDM on a small
    #    tile_m with a deeper (nb3) ring to overlap DMA with compute. Mem-bound perf,
    #    a8w4 bmn 32x64x7168x2048 (tile 64x256x128, w1x2, nb3): ~29.6 us, ~9.5 TB/s
    #    HBM (~2030 TF); the ~283 MB weight+scale stream is the limiter.
    compute_bound = (M >= 1024) and not bw_bound
    if compute_bound and N % 256 == 0:
        tile_m, tile_n = 256, 256
        tile_k = 256 if K % 256 == 0 else 128
        k_tiles = K // tile_k
        m_warp, n_warp = 4, 2
        num_buffers = min(2, k_tiles)
    else:
        # Shrink tile_m to the smallest {32,64,128} covering M when the batch x N-tile
        # grid already fills the GPU (BW-bound MoE); small grids keep the larger tile_m
        # for latency hiding. Never grow past the caller's tile_m.
        if bw_bound:
            _cands = [t for t in (32, 64, 128) if t <= tile_m]
            if _cands:
                tile_m = next((t for t in _cands if t >= M), _cands[-1])
            if N % 256 == 0:
                tile_n = 256
            if K % 128 == 0:
                tile_k = 128
                k_tiles = K // tile_k
        # BW-bound: pack many WMMAs per wave in a small workgroup (128-wide N tiles,
        # warp_tile_m<=64). Non-BW latency-bound: wider workgroup for latency hiding.
        if bw_bound:
            n_warp = max(1, tile_n // 128)
            m_warp = max(1, tile_m // 64)
        else:
            n_warp = max(1, tile_n // 64)
            m_warp = (tile_m // 16) if tile_m <= 64 else (tile_m // 32)
        # 3-deep ring overlaps the 4 TDM streams (A+B+scaleA+scaleB) with compute.
        num_buffers = min(3, k_tiles)

    if m_warp * n_warp * 32 > 1024:
        raise RuntimeError(
            f"[FlyDSL gfx1250] block {m_warp * n_warp * 32} > 1024 for tile "
            f"{tile_m}x{tile_n}"
        )

    out_is_f16 = 1 if dtype == torch.float16 else 0
    shape = (M, B, N) if layout == "mbn" else (B, M, N)
    out_phys = (
        out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    )

    # Build a trivial m_tile_map for single-batch (n_experts=1): the one "expert"
    # owns all M rows, so the exclusive-end psum is just [contiguous_m].
    contiguous_m = B * M
    m_tile_map = torch.tensor([contiguous_m], dtype=torch.int32, device=a.device)

    # A/B only need the base address (views built in-kernel) -> pass as pointers;
    # bind the contiguous A to a local so its storage outlives the async launch.
    a_c = a.contiguous()
    _run_compiled(
        launch_gemm_a8w4_tdm,
        ptr_arg(out_phys),
        ptr_arg(a_c),
        ptr_arg(w),
        ptr_arg(a_scales.view(torch.int32)),
        ptr_arg(w_scales.view(torch.int32)),
        contiguous_m,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        num_buffers,
        a_is_fp4,
        ptr_arg(m_tile_map),
        1,            # n_experts
        0,            # stage1_act (no activation)
        0,            # has_bias
        ptr_arg(a_c), # arg_bias (dummy, unused)
        float("inf"), # f32_swiglu_limit (unused)
        0,            # stage1_quant_out
        1,            # quant_wmma_rep
        ptr_arg(out_phys),  # arg_quant_scale (dummy, unused)
    )
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys
