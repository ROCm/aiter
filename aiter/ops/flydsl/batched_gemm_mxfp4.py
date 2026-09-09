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

import csv
import functools
import os

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.tensor_shim import ptr_arg

SCALE_GROUP_SIZE = 32
WMMA_K_GFX1250 = 128

# a_dtype -> A bytes per code (fp4 = 2 codes/byte; fp6/fp8 = 1 byte/code).
_A_CODES_PER_BYTE = {"fp4": 2, "fp6": 1, "fp8": 1}

# gfx950 launch_gemm CompiledFunctions, keyed by their full Constexpr config.
_GFX950_CF_CACHE: dict = {}


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
    k_batch: int = 1,
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
    # Only k_batch > 1 takes the split-K branch, so a non-positive value would
    # slip through to the kernel and evaluate K_TILES % k_batch -- a constexpr
    # division by zero at k_batch=0, and a negative grid.z extent below that.
    if not isinstance(k_batch, int) or isinstance(k_batch, bool) or k_batch < 1:
        raise ValueError(f"[FlyDSL] k_batch must be an int >= 1; got {k_batch!r}")

    if gfx == "gfx1250":
        return _run_gfx1250(
            a,
            w,
            a_scales,
            w_scales,
            N,
            dtype,
            a_dtype=a_dtype,
            layout=layout,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            out=out,
        )
    if gfx != "gfx950":
        raise RuntimeError(
            f"[FlyDSL] MXFP preshuffle GEMM requires gfx950/gfx1250, got {gfx}"
        )
    from .kernels.mxfp4_preshuffle import launch_gemm  # gfx950 wave64 MFMA path

    B, M = (a.shape[0], a.shape[1]) if layout == "bmn" else (a.shape[1], a.shape[0])
    a_row_bytes = a.shape[-1]
    K = a_row_bytes * _A_CODES_PER_BYTE[a_dtype]

    # tile_m % 32 / tile_n % 64 must be exact or the kernel's chunk counts silently drop work.
    # tile_m was checked against 16 here, but the kernel itself asserts % 32 (the A
    # e8m0 scale is 32-row granular), so tile_m=16 slipped past this guard and died
    # on a raw assert inside launch_gemm instead of a readable error.
    if tile_m % 32 != 0:
        raise RuntimeError(
            f"[FlyDSL] tile_m ({tile_m}) must be a multiple of 32 "
            "(the A e8m0 scale is 32-row granular)"
        )
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
    # The A tile is staged into LDS in whole cooperative rounds of num_threads*16
    # bytes, so an A tile that is not a multiple of one round would leave part of
    # itself never DMA'd. The kernel asserts this, but the per-dimension checks
    # above cannot see it: it couples tile_m, tile_k, tile_n *and* a_dtype --
    # (32, 128, 128) is fine for fp6 but not fp4, for instance. tiles_are_valid()
    # applies the same rule when picking tiles; this guard is for direct callers,
    # which would otherwise hit a raw kernel assert.
    if not tiles_are_valid(tile_m, tile_n, tile_k, N, K, a_dtype):
        a_row_b = tile_k // _A_CODES_PER_BYTE[a_dtype]
        raise RuntimeError(
            f"[FlyDSL] tile_m*tile_k/{_A_CODES_PER_BYTE[a_dtype]} "
            f"({tile_m * a_row_b}B of A tile) must be a multiple of "
            f"{min(4, tile_n // 16) * 64 * 16}B (num_threads*16) for "
            f"a_dtype={a_dtype!r}; got tile_m={tile_m}, tile_n={tile_n}, "
            f"tile_k={tile_k}"
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
    # ptr_arg erases dtype/shape/device, so a mismatched `out` is not a wrong
    # result but an out-of-bounds write: the kernel would store bf16 elements
    # into, say, an fp32 or an undersized allocation. Nothing downstream can
    # catch that, so validate before taking the pointer.
    if out is not None:
        if out.dtype != dtype:
            raise RuntimeError(f"[FlyDSL] out.dtype ({out.dtype}) must be {dtype}")
        if tuple(out.shape) != tuple(shape):
            raise RuntimeError(
                f"[FlyDSL] out.shape ({tuple(out.shape)}) must be {tuple(shape)}"
            )
        if out.device != a.device:
            raise RuntimeError(
                f"[FlyDSL] out.device ({out.device}) must be {a.device}"
            )
        if not out.is_contiguous():
            raise RuntimeError(
                "[FlyDSL] out must be contiguous; the kernel writes it through a "
                "raw pointer using the strides computed above"
            )
    out_phys = (
        out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    )

    # Split-K. Without it the grid is ceil(M/tile_m) * (N/tile_n) workgroups, so a
    # decode-shaped M leaves most of the device idle -- at M=1, N=4096, tile_n=64
    # that is 64 workgroups against 256 CUs. k_batch splits the K loop across
    # grid.z into fp32 partials that launch_splitk_reduce then sums, which
    # measured 2.04x at M=1 with bit-identical output.
    if k_batch > 1:
        if (K // tile_k) % k_batch:
            raise RuntimeError(
                f"[FlyDSL] K/tile_k ({K // tile_k}) must be divisible by "
                f"k_batch ({k_batch})"
            )
        if layout != "bmn" or B != 1:
            raise RuntimeError(
                "[FlyDSL] split-K needs the contiguous single-batch layout "
                f"(bmn, B=1); got layout={layout!r}, B={B}"
            )
        gemm_dst = torch.empty((k_batch, M, N), dtype=torch.float32,
                               device=a.device)
    else:
        gemm_dst = out_phys

    # Everything the kernel bakes in. M is absent on purpose -- it rides i32_m at
    # runtime, so one compiled config serves every batch size (verified: a config
    # compiled at M=1 gives correct results for M up to 1000).
    cfg_key = (N, K, tile_m, tile_n, tile_k, a_dtype, out_dtype, B, strides,
               k_batch)

    # Operands go in as ptr_arg (raw data_ptr) so each launch skips per-tensor
    # DLPack conversion.
    launch_args = (
        ptr_arg(gemm_dst.view(-1)),
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
        # B operand is always MXFP4 here (see the function name / docstring);
        # launch_gemm grew b_dtype when FlyDSL added the a8w8 (fp8 B) path.
        "fp4",
        B,
        *strides,
        0,          # waves_per_eu
        0,          # xcd_swizzle
        k_batch,
    )

    # Re-entering the @flyc.jit wrapper re-resolves the config from its 24
    # arguments on every call, which measured ~18us of CPU per launch on MI355X.
    # Holding the CompiledFunction it produces and calling that directly drops
    # CPU dispatch to ~3.4us, so cache on the full Constexpr tuple.
    #
    # Deliberately NOT tensor_shim._run_compiled: that stashes one cf on the
    # callable itself, which is right for the per-config closures the other
    # kernels build but wrong here -- launch_gemm is a single shared module-level
    # jit, so the first config's kernel would be handed to every later config.
    cf = _GFX950_CF_CACHE.get(cfg_key)
    if cf is None:
        import flydsl.compiler as flyc

        # flyc.compile both compiles and runs this first call.
        _GFX950_CF_CACHE[cfg_key] = flyc.compile(launch_gemm, *launch_args)
    else:
        cf(*launch_args)

    if k_batch > 1:
        from .kernels.mxfp4_preshuffle import launch_splitk_reduce

        reduce_args = (
            ptr_arg(gemm_dst.view(-1)),
            ptr_arg(out_phys.view(-1)),
            M * N // 2,   # output dwords: 2 out elems per dword
            M * N,        # dwords per fp32 slab
            torch.cuda.current_stream(),
            k_batch,
            out_dtype,
        )
        reduce_key = ("reduce", M * N, k_batch, out_dtype)
        rcf = _GFX950_CF_CACHE.get(reduce_key)
        if rcf is None:
            import flydsl.compiler as flyc

            _GFX950_CF_CACHE[reduce_key] = flyc.compile(
                launch_splitk_reduce, *reduce_args
            )
        else:
            rcf(*reduce_args)

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
        raise RuntimeError(
            f"[FlyDSL gfx1250] tile_n ({tile_n}) must be a multiple of 32"
        )
    if N % tile_n != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250] N ({N}) must be a multiple of tile_n ({tile_n})"
        )
    if K % WMMA_K_GFX1250 != 0:
        raise RuntimeError(f"[FlyDSL gfx1250] K ({K}) must be a multiple of 128")

    # Pipeline K-tile: tile_k must be a multiple of 128 dividing K. Default 256.
    if tile_k % WMMA_K_GFX1250 != 0 or K % tile_k != 0:
        tile_k = WMMA_K_GFX1250 if K % 256 != 0 else 256
    k_tiles = K // tile_k

    try:
        _num_cu = torch.cuda.get_device_properties(a.device).multi_processor_count
    except Exception:  # noqa: BLE001
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
    layout_mbn = 1 if layout == "mbn" else 0
    shape = (M, B, N) if layout == "mbn" else (B, M, N)
    out_phys = (
        out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    )

    # A/B only need the base address (views built in-kernel) -> pass as pointers;
    # bind the contiguous A to a local so its storage outlives the async launch.
    a_c = a.contiguous()
    launch_gemm_a8w4_tdm(
        out_phys,
        ptr_arg(a_c),
        ptr_arg(w),
        a_scales.view(torch.int32),
        w_scales.view(torch.int32),
        M,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        B,
        layout_mbn,
        num_buffers,
        a_is_fp4,
        ptr_arg(a_c),  # masked_m unused when grouped_masked=0
        0,
    )
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys


# ===========================================================================
# a8w4 batched GEMM on the *dedicated* strided-batched TDM kernel
# (kernels/mxfp4_preshuffle_batched_gemm_gfx1250_tdm.py). This is a cleaner
# batched-only kernel (adds cluster support, drops the MoE grouped-masked /
# bias / swiglu / quant-epilogue baggage of mxfp4_preshuffle_gfx1250_tdm.py).
#
# Operand-prep helpers below build exactly the layout the kernel reads:
#   - weight codes : per-batch shuffle_weight_gfx1250 -> [B, N//16, (K//2)*16]
#   - weight scale : n32k4 -> [B, N//32, (K//32)*32]
#   - act codes    : MXFP8 e4m3 payload, mbn physical [M, B, K]
#   - act scale    : n32k4 on the 32-row M-super, mbn physical
#                    [M//32, B, (K//32)*32] (super-major, batch middle)
# The A-scale n32k4 fold is bit-identical to the weight fold (the kernel's
# load_sa reads `super*SC_INNER + ksl*32 + row%32` int32, each int32 = the 4
# e8m0 of one WMMA-K=128 step -> column order remain_k*128 + row32*4 + r).
# ===========================================================================


def preshuffle_a8w4_weight_mbn(w_bf16: torch.Tensor):
    """Quantize + preshuffle a BF16 weight batch to the a8w4 kernel layout.

    Args:
        w_bf16: [B, N, K] bf16/fp16 weight (per-batch [N, K], row = output N).
    Returns:
        (w_codes, w_scales):
          w_codes  : [B, N//16, (K//2)*16] uint8  (MXFP4 codes, WMMA-shuffled)
          w_scales : [B, N//32, (K//32)*32] uint8 (e8m0 n32k4)
    """
    from aiter.ops.shuffle import shuffle_scale_n32k4, shuffle_weight_gfx1250
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    assert w_bf16.dim() == 3, f"expected [B,N,K], got {tuple(w_bf16.shape)}"
    B, N, K = w_bf16.shape
    assert N % 32 == 0 and K % 128 == 0, f"need N%32==0,K%128==0 got N={N},K={K}"

    codes, scales = [], []
    for b in range(B):
        c, s = dynamic_mxfp4_quant(w_bf16[b].contiguous())  # c:[N,K//2] s:[N,K//32]
        codes.append(shuffle_weight_gfx1250(c))  # [N//16, (K//2)*16]
        scales.append(s.view(torch.uint8))
    w_codes = torch.stack(codes, 0).contiguous()
    w_scale_raw = torch.stack(scales, 0).contiguous()  # [B, N, K//32]
    w_scales = shuffle_scale_n32k4(w_scale_raw, experts_cnt=B).contiguous()
    return w_codes, w_scales


def quant_act_mxfp8_mbn(o_mbn: torch.Tensor):
    """Fused MXFP8 (e4m3 + e8m0) quant of an mbn activation, kernel-ready.

    Single Triton launch (``dynamic_mxfp8_quant_n32k4_mbn``): the e8m0 scale is
    written *directly* in the n32k4 layout the batched a8w4 kernel reads, so
    there are NO post-quant transpose/permute/contiguous copies.

    Args:
        o_mbn: [M, B, K] bf16/fp16 activation (deepseek-v4 grouped output,
               M = tokens, B = groups). Physical layout must be M-outer.
    Returns:
        (a_fp8, a_scales):
          a_fp8   : [M, B, K] float8_e4m3fn  (mbn physical, M-outer)
          a_scales: [ceil(M/32), B, (K//32)*32] uint8  (e8m0 n32k4, super-major;
                    padded supers are pre-zeroed and OOB-masked by the kernel).
    """
    from aiter.ops.triton.quant import dynamic_mxfp8_quant_n32k4_mbn

    assert o_mbn.dim() == 3, f"expected [M,B,K], got {tuple(o_mbn.shape)}"
    _, _, K = o_mbn.shape
    assert K % 128 == 0, f"need K%128==0, got K={K}"

    return dynamic_mxfp8_quant_n32k4_mbn(o_mbn)


def flydsl_batched_gemm_a8w4_v2(
    a: torch.Tensor,
    w: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    layout: str = "mbn",
    tile_m: int = 128,
    tile_n: int = 256,
    tile_k: int = 256,
    cluster_m: int = 1,
    cluster_n: int = 1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Strided-batched a8w4 (MXFP8 A x MXFP4 B) GEMM on the dedicated batched
    TDM kernel. Operands are ALREADY prepared (see ``preshuffle_a8w4_weight_mbn``
    / ``quant_act_mxfp8_mbn``); no quant/shuffle happens here.

    Args:
        a        : MXFP8 codes. mbn: [M, B, K] fp8/uint8 (M-outer). bmn: [B, M, K].
        w        : MXFP4 codes, WMMA-shuffled [B, N//16, (K//2)*16] uint8.
        a_scales : e8m0 n32k4. mbn: [M//32, B, (K//32)*32] uint8. viewed int32.
        w_scales : e8m0 n32k4 [B, N//32, (K//32)*32] uint8. viewed int32.
        N        : output N (o_lora_rank for wo_a).
        layout   : 'mbn' (deepseek-v4 grouped output) or 'bmn'.
    Returns:
        [B, M, N] (mbn: a non-contiguous view of the [M, B, N] physical buffer).
    """
    from .kernels.mxfp4_preshuffle_batched_gemm_gfx1250_tdm import (
        launch_gemm_a8w4_tdm as _launch_v2,
    )

    gfx = get_gfx()
    if gfx != "gfx1250":
        raise RuntimeError(f"[FlyDSL] batched a8w4 v2 requires gfx1250, got {gfx}")
    if layout not in ("bmn", "mbn"):
        raise ValueError(f"[FlyDSL] layout must be 'bmn' or 'mbn'; got {layout!r}")
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"[FlyDSL] unsupported out dtype {dtype}")

    a_is_fp4 = 0  # a8w4: fp8 activation
    B, M = (a.shape[0], a.shape[1]) if layout == "bmn" else (a.shape[1], a.shape[0])
    K = a.shape[-1]  # fp8: 1 byte/code

    # tile constraints (mirror _run_gfx1250): supers of 32, N%tile_n, K%128.
    if tile_m % 32 != 0:
        tile_m = ((tile_m + 31) // 32) * 32
    if tile_n % 32 != 0:
        raise RuntimeError(f"[FlyDSL] tile_n ({tile_n}) must be a multiple of 32")
    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL] N ({N}) must be a multiple of tile_n ({tile_n})")
    if K % WMMA_K_GFX1250 != 0:
        raise RuntimeError(f"[FlyDSL] K ({K}) must be a multiple of 128")
    if tile_k % WMMA_K_GFX1250 != 0 or K % tile_k != 0:
        tile_k = WMMA_K_GFX1250 if K % 256 != 0 else 256
    k_tiles = K // tile_k

    try:
        _num_cu = torch.cuda.get_device_properties(a.device).multi_processor_count
    except Exception:  # noqa: BLE001
        _num_cu = 256
    _n_bn = B * ((N + tile_n - 1) // tile_n)
    bw_bound = _n_bn >= 2 * _num_cu
    compute_bound = (M >= 1024) and not bw_bound

    if compute_bound and N % 256 == 0:
        tile_m, tile_n = 256, 256
        tile_k = 256 if K % 256 == 0 else 128
        k_tiles = K // tile_k
        m_warp, n_warp = 4, 2
        num_buffers = min(2, k_tiles)
    else:
        if bw_bound:
            _cands = [t for t in (32, 64, 128) if t <= tile_m]
            if _cands:
                tile_m = next((t for t in _cands if t >= M), _cands[-1])
            if N % 256 == 0:
                tile_n = 256
            if K % 128 == 0:
                tile_k = 128
                k_tiles = K // tile_k
        if bw_bound:
            n_warp = max(1, tile_n // 128)
            m_warp = max(1, tile_m // 64)
        else:
            n_warp = max(1, tile_n // 64)
            m_warp = (tile_m // 16) if tile_m <= 64 else (tile_m // 32)
        num_buffers = min(3, k_tiles)

    if m_warp * n_warp * 32 > 1024:
        raise RuntimeError(
            f"[FlyDSL] block {m_warp * n_warp * 32} > 1024 for tile {tile_m}x{tile_n}"
        )

    out_is_f16 = 1 if dtype == torch.float16 else 0
    layout_mbn = 1 if layout == "mbn" else 0
    shape = (M, B, N) if layout == "mbn" else (B, M, N)
    out_phys = (
        out if out is not None else torch.empty(shape, dtype=dtype, device=a.device)
    )

    a_c = a.contiguous()
    _launch_v2(
        out_phys,
        ptr_arg(a_c),
        ptr_arg(w),
        a_scales.view(torch.int32),
        w_scales.view(torch.int32),
        M,
        torch.cuda.current_stream(),
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        B,
        layout_mbn,
        num_buffers,
        a_is_fp4,
        cluster_m,
        cluster_n,
    )
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys


# ---------------------------------------------------------------------------
# gfx950 (wave64 MFMA) operand prep + tile selection.
#
# The helpers above (preshuffle_a8w4_weight_mbn / quant_act_mxfp8_mbn) target the
# gfx1250 WMMA layout. These are the CDNA4 equivalents, and they are what the
# module docstring means by "preshuffle_operands": callers should not have to
# rediscover which of aiter's several shuffles this kernel wants.
# ---------------------------------------------------------------------------

# A dtypes quant_mx_act can produce. The kernel also takes fp8 A, which is where
# a8w4 lives on gfx950 (aiter's gemm_a8w4_mxfp8 asm path is gfx1250-only).
_MX_A_SUPPORTED = ("fp4", "fp6", "fp8")

# In-process overrides, keyed by (M, N, K, a_dtype). Takes precedence over the
# tuned CSV; mainly for tuners and tests.
MX_TUNED_TILES: dict[tuple[int, int, int, str], tuple[int, int, int]] = {}

# Shape-tuned tiles, same spirit as aiter/configs/a6w6_blockscale_tuned_gemm.csv:
# exact (gfx, cu_num, M, N, K, a_dtype) matches only, everything else falls
# through to the heuristic in pick_mx_tiles. Regenerate with
# op_tests/op_benchmarks/hip/tune_gemm_mx_a6w4.py.
#
# Resolved through AITER_CONFIGS like every other tuned GEMM table, rather than
# by walking up from __file__: that also picks up per-model tables dropped in
# aiter/configs/model_configs/ and honours the AITER_CONFIG_GEMM_A6W4 override.


def _mx_tuned_csv() -> str:
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_GEMM_A6W4_FILE


@functools.cache
def _load_mx_tuned_csv() -> dict[tuple[str, int, int, int, int, str],
                                 tuple[int, int, int]]:
    MX_TUNED_CSV = _mx_tuned_csv()
    if not os.path.exists(MX_TUNED_CSV):
        return {}
    table: dict[tuple[str, int, int, int, int, str], tuple[int, int, int]] = {}
    with open(MX_TUNED_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (row["gfx"].strip(), int(row["cu_num"]), int(row["M"]),
                       int(row["N"]), int(row["K"]), row["a_dtype"].strip())
                tiles = (int(row["tile_m"]), int(row["tile_n"]),
                         int(row["tile_k"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{MX_TUNED_CSV}: bad row {row!r}") from exc
            if key in table:
                raise ValueError(f"{MX_TUNED_CSV}: duplicate key {key}")
            table[key] = tiles
    return table


def clear_mx_tuned_cache() -> None:
    """Drop the cached tuned table after a tuner rewrites the CSV."""
    _load_mx_tuned_csv.cache_clear()


@functools.cache
def _cu_count() -> int:
    """Compute-unit count, for the occupancy check in pick_mx_tiles."""
    try:
        from aiter.jit.utils.chip_info import get_cu_num

        return int(get_cu_num())
    except Exception:  # noqa: BLE001
        return 256  # MI355X



def tiles_are_valid(
    tile_m: int, tile_n: int, tile_k: int, N: int, K: int, a_dtype: str = "fp6"
) -> bool:
    """Is this tile triple launchable through :func:`flydsl_batched_gemm_mxfp4`?

    Kept in sync with the asserts at the top of ``launch_gemm``, tightened to the
    subset the public launcher exposes; picking a tile combination that violates
    them otherwise dies on a raw kernel assert. The last condition is the
    non-obvious one: the A tile is staged into LDS by whole cooperative rounds of
    ``num_threads * 16`` bytes, so an A tile that is not a multiple of that would
    leave part of itself never DMA'd.
    """
    # tile_n is held to % 64, not the kernel's own % 16: flydsl_batched_gemm_mxfp4
    # requires 64, so a 16- or 32-wide tile that the kernel would accept is still
    # unreachable through the public launcher. Keeping the looser rule here would
    # let pick_mx_tiles hand back a tile the documented "pass it straight
    # through" call then rejects.
    if tile_m % 32 != 0 or tile_n % 64 != 0 or tile_k not in (128, 256):
        return False
    if N % tile_n != 0 or K % tile_k != 0:
        return False
    a_row_b = tile_k // 2 if a_dtype == "fp4" else tile_k
    a_lds_b = tile_m * a_row_b
    num_threads = min(4, tile_n // 16) * 64
    return a_lds_b % (num_threads * 16) == 0


def pick_mx_tiles(
    M: int, N: int, K: int, a_dtype: str = "fp6"
) -> tuple[int, int, int]:
    """Pick (tile_m, tile_n, tile_k) for the gfx950 MX preshuffle GEMM.

    Order: tuned table -> preference-ordered search filtered by
    :func:`tiles_are_valid`, so the result is always something the kernel
    accepts and callers can pass it straight through.

    Preferences come from a tile sweep on MI355X (gfx950) over
    (M,N,K) = (1024,4096,4096), (4096,4096,4096), (2048,8192,8192):

      * tile_k=128 beat 256 by 83% and 101% on the two largest shapes -- 256
        halves the number of K chunks and starves the pipeline. Prefer 128.
      * tile_n=256 was in the top configs for every shape measured.
      * tile_m tracks M; 32 is the floor (the A e8m0 scale is 32-row granular)
        and 128 only pays off once M is large enough to fill it.

    Below M~2048 the call is dominated by per-launch dispatch rather than
    compute, so tile choice barely moves the needle there.
    """
    if K % 256 != 0:
        raise RuntimeError(f"[FlyDSL] K ({K}) must be a multiple of 256")

    # Tuned entries are data, and a hand-edited or stale row is not obviously
    # wrong at parse time: tile_m=0 parses fine and then divides by zero in
    # pick_mx_split_k, and a negative tile passes the modulo-only checks in
    # tiles_are_valid. This function promises every result is launchable, so
    # hold tuned rows to the same bar as searched ones.
    def _checked(tiles, origin):
        tile_m, tile_n, tile_k = tiles
        if min(tiles) <= 0 or not tiles_are_valid(
            tile_m, tile_n, tile_k, N, K, a_dtype
        ):
            raise RuntimeError(
                f"[FlyDSL] {origin} gives tiles {tiles} for M={M} N={N} K={K} "
                f"a_dtype={a_dtype!r}, which the kernel cannot launch"
            )
        return tiles

    tuned = MX_TUNED_TILES.get((M, N, K, a_dtype))
    if tuned is not None:
        return _checked(tuned, "MX_TUNED_TILES")
    key = (get_gfx(), _cu_count(), M, N, K, a_dtype)
    tuned = _load_mx_tuned_csv().get(key)
    if tuned is not None:
        return _checked(tuned, f"{_mx_tuned_csv()} row {key}")

    # tile_m: a sweep over the R1 / Qwen3.8-27B shapes picks 32 for essentially
    # every M up to 128 and splits between 32/64/128 above that, so grow it only
    # slowly. A wider M tile mostly costs occupancy on these N/K-heavy shapes.
    if M <= 128:
        m_pref = (32, 64, 128, 256)
    elif M <= 512:
        m_pref = (64, 32, 128, 256)
    else:
        m_pref = (128, 64, 32, 256)
    n_pref = sorted((t for t in (256, 128, 64) if N % t == 0), reverse=True)
    if not n_pref:
        raise RuntimeError(
            f"[FlyDSL] N ({N}) must be a multiple of 64 for the MX preshuffle GEMM"
        )

    # Fat N tiles are only worth it if the grid still fills the device. First
    # pass takes the widest tile_n that still covers the CUs. If nothing covers
    # them -- the common case for decode-shaped M -- the second pass runs tile_n
    # ascending instead, because then the goal flips to *maximising* workgroups:
    # over the tuned R1 / Qwen3.8-27B rows the underfilled cases chose tile_n=64
    # 127 times against 39 for tile_n=256.
    cu = _cu_count()
    for tile_n_order in (n_pref, sorted(n_pref)):
        require_occupancy = tile_n_order is n_pref
        for tile_n in tile_n_order:
            for tile_m in m_pref:
                # tile_k=256 won at every K in the tuned sweep (loop overhead
                # dominates the extra pipelining a 128 chunk buys).
                for tile_k in (256, 128):
                    if not tiles_are_valid(tile_m, tile_n, tile_k, N, K, a_dtype):
                        continue
                    if require_occupancy:
                        workgroups = -(-M // tile_m) * (N // tile_n)
                        if workgroups < cu:
                            continue
                    return tile_m, tile_n, tile_k
    raise RuntimeError(
        f"[FlyDSL] no valid tile combination for M={M} N={N} K={K} "
        f"a_dtype={a_dtype!r}"
    )


def pick_mx_split_k(M: int, N: int, K: int, tile_m: int, tile_n: int,
                    tile_k: int) -> int:
    """Pick ``k_batch`` so the grid covers the device.

    Without split-K the grid is ``ceil(M/tile_m) * (N/tile_n)`` workgroups, which
    for a decode-shaped M leaves most CUs idle: at M=1, N=4096, tile_n=64 that is
    64 workgroups on 256 CUs. Splitting the K loop across ``grid.z`` fills them.

    Take the largest divisor of ``K/tile_k`` that does not overshoot the CU count
    -- measured on MI355X, landing exactly on 256 workgroups gave 2.04x at M=1
    while oversubscribing (448 or 896) gave some of it back.
    """
    base = -(-M // tile_m) * (N // tile_n)
    cu = _cu_count()
    if base >= cu:
        return 1
    k_tiles = K // tile_k
    best = 1
    for kb in range(2, k_tiles + 1):
        if k_tiles % kb:
            continue
        if base * kb > cu:
            break
        best = kb
    return best


def preshuffle_mx_weight(w_bf16: torch.Tensor):
    """Quantize + preshuffle a BF16 weight to the gfx950 MXFP4 B layout.

    Args:
        w_bf16: ``[N, K]`` bf16/fp16 weight (row = output N).
    Returns:
        ``(w_codes [N, K//2] uint8, w_scales [N, K//32] uint8)``, both already
        shuffled -- pass straight to :func:`flydsl_batched_gemm_mxfp4`.
    """
    from aiter.ops.flydsl.mxfp6_utils import shuffle_scale_w4
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import shuffle_weight_NK

    assert w_bf16.dim() == 2, f"expected [N,K], got {tuple(w_bf16.shape)}"
    w_q, w_scale = per_1x32_f4_quant(w_bf16.float())[:2]
    # shuffle_weight_NK(w, 16, 64) is bit-identical to FlyDSL's shuffle_weight_w4;
    # only the scale needs the CDNA4-specific shuffle (shuffle_scale_n32k4 is the
    # gfx1250 WMMA layout, not this one).
    return shuffle_weight_NK(w_q, 16, 64), shuffle_scale_w4(w_scale, 1, False)


def quant_mx_act(a_bf16: torch.Tensor, a_dtype: str = "fp6"):
    """Quantize a BF16 activation to the gfx950 MX A layout.

    Args:
        a_bf16: ``[M, K]`` bf16/fp16 activation.
        a_dtype: ``"fp6"`` (MXFP6-E2M3), ``"fp4"`` (MXFP4-E2M1) or ``"fp8"``
            (MXFP8-E4M3).
    Returns:
        ``(a_codes, a_scales)``. ``a_codes`` is ``[M, K]`` for fp6/fp8 (1 byte
        per code; fp6 is FP8-padded packed) or ``[M, K//2]`` for fp4;
        ``a_scales`` is the shuffled E8M0 scale.
    """
    from aiter import dtypes
    from aiter.ops.flydsl.mxfp6_utils import per_1x32_f6_quant, shuffle_scale_w4
    from aiter.ops.quant import per_1x32_f4_quant, per_1x32_mx_quant_hip

    if a_dtype not in _MX_A_SUPPORTED:
        raise ValueError(
            f"[FlyDSL] quant_mx_act supports {list(_MX_A_SUPPORTED)}; got {a_dtype!r}"
        )
    assert a_bf16.dim() == 2, f"expected [M,K], got {tuple(a_bf16.shape)}"

    M = a_bf16.shape[0]
    # Every quantizer here reduces over 32-row groups, so pad M up before
    # quantizing and keep the scale at that padded height (the kernel indexes it
    # in 32-row supers); only the codes are trimmed back to the real M.
    m_pad = max(32, (M + 31) // 32 * 32)
    if m_pad != M:
        a_bf16 = torch.nn.functional.pad(a_bf16, (0, 0, 0, m_pad - M))

    if a_dtype == "fp6":
        codes, scale, _ = per_1x32_f6_quant(a_bf16)
    elif a_dtype == "fp8":
        codes, scale = per_1x32_mx_quant_hip(
            a_bf16, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    else:
        codes, scale = per_1x32_f4_quant(a_bf16.float())[:2]
    return codes[:M].contiguous(), shuffle_scale_w4(
        scale.view(torch.uint8), 1, False
    )


def preshuffle_operands(a_bf16: torch.Tensor, w_bf16: torch.Tensor, *,
                        a_dtype: str = "fp6"):
    """Convenience wrapper: prepare both operands for a single (B=1) GEMM.

    Real consumers usually split these -- the weight is prepared once at load
    time (:func:`preshuffle_mx_weight`) and the activation every forward
    (:func:`quant_mx_act`).

    Returns ``(a_codes, w_codes, a_scales, w_scales)`` in
    :func:`flydsl_batched_gemm_mxfp4` argument order.
    """
    a_codes, a_scales = quant_mx_act(a_bf16, a_dtype)
    w_codes, w_scales = preshuffle_mx_weight(w_bf16)
    return a_codes, w_codes, a_scales, w_scales


def flydsl_gemm_mxfp4(
    a: torch.Tensor,
    w: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    N: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    a_dtype: str = "fp6",
    tile_m: int | None = None,
    tile_n: int | None = None,
    tile_k: int | None = None,
    k_batch: int | None = None,
) -> torch.Tensor:
    """Un-batched ``[M,K] x [N,K].T -> [M,N]`` MX preshuffle GEMM (gfx950).

    Thin 2D front-end over :func:`flydsl_batched_gemm_mxfp4` (B=1) for linear
    layers, which do not have a batch dimension. Tiles default to
    :func:`pick_mx_tiles`. Operands come from :func:`quant_mx_act` /
    :func:`preshuffle_mx_weight`.
    """
    assert a.dim() == 2, f"expected [M,K] activation, got {tuple(a.shape)}"
    M, a_row = a.shape
    K = a_row * _A_CODES_PER_BYTE[a_dtype]
    if tile_m is None or tile_n is None or tile_k is None:
        auto_m, auto_n, auto_k = pick_mx_tiles(M, N, K, a_dtype)
        tile_m = auto_m if tile_m is None else tile_m
        tile_n = auto_n if tile_n is None else tile_n
        tile_k = auto_k if tile_k is None else tile_k
    if k_batch is None:
        k_batch = pick_mx_split_k(M, N, K, tile_m, tile_n, tile_k)
    out = flydsl_batched_gemm_mxfp4(
        a.view(1, M, a_row), w, a_scales, w_scales, N, dtype,
        a_dtype=a_dtype, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        k_batch=k_batch,
    )
    return out.view(M, N)
