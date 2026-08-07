# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strided-batched a8w4 (MXFP8 E4M3 A x MXFP4 B) preshuffle GEMM launcher for the
gfx1250 wave32 WMMA path: out[b] = dequant(x[b]) @ dequant(w[b]).T, per-1x32 e8m0
scales folded into a scaled 16x16x128 matrix op. Operands are preshuffled + laid out
by the caller (once, off the launch path) -- see ``preshuffle_a8w4_weight_mbn`` and
``quant_act_mxfp8_mbn``. layout 'bmn' = contiguous [B,M,N], 'mbn' = the deepseek-v4
grouped-output [M,B,N] (returned as a non-contiguous [B,M,N] view)."""

from __future__ import annotations

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.tensor_shim import ptr_arg

WMMA_K_GFX1250 = 128


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
    except Exception:
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
