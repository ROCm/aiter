# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Strided-batched MXFP4/MXFP6/MXFP8 preshuffle GEMM, ported from FlyDSL.

out[b] = dequant(x[b]) @ dequant(w[b]).T  (per-1x32 e8m0 scales folded into a scaled
16x16x128 MFMA). gfx950 only. The batch index rides grid.z; weights and e8m0 scales are
CK-preshuffled host-side before launch.

Two batched layouts for the A input and C output (weights stay batch-contiguous):
  - bmn: contiguous [B, M, *] / [B, M, N] (plain per-batch).
  - mbn: physical [M, B, *] / [M, B, N], the deepseek-v4 grouped-output path. The
    interleaving + explicit kernel strides are handled here; callers pass logical
    [B, M, *] data either way and get a logical [B, M, N] result. NOTE: the mbn result
    is a non-contiguous view (physically [M, B, N]) — call .contiguous() before ops
    that require contiguity.
"""

from __future__ import annotations

import functools

import torch

import flydsl.compiler as flyc

from aiter.jit.utils.chip_info import get_gfx

from .kernels.mxfp4_preshuffle import compile_mxfp4_gemm
from .kernels.mxfp4_preshuffle_utils import shuffle_scale_w4, shuffle_weight_w4

SCALE_GROUP_SIZE = 32

# a_dtype -> A bytes per code (fp4 = 2 codes/byte; fp6/fp8 = 1 byte/code).
_A_CODES_PER_BYTE = {"fp4": 2, "fp6": 1, "fp8": 1}


@functools.lru_cache(maxsize=None)
def _get_launcher(
    N, K, tile_m, tile_n, tile_k, a_dtype, out_dtype, use_async_copy, batch, strides
):
    return compile_mxfp4_gemm(
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype=a_dtype,
        out_dtype=out_dtype,
        use_async_copy=use_async_copy,
        batch=batch,
        **dict(strides),
    )


def _shuffled_operands(x, w, x_scales, w_scales, B, M, K):
    """Per-batch preshuffle: A codes plain (real M rows), B + both scales CK-shuffled,
    A-scales padded to M/32. Returns lists (a, sa) and the concatenated (B_flat, SB_flat).
    """
    M32 = (M + 31) // 32 * 32
    a_list, sa_list, b_list, sb_list = [], [], [], []
    for b in range(B):
        a_list.append(x[b].contiguous().view(-1))
        b_list.append(shuffle_weight_w4(w[b].contiguous(), 16, False, False).view(-1))
        sa = x_scales[b]
        if M32 != M:
            pad = torch.zeros(
                (M32 - M, K // SCALE_GROUP_SIZE), dtype=sa.dtype, device=sa.device
            )
            sa = torch.cat([sa, pad], dim=0)
        sa_list.append(shuffle_scale_w4(sa.contiguous(), 1, False).view(-1))
        sb_list.append(shuffle_scale_w4(w_scales[b].contiguous(), 1, False).view(-1))
    return a_list, sa_list, torch.cat(b_list), torch.cat(sb_list)


def flydsl_batched_gemm_mxfp4(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    *,
    a_dtype: str = "fp4",
    layout: str = "bmn",
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 256,
    use_async_copy: bool = False,
) -> torch.Tensor:
    """Strided-batched MXFP A x MXFP4 B preshuffle GEMM (gfx950).

    a_dtype='fp4' (a4w4): x is (B, M, K//2) uint8 fp4 codes (2 codes/byte).
    a_dtype='fp8' (a8w4): x is (B, M, K) uint8 MXFP8 E4M3 codes (1 byte/code).
    a_dtype='fp6' (a6w4): x is (B, M, K) uint8 FP8-padded MXFP6 E2M3 codes; the kernel
        supports it but this entry point is NOT exercised by the test suite (unvalidated).
    w:        (B, N, K//2) uint8 fp4 codes (plain [N, K//2], shuffled host-side).
    x_scales: (B, M, K//32) uint8 e8m0; w_scales: (B, N, K//32) uint8 e8m0.
    layout='bmn' -> [B,M,*] A/C; 'mbn' -> [M,B,*] A/C (deepseek grouped-output). The mbn
        result is a non-contiguous [B,M,N] view of a physical [M,B,N] buffer.
    Returns out (B, M, N) in `dtype` (bf16/fp16).
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

    B, M, _ = x.shape
    N = w.shape[1]
    K = x_scales.shape[-1] * SCALE_GROUP_SIZE

    # tile_m % 16 (m_chunks = BM//16) and tile_n % 64 (num_acc_n = (BN//4)//16) must be
    # exact or the kernel's chunk counts truncate and silently drop work.
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
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"[FlyDSL] unsupported out dtype {dtype}")

    a_list, sa_list, B_flat, SB_flat = _shuffled_operands(
        x, w, x_scales, w_scales, B, M, K
    )
    a_row_bytes = a_list[0].numel() // M
    mchunks = (M + 31) // 32
    chunk_bytes = sa_list[0].numel() // mchunks

    strides = {}
    if layout == "mbn":
        # A / scale_a / C interleaved as [M,B,*] / [ceil(M/32),B,chunk] / [M,B,N].
        A_flat = (
            torch.stack([a.view(M, a_row_bytes) for a in a_list], dim=1)
            .contiguous()
            .view(-1)
        )
        SA_flat = (
            torch.stack([s.view(mchunks, chunk_bytes) for s in sa_list], dim=1)
            .contiguous()
            .view(-1)
        )
        strides = dict(
            a_row_stride=B * a_row_bytes,
            a_batch_stride=a_row_bytes,
            sca_row_stride=B * (chunk_bytes // 4),
            sca_batch_stride=chunk_bytes,
            c_row_stride=B * N,
            c_batch_stride=N * 2,
        )
        out_phys = torch.empty((M, B, N), dtype=dtype, device=x.device)
    else:
        A_flat = torch.cat(a_list)
        SA_flat = torch.cat(sa_list)
        out_phys = torch.empty((B, M, N), dtype=dtype, device=x.device)

    C_flat = out_phys.view(-1)
    bias = torch.empty(0, dtype=dtype, device=x.device)
    stream = torch.cuda.current_stream()

    launcher = _get_launcher(
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        a_dtype,
        out_dtype,
        use_async_copy,
        B,
        tuple(sorted(strides.items())),
    )
    args = (C_flat, A_flat, B_flat, SA_flat, SB_flat, bias, M, N, stream)

    # flyc.compile bakes tensor sizes, so cache the CompiledFunction per M.
    cf = getattr(launcher, "_cf_cache", {}).get(M)
    if cf is not None:
        cf(*args)
    else:
        try:
            cf = flyc.compile(launcher, *args)  # first call compiles AND executes
        except Exception:
            # flyc.compile leaks ir.Context on failure, poisoning subsequent JIT calls;
            # drain it to isolate the failure (see moe_kernels._run_compiled).
            try:
                from flydsl._mlir import ir

                while ir.Context.current is not None:
                    ir.Context.current.__exit__(None, None, None)
            except Exception:
                pass
            raise
        cache = getattr(launcher, "_cf_cache", None)
        if cache is None:
            cache = launcher._cf_cache = {}
        cache[M] = cf

    # mbn C physical [M,B,N] -> logical [B,M,N] view.
    return out_phys.transpose(0, 1) if layout == "mbn" else out_phys
