# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for the a8w8 blockscale GEMM with f32 block scales.

Scales are f32 and applied after each WMMA
"""

from __future__ import annotations

import torch

_compile_gemm_a8w8_blockscale = None


def _lazy_import():
    """Defer the kernel import so ``import aiter`` does not pull in flydsl."""
    global _compile_gemm_a8w8_blockscale
    if _compile_gemm_a8w8_blockscale is not None:
        return
    # Absolute (not relative) so this module also works when loaded by file path.
    from aiter.ops.flydsl.kernels.gemm_a8w8_blockscale_f32_kernel_gfx1250 import (
        compile_gemm_a8w8_blockscale,
    )

    _compile_gemm_a8w8_blockscale = compile_gemm_a8w8_blockscale


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    y: torch.Tensor = None,
    dtype: torch.dtype = torch.bfloat16,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    variant: str = "compute_bound",
    kernarg_preload: bool = False,
    split_k: int = 1,
):
    """Compute Y = (X @ W^T) with per-block f32 scales (A8W8 blockscale).

    variant: "compute_bound" (default) or "memory_bound".
    """
    _lazy_import()
    assert x.ndim == 2 and w.ndim == 2, "X and W must be 2D"
    M, K = x.shape
    N = w.shape[0] * 16
    K_w = w.shape[1] // 16
    assert K == K_w, f"K mismatch: X has {K}, W has {K_w}"
    assert x_scale.ndim == 2 and w_scale.ndim == 2, "scales must be 2D"
    assert x_scale.shape[0] == M, f"x_scale rows {x_scale.shape[0]} != M {M}"
    scale_k_x = x_scale.shape[1]
    scale_n, scale_k_w = w_scale.shape
    assert scale_k_x == scale_k_w, f"scale_k {scale_k_x} != {scale_k_w}"
    scale_k = scale_k_x
    assert scale_k == -(-K // scale_block_k), f"bad scale_k {scale_k} for K {K}"
    assert scale_n == -(-N // scale_block_n), f"bad scale_n {scale_n} for N {N}"
    assert dtype in (torch.bfloat16, torch.float16, torch.float32), f"bad dtype {dtype}"
    _splitk_f32_accum = split_k > 1 and dtype in (torch.bfloat16, torch.float16)
    buf_dtype = torch.float32 if _splitk_f32_accum else dtype
    _half_str = "fp16" if dtype == torch.float16 else "bf16"
    out_dtype_str = "f32" if _splitk_f32_accum or dtype == torch.float32 else _half_str

    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size * 16))
        new_scale_k = K_padded // scale_block_k
        scale_pad = new_scale_k - scale_k
        if scale_pad > 0:
            x_scale = torch.nn.functional.pad(x_scale, (0, scale_pad))
            w_scale = torch.nn.functional.pad(w_scale, (0, scale_pad))
        K = K_padded

    # Pad N up to tile_n so the kernel's WMMAs and stores land inside the allocated output.
    N_stride = ((N + tile_n - 1) // tile_n) * tile_n

    if y is not None:
        assert y.shape == (M, N), f"y shape {y.shape} != ({M}, {N})"
        assert y.dtype == dtype, f"y dtype {y.dtype} != {dtype}"

    _alloc = torch.zeros if split_k > 1 else torch.empty
    if _splitk_f32_accum or N_stride != N:
        y_buf = _alloc((M, N_stride), dtype=buf_dtype, device=x.device)
    elif y is not None:
        y_buf = y
        if split_k > 1:
            y_buf.zero_()
    else:
        y_buf = _alloc((M, N), dtype=dtype, device=x.device)

    launcher = _compile_gemm_a8w8_blockscale(
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        out_dtype=out_dtype_str,
        variant=variant,
        kernarg_preload=kernarg_preload,
        split_k=split_k,
    )

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    launcher(y_buf, x, w, x_scale, w_scale, M, N_stride, stream=stream)

    result = y_buf[:, :N] if N_stride != N else y_buf
    if _splitk_f32_accum:
        result = result.to(dtype)
    if y is None or result is y:
        return result
    y.copy_(result)
    return y


__all__ = ["gemm_a8w8_blockscale"]
