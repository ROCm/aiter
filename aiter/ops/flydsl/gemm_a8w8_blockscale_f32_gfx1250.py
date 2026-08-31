# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for the a8w8 blockscale GEMM with f32 block scales.

Scales are f32 and applied after each WMMA
"""

from __future__ import annotations

import functools

import torch

_compile_gemm_a8w8_blockscale = None
_run_compiled = None
_flyc = None
_fx = None


def _lazy_import():
    global _compile_gemm_a8w8_blockscale, _run_compiled, _flyc, _fx
    if _compile_gemm_a8w8_blockscale is not None:
        return
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a8w8_blockscale_f32_kernel_gfx1250 import (
        compile_gemm_a8w8_blockscale,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled as run_compiled

    _compile_gemm_a8w8_blockscale = compile_gemm_a8w8_blockscale
    _run_compiled = run_compiled
    _flyc = flyc
    _fx = fx


_FX_DTYPE = {}


def _p(t):
    if not _FX_DTYPE:
        _FX_DTYPE.update(
            {
                torch.float8_e4m3fn: _fx.Float8E4M3FN,
                torch.bfloat16: _fx.BFloat16,
                torch.float16: _fx.Float16,
                torch.float32: _fx.Float32,
            }
        )
    return _flyc.from_c_void_p(_FX_DTYPE[t.dtype], t.data_ptr())


@functools.lru_cache(maxsize=1024)
def _cached_launcher(*cfg):
    keys = (
        "N",
        "K",
        "tile_m",
        "tile_n",
        "tile_k",
        "m_warp",
        "n_warp",
        "scale_block_k",
        "scale_block_n",
        "num_buffers",
        "waves_per_eu",
        "out_dtype",
        "variant",
        "kernarg_preload",
        "split_k",
    )
    return _compile_gemm_a8w8_blockscale(**dict(zip(keys, cfg)))


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
    assert (
        x.stride(1) == 1 or K == 1
    ), f"x needs unit-stride K for TDM, got strides {tuple(x.stride())}"
    assert (
        w.stride(1) == 1 or w.shape[1] == 1
    ), f"w needs unit-stride K for TDM, got strides {tuple(w.stride())}"
    assert (
        x_scale.stride(1) == 1 or scale_k == 1
    ), f"x_scale needs unit-stride K, got strides {tuple(x_scale.stride())}"
    assert w_scale.is_contiguous(), "w_scale must be contiguous"
    _splitk_f32_accum = split_k > 1 and dtype in (torch.bfloat16, torch.float16)
    buf_dtype = torch.float32 if _splitk_f32_accum else dtype
    _half_str = "fp16" if dtype == torch.float16 else "bf16"
    out_dtype_str = "f32" if _splitk_f32_accum or dtype == torch.float32 else _half_str

    if y is not None:
        assert y.shape == (M, N), f"y shape {y.shape} != ({M}, {N})"
        assert y.dtype == dtype, f"y dtype {y.dtype} != {dtype}"

    _alloc = torch.zeros if split_k > 1 else torch.empty
    if _splitk_f32_accum or y is None:
        y_buf = _alloc((M, N), dtype=buf_dtype, device=x.device)
    else:
        y_buf = y
        if split_k > 1:
            y_buf.zero_()

    launcher = _cached_launcher(
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        scale_block_k,
        scale_block_n,
        num_buffers,
        waves_per_eu,
        out_dtype_str,
        variant,
        kernarg_preload,
        split_k,
    )

    lda = x.stride(0) if M > 1 else K
    ldb = w.stride(0) if w.shape[0] > 1 else w.shape[1]
    lds = (x_scale.stride(0) if M > 1 else x_scale.shape[1]) * 4
    assert y_buf.stride(1) == 1, "output must have unit-stride N"
    ldc = y_buf.stride(0) if y_buf.shape[0] > 1 else y_buf.shape[1]

    stream = torch.cuda.current_stream(device=x.device).cuda_stream
    _run_compiled(
        launcher,
        _p(y_buf),
        _p(x),
        _p(w),
        _p(x_scale),
        _p(w_scale),
        M,
        N,
        ldc,
        lda,
        ldb,
        lds,
        _fx.Stream(stream),
    )

    if not _splitk_f32_accum:
        return y_buf
    if y is not None:
        y.copy_(y_buf)
        return y
    return y_buf.to(dtype)


__all__ = ["gemm_a8w8_blockscale"]
