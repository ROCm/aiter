# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for the A16W16 GEMM: torch-facing wrapper.

The compiled kernel lives in ``kernels/gemm_a16w16_kernel_gfx1250.py``; this
module owns layout detection, padding, output allocation and dtype mapping.
"""

from __future__ import annotations

import torch

_compile_gemm_a16w16 = None


def _lazy_import():
    """Defer the kernel import so ``import aiter`` does not pull in flydsl."""
    global _compile_gemm_a16w16
    if _compile_gemm_a16w16 is not None:
        return
    # Absolute (not relative) so this module also works when loaded by file path.
    from aiter.ops.flydsl.kernels.gemm_a16w16_kernel_gfx1250 import (
        compile_gemm_a16w16,
    )

    _compile_gemm_a16w16 = compile_gemm_a16w16


def gemm_a16w16(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float16,
    y: torch.Tensor | None = None,
    activation: str | None = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 32,
    m_warp: int = 2,
    n_warp: int = 4,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    l2_prefetch_distance: int = 2,
    kernarg_preload: bool = False,
    split_k: int = 1,
    sched_strategy: str | None = None,
    main_loop_unroll: bool = False,
    variant: str = "bandwidth_bound",
):
    """Compute Y = X @ W^T + bias. Auto-detects physical layout from strides."""
    _lazy_import()
    _half = (torch.float16, torch.bfloat16)
    assert x.dtype in _half, f"x must be fp16/bf16, got {x.dtype}"
    assert w.dtype in _half, f"w must be fp16/bf16, got {w.dtype}"
    assert x.shape[1] == w.shape[1], "Incompatible K dimensions"

    M, K, N = x.shape[0], x.shape[1], w.shape[0]
    physical_mk = x.stride(1) == 1
    physical_kn = w.stride(1) != 1

    K_padded = ((K + tile_k - 1) // tile_k) * tile_k
    if K_padded != K:
        pad_size = K_padded - K
        if physical_mk:
            x = torch.nn.functional.pad(x, (0, pad_size))
        else:
            x = torch.nn.functional.pad(x.T, (0, 0, 0, pad_size)).T

        if physical_kn:
            if w.stride(1) == 1:
                w = torch.nn.functional.pad(w, (0, 0, 0, pad_size))
            else:
                w = torch.nn.functional.pad(w.T, (0, 0, 0, pad_size)).T
        else:
            w = torch.nn.functional.pad(w, (0, pad_size))
        K = K_padded

    N_stride = ((N + tile_n - 1) // tile_n) * tile_n
    _splitk_f32_accum = split_k > 1 and dtype in _half
    in_dtype_str = "fp16" if x.dtype == torch.float16 else "bf16"
    out_dtype_str = {torch.float16: "f16", torch.bfloat16: "bf16"}.get(dtype, "f32")
    buf_dtype = dtype
    if _splitk_f32_accum:
        out_dtype_str, buf_dtype = "f32", torch.float32

    _alloc = torch.zeros if split_k > 1 else torch.empty
    if y is not None and not _splitk_f32_accum:
        y_buf = (
            y
            if N_stride == N
            else _alloc((M, N_stride), device=x.device, dtype=buf_dtype)
        )
        if split_k > 1 and y_buf is y:
            y_buf.zero_()
    else:
        y_buf = (
            _alloc((M, N_stride), device=x.device, dtype=buf_dtype)
            if N_stride != N
            else _alloc((M, N), device=x.device, dtype=buf_dtype)
        )

    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=dtype)

    launch_fn = _compile_gemm_a16w16(
        M=M if not physical_mk else 0,
        N=N,
        K=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        in_dtype=in_dtype_str,
        out_dtype=out_dtype_str,
        num_buffers=num_buffers,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        activation=activation,
        add_bias=(bias.numel() > 0),
        physical_mk=physical_mk,
        physical_kn=physical_kn,
        kernarg_preload=kernarg_preload,
        split_k=split_k,
        sched_strategy=sched_strategy,
        main_loop_unroll=main_loop_unroll,
        variant=variant,
    )

    launch_fn(y_buf, x, w, bias, M, N_stride, stream=torch.cuda.current_stream())

    result = y_buf[:, :N] if N_stride != N else y_buf
    if _splitk_f32_accum:
        result = result.to(dtype)
    if y is None or result is y:
        return result
    y.copy_(result)
    return y


__all__ = ["gemm_a16w16"]
