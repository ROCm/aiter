# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A8W8 FP8 GEMM wrapper with automatic config lookup.

Computes: Y = (X @ W^T) * (x_scale * w_scale) [+ bias]

Mirrors the aiter gfx1250_gemm_a8w8 interface: pass tensors in, get results out.
Handles padding, config loading, compilation, and launch internally.

Usage:
    from aiter.ops.flydsl.gemm_a8w8 import flydsl_gemm_a8w8

    y = flydsl_gemm_a8w8(x_fp8, w_fp8, x_scale, w_scale)
    y = flydsl_gemm_a8w8(x_fp8, w_fp8, x_scale, w_scale, bias=bias, dtype=torch.float16)
    y = flydsl_gemm_a8w8(x_fp8, w_fp8, x_scale, w_scale, config={"tile_m": 64, ...})
"""

import warnings
from typing import Optional

import torch

from .kernels.gemm_a8w8_gfx1250 import compile_gemm_a8w8
from .gemm_config_utils import get_gemm_config


def _pad_to(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile


def flydsl_gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
) -> torch.Tensor:
    """Compute 8-bit GEMM: Y = (X @ W^T) * (x_scale * w_scale) [+ bias].

    Args:
        x: FP8 input [M, K], contiguous in K.
        w: FP8 weight [N, K], contiguous in K.
        x_scale: Per-token scale [M] or [M, 1], float32.
        w_scale: Per-channel scale [N] or [1, N] or [N], float32.
        bias: Optional bias [N], float32.
        dtype: Output dtype (torch.bfloat16, torch.float16, or torch.float32).
        y: Optional pre-allocated output [M, N].
        config: Optional dict overriding auto-selected config params.

    Returns:
        Output tensor [M, N] in the requested dtype.
    """
    assert x.shape[1] == w.shape[1], f"Incompatible K dims: x={x.shape[1]}, w={w.shape[1]}"
    assert x.dtype == w.dtype, f"Input dtypes must match: {x.dtype} vs {w.dtype}"

    M, K = x.shape
    N = w.shape[0]

    _dtype_map = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
    out_dtype = _dtype_map.get(dtype)
    if out_dtype is None:
        raise ValueError(f"Unsupported output dtype: {dtype}")

    # Load config if not provided
    if config is None:
        config, _ = get_gemm_config("GEMM-A8W8", M, N=N, K=K)

    tile_m = config.get("tile_m", 128)
    tile_n = config.get("tile_n", 256)
    tile_k = config.get("tile_k", 128)
    m_warp = config.get("m_warp", 2)
    n_warp = config.get("n_warp", 4)
    num_buffers = config.get("num_buffers", 2)
    use_tdm_load = config.get("use_tdm_load", True)
    waves_per_eu = config.get("waves_per_eu", None)
    cluster_m = config.get("cluster_m", 1)
    cluster_n = config.get("cluster_n", 1)
    l2_prefetch_distance = config.get("l2_prefetch_distance", 0)
    has_bias = bias is not None

    # Pad dimensions to tile boundaries
    mpad = _pad_to(M, tile_m)
    npad = _pad_to(N, tile_n)
    kpad = _pad_to(K, tile_k)

    # Ensure num_buffers doesn't exceed K tiles
    num_k_tiles = kpad // tile_k
    if num_buffers > num_k_tiles:
        num_buffers = 2

    # Ensure K contiguity
    if x.stride(1) != 1:
        warnings.warn(
            "x is not contiguous in K. Making a contiguous copy.",
            stacklevel=2,
        )
        x = x.contiguous()
    if w.stride(1) != 1:
        warnings.warn(
            "w is not contiguous in K. Making a contiguous copy.",
            stacklevel=2,
        )
        w = w.contiguous()

    device = x.device

    # Pad inputs if needed
    if mpad != M or kpad != K:
        a_pad = torch.zeros((mpad, kpad), dtype=x.dtype, device=device)
        a_pad[:M, :K] = x
    else:
        a_pad = x

    if npad != N or kpad != K:
        b_pad = torch.zeros((npad, kpad), dtype=w.dtype, device=device)
        b_pad[:N, :K] = w
    else:
        b_pad = w

    # Flatten scales
    x_scale_1d = x_scale.reshape(-1)
    w_scale_1d = w_scale.reshape(-1)

    if mpad != M:
        xs_pad = torch.zeros(mpad, dtype=torch.float32, device=device)
        xs_pad[:M] = x_scale_1d
        x_scale_1d = xs_pad

    if npad != N:
        ws_pad = torch.zeros(npad, dtype=torch.float32, device=device)
        ws_pad[:N] = w_scale_1d
        w_scale_1d = ws_pad

    bias_1d = torch.zeros(npad, dtype=torch.float32, device=device)
    if has_bias:
        bias_1d[:N] = bias.reshape(-1)

    # Output
    c_pad = torch.zeros((mpad, npad), dtype=dtype, device=device)

    # Compile
    launch_fn = compile_gemm_a8w8(
        K=kpad,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        num_buffers=num_buffers,
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_load=use_tdm_load,
        out_dtype=out_dtype,
        has_bias=has_bias,
    )

    # Launch
    launch_fn(
        c_pad.contiguous().view(-1),
        a_pad.contiguous().view(-1),
        b_pad.contiguous().view(-1),
        x_scale_1d.contiguous(),
        w_scale_1d.contiguous(),
        bias_1d.contiguous(),
        mpad,
        npad,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    # Extract unpadded result
    result = c_pad[:M, :N]

    if y is not None:
        y.copy_(result)
        return y
    return result.contiguous()
