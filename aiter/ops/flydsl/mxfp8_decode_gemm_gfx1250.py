# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 decode GEMM on the native-MXFP8 path:
kernels/mxfp8_decode_gemm_gfx1250.py.

Same operands and bf16 output as ``gemm_a8w8_blockscale(isBpreshuffled=True)``;
the values differ from the tuned GEMM's only by the FP32 K order (rare one-ulp
bf16 flips). Callers check ``mxfp8_decode_gemm_supported`` and keep the tuned
GEMM for the shapes it declines.
"""

from __future__ import annotations

import functools

import torch

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

# Decode token counts only: past 32 rows the tuned GEMM is as fast or faster.
_MAX_ROWS = 32
# K-waves of a workgroup: with one 128-wide K step each, every weight load of
# the workgroup is in flight at once.
_MAX_K_WAVES = 16


@functools.cache
def mxfp8_decode_gemm_config(m: int, n: int, k: int):
    """The kernel config for ``m`` rows of an (n, k) GEMM, or None (not served).

    One M tile (16 rows up to 16 rows, else 32); one K-wave per 128-wide K step
    (at most 16, each taking an equal share of the K steps); and the fewest
    16-column blocks per workgroup that keep the grid within one workgroup per
    CU. Weight loads are non-temporal and the activation goes to LDS by TDM.
    """
    from aiter.jit.utils.chip_info import get_cu_num

    from .kernels.mxfp8_decode_gemm_gfx1250 import decode_gemm_config_error

    if not 0 < m <= _MAX_ROWS or n % 32 or k % 128:
        return None
    tile_m = 16 if m <= 16 else 32
    steps = k // 128
    kw = max(w for w in range(1, _MAX_K_WAVES + 1) if steps % w == 0)
    n_blocks = n // 16
    nb = next(
        nb
        for nb in range(1, n_blocks + 1)
        if n_blocks % nb == 0 and n_blocks // nb <= get_cu_num()
    )
    # (tile_m, nb, ks, kw, wn, wdepth, wcpol, atdm)
    cfg = (tile_m, nb, 1, kw, 1, 1, 1, 1)
    if decode_gemm_config_error(n, k, tile_m, nb, 1, kw, 1, 1, atdm=1):
        return None
    return cfg


def mxfp8_decode_gemm_supported(m: int, n: int, k: int) -> bool:
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return (
        get_gfx_runtime() == "gfx1250" and mxfp8_decode_gemm_config(m, n, k) is not None
    )


@functools.cache
def _launcher():
    from .kernels.mxfp8_decode_gemm_gfx1250 import launch_mxfp8_decode_gemm

    return launch_mxfp8_decode_gemm


@capture_scoped_cache
def _slab_cached(device: torch.device, stream_id: int, numel: int) -> torch.Tensor:
    """FP32 split partials, one per (device, stream); a capture gets its own."""
    del stream_id
    with persistent_alloc(device):
        return torch.empty(numel, dtype=torch.float32, device=device)


def _slab(device, numel):
    bucket = 1 << max(20, (numel - 1).bit_length())
    return _slab_cached(device, torch.cuda.current_stream(device).cuda_stream, bucket)


def mxfp8_decode_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    config: tuple | None = None,
) -> torch.Tensor:
    """out[M, N] bf16 = x [M, K] (e4m3, unit column stride, 16-byte row stride) with
    x_scale [M, K / 32] (e8m0, row-major) times w [N, K] ((16, 16)-preshuffled e4m3)
    with w_scale (32x4-shuffled e8m0, shuffle_mxfp8_bpreshuffle_scale)."""
    import flydsl.expr as fx

    from .kernels.tensor_shim import _run_compiled, ptr_arg

    m, k = x.shape
    n = w.shape[0]
    cfg = config or mxfp8_decode_gemm_config(m, n, k)
    if cfg is None:
        raise ValueError(f"mxfp8_decode_gemm: no config for M={m} N={n} K={k}")
    if (
        x.dtype != torch.float8_e4m3fn
        or tuple(w.shape) != (n, k)
        or x.stride(1) != 1
        or x.stride(0) % 16
        or tuple(x_scale.shape) != (m, k // 32)
        or not x_scale.is_contiguous()
        or tuple(w_scale.shape) != (n, k // 32)
        or not (w.is_contiguous() and w_scale.is_contiguous())
    ):
        raise ValueError("mxfp8_decode_gemm: operand shapes / strides")
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    elif (
        out.dtype != torch.bfloat16
        or tuple(out.shape) != (m, n)
        or not out.is_contiguous()
    ):
        raise ValueError("mxfp8_decode_gemm: out must be a contiguous bf16 [M, N]")
    if m == 0:
        return out
    cfg = tuple(cfg) + (0,) * (8 - len(cfg))
    tile_m, nb, ks, kw, wn, wdepth, wcpol, atdm = cfg
    mpad = (m + tile_m - 1) // tile_m * tile_m
    slab = _slab(x.device, ks * mpad * n) if ks > 1 else out
    stream = torch.cuda.current_stream(x.device)
    spec = (n, k, tile_m, nb, ks, kw, wn, wdepth, wcpol, atdm)
    _run_compiled(
        _launcher(),
        ptr_arg(x), ptr_arg(x_scale), ptr_arg(w), ptr_arg(w_scale), ptr_arg(out), ptr_arg(slab),
        m, x.stride(0), mpad, fx.Stream(stream),
        *spec,
        specialization_key=("mxfp8_decode_gemm", x.device.index) + spec,
    )  # fmt: skip
    return out
