# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for the A16W16 GEMM: torch-facing wrapper."""

from __future__ import annotations

import functools
import re

import torch

_compile_gemm_a16w16 = None
_run_compiled = None
_ptr_arg = None
_fx = None


def _lazy_import():
    global _compile_gemm_a16w16, _run_compiled, _ptr_arg, _fx
    if _compile_gemm_a16w16 is not None:
        return
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a16w16_kernel_gfx1250 import (
        compile_gemm_a16w16,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled as run_compiled
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compile_gemm_a16w16 = compile_gemm_a16w16
    _run_compiled = run_compiled
    _ptr_arg = ptr_arg
    _fx = fx


_FX_DTYPE = {}


def _p(t):
    if not _FX_DTYPE:
        _FX_DTYPE.update(
            {
                torch.bfloat16: _fx.BFloat16,
                torch.float16: _fx.Float16,
                torch.float32: _fx.Float32,
                torch.int32: _fx.Int32,
            }
        )
    return _ptr_arg(t, _FX_DTYPE[t.dtype])


_CFG_KEYS = (
    "N",
    "K",
    "tile_m",
    "tile_n",
    "tile_k",
    "m_warp",
    "n_warp",
    "in_dtype",
    "out_dtype",
    "num_buffers",
    "waves_per_eu",
    "activation",
    "add_bias",
    "physical_mk",
    "physical_kn",
    "kernarg_preload",
    "split_k",
    "sched_strategy",
    "main_loop_unroll",
    "variant",
)


@functools.lru_cache(maxsize=1024)
def _cached_launcher(*cfg):
    return _compile_gemm_a16w16(**dict(zip(_CFG_KEYS, cfg)))


# flydsl_a16w16_gfx1250_t{tm}x{tn}x{tk}_mw{mw}_nw{nw}_nb{nb}_sk{sk}_unroll{0|1}
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_a16w16_gfx1250_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_unroll(?P<main_loop_unroll>[01])$"
)


def parse_kernel_name(name: str):
    """Parse a flydsl_a16w16_gfx1250_ kernelName into its config dict, or None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    if m is None:
        return None
    cfg = {k: int(v) for k, v in m.groupdict().items()}
    cfg["main_loop_unroll"] = bool(cfg["main_loop_unroll"])
    return cfg


def run_gemm_a16w16_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
    dtype: torch.dtype,
    kernel_name: str,
) -> torch.Tensor:
    """Dispatch entry: decode a tuned kernelName and run the kernel."""
    cfg = parse_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(f"[FlyDSL gfx1250] unrecognised kernelName: {kernel_name!r}")
    return gemm_a16w16(x, w, bias=bias, dtype=dtype, **cfg)


_SPLIT_K_MAX_TILES = 1 << 16


@functools.cache
def _split_k_counters(device, stream):
    return torch.zeros(_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device)


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
    assert (
        x.stride(1) == 1 or x.stride(0) == 1 or 1 in (M, K)
    ), f"gemm_a16w16: x needs a unit-stride dim for TDM, got strides {tuple(x.stride())}"
    assert (
        w.stride(1) == 1 or w.stride(0) == 1 or 1 in (N, K)
    ), f"gemm_a16w16: w needs a unit-stride dim for TDM, got strides {tuple(w.stride())}"
    physical_mk = x.stride(1) == 1 or K == 1
    physical_kn = w.stride(1) != 1 and N > 1

    in_dtype_str = "fp16" if x.dtype == torch.float16 else "bf16"
    out_dtype_str = {torch.float16: "f16", torch.bfloat16: "bf16"}.get(dtype, "f32")

    if y is not None:
        assert (
            y.shape[0] == M and y.shape[1] == N
        ), f"y must be ({M}, {N}), got {tuple(y.shape)}"
        assert y.stride(1) == 1 or 1 in (
            M,
            N,
        ), f"gemm_a16w16: y needs unit column stride, got strides {tuple(y.stride())}"
        y_buf = y
    else:
        y_buf = torch.empty((M, N), device=x.device, dtype=dtype)
    ldy = y_buf.stride(0) if M > 1 else N
    assert ldy >= N, f"gemm_a16w16: y row stride {ldy} < N {N}"

    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=dtype)

    stream = torch.cuda.current_stream(device=x.device)
    sem = _split_k_counters(x.device, stream)
    if split_k > 1:
        tiles = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
        assert (
            tiles <= sem.numel()
        ), f"gemm_a16w16: split_k needs {tiles} tile counters, max {sem.numel()}"
        m_pad = ((M + tile_m - 1) // tile_m) * tile_m
        n_pad = ((N + tile_n - 1) // tile_n) * tile_n
        ws = torch.empty((split_k, m_pad, n_pad), device=x.device, dtype=torch.float32)
    else:
        ws = y_buf

    launch_fn = _cached_launcher(
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        in_dtype_str,
        out_dtype_str,
        num_buffers,
        waves_per_eu,
        activation,
        bias.numel() > 0,
        physical_mk,
        physical_kn,
        kernarg_preload,
        split_k,
        sched_strategy,
        main_loop_unroll,
        variant,
    )

    if physical_mk:
        lda = x.stride(0) if M > 1 else K
    else:
        lda = x.stride(1) if K > 1 else M
    if physical_kn:
        ldb = w.stride(1) if K > 1 else N
    else:
        ldb = w.stride(0) if N > 1 else K
    _run_compiled(
        launch_fn,
        _p(y_buf),
        _p(x),
        _p(w),
        _p(bias),
        _p(ws),
        _p(sem),
        M,
        ldy,
        lda,
        ldb,
        _fx.Stream(stream.cuda_stream),
    )
    return y_buf


__all__ = ["gemm_a16w16", "parse_kernel_name", "run_gemm_a16w16_gfx1250"]
