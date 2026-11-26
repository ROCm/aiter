# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for the A16W16 GEMM: torch-facing wrapper."""

from __future__ import annotations

import functools

import torch

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

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
    "split_cat",
)


@functools.lru_cache(maxsize=1024)
def _cached_launcher(*cfg):
    return _compile_gemm_a16w16(**dict(zip(_CFG_KEYS, cfg)))


_SPLIT_K_MAX_TILES = 4096


@capture_scoped_cache
def _split_k_counters(device, stream):
    with persistent_alloc(torch.device(device)):
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
    k_out: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
    k_rope: torch.Tensor | None = None,
    split_sizes: tuple[int, int] | None = None,
):
    """Compute Y = X @ W^T + bias. Auto-detects physical layout from strides.

    With ``split_sizes=(k_size, v_size)``, Y's N columns are heads of
    ``k_size + v_size`` columns, as in MLA's ``kv_b_proj`` output. They are
    stored into ``k_out[M, heads, k_size + rope]`` and ``v_out[M, heads,
    v_size]`` instead of Y, with ``k_rope`` ([M, rope] or [M, 1, rope]) in the
    tail of every K head, and ``(k_out, v_out)`` is returned.
    """
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

    split_cat, split_cat_args = None, ()
    if split_sizes is not None:
        # The kernel stores K through Y and takes V and k_rope as extra arguments.
        k_size, v_size = split_sizes
        heads, rope_size = k_out.shape[1], k_rope.shape[-1]
        assert y is None and dtype in _half, "split-cat stores fp16/bf16 K/V, not y"
        assert N == heads * (k_size + v_size), f"N={N} is not heads * sum(split_sizes)"
        assert k_out.shape == (M, heads, k_size + rope_size), "bad k_out shape"
        assert v_out.shape == (M, heads, v_size), "bad v_out shape"
        assert k_rope.shape[0] == M and k_rope.numel() == M * rope_size, "bad k_rope"
        assert all(
            t.dtype == dtype and t.device == x.device and t.stride(-1) == 1
            for t in (k_out, v_out, k_rope)
        ), "k_out, v_out and k_rope need y's dtype and device and a unit last stride"
        assert rope_size <= v_size and all(
            size % 8 == 0 for size in (k_size, v_size, rope_size)
        ), "split-cat needs sizes that are multiples of 8 and rope <= v_size"
        split_cat = (k_size, v_size, rope_size)
        y_buf, ldy = k_out, k_out.stride(0)
        split_cat_args = (
            _p(v_out),
            _p(k_rope),
            k_out.stride(1),
            v_out.stride(0),
            v_out.stride(1),
            k_rope.stride(0),
        )
    elif y is not None:
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
    if split_cat is None:
        ldy = y_buf.stride(0) if M > 1 else N
        assert ldy >= N, f"gemm_a16w16: y row stride {ldy} < N {N}"

    if bias is None:
        bias = torch.empty(0, device=x.device, dtype=dtype)

    stream = torch.cuda.current_stream(device=x.device)
    sem = _split_k_counters(x.device, stream)
    if split_k > 1:
        tiles = ((M + tile_m - 1) // tile_m) * ((N + tile_n - 1) // tile_n)
        if tiles > _SPLIT_K_MAX_TILES:
            raise RuntimeError(
                f"[FlyDSL gfx1250] split_k needs {tiles} tile counters, "
                f"more than {_SPLIT_K_MAX_TILES}"
            )
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
        split_cat,
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
        *split_cat_args,
        _fx.Stream(stream.cuda_stream),
    )
    return y_buf if split_cat is None else (k_out, v_out)


__all__ = ["gemm_a16w16"]
