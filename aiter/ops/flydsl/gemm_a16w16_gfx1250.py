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
)


@functools.lru_cache(maxsize=1024)
def _cached_launcher(*cfg):
    return _compile_gemm_a16w16(**dict(zip(_CFG_KEYS, cfg)))


_SPLIT_CAT_CFG_KEYS = _CFG_KEYS + (
    "split_cat",
    "k_split_size",
    "v_split_size",
    "rope_size",
)


@functools.lru_cache(maxsize=128)
def _cached_split_cat_launcher(*cfg):
    return _compile_gemm_a16w16(**dict(zip(_SPLIT_CAT_CFG_KEYS, cfg)))


_SPLIT_K_MAX_TILES = 4096


@capture_scoped_cache
def _split_k_counters(device, stream):
    with persistent_alloc(torch.device(device)):
        return torch.zeros(_SPLIT_K_MAX_TILES, dtype=torch.int32, device=device)


def _validate_split_cat(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype,
    y: torch.Tensor | None,
    k_out: torch.Tensor | None,
    v_out: torch.Tensor | None,
    k_rope: torch.Tensor | None,
    split_sizes: tuple[int, int] | None,
):
    """Validate and normalize the opt-in MLA split-cat output contract."""
    values = (k_out, v_out, k_rope, split_sizes)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "k_out, v_out, k_rope, and split_sizes must be provided together"
        )
    if y is not None:
        raise ValueError("y cannot be combined with split-cat outputs")
    if (
        not isinstance(split_sizes, (tuple, list))
        or len(split_sizes) != 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in split_sizes
        )
    ):
        raise ValueError("split_sizes must be two positive integers")
    k_split_size, v_split_size = split_sizes
    if k_split_size % 8 or v_split_size % 8:
        raise ValueError("split sizes must be multiples of 8")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError("gfx1250 split-cat requires BF16 x and w")
    if dtype != torch.bfloat16:
        raise ValueError("gfx1250 split-cat requires BF16 output")

    m, n = x.shape[0], w.shape[0]
    if k_out.dim() != 3 or v_out.dim() != 3:
        raise ValueError("k_out and v_out must be [M, heads, width] tensors")
    if k_out.shape[:2] != v_out.shape[:2] or k_out.shape[0] != m:
        raise ValueError("k_out and v_out must have matching [M, heads] dimensions")
    heads = k_out.shape[1]
    if heads <= 0 or n != heads * (k_split_size + v_split_size):
        raise ValueError(
            f"GEMM N={n} must equal heads * sum(split_sizes), got "
            f"{heads} * ({k_split_size} + {v_split_size})"
        )
    if v_out.shape[2] != v_split_size:
        raise ValueError(
            f"v_out width must equal split_sizes[1]={v_split_size}, "
            f"got {v_out.shape[2]}"
        )

    if k_rope.dim() == 2:
        rope_shape_ok = k_rope.shape[0] == m
        rope_size = k_rope.shape[1]
        rope_row_stride = k_rope.stride(0)
        rope_last_stride = k_rope.stride(1)
    elif k_rope.dim() == 3:
        rope_shape_ok = k_rope.shape[0] == m and k_rope.shape[1] == 1
        rope_size = k_rope.shape[2]
        rope_row_stride = k_rope.stride(0)
        rope_last_stride = k_rope.stride(2)
    else:
        raise ValueError("k_rope must have shape [M, rope] or [M, 1, rope]")
    if not rope_shape_ok or rope_size <= 0:
        raise ValueError("k_rope must have shape [M, rope] or [M, 1, rope]")
    if rope_size % 8:
        raise ValueError("k_rope width must be a multiple of 8")
    if rope_size > v_split_size:
        raise ValueError("k_rope width cannot exceed the V split size")
    if k_out.shape[2] != k_split_size + rope_size:
        raise ValueError(
            f"k_out width must be k split + rope = {k_split_size + rope_size}, "
            f"got {k_out.shape[2]}"
        )

    tensors = (k_out, v_out, k_rope)
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("k_out, v_out, and k_rope must all be BF16")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("split-cat tensors must be on the same device as x")

    def _check_output(name, tensor):
        width = tensor.shape[2]
        if tensor.stride(2) != 1 or tensor.stride(1) < width:
            raise ValueError(
                f"{name} must be last-dimension contiguous with non-overlapping "
                f"heads; got strides "
                f"{tuple(tensor.stride())}"
            )
        min_row_stride = (heads - 1) * tensor.stride(1) + width
        if tensor.stride(0) < min_row_stride:
            raise ValueError(
                f"{name} row stride {tensor.stride(0)} is smaller than "
                f"{min_row_stride}"
            )

    _check_output("k_out", k_out)
    _check_output("v_out", v_out)
    if rope_last_stride != 1 or rope_row_stride < rope_size:
        raise ValueError(
            "k_rope must be last-dimension contiguous with a non-overlapping row pitch"
        )
    return {
        "k_split_size": k_split_size,
        "v_split_size": v_split_size,
        "rope_size": rope_size,
        "heads": heads,
        "k_row_stride": k_out.stride(0),
        "k_head_stride": k_out.stride(1),
        "v_row_stride": v_out.stride(0),
        "v_head_stride": v_out.stride(1),
        "rope_row_stride": rope_row_stride,
    }


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
    """Compute ``Y = X @ W.T + bias``, optionally into MLA K/V final layouts.

    The default path is unchanged. To select the split-cat epilogue, provide all
    of ``k_out``, ``v_out``, ``k_rope``, and ``split_sizes=(k_nope, v_dim)``.
    The GEMM's logical N dimension is interpreted as
    ``[head0 k_nope, head0 v, head1 k_nope, head1 v, ...]``. The function writes
    ``k_out=[k_nope, k_rope]`` and ``v_out=v`` and returns ``(k_out, v_out)``.
    Output rows may have padding, but every head must be last-dimension
    contiguous.
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
    split_cat = _validate_split_cat(x, w, dtype, y, k_out, v_out, k_rope, split_sizes)

    if split_cat is not None:
        # arg_y/ldy preserve the legacy internal launcher contract but are not
        # touched by the split-cat epilogue.
        y_buf = k_out
        ldy = N
    else:
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

    launcher_cfg = (
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
    if split_cat is None:
        launch_fn = _cached_launcher(*launcher_cfg)
    else:
        launch_fn = _cached_split_cat_launcher(
            *launcher_cfg,
            True,
            split_cat["k_split_size"],
            split_cat["v_split_size"],
            split_cat["rope_size"],
        )

    if physical_mk:
        lda = x.stride(0) if M > 1 else K
    else:
        lda = x.stride(1) if K > 1 else M
    if physical_kn:
        ldb = w.stride(1) if K > 1 else N
    else:
        ldb = w.stride(0) if N > 1 else K
    args = (
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
    )
    if split_cat is None:
        _run_compiled(*args, _fx.Stream(stream.cuda_stream))
        return y_buf
    _run_compiled(
        *args,
        _p(k_out),
        _p(v_out),
        _p(k_rope),
        split_cat["k_row_stride"],
        split_cat["k_head_stride"],
        split_cat["v_row_stride"],
        split_cat["v_head_stride"],
        split_cat["rope_row_stride"],
        _fx.Stream(stream.cuda_stream),
    )
    return k_out, v_out


__all__ = ["gemm_a16w16"]
