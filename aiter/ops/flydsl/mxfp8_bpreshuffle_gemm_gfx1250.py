# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for mxfp8_128 / mxfp8_32 bpreshuffle GEMM.

mxfp8_128: fp8 e4m3 activations/weights with fp8_e8m0 block scales over a
128-element block (distinct from the fp32-scale blockscale GEMM).

mxfp8_32: A scales are row-major ``(M, K//32)`` by default, optionally
32x4 shuffled; B scales are 32x4 shuffled ``(N, K//32)``, or row-major
``(N//32, K//32)`` with ``w_scale_32x32``.
"""

from __future__ import annotations

import dataclasses
import functools
import re

import torch
from torch import Tensor

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

_launch_gemm_a8w8 = None
_launch_gemm_a8w8_compute_bound = None
_compile_splitk_reduce = None
_run_compiled = None
_ptr_arg = None
_fx = None

BLOCK_K = 128
MX32_BLOCK_K = 32
_BLOCK_N = 128
WMMA_NAME_PREFIX = "flydsl_mxfp8_128_bpreshuffle_wmma"
COMPUTE_WMMA_NAME_PREFIX = "flydsl_mxfp8_128_bpreshuffle_compute_wmma"
MX32_WMMA_NAME_PREFIX = "flydsl_mxfp8_32_bpreshuffle_wmma"
MX32_COMPUTE_WMMA_NAME_PREFIX = "flydsl_mxfp8_32_bpreshuffle_compute_wmma"
MX32_SMALL_WMMA_NAME_PREFIX = "flydsl_mxfp8_32_bpreshuffle_small_wmma"
_SUPPORTED_NUM_BUFFERS = (2, 3, 4)
_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_MAX_SPLIT_K = 8


@capture_scoped_cache
def _fused_splitk_workspace(
    device: torch.device,
    stream_id: int,
    shape: tuple[int, ...],
) -> Tensor:
    """Reuse stream-private fused split-K scratch in eager and graph modes."""
    del stream_id
    with persistent_alloc(device):
        return torch.empty(shape, dtype=torch.float32, device=device)


def clear_mxfp8_32_workspace_cache() -> None:
    """Release graph-scoped split-K scratch between independent tuner candidates."""
    _fused_splitk_workspace.cache_clear()


def _fused_splitk_ok(
    tile_m,
    tile_n,
    cluster_m,
    cluster_n,
    split_k,
    native_fused_splitk,
    m_warp,
    n_warp,
) -> bool:
    """Whether this launch can run the clustered fused split-K epilogue."""
    return bool(
        split_k > 1
        and tile_m % split_k == 0
        and (m_warp * n_warp * 32) % (tile_n // 4) == 0
        and cluster_m * cluster_n * split_k <= 16
        and (
            native_fused_splitk
            or (
                tile_m & (tile_m - 1) == 0
                and cluster_m == cluster_n == 1
                and m_warp * n_warp >= 2
            )
        )
    )


def _lazy_import():
    global _launch_gemm_a8w8, _launch_gemm_a8w8_compute_bound
    global _compile_splitk_reduce, _run_compiled, _ptr_arg, _fx
    if _launch_gemm_a8w8 is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_256x256_gfx1250 import (
        launch_gemm_a8w8_256x256,
    )
    from .kernels.gemm_a8w8_gfx1250 import launch_gemm_a8w8
    from .kernels.gemm_a8w8_splitk_reduce_gfx1250 import (
        compile_gemm_a8w8_splitk_reduce,
    )
    from .kernels.tensor_shim import _run_compiled as run_compiled
    from .kernels.tensor_shim import ptr_arg

    _launch_gemm_a8w8 = launch_gemm_a8w8
    _launch_gemm_a8w8_compute_bound = launch_gemm_a8w8_256x256
    _compile_splitk_reduce = compile_gemm_a8w8_splitk_reduce
    _run_compiled = run_compiled
    _ptr_arg = ptr_arg
    _fx = fx_mod


def _require_e8m0_scale(scale: Tensor, shape: tuple[int, int], name: str) -> Tensor:

    from aiter.utility import dtypes

    if tuple(scale.shape) != shape:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] {name} must have shape {shape}, "
            f"got {tuple(scale.shape)}"
        )
    if scale.dtype != dtypes.fp8_e8m0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] {name} must be fp8_e8m0, got {scale.dtype}"
        )
    return scale


def check_persistent_n_tiles(
    persistent_n_tiles: int,
    N: int,
    tile_n: int,
    cluster_n: int,
    split_k: int,
    compute_bound: bool,
) -> None:
    """Validate a persistent (multi-tile-per-CTA) config against the shape."""
    if persistent_n_tiles < 1:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] persistent_n_tiles must be >= 1, "
            f"got {persistent_n_tiles}"
        )
    if persistent_n_tiles == 1:
        return
    if not compute_bound:
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] persistent_n_tiles>1 is compute-bound only"
        )
    if split_k != 1:
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] persistent_n_tiles>1 requires split_k=1"
        )
    n_tiles = N // tile_n
    if n_tiles % persistent_n_tiles or (n_tiles // persistent_n_tiles) % cluster_n:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] persistent_n_tiles={persistent_n_tiles} needs "
            f"N/tile_n={n_tiles} divisible by it and the quotient a multiple "
            f"of cluster_n={cluster_n}"
        )


def _run_mxfp8_bpreshuffle_gemm_a8_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
    kernel_name: str,
    num_buffers: int = 2,
    m_warp: int = 2,
    n_warp: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    split_k: int = 1,
    x_scale_transposed: bool = True,
    a_preshuffle: bool = False,
    a_scale_preshuffled: bool = False,
    persistent_n_tiles: int = 1,
    early_a: bool = False,
    early_b: bool = False,
    k_pipe: int = 0,
    fused_splitk: bool = True,
    scale_block: int = BLOCK_K,
    w_scale_32x32: bool = False,
    batch: int = 1,
    y_scale_out: Tensor | None = None,
    l2_prefetch: int = 0,
) -> Tensor:
    """Run the gfx1250 WMMA mxfp8_128 / mxfp8_32 bpreshuffle GEMM.

    XQ: ``(M, K)`` FP8 E4M3. WQ: ``(N, K)`` FP8 E4M3, already 16x16
    preshuffled. Out: ``(M, N)`` bf16/f16.

    ``scale_block=128``: x_scale ``(M, K//128)`` fp8_e8m0; when
    ``x_scale_transposed=True`` the backing storage is interpreted as
    ``(K//128, M)`` without copying. w_scale ``(N//128, K//128)`` fp8_e8m0
    dense row-major.

    ``scale_block=32``: x_scale ``(M, K//32)`` row-major by default, or
    ``(ceil(M/32)*32, K//32)`` in 32x4 shuffled storage when
    ``a_scale_preshuffled=True``. w_scale ``(N, K//32)`` is 32x4 shuffled, or
    ``(N//32, K//32)`` row-major with ``w_scale_32x32``. Both scales are fp8_e8m0.

    ``batch > 1`` (compute mxfp8_32, split_k 1): the operands are batch 0 of
    ``[M, batch, K]`` XQ / x_scale, ``[batch, N, K]`` WQ / w_scale and
    ``[M, batch, N]`` Out.

    ``y_scale_out`` (compute mxfp8_32, split_k 1, no persistent tiles): the
    epilogue quantizes C to group-32 MXFP8 instead of writing bf16. Out is then
    the e4m3 C and ``y_scale_out`` its row-major ``[M, Out.stride(0) // 32]``
    e8m0 scale.
    """
    compute_bound = is_compute_wmma_kernel_name(kernel_name)
    fuse_quant = y_scale_out is not None
    small_native = is_small_wmma_kernel_name(kernel_name)
    mx32 = scale_block == MX32_BLOCK_K
    if not mx32 and scale_block != BLOCK_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] scale_block must be {BLOCK_K} or "
            f"{MX32_BLOCK_K}, got {scale_block}"
        )
    if a_scale_preshuffled and not mx32:
        raise RuntimeError("a_scale_preshuffled is supported only for mxfp8_32")
    if (early_a or early_b) and (not compute_bound or int(split_k) != 1):
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] early A/B merge requires a compute kernel "
            "with split_k=1"
        )
    if l2_prefetch and (
        not compute_bound
        or int(split_k) != 1
        or int(persistent_n_tiles) != 1
        or batch > 1
    ):
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] L2 prefetch (_pf) needs a compute kernel with "
            "split_k=1, no persistent tiles and batch 1"
        )
    if w_scale_32x32 and not mx32:
        raise RuntimeError("w_scale_32x32 is supported only for mxfp8_32")
    if batch > 1 and not (
        compute_bound
        and mx32
        and int(split_k) == 1
        and not a_preshuffle
        and not a_scale_preshuffled
    ):
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] batch > 1 needs a compute mxfp8_32 kernel "
            "with split_k=1 and row-major A and A scale"
        )
    if fuse_quant and not (
        compute_bound
        and mx32
        and int(split_k) == 1
        and int(persistent_n_tiles) == 1
        and not a_preshuffle
        and not a_scale_preshuffled
    ):
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] y_scale_out needs a compute mxfp8_32 kernel "
            "with split_k=1, no persistent tiles and row-major A and A scale"
        )

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8] A/B must be 2-D, got "
            f"{tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250 mxfp8] A/B must be 1-byte fp8 storage")

    a_rows, K = XQ.shape
    M = Out.shape[0] if a_preshuffle else a_rows
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] K mismatch: A.K={K} vs B.K={WQ.shape[1]}"
        )
    split_k = int(split_k)
    cluster_m = int(cluster_m)
    cluster_n = int(cluster_n)
    if split_k < 1 or not (1 <= cluster_m < 16 and 1 <= cluster_n < 16):
        raise RuntimeError(
            "split_k must be positive and cluster dimensions must be 1..15"
        )
    if min(M, N, K, tile_m, tile_n, tile_k) < 1:
        raise RuntimeError("GEMM dimensions and tiles must be positive")

    if N % _BLOCK_N != 0 or K % BLOCK_K != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] N/K must be multiples of "
            f"{_BLOCK_N}/{BLOCK_K}, got N={N}, K={K}"
        )
    if cluster_m * cluster_n > 16:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] a gfx1250 cluster holds at most 16 "
            f"workgroups, got {cluster_m}x{cluster_n}"
        )
    # A ragged last N tile (B / B-scale rows past N zero-filled, C clipped) is
    # served by the compute kernel when B is not multicast; see ragged_n there.
    ragged_n = ragged_n_ok(
        N,
        tile_n,
        cluster_m,
        cluster_n,
        split_k,
        persistent_n_tiles,
        compute_bound and mx32 and not w_scale_32x32 and batch == 1 and not fuse_quant,
    )
    if N % (tile_n * cluster_n) != 0 and not ragged_n:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] N={N} must be a multiple of "
            f"tile_n*cluster_n={tile_n}*{cluster_n}={tile_n * cluster_n}"
        )
    if not cluster_m_grid_ok(M, tile_m, cluster_m):
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] M={M} gives "
            f"ceil(M/tile_m)={(M + tile_m - 1) // tile_m} M-tiles, which is not a "
            f"multiple of cluster_m={cluster_m}; grid.x would be padded with "
            f"workgroups that own no M row and deadlock the cluster"
        )
    if K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] K={K} not a multiple of tile_k={tile_k}"
        )

    out_dtype = "bf16" if fuse_quant else _OUT_DTYPE_NAME.get(Out.dtype)
    if fuse_quant:
        ldc_q = Out.stride(0)
        if (
            Out.element_size() != 1
            or Out.stride(1) != 1
            or ldc_q % 32
            or y_scale_out.element_size() != 1
            or not y_scale_out.is_contiguous()
            or y_scale_out.numel() != M * (ldc_q // 32)
            or y_scale_out.data_ptr() % 8
        ):
            raise RuntimeError(
                "[FlyDSL gfx1250 mxfp8] the fused-quant Out must be 1-byte e4m3 "
                "with unit column stride and a row stride divisible by 32, and "
                f"y_scale_out {M * (ldc_q // 32)} contiguous 8-byte-aligned e8m0 "
                f"bytes; got Out {tuple(Out.shape)} {Out.dtype} "
                f"stride={tuple(Out.stride())}, y_scale_out "
                f"{tuple(y_scale_out.shape)} {y_scale_out.dtype}"
            )
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] unsupported out dtype {Out.dtype}; "
            "expected bf16/fp16"
        )

    if split_k > _MAX_SPLIT_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] split_k={split_k} exceeds the "
            f"supported maximum of {_MAX_SPLIT_K}"
        )

    nb = int(num_buffers)
    if nb not in _SUPPORTED_NUM_BUFFERS:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] num_buffers must be one of "
            f"{_SUPPORTED_NUM_BUFFERS}, got {nb}"
        )
    if K % (split_k * tile_k) != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] K={K} must be divisible by "
            f"split_k*tile_k={split_k}*{tile_k}={split_k * tile_k}"
        )
    num_k_tiles = (K // split_k) // tile_k
    # Generic K=128 with two buffers uses only the prologue and drain.
    # Other configurations, including compute-bound kernels, keep their limits.
    min_k_tiles = 1 if not compute_bound and nb == 2 and K == tile_k == 128 else nb
    if num_k_tiles < min_k_tiles:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] {nb}-buffer pipeline needs >= {min_k_tiles} "
            f"K-tiles per split-k chunk, got {num_k_tiles}"
        )
    if compute_bound:
        k_pair = compute_kernel_k_pair(nb, tile_n)
        if K // split_k < 512 or num_k_tiles % k_pair != 0:
            raise RuntimeError(
                f"[FlyDSL gfx1250 mxfp8 compute] each split needs at least "
                f"512 K elements and a whole K-pair; got K={K}, split_k={split_k}, "
                f"tile_k={tile_k}, K-pair={k_pair}"
            )
    if split_k > 1:
        if Out.stride(1) != 1:
            raise RuntimeError(
                "[FlyDSL gfx1250 mxfp8] split_k>1 needs contiguous Out rows, "
                f"got strides={tuple(Out.stride())} for {M}x{N}"
            )
        # A padded row stride is reduced row-by-row, so each row start must keep
        # the 16B alignment the vectorised copy needs.
        if Out.stride(0) != N and (Out.stride(0) & 7 or N & 7):
            raise RuntimeError(
                f"[FlyDSL gfx1250 mxfp8] split_k>1 with padded Out rows needs "
                f"N and stride(0) to be multiples of 8, got N={N}, "
                f"stride(0)={Out.stride(0)}"
            )

    check_persistent_n_tiles(
        persistent_n_tiles, N, tile_n, cluster_n, split_k, compute_bound
    )

    if mx32 and tile_n % 32 != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_32] tile_n={tile_n} must be a multiple of 32 "
            "(one WMMA scale lane group per 32 columns)"
        )

    if a_preshuffle and a_rows != M + (M & 1):
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] a_preshuffle needs A padded to an even "
            f"row count: Out gives M={M}, so A must have {M + (M & 1)} rows, got "
            f"{a_rows}.  The last A row pair is read whole, so an odd-M A buffer "
            "would be a short read; pad A (not x_scale, not Out) before shuffling."
        )

    if mx32:
        k_blocks = K // MX32_BLOCK_K
        a_rows = ((M + 31) // 32) * 32 if a_scale_preshuffled else M
        a_scale = _require_e8m0_scale(x_scale, (a_rows, k_blocks), "x_scale")
        b_rows = N // 32 if w_scale_32x32 else N
        b_scale = _require_e8m0_scale(w_scale, (b_rows, k_blocks), "w_scale")
        a_dense = (
            a_scale.is_contiguous()
            if batch == 1
            else a_scale.stride() == (batch * k_blocks, 1)
        )
        if not (a_dense and b_scale.is_contiguous()):
            raise RuntimeError(
                "[FlyDSL gfx1250 mxfp8_32] x_scale/w_scale must be contiguous"
            )
        stride_ascale_k = batch * k_blocks
    else:
        if not x_scale_transposed:
            raise RuntimeError(
                "[FlyDSL gfx1250 mxfp8] x_scale_transposed=False is not "
                "supported by the dedicated mxfp8_128 kernel (A-scale must be "
                "M-contiguous)"
            )
        k_blocks = K // BLOCK_K
        a_scale = _require_e8m0_scale(x_scale, (M, k_blocks), "x_scale")
        b_scale = _require_e8m0_scale(w_scale, (N // _BLOCK_N, k_blocks), "w_scale")
        stride_ascale_k = a_scale.stride(1) if a_scale.stride(0) == 1 else M

    if batch > 1 and (XQ.stride() != (batch * K, 1) or Out.stride() != (batch * N, 1)):
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8] batch > 1 needs XQ strides ({batch * K}, 1) "
            f"and Out strides ({batch * N}, 1), got {tuple(XQ.stride())} and "
            f"{tuple(Out.stride())}"
        )

    lda = XQ.stride(0)
    ldc = Out.stride(0)
    _lazy_import()
    torch_stream = torch.cuda.current_stream(device=XQ.device)
    stream = _fx.Stream(torch_stream)
    fused = fused_splitk and _fused_splitk_ok(
        tile_m,
        tile_n,
        cluster_m,
        cluster_n,
        split_k,
        compute_bound or small_native,
        m_warp,
        n_warp,
    )
    if fused:
        # One contiguous plane per split of an output tile, holding only the
        # peer rows -- the local stripe never leaves LDS.  Round M up for the
        # last tile; the output store's M bound masks its spare rows.
        partial_shape = (
            (
                ((M + tile_m - 1) // tile_m) * (N // tile_n),
                split_k,
                tile_m - tile_m // split_k if split_k == 2 else tile_m,
                tile_n,
            )
            if (compute_bound or small_native)
            else (
                ((M + tile_m - 1) // tile_m) * (N // tile_n),
                split_k,
                tile_m,
                tile_n,
            )
        )
    elif split_k > 1:
        partial_shape = (split_k, M, ldc)
    else:
        partial_shape = None
    partials = (
        _fused_splitk_workspace(
            Out.device, int(torch_stream.cuda_stream), tuple(partial_shape)
        )
        if fused
        else (
            None
            if partial_shape is None
            else torch.empty(partial_shape, dtype=torch.float32, device=Out.device)
        )
    )
    gemm_out = Out if partials is None else partials
    out_is_f16 = 1 if out_dtype == "f16" else 0

    launch_args = (
        _ptr_arg(gemm_out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(a_scale),
        _ptr_arg(b_scale),
        M,
        stream,
        N,
        K,
        stride_ascale_k,
        lda,
        ldc,
        tile_m,
        tile_n,
        tile_k,
        m_warp,
        n_warp,
        out_is_f16,
        nb,
        cluster_m,
        cluster_n,
        True,
    )
    # Dispatch through the cached CompiledFunction of each constexpr set: calling
    # the JitFunction re-derives its specialization on the host every launch.
    runtime_args, constexpr_args = launch_args[:12], launch_args[12:]
    if compute_bound and k_pipe:
        raise ValueError(
            "[FlyDSL gfx1250] k_pipe (_pd) applies to the direct kernel only"
        )
    if compute_bound:
        # A launch of one workgroup is no cluster: it runs the plain kernel.
        use_cluster = cluster_m * cluster_n * (split_k if fused else 1) > 1
        specialization = (*constexpr_args[:7], use_cluster, *constexpr_args[8:]) + (
            scale_block,
            split_k,
            a_preshuffle,
            persistent_n_tiles,
            early_a,
            early_b,
            fused,
            a_scale_preshuffled,
            compute_kernel_k_tiles_class(num_k_tiles),
            w_scale_32x32,
            batch > 1,
        )
        _run_compiled(
            _launch_gemm_a8w8_compute_bound,
            *runtime_args,
            _ptr_arg(y_scale_out) if fuse_quant else _ptr_arg(Out),
            cluster_m,
            *specialization,
            batch,
            fuse_quant,
            ragged_n,
            int(l2_prefetch),
            specialization_key=specialization
            + (fuse_quant, ragged_n, int(l2_prefetch)),
        )
    else:
        rank0_splitk = fused and not small_native
        specialization = constexpr_args + (
            scale_block,
            split_k,
            False,  # batched
            0,  # preload_ks
            a_preshuffle,
            a_scale_preshuffled,
            w_scale_32x32,
            fused,
            rank0_splitk,
            k_pipe,
        )
        _run_compiled(
            _launch_gemm_a8w8,
            *launch_args,
            scale_block,
            split_k,
            False,  # batched
            0,  # preload_ks
            1,  # batch (runtime)
            a_preshuffle,
            a_scale_preshuffled,
            w_scale_32x32,  # b_scale_32x32
            False,  # k_tail
            _ptr_arg(Out),  # arg_scale_out
            False,  # fuse_quant_out
            _ptr_arg(Out),  # arg_out
            fused,
            rank0_splitk,
            k_pipe,
            specialization_key=specialization,
        )
    if partials is not None and not fused:
        dense = ldc == N
        _run_compiled(
            _compile_splitk_reduce(
                split_k=split_k, out_dtype_str=out_dtype, partial_dtype_str="f32"
            ),
            _ptr_arg(partials),
            _ptr_arg(Out),
            M * N if dense else N,
            1 if dense else M,
            ldc,
            M * ldc * partials.element_size(),
            stream,
        )
    return Out


BASE_NAME_SUFFIX_RE = (
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)"
)
NAME_SUFFIX_RE = (
    BASE_NAME_SUFFIX_RE + r"(?P<fused_splitk>_fsk)?"
    r"(?P<a_preshuffle>_apre)?"
    r"(?:_ps(?P<persistent_n_tiles>\d+))?"
    r"(?P<early_merge>_eab|_ea|_eb)?"
    r"(?:_pf(?P<l2_prefetch>[12]))?"
)
MX32_NAME_SUFFIX_RE = (
    NAME_SUFFIX_RE + r"(?P<a_scale_preshuffled>_aspre)?" + r"(?:_pd(?P<k_pipe>\d+))?"
)
_KERNEL_NAME_RE = re.compile(rf"^{re.escape(WMMA_NAME_PREFIX)}_{NAME_SUFFIX_RE}")
_COMPUTE_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(COMPUTE_WMMA_NAME_PREFIX)}_{NAME_SUFFIX_RE}"
)
_MX32_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(MX32_WMMA_NAME_PREFIX)}_{MX32_NAME_SUFFIX_RE}"
)
_MX32_COMPUTE_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(MX32_COMPUTE_WMMA_NAME_PREFIX)}_{MX32_NAME_SUFFIX_RE}"
)
_MX32_SMALL_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(MX32_SMALL_WMMA_NAME_PREFIX)}_{MX32_NAME_SUFFIX_RE}"
)


def _parse_name(match):
    if match is None:
        return None
    groups = match.groupdict()
    a_preshuffle = groups.pop("a_preshuffle", None) is not None
    a_scale_preshuffled = groups.pop("a_scale_preshuffled", None) is not None
    fused_splitk = groups.pop("fused_splitk", None) is not None
    early_merge = groups.pop("early_merge", None)
    persistent_n_tiles = groups.pop("persistent_n_tiles", None)
    l2_prefetch = groups.pop("l2_prefetch", None)
    k_pipe = groups.pop("k_pipe", None)
    cfg = {key: int(value) for key, value in groups.items()}
    cfg["k_pipe"] = int(k_pipe) if k_pipe else 0
    cfg["a_preshuffle"] = a_preshuffle
    cfg["a_scale_preshuffled"] = a_scale_preshuffled
    cfg["fused_splitk"] = fused_splitk
    cfg["early_a"] = early_merge in ("_ea", "_eab")
    cfg["early_b"] = early_merge in ("_eb", "_eab")
    cfg["persistent_n_tiles"] = int(persistent_n_tiles) if persistent_n_tiles else 1
    cfg["l2_prefetch"] = int(l2_prefetch) if l2_prefetch else 0
    return cfg


def parse_wmma_kernel_name(name: str):
    """Parse a generic or compute-bound mxfp8_128 kernelName."""
    return _parse_name(
        _COMPUTE_KERNEL_NAME_RE.fullmatch(name) or _KERNEL_NAME_RE.fullmatch(name)
    )


def parse_mxfp8_32_wmma_kernel_name(name: str):
    """Parse a generic, compute-bound, or low-resource mxfp8_32 name."""
    return _parse_name(
        _MX32_COMPUTE_KERNEL_NAME_RE.fullmatch(name)
        or _MX32_SMALL_KERNEL_NAME_RE.fullmatch(name)
        or _MX32_KERNEL_NAME_RE.fullmatch(name)
    )


def mxfp8_32_k_pipe_ok(cfg: dict) -> bool:
    """A ``_pd`` (k_pipe) name: the direct kernel's row path, at most tile_k/128 deep."""
    wm = cfg["tile_m"] // cfg["m_warp"] // 16
    wn = cfg["tile_n"] // cfg["n_warp"] // 16
    quadrant = wm % 2 == 0 and wn % 2 == 0 and wm * wn >= 8
    return 0 < cfg["k_pipe"] <= cfg["tile_k"] // 128 and not quadrant


@functools.lru_cache(maxsize=4096)
def mxfp8_32_kernel_launchable(kernel_name: str, M: int, N: int, K: int) -> bool:
    """Whether a tuned mx32 kernelName can launch for the real, unpadded shape."""
    cfg = parse_mxfp8_32_wmma_kernel_name(kernel_name)
    if cfg is None:
        return False

    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        kernels_list,
        mxfp8_32_kernel_fits_shape,
    )

    if cfg["k_pipe"] and not mxfp8_32_k_pipe_ok(cfg):
        return False
    base_name = re.sub(r"_pd\d+$", "", kernel_name).removesuffix("_aspre")
    for instance in kernels_list.values():
        if instance.name.replace("mxfp8_128", "mxfp8_32") == base_name:
            # The launch fits cluster_m to M (_resolved_wmma_config); check that.
            cluster_m = resolve_cluster_m(M, instance.tile_m, instance.cluster_m)
            instance = dataclasses.replace(
                instance, cluster_m=cluster_m or instance.cluster_m
            )
            return not instance.a_preshuffle and mxfp8_32_kernel_fits_shape(
                instance, M, N, K
            )
    return False


def ragged_n_ok(
    N: int,
    tile_n: int,
    cluster_m: int,
    cluster_n: int,
    split_k: int,
    persistent_n_tiles: int,
    compute_mx32: bool,
) -> bool:
    """Whether the compute kernel runs N with a ragged last tile: N a multiple of
    32 but not of tile_n, whole clusters of N tiles, and B never multicast
    (cluster_m 1), no split-K or persistent tiles."""
    n_tiles = -(-N // tile_n)
    return bool(
        compute_mx32
        and N % tile_n != 0
        and N % 32 == 0
        and cluster_m == 1
        and n_tiles % cluster_n == 0
        and split_k == 1
        and persistent_n_tiles == 1
    )


def compute_kernel_k_pair(num_buffers: int, tile_n: int) -> int:
    """K-tiles the compute-bound kernel stages per TDM super."""
    return 1 if num_buffers == 4 and tile_n == 256 else 2


def compute_kernel_k_tiles_class(num_k_tiles: int) -> int:
    """K-loop length bucket of the compute-bound kernel: 0 for <= 16 K-tiles
    per split, 1 for <= 64, 2 above."""
    return 0 if num_k_tiles <= 16 else (1 if num_k_tiles <= 64 else 2)


def cluster_m_grid_ok(M: int, tile_m: int, cluster_m: int) -> bool:
    """Whether ``M`` fills every cluster row of the launch grid."""
    if M <= 0 or tile_m <= 0 or cluster_m < 1:
        return False
    m_blocks = (M + tile_m - 1) // tile_m
    return m_blocks % cluster_m == 0


def resolve_cluster_m(M: int, tile_m: int, cluster_m: int) -> int | None:
    """The tuned cluster_m if it tiles M, else the largest power of two that does."""
    for cm in range(max(1, int(cluster_m)), 0, -1):
        if (cm == cluster_m or not cm & (cm - 1)) and cluster_m_grid_ok(M, tile_m, cm):
            return cm
    return None


def is_compute_wmma_kernel_name(name: str) -> bool:
    """Return whether ``name`` selects the compute-bound implementation."""
    return (
        _COMPUTE_KERNEL_NAME_RE.fullmatch(name) is not None
        or _MX32_COMPUTE_KERNEL_NAME_RE.fullmatch(name) is not None
    )


def is_small_wmma_kernel_name(name: str) -> bool:
    """Return whether ``name`` selects the low-resource small-M schedule."""
    return _MX32_SMALL_KERNEL_NAME_RE.fullmatch(name) is not None


@functools.lru_cache(maxsize=1024)
def _resolved_wmma_config(
    kernel_name: str, true_m: int, allow_cluster_m_fallback: bool, mx32: bool
) -> tuple[tuple[str, int | bool], ...]:
    """Parsed kernelName with its cluster_m fitted to M, cached off the hot path."""
    parse = parse_mxfp8_32_wmma_kernel_name if mx32 else parse_wmma_kernel_name
    cfg = parse(kernel_name)
    tag = "mxfp8_32" if mx32 else "mxfp8_128"
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 {tag}] unrecognised kernelName: {kernel_name!r}"
        )
    if allow_cluster_m_fallback:
        cfg["cluster_m"] = (
            resolve_cluster_m(true_m, cfg["tile_m"], cfg["cluster_m"])
            or cfg["cluster_m"]
        )
    return tuple(cfg.items())


def run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
    a_is_preshuffled: bool = False,
    allow_cluster_m_fallback: bool = True,
) -> Tensor:
    """Decode a tuned kernelName and dispatch its internal implementation."""
    # Out always carries true M; a preshuffled XQ may include one pad row.
    cfg = dict(
        _resolved_wmma_config(
            kernel_name, Out.shape[0], allow_cluster_m_fallback, False
        )
    )
    if cfg["a_preshuffle"] and not a_is_preshuffled:
        raise ValueError(
            f"[FlyDSL gfx1250 mxfp8_128] kernelName {kernel_name!r} selects "
            "a_preshuffle, but the caller did not declare a preshuffled A. The "
            "tuned-config dispatch forwards the model's row-major activation, so "
            "this kernel would read it as (2, 128)-tiled and return wrong results. "
            "Feed it shuffle_mxfp8fp4_a(A) and pass a_is_preshuffled=True, or "
            "call gemm_a8w8_blockscale_abpreshuffle, to opt in."
        )
    return _run_mxfp8_bpreshuffle_gemm_a8_gfx1250(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        kernel_name=kernel_name,
        num_buffers=cfg["num_buffers"],
        split_k=cfg["split_k"],
        cluster_m=cfg["cluster_m"],
        cluster_n=cfg["cluster_n"],
        m_warp=cfg["m_warp"],
        n_warp=cfg["n_warp"],
        x_scale_transposed=True,
        a_preshuffle=cfg["a_preshuffle"],
        fused_splitk=cfg["fused_splitk"],
        persistent_n_tiles=cfg["persistent_n_tiles"],
        early_a=cfg["early_a"],
        early_b=cfg["early_b"],
        l2_prefetch=cfg["l2_prefetch"],
    )


def run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
    a_scale_preshuffled: bool = False,
    w_scale_32x32: bool = False,
    batch: int = 1,
    y_scale_out: Tensor | None = None,
) -> Tensor:
    """Decode a tuned mxfp8_32 kernelName and run it.

    x_scale is row-major by default, or 32x4 shuffled with
    ``a_scale_preshuffled=True`` and an ``_aspre`` kernelName suffix.
    w_scale is 32x4 shuffled, or row-major 32x32 blocks with
    ``w_scale_32x32``; WQ is 16x16 preshuffled. ``batch`` and
    ``y_scale_out``: see ``_run_mxfp8_bpreshuffle_gemm_a8_gfx1250``.
    """
    cfg = dict(_resolved_wmma_config(kernel_name, Out.shape[0], True, True))
    if cfg["a_preshuffle"]:
        raise ValueError(
            f"[FlyDSL gfx1250 mxfp8_32] a_preshuffle is not supported: {kernel_name!r}"
        )
    if cfg["a_scale_preshuffled"] != a_scale_preshuffled:
        raise ValueError("a_scale_preshuffled must match the kernelName _aspre suffix")
    return _run_mxfp8_bpreshuffle_gemm_a8_gfx1250(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        kernel_name=kernel_name,
        num_buffers=cfg["num_buffers"],
        split_k=cfg["split_k"],
        cluster_m=cfg["cluster_m"],
        cluster_n=cfg["cluster_n"],
        m_warp=cfg["m_warp"],
        n_warp=cfg["n_warp"],
        fused_splitk=cfg["fused_splitk"],
        persistent_n_tiles=cfg["persistent_n_tiles"],
        early_a=cfg["early_a"],
        early_b=cfg["early_b"],
        l2_prefetch=cfg["l2_prefetch"],
        k_pipe=cfg["k_pipe"],
        scale_block=MX32_BLOCK_K,
        a_scale_preshuffled=a_scale_preshuffled,
        w_scale_32x32=w_scale_32x32,
        batch=batch,
        y_scale_out=y_scale_out,
    )


# ---------------------------------------------------------------------------
# Decode-sized weight-streaming mxfp8_32 GEMM (kernels/gemm_a8w8_mx32_stream_gfx1250)
# ---------------------------------------------------------------------------

MX32_STREAM_NAME_PREFIX = "flydsl_mxfp8_32_bpreshuffle_stream"
_MX32_STREAM_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(MX32_STREAM_NAME_PREFIX)}_t(?P<tile_m>\d+)x(?P<tile_n>\d+)"
    r"_wn(?P<wn>\d+)_wk(?P<wk>\d+)_sk(?P<split_k>\d+)(?P<cluster>c)?"
    r"_d(?P<depth>\d+)(?P<a_lds>_al)?(?P<a_wave>_aw)?(?P<k_interleave>_ki)?$"
)
_launch_stream = None


def mxfp8_32_stream_kernel_name(
    tile_m, tile_n, wn, wk, split_k, depth, a_lds=0, cluster=False, k_interleave=False
) -> str:
    """a_lds: 0 direct A, 1 workgroup-staged A (``_al``), 2 per-wave A transpose (``_aw``).
    k_interleave (``_ki``): a K-wave takes every wk-th K step instead of a contiguous range.
    """
    return (
        f"{MX32_STREAM_NAME_PREFIX}_t{tile_m}x{tile_n}_wn{wn}_wk{wk}"
        f"_sk{split_k}{'c' if cluster else ''}_d{depth}"
        f"{ {0: '', 1: '_al', 2: '_aw'}[int(a_lds)] }{'_ki' if k_interleave else ''}"
    )


def is_mxfp8_32_stream_kernel_name(name: str) -> bool:
    return _MX32_STREAM_KERNEL_NAME_RE.fullmatch(name) is not None


@functools.lru_cache(maxsize=1024)
def parse_mxfp8_32_stream_kernel_name(name: str) -> dict | None:
    match = _MX32_STREAM_KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    g = match.groupdict()
    cfg = {
        key: int(g[key]) for key in ("tile_m", "tile_n", "wn", "wk", "split_k", "depth")
    }
    cfg["cluster"] = g["cluster"] is not None
    if g["a_lds"] is not None and g["a_wave"] is not None:
        return None
    cfg["a_lds"] = (
        1 if g["a_lds"] is not None else (2 if g["a_wave"] is not None else 0)
    )
    cfg["k_interleave"] = g["k_interleave"] is not None
    return cfg


def _stream_shape_error(cfg: dict, M: int, N: int, K: int) -> str | None:
    from aiter.jit.utils.lds_capacity import get_lds_capacity_bytes

    from .kernels.gemm_a8w8_mx32_stream_gfx1250 import (
        stream_k_steps_per_wave,
        stream_lds_bytes,
    )

    tile_m, tile_n = cfg["tile_m"], cfg["tile_n"]
    wn, wk, split_k = cfg["wn"], cfg["wk"], cfg["split_k"]
    waves = wn * wk
    if tile_m % 16 or not 16 <= tile_m <= 256 or tile_n % (16 * wn):
        return f"unsupported tile {tile_m}x{tile_n} for wn={wn}"
    n_blocks = tile_n // (16 * wn)
    if not 1 <= n_blocks <= 8 or not 1 <= waves <= 16 or not 1 <= split_k <= 16:
        return "unsupported waves / n_blocks / split_k"
    if N % tile_n:
        return f"N={N} is not a multiple of tile_n={tile_n}"
    if stream_k_steps_per_wave(K, wk, split_k) < 1:
        return f"K={K} does not split into 128-wide steps over {wk}x{split_k}"
    tile_vec, block = tile_m * tile_n // 4, waves * 32
    if tile_vec % block and block % tile_vec:
        return f"tile {tile_m}x{tile_n} does not divide over {waves} waves"
    if split_k > 1 and waves == 1:
        return "split_k > 1 needs two or more waves"
    if cfg["cluster"] and (tile_m * tile_n // 4) % split_k:
        return "cluster split_k must divide the tile"
    if cfg["a_lds"] == 1 and (tile_m * (K // split_k) // 16) % (waves * 32):
        return "staged A tile does not divide over the block"
    lds = stream_lds_bytes(K, tile_m, n_blocks, wn, wk, split_k, cfg["a_lds"])
    if lds > get_lds_capacity_bytes("gfx1250"):
        return f"needs {lds} bytes of LDS"
    return None


@functools.lru_cache(maxsize=4096)
def mxfp8_32_stream_kernel_launchable(kernel_name: str, M: int, N: int, K: int) -> bool:
    cfg = parse_mxfp8_32_stream_kernel_name(kernel_name)
    return cfg is not None and M > 0 and _stream_shape_error(cfg, M, N, K) is None


@capture_scoped_cache
def _stream_semaphore(device: torch.device, stream_id: int, tiles: int) -> Tensor:
    """Per-tile arrival counters; every reducer hands its counter back zeroed."""
    del stream_id
    with persistent_alloc(device):
        return torch.zeros((tiles,), dtype=torch.int32, device=device)


def run_gemm_a8w8_mxfp8_32_stream_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
) -> Tensor:
    """Run the decode weight-streaming mxfp8_32 GEMM named by ``kernel_name``.

    XQ ``(M, K)`` e4m3 with unit column stride and a 16-byte-multiple row
    stride; WQ ``(N, K)`` e4m3 16x16 preshuffled; x_scale ``(M, K//32)`` e8m0
    row-major; w_scale ``(N, K//32)`` e8m0 32x4 shuffled; Out ``(M, N)``
    bf16/f16 with unit column stride.
    """
    cfg = parse_mxfp8_32_stream_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 mxfp8_32 stream] bad kernelName {kernel_name!r}"
        )
    if XQ.dim() != 2 or WQ.dim() != 2 or XQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250 mxfp8_32 stream] A/B must be 2-D fp8")
    M, K = XQ.shape
    N = WQ.shape[0]
    if WQ.shape[1] != K or tuple(Out.shape) != (M, N):
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_32 stream] shape mismatch A{tuple(XQ.shape)} "
            f"B{tuple(WQ.shape)} C{tuple(Out.shape)}"
        )
    if M == 0:
        return Out
    err = _stream_shape_error(cfg, M, N, K)
    if err is not None:
        raise RuntimeError(f"[FlyDSL gfx1250 mxfp8_32 stream] {kernel_name}: {err}")
    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(f"[FlyDSL gfx1250 mxfp8_32 stream] out dtype {Out.dtype}")
    if (
        XQ.stride(1) != 1
        or XQ.stride(0) % 16
        or not WQ.is_contiguous()
        or Out.stride(1) != 1
        or Out.stride(0) % 4
    ):
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8_32 stream] needs unit column strides, a "
            "16-byte-multiple A row stride, an 8-byte-aligned C row and a "
            "contiguous B"
        )
    k_blocks = K // MX32_BLOCK_K
    a_scale = _require_e8m0_scale(x_scale, (M, k_blocks), "x_scale")
    b_scale = _require_e8m0_scale(w_scale, (N, k_blocks), "w_scale")
    if not (a_scale.is_contiguous() and b_scale.is_contiguous()):
        raise RuntimeError("[FlyDSL gfx1250 mxfp8_32 stream] scales must be contiguous")

    global _launch_stream
    _lazy_import()
    if _launch_stream is None:
        from .kernels.gemm_a8w8_mx32_stream_gfx1250 import (
            launch_gemm_a8w8_mx32_stream,
        )

        _launch_stream = launch_gemm_a8w8_mx32_stream
    tile_m, tile_n = cfg["tile_m"], cfg["tile_n"]
    split_k = cfg["split_k"]
    torch_stream = torch.cuda.current_stream(device=XQ.device)
    tiles = ((M + tile_m - 1) // tile_m) * (N // tile_n)
    if split_k > 1:
        stream_id = int(torch_stream.cuda_stream)
        ws = _fused_splitk_workspace(
            Out.device, stream_id, (tiles, split_k, tile_m * tile_n)
        )
        sem = _stream_semaphore(Out.device, stream_id, tiles)
    else:
        ws = sem = Out
    spec = (
        K,
        tile_m,
        tile_n // (16 * cfg["wn"]),
        cfg["wn"],
        cfg["wk"],
        split_k,
        cfg["depth"],
        cfg["a_lds"],
        cfg["cluster"],
        1 if out_dtype == "f16" else 0,
        cfg["k_interleave"],
    )
    _run_compiled(
        _launch_stream,
        _ptr_arg(Out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(a_scale),
        _ptr_arg(b_scale),
        _ptr_arg(ws),
        _ptr_arg(sem),
        M,
        _fx.Stream(torch_stream),
        N,
        XQ.stride(0),
        Out.stride(0),
        *spec,
        specialization_key=("mx32_stream", Out.device.index) + spec,
    )
    return Out


def mxfp8_32_stream_candidates(M: int, N: int, K: int) -> list[str]:
    """Stream kernelNames the tuner times for a decode-sized shape.

    Only M <= 16 is offered: one 16-row tile, the whole K inside the workgroup
    (``wk`` K-waves, no cross-workgroup split), which is where the kernel beats
    the TDM kernels on gfx1250.
    """
    from .kernels.gemm_a8w8_mx32_stream_gfx1250 import stream_k_steps_per_wave

    if not 0 < M <= 16:
        return []
    names = []
    for tile_n in (16, 32):
        for wk in (2, 4, 6, 8, 12, 16):
            kpw = stream_k_steps_per_wave(K, wk, 1)
            if kpw < 1:
                continue
            for depth in sorted({min(kpw, 3), min(kpw, 6)}):
                # Interleaved K steps (_ki) only change the order when a wave
                # owns two or more steps.
                for k_interleave in (False, True) if kpw > 1 else (False,):
                    name = mxfp8_32_stream_kernel_name(
                        16, tile_n, 1, wk, 1, depth, k_interleave=k_interleave
                    )
                    if mxfp8_32_stream_kernel_launchable(name, M, N, K):
                        names.append(name)
    return names
