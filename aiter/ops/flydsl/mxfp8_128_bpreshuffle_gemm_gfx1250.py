# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 FlyDSL backend for mxfp8_128 bpreshuffle GEMM.

mxfp8_128: fp8 e4m3 activations/weights with fp8_e8m0 block scales over a
128-element block (distinct from the 32-element-block MXFP8 and from the
fp32-scale blockscale GEMM).
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

_launch_gemm_a8w8 = None
_launch_gemm_a8w8_compute_bound = None
_compile_splitk_reduce = None
_run_compiled = None
_ptr_arg = None
_fx = None

BLOCK_K = 128
_BLOCK_N = 128
WMMA_NAME_PREFIX = "flydsl_mxfp8_128_bpreshuffle_wmma"
COMPUTE_WMMA_NAME_PREFIX = "flydsl_mxfp8_128_bpreshuffle_compute_wmma"
_SUPPORTED_NUM_BUFFERS = (2, 3, 4)
_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_MAX_SPLIT_K = 8


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
            f"[FlyDSL gfx1250 mxfp8_128] {name} must have shape {shape}, "
            f"got {tuple(scale.shape)}"
        )
    if scale.dtype != dtypes.fp8_e8m0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] {name} must be fp8_e8m0, got {scale.dtype}"
        )
    return scale


def _run_mxfp8_128_preshuffle_gemm_a8_gfx1250(
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
) -> Tensor:
    """Run the gfx1250 WMMA mxfp8_128 bpreshuffle GEMM.

    XQ: ``(M, K)`` FP8 E4M3. WQ: ``(N, K)`` FP8 E4M3, already 16x16
    preshuffled. x_scale: ``(M, K//128)`` fp8_e8m0; when
    ``x_scale_transposed=True`` the backing storage is interpreted as
    ``(K//128, M)`` without copying. w_scale: ``(N//128, K//128)`` fp8_e8m0
    dense row-major. Out: ``(M, N)`` bf16/f16.
    """
    _lazy_import()
    compute_bound = is_compute_wmma_kernel_name(kernel_name)

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8_128] A/B must be 2-D, got "
            f"{tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250 mxfp8_128] A/B must be 1-byte fp8 storage")

    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] K mismatch: A.K={K} vs B.K={WQ.shape[1]}"
        )
    split_k = max(1, int(split_k))
    cluster_m = max(1, int(cluster_m))
    cluster_n = max(1, int(cluster_n))

    if N % _BLOCK_N != 0 or K % BLOCK_K != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] N/K must be multiples of "
            f"{_BLOCK_N}/{BLOCK_K}, got N={N}, K={K}"
        )
    if cluster_m * cluster_n > 16:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] a gfx1250 cluster holds at most 16 "
            f"workgroups, got {cluster_m}x{cluster_n}"
        )
    if N % (tile_n * cluster_n) != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] N={N} must be a multiple of "
            f"tile_n*cluster_n={tile_n}*{cluster_n}={tile_n * cluster_n}"
        )
    if not cluster_m_grid_ok(M, tile_m, cluster_m):
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] M={M} gives "
            f"ceil(M/tile_m)={(M + tile_m - 1) // tile_m} M-tiles, which is not a "
            f"multiple of cluster_m={cluster_m}; grid.x would be padded with "
            f"workgroups that own no M row and deadlock the cluster"
        )
    if K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] K={K} not a multiple of tile_k={tile_k}"
        )

    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] unsupported out dtype {Out.dtype}; "
            "expected bf16/fp16"
        )

    if split_k > _MAX_SPLIT_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] split_k={split_k} exceeds the "
            f"supported maximum of {_MAX_SPLIT_K}"
        )

    nb = int(num_buffers)
    if nb not in _SUPPORTED_NUM_BUFFERS:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] num_buffers must be one of "
            f"{_SUPPORTED_NUM_BUFFERS}, got {nb}"
        )
    if K % (split_k * tile_k) != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] K={K} must be divisible by "
            f"split_k*tile_k={split_k}*{tile_k}={split_k * tile_k}"
        )
    num_k_tiles = (K // split_k) // tile_k
    if num_k_tiles < nb:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] {nb}-buffer pipeline needs >= {nb} "
            f"K-tiles per split-k chunk, got {num_k_tiles}"
        )
    if compute_bound:
        k_pair = compute_kernel_k_pair(nb, tile_n)
        if K // split_k < 512 or num_k_tiles % k_pair != 0:
            raise RuntimeError(
                f"[FlyDSL gfx1250 mxfp8_128 compute] each split needs at least "
                f"512 K elements and a whole K-pair; got K={K}, split_k={split_k}, "
                f"tile_k={tile_k}, K-pair={k_pair}"
            )
        if cluster_m * cluster_n < 2:
            raise RuntimeError(
                f"[FlyDSL gfx1250 mxfp8_128 compute] needs a real cluster, got "
                f"{cluster_m}x{cluster_n}"
            )
    if split_k > 1:
        if Out.stride(1) != 1:
            raise RuntimeError(
                "[FlyDSL gfx1250 mxfp8_128] split_k>1 needs contiguous Out rows, "
                f"got strides={tuple(Out.stride())} for {M}x{N}"
            )
        # A padded row stride is reduced row-by-row, so each row start must keep
        # the 16B alignment the vectorised copy needs.
        if Out.stride(0) != N and (Out.stride(0) & 7 or N & 7):
            raise RuntimeError(
                f"[FlyDSL gfx1250 mxfp8_128] split_k>1 with padded Out rows needs "
                f"N and stride(0) to be multiples of 8, got N={N}, "
                f"stride(0)={Out.stride(0)}"
            )

    if not x_scale_transposed:
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8_128] x_scale_transposed=False is not supported "
            "by the dedicated mxfp8_128 kernel (A-scale must be M-contiguous)"
        )

    k_blocks = K // BLOCK_K
    a_scale = _require_e8m0_scale(x_scale, (M, k_blocks), "x_scale")
    b_scale = _require_e8m0_scale(w_scale, (N // _BLOCK_N, k_blocks), "w_scale")
    stride_ascale_k = a_scale.stride(1) if a_scale.stride(0) == 1 else M

    lda = XQ.stride(0)
    ldc = Out.stride(0)
    partials = (
        torch.empty((split_k, M, ldc), dtype=Out.dtype, device=Out.device)
        if split_k > 1
        else None
    )
    gemm_out = Out if partials is None else partials
    out_is_f16 = 1 if out_dtype == "f16" else 0

    stream = _fx.Stream(torch.cuda.current_stream(device=XQ.device))
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
    launch = _launch_gemm_a8w8_compute_bound if compute_bound else _launch_gemm_a8w8
    launch(*launch_args, BLOCK_K, split_k)
    if partials is not None:
        dense = ldc == N
        _run_compiled(
            _compile_splitk_reduce(split_k=split_k, out_dtype_str=out_dtype),
            _ptr_arg(partials),
            _ptr_arg(Out),
            M * N if dense else N,
            1 if dense else M,
            ldc,
            M * ldc * Out.element_size(),
            stream,
        )
    return Out


NAME_SUFFIX_RE = (
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)$"
)
_KERNEL_NAME_RE = re.compile(rf"^{re.escape(WMMA_NAME_PREFIX)}_{NAME_SUFFIX_RE}")
_COMPUTE_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(COMPUTE_WMMA_NAME_PREFIX)}_{NAME_SUFFIX_RE}"
)


def parse_wmma_kernel_name(name: str):
    """Parse a generic or compute-bound mxfp8_128 kernelName."""
    match = _COMPUTE_KERNEL_NAME_RE.fullmatch(name) or _KERNEL_NAME_RE.fullmatch(name)
    return (
        {key: int(value) for key, value in match.groupdict().items()} if match else None
    )


def compute_kernel_k_pair(num_buffers: int, tile_n: int) -> int:
    """K-tiles the compute-bound kernel stages per TDM super."""
    return 1 if num_buffers == 4 and tile_n == 256 else 2


def cluster_m_grid_ok(M: int, tile_m: int, cluster_m: int) -> bool:
    """Whether ``M`` fills every cluster row of the launch grid."""
    if M <= 0 or tile_m <= 0 or cluster_m < 1:
        return False
    m_blocks = (M + tile_m - 1) // tile_m
    return m_blocks % cluster_m == 0


def is_compute_wmma_kernel_name(name: str) -> bool:
    """Return whether ``name`` selects the compute-bound implementation."""
    return _COMPUTE_KERNEL_NAME_RE.fullmatch(name) is not None


def run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
) -> Tensor:
    """Decode a tuned kernelName and dispatch its internal implementation."""
    cfg = parse_wmma_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 mxfp8_128] unrecognised kernelName: {kernel_name!r}"
        )
    return _run_mxfp8_128_preshuffle_gemm_a8_gfx1250(
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
    )
