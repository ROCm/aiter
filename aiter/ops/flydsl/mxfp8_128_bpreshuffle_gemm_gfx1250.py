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
_ptr_arg = None
_fx = None

_BLOCK_K = 128
_BLOCK_N = 128
_SUPPORTED_NUM_BUFFERS = (2, 3, 4)
_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_MAX_SPLIT_K = 1


def _lazy_import():
    global _launch_gemm_a8w8, _ptr_arg, _fx
    if _launch_gemm_a8w8 is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_gfx1250 import launch_gemm_a8w8
    from .kernels.tensor_shim import ptr_arg

    _launch_gemm_a8w8 = launch_gemm_a8w8
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


def run_mxfp8_128_preshuffle_gemm_a8_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    *,
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
    if N % _BLOCK_N != 0 or K % _BLOCK_K != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] N/K must be multiples of "
            f"{_BLOCK_N}/{_BLOCK_K}, got N={N}, K={K}"
        )
    if N % tile_n != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] N={N} not a multiple of tile_n={tile_n}"
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

    split_k = max(1, int(split_k))
    cluster_m = max(1, int(cluster_m))
    cluster_n = max(1, int(cluster_n))

    if split_k > _MAX_SPLIT_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 mxfp8_128] split_k={split_k} exceeds the "
            f"bf16/f16 atomic-add precision cap of {_MAX_SPLIT_K}"
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

    if not x_scale_transposed:
        raise RuntimeError(
            "[FlyDSL gfx1250 mxfp8_128] x_scale_transposed=False is not supported "
            "by the dedicated mxfp8_128 kernel (A-scale must be M-contiguous)"
        )

    k_blocks = K // _BLOCK_K
    a_scale = _require_e8m0_scale(x_scale, (M, k_blocks), "x_scale")
    b_scale = _require_e8m0_scale(w_scale, (N // _BLOCK_N, k_blocks), "w_scale")
    stride_ascale_k = a_scale.numel() // a_scale.stride(0)

    lda = XQ.stride(0)
    ldc = Out.stride(0)
    if split_k > 1:
        Out.zero_()
    out_is_f16 = 1 if out_dtype == "f16" else 0

    stream = _fx.Stream(torch.cuda.current_stream(device=XQ.device))
    _launch_gemm_a8w8(
        _ptr_arg(Out),
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
    return Out


_KERNEL_NAME_RE = re.compile(
    r"^flydsl_mxfp8_128_bpreshuffle_wmma_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)$"
)


def parse_wmma_kernel_name(name: str):
    """Parse a flydsl_mxfp8_128_bpreshuffle_wmma_ kernelName, or return None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str,
) -> Tensor:
    """Dispatch entry: decode a tuned wmma kernelName and run the kernel."""
    cfg = parse_wmma_kernel_name(kernel_name)
    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 mxfp8_128] unrecognised kernelName: {kernel_name!r}"
        )
    return run_mxfp8_128_preshuffle_gemm_a8_gfx1250(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        num_buffers=cfg["num_buffers"],
        split_k=cfg["split_k"],
        cluster_m=cfg["cluster_m"],
        cluster_n=cfg["cluster_n"],
        m_warp=cfg["m_warp"],
        n_warp=cfg["n_warp"],
        x_scale_transposed=True,
    )
