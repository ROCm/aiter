# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 (WMMA) backend for the FlyDSL a8w8 bpreshuffle GEMM."""

from __future__ import annotations

import re

import torch
from torch import Tensor

# Lazily bound flydsl symbols (kept out of import path when flydsl is absent).
_launch_gemm_a8w8_ptpc = None
_ptr_arg = None
_fx = None

_WMMA_K = 128
_SUPPORTED_NUM_BUFFERS = (2, 3, 4)
_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_MAX_SPLIT_K = 1


def _lazy_import():
    global _launch_gemm_a8w8_ptpc, _ptr_arg, _fx
    if _launch_gemm_a8w8_ptpc is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.gemm_a8w8_ptpc_gfx1250 import launch_gemm_a8w8_ptpc
    from .kernels.tensor_shim import ptr_arg

    _launch_gemm_a8w8_ptpc = launch_gemm_a8w8_ptpc
    _ptr_arg = ptr_arg
    _fx = fx_mod


def _as_1d_fp32(scale: Tensor, length: int, name: str) -> Tensor:
    flat = scale.reshape(-1)
    if flat.numel() != length:
        raise RuntimeError(
            f"[FlyDSL gfx1250] {name} must have {length} elements, "
            f"got {tuple(scale.shape)}"
        )
    if flat.dtype != torch.float32:
        flat = flat.to(torch.float32)
    return flat.contiguous()


def run_preshuffle_gemm_a8_gfx1250(
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
    waves_per_eu: int = 0,
    m_warp: int = 2,
    n_warp: int = 2,
    cluster_m: int = 1,
    cluster_n: int = 1,
    split_k: int = 1,
) -> Tensor:
    """Run the gfx1250 WMMA a8w8 bpreshuffle (ptpc) GEMM; writes into ``Out``.

    XQ: (M, K) FP8 E4M3. WQ: (N, K) FP8 E4M3, 16x16 ``shuffle_weight``.
    x_scale: per-token fp32 ``(M,)`` or ``(M, 1)``. w_scale: per-channel fp32
    ``(N,)`` or ``(N, 1)``. ``Out``: (M, N) bf16/f16.
    """
    _lazy_import()

    if XQ.dim() != 2 or WQ.dim() != 2:
        raise RuntimeError(
            f"[FlyDSL gfx1250] A/B must be 2-D, got {tuple(XQ.shape)}, {tuple(WQ.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250] A/B must be 1-byte fp8 storage")

    M, K = XQ.shape
    N = WQ.shape[0]
    if K != WQ.shape[1]:
        raise RuntimeError(f"[FlyDSL gfx1250] K mismatch: A.K={K} vs B.K={WQ.shape[1]}")
    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL gfx1250] N={N} not a multiple of tile_n={tile_n}")
    if K % _WMMA_K != 0 or K % tile_k != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250] K={K} must be a multiple of WMMA_K={_WMMA_K} and "
            f"tile_k={tile_k}"
        )

    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250] unsupported out dtype {Out.dtype}; expected bf16/fp16"
        )

    split_k = max(1, int(split_k))
    cluster_m = max(1, int(cluster_m))
    cluster_n = max(1, int(cluster_n))

    if split_k > _MAX_SPLIT_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250] split_k={split_k} exceeds the bf16/f16 atomic-add "
            f"precision cap of {_MAX_SPLIT_K}"
        )

    # Validate (tuned names always pass); fail loudly rather than silently clamp.
    nb = int(num_buffers)
    if nb not in _SUPPORTED_NUM_BUFFERS:
        raise RuntimeError(
            f"[FlyDSL gfx1250] num_buffers must be one of {_SUPPORTED_NUM_BUFFERS}, "
            f"got {nb}"
        )
    if K % (split_k * tile_k) != 0:
        raise RuntimeError(
            f"[FlyDSL gfx1250] K={K} must be divisible by split_k*tile_k="
            f"{split_k}*{tile_k}={split_k * tile_k}"
        )
    num_k_tiles = (K // split_k) // tile_k
    if num_k_tiles < nb:
        raise RuntimeError(
            f"[FlyDSL gfx1250] {nb}-buffer pipeline needs >= {nb} K-tiles per "
            f"split-k chunk, got {num_k_tiles} (K={K}, split_k={split_k}, tile_k={tile_k})"
        )

    sa = _as_1d_fp32(x_scale, M, "x_scale")
    sb = _as_1d_fp32(w_scale, N, "w_scale")

    lda = XQ.stride(0)
    ldc = Out.stride(0)
    out_is_f16 = 1 if out_dtype == "f16" else 0
    if split_k > 1:
        Out.zero_()  # split-k atomic-accumulates into Out

    stream = _fx.Stream(torch.cuda.current_stream(device=XQ.device))
    _launch_gemm_a8w8_ptpc(
        Out,
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        sa,
        sb,
        M,
        stream,
        N,
        K,
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
    )
    return Out


# flydsl_bpreshuffle_wmma_t{tm}x{tn}x{tk}_mw{mw}_nw{nw}_nb{nb}_sk{sk}_cm{cm}_cn{cn}
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_bpreshuffle_wmma_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)$"
)


def parse_wmma_kernel_name(name: str):
    """Parse a flydsl_bpreshuffle_wmma_ kernelName into its config dict, or None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def run_gemm_a8w8_bpreshuffle_gfx1250(
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
        raise ValueError(f"[FlyDSL gfx1250] unrecognised kernelName: {kernel_name!r}")
    return run_preshuffle_gemm_a8_gfx1250(
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
    )
