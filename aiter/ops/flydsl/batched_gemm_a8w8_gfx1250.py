# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 strided-batched mxfp8_128 GEMM.

===========  ==========================  ===========================
operand      shape                       notes
===========  ==========================  ===========================
``XQ``       ``[M, B, K]`` fp8           M-outer, K-contiguous
``WQ``       ``[B, N, K]`` fp8           16x16 preshuffled (ops.shuffle.shuffle_weight)
``x_scale``  ``[M, B, K//128]`` e8m0     row-major, same as the 2-D op's contract
``w_scale``  ``[B, N//128, K//128]`` e8m0
``Out``      ``[M, B, N]`` bf16/fp16
===========  ==========================  ===========================
"""

from __future__ import annotations

import functools
import re

import torch
from torch import Tensor

from aiter.jit.utils.chip_info import get_cu_num, get_gfx

from .mxfp8_128_bpreshuffle_gemm_gfx1250 import NAME_SUFFIX_RE
from .utils import get_shared_memory_per_block

BLOCK_K = 128
BLOCK_N = 128
BMM_WMMA_NAME_PREFIX = "flydsl_bmm_mxfp8_128_wmma"

_OUT_DTYPE_NAME = {torch.bfloat16: "bf16", torch.float16: "f16"}
_PRELOAD_MIN_TILE_M = 128
_MAX_HEURISTIC_TILE_M = 128
_LDS_BYTES = get_shared_memory_per_block(fallback_gfx="gfx1250")
_SUFFIX_RE = NAME_SUFFIX_RE.removesuffix("$")
_BMM_KERNEL_NAME_RE = re.compile(
    rf"^{re.escape(BMM_WMMA_NAME_PREFIX)}_{_SUFFIX_RE}(?P<preload>_pre)?$"
)

_launch_gemm_a8w8 = None
_ptr_arg = None
_fx = None


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


def parse_bmm_kernel_name(name: str) -> dict[str, int] | None:
    """Parse a tuned batched kernelName, or ``None`` if it is not one of ours."""
    match = _BMM_KERNEL_NAME_RE.fullmatch(name)
    if match is None:
        return None
    groups = match.groupdict()
    preload = groups.pop("preload")
    cfg = {key: int(value) for key, value in groups.items()}
    cfg["preload"] = int(preload is not None)
    return cfg


def bmm_kernel_name(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    m_warp: int,
    n_warp: int,
    num_buffers: int,
    preload: bool = False,
) -> str:
    """Canonical tuned-CSV name for a batched config (split-K/cluster are fixed)."""
    return (
        f"{BMM_WMMA_NAME_PREFIX}_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk1_cm1_cn1"
        f"{'_pre' if preload else ''}"
    )


def _align(value: int, to: int) -> int:
    return -(-value // to) * to


def preload_lds_bytes(c: dict[str, int], k: int) -> int:
    """LDS a config needs with the whole-K scale panels staged."""
    stage_a = _align(c["tile_m"] * (c["tile_k"] + 16), 16)
    stage_b = _align(c["tile_n"] * c["tile_k"], 16)
    pitch = _align(stage_a + stage_b, 1024)
    ks = k // BLOCK_K
    panel = _align(c["tile_m"] * ks, 16) + _align(
        max(1, c["tile_n"] // BLOCK_N) * ks, 16
    )
    c_pad = 8 if c["tile_n"] >= 128 else 0
    c_store = _align(c["tile_m"] * (c["tile_n"] + c_pad) * 2, 128)
    return max(pitch * c["num_buffers"] + panel, c_store)


def bmm_candidates(b: int, m: int, n: int, k: int) -> list[dict[str, int]]:
    """Every batched config that can legally run this shape, tuner-ordered."""
    from .gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        is_compute_kernel,
        kernel_fits_shape,
        kernels_list,
    )

    out = []
    for ki in kernels_list.values():
        if is_compute_kernel(ki) or ki.split_k != 1:
            continue
        if ki.cluster_m != 1 or ki.cluster_n != 1:
            continue
        if not kernel_fits_shape(ki, m, n, k):
            continue
        out.append(
            {
                "tile_m": ki.tile_m,
                "tile_n": ki.tile_n,
                "tile_k": ki.tile_k,
                "m_warp": ki.m_warp,
                "n_warp": ki.n_warp,
                "num_buffers": ki.num_buffers,
            }
        )
    return out


@functools.lru_cache(maxsize=1024)
def pick_bmm_kernel_name(b: int, m: int, n: int, k: int) -> str:
    """Heuristic config for a shape with no tuned row."""
    cands = bmm_candidates(b, m, n, k)
    if not cands:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] no batched mxfp8_128 kernel fits "
            f"B={b}, M={m}, N={n}, K={k}"
        )
    num_cu = get_cu_num() or 256
    want_tm = min(_MAX_HEURISTIC_TILE_M, max(16, 1 << (m - 1).bit_length()))

    def rank(c):
        wgs = -(-m // c["tile_m"]) * (n // c["tile_n"]) * b
        tail = -(-wgs // num_cu) * num_cu - wgs
        return (
            tail,
            abs(c["tile_m"] - want_tm),
            -c["tile_n"],
            -c["num_buffers"],
            -c["tile_k"],
        )

    c = min(cands, key=rank)
    preload = (
        c["tile_m"] >= _PRELOAD_MIN_TILE_M and preload_lds_bytes(c, k) <= _LDS_BYTES
    )
    return bmm_kernel_name(
        c["tile_m"],
        c["tile_n"],
        c["tile_k"],
        c["m_warp"],
        c["n_warp"],
        c["num_buffers"],
        preload=preload,
    )


def _check_e8m0(scale: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    """Validate an e8m0 scale operand."""
    from aiter.utility import dtypes

    if tuple(scale.shape) != shape:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name} must have shape {shape}, "
            f"got {tuple(scale.shape)}"
        )
    if scale.dtype not in (dtypes.fp8_e8m0, torch.uint8):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name} must be e8m0/uint8, got {scale.dtype}"
        )
    if not scale.is_contiguous():
        raise RuntimeError(f"[FlyDSL gfx1250 bmm] {name} must be contiguous")
    return scale


def run_bmm_a8w8_mxfp8_128_gfx1250(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str | None = None,
) -> Tensor:
    """Run the batched mxfp8_128 GEMM; writes into ``Out`` and returns it."""
    if get_gfx() != "gfx1250":
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] batched mxfp8_128 requires gfx1250, "
            f"got {get_gfx()}"
        )
    _lazy_import()

    if XQ.dim() != 3 or WQ.dim() != 3 or Out.dim() != 3:
        raise RuntimeError(
            "[FlyDSL gfx1250 bmm] XQ/WQ/Out must be 3-D, got "
            f"{tuple(XQ.shape)}, {tuple(WQ.shape)}, {tuple(Out.shape)}"
        )
    m, b, k = XQ.shape
    wb, n, wk = WQ.shape
    if (wb, wk) != (b, k):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] WQ {tuple(WQ.shape)} does not match "
            f"XQ {tuple(XQ.shape)}; expected [{b}, N, {k}]"
        )
    if tuple(Out.shape) != (m, b, n):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] Out must be [{m}, {b}, {n}], "
            f"got {tuple(Out.shape)}"
        )
    if XQ.element_size() != 1 or WQ.element_size() != 1:
        raise RuntimeError("[FlyDSL gfx1250 bmm] XQ/WQ must be 1-byte fp8 storage")
    if n % BLOCK_N or k % BLOCK_K:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] N/K must be multiples of {BLOCK_N}/{BLOCK_K}, "
            f"got N={n}, K={k}"
        )
    out_dtype = _OUT_DTYPE_NAME.get(Out.dtype)
    if out_dtype is None:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] unsupported out dtype {Out.dtype}; expected "
            "bf16/fp16"
        )
    if XQ.stride() != (b * k, k, 1):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] XQ must be contiguous [M,B,K], got strides "
            f"{tuple(XQ.stride())}"
        )
    if Out.stride() != (b * n, n, 1):
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] Out must be contiguous [M,B,N], got strides "
            f"{tuple(Out.stride())}"
        )
    if not WQ.is_contiguous():
        raise RuntimeError("[FlyDSL gfx1250 bmm] WQ must be contiguous")

    k_blocks = k // BLOCK_K
    _check_e8m0(x_scale, (m, b, k_blocks), "x_scale")
    _check_e8m0(w_scale, (b, n // BLOCK_N, k_blocks), "w_scale")

    name = kernel_name or pick_bmm_kernel_name(b, m, n, k)
    cfg = parse_bmm_kernel_name(name)
    if cfg is None:
        raise ValueError(f"[FlyDSL gfx1250 bmm] unrecognised kernelName: {name!r}")
    if cfg["split_k"] != 1 or cfg["cluster_m"] != 1 or cfg["cluster_n"] != 1:
        raise ValueError(
            f"[FlyDSL gfx1250 bmm] split-K / clusters are not supported on the "
            f"batched path, got {name!r}"
        )
    tile_n, tile_k = cfg["tile_n"], cfg["tile_k"]
    preload_ks = k_blocks if cfg["preload"] else 0
    if preload_ks:
        need = preload_lds_bytes(cfg, k)
        if need > _LDS_BYTES:
            raise RuntimeError(
                f"[FlyDSL gfx1250 bmm] {name!r} needs {need} B of LDS with the "
                f"K={k} scale panels, over the {_LDS_BYTES} B budget"
            )
    if n % tile_n or k % tile_k:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name!r} needs N%{tile_n}==0 and K%{tile_k}==0, "
            f"got N={n}, K={k}"
        )
    if k // tile_k < cfg["num_buffers"]:
        raise RuntimeError(
            f"[FlyDSL gfx1250 bmm] {name!r} needs >= {cfg['num_buffers']} K-tiles, "
            f"got {k // tile_k}"
        )

    _launch_gemm_a8w8(
        _ptr_arg(Out),
        _ptr_arg(XQ),
        _ptr_arg(WQ),
        _ptr_arg(x_scale),
        _ptr_arg(w_scale),
        m,
        _fx.Stream(torch.cuda.current_stream(device=XQ.device)),
        n,
        k,
        b * k_blocks,  # stride_ascale_k: x_scale row stride, [M, B, K//128]
        b * k,  # lda: XQ is [M, B, K]
        b * n,  # ldc: Out is [M, B, N]
        cfg["tile_m"],
        tile_n,
        tile_k,
        cfg["m_warp"],
        cfg["n_warp"],
        1 if out_dtype == "f16" else 0,
        cfg["num_buffers"],
        1,  # cluster_m
        1,  # cluster_n
        True,  # is_mxscale
        BLOCK_K,
        1,  # split_k
        True,  # batched
        preload_ks,
        b,
    )
    return Out


__all__ = [
    "run_bmm_a8w8_mxfp8_128_gfx1250",
]
