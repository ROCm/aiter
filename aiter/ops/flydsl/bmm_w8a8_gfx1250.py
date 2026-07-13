# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 (WMMA) public interface for the FlyDSL W8A8 blockwise batched GEMM.

Wraps :func:`aiter.ops.flydsl.kernels.bmm_w8a8_gfx1250.compile_bmm_w8a8_bpreshuffle_gfx1250`.

Operation (both operands fp8_e4m3fn, E8M0 uint8 blockwise dequant)::

    C[.., n] = sum_k (A_fp8[.., k] * a_scale[.., k//gk])
                    * (B_fp8[b, n, k] * b_scale[b, n//gn, k//gk])

Layouts (``B`` / ``b_scale`` are always ``[B, N, K]`` / ``[B, N//gn, K//gk]``)::

    layout="mbn":  A=[M, B, K]  a_scale=[M, B, K//gk]  C=[M, B, N]
    layout="bmn":  A=[B, M, K]  a_scale=[B, M, K//gk]  C=[B, M, N]

The kernel clips A loads / C stores to the runtime M via hardware OOB, so M may be
ragged (no host padding of A / a_scale). N / K must divide their tiles.
"""

from __future__ import annotations

import re

import torch
from torch import Tensor

from aiter import dtypes

# Lazily bound flydsl symbols (kept out of import path when flydsl is absent).
_compile_bmm = None
_run_compiled = None
_fx = None

_WMMA_K = 128
_SUPPORTED_NUM_BUFFERS = (2, 3, 4)
_OUT_DTYPE_NAME = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}
_MAX_SPLIT_K = 4


def _lazy_import():
    global _compile_bmm, _run_compiled, _fx
    if _compile_bmm is not None:
        return
    import flydsl.expr as fx_mod

    from .kernels.bmm_w8a8_gfx1250 import compile_bmm_w8a8_bpreshuffle_gfx1250
    from .kernels.tensor_shim import _run_compiled as runner

    _compile_bmm = compile_bmm_w8a8_bpreshuffle_gfx1250
    _run_compiled = runner
    _fx = fx_mod


def _shapes_from_layout(A: Tensor, C: Tensor, layout: str):
    """Return (B, M, N, K) derived from the A / C shapes for the given layout."""
    if A.dim() != 3 or C.dim() != 3:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] A and C must be 3-D, got A={tuple(A.shape)}, C={tuple(C.shape)}")
    if layout == "mbn":
        M, B, K = A.shape
        Mc, Bc, N = C.shape
    elif layout == "bmn":
        B, M, K = A.shape
        Bc, Mc, N = C.shape
    else:
        raise ValueError(f"[FlyDSL bmm_w8a8] layout must be 'mbn' or 'bmn', got {layout!r}")
    if (Mc, Bc) != (M, B):
        raise RuntimeError(f"[FlyDSL bmm_w8a8] A / C (M, B) mismatch for layout={layout!r}: A gives (M={M}, B={B}), C gives (M={Mc}, B={Bc})")
    return B, M, N, K

_KERNEL_NAME_RE = re.compile(
    r"^flydsl_blockscale_bpreshuffle_bmm_"
    r"t(?P<tile_m>\d+)x(?P<tile_n>\d+)x(?P<tile_k>\d+)_"
    r"mw(?P<m_warp>\d+)_nw(?P<n_warp>\d+)_"
    r"nb(?P<num_buffers>\d+)_sk(?P<split_k>\d+)_"
    r"cm(?P<cluster_m>\d+)_cn(?P<cluster_n>\d+)$"
)

def parse_wmma_kernel_name(name: str):
    """Parse a flydsl_blockscale_bpreshuffle_wmma_ kernelName, or return None."""
    m = _KERNEL_NAME_RE.fullmatch(name)
    return {k: int(v) for k, v in m.groupdict().items()} if m else None


def run_bmm_w8a8_gfx1250(
    C: Tensor,
    A: Tensor,
    B_mat: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = 128,
    *,
    layout: str = "mbn",
    group_k: int = 128,
    group_n: int = 128,
    num_buffers: int = 2,
    m_warp: int = 2,
    n_warp: int = 2,
    waves_per_eu: int = 0,
    cluster_m: int = 1,
    cluster_n: int = 1,
    split_k: int = 1,
    preshuffle_b: bool = True,
) -> Tensor:
    """Run the gfx1250 WMMA W8A8 blockwise batched GEMM; writes into ``C``.

    A / B are fp8_e4m3fn (1-byte); ``a_scale`` / ``b_scale`` are uint8 E8M0
    block scales. Shapes follow ``layout`` (see module docstring); ``B_mat`` is
    ``[B, N, K]`` fp8 (16x16-shuffled when ``preshuffle_b=True``) and
    ``b_scale`` is ``[B, N//group_n, K//group_k]``. ``C`` is the output
    (bf16 / f16 / f32) laid out per ``layout``.
    """
    _lazy_import()

    Bn, M, N, K = _shapes_from_layout(A, C, layout)

    if B_mat.dim() != 3:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] B must be 3-D, got {tuple(B_mat.shape)}")
    if not preshuffle_b:
        Bb, Nb, Kb = B_mat.shape
        if (Bb, Nb, Kb) != (Bn, N, K):
            raise RuntimeError(f"[FlyDSL bmm_w8a8] B must be [B={Bn}, N={N}, K={K}], got {tuple(B_mat.shape)}")

    if A.element_size() != 1 or B_mat.element_size() != 1:
        raise RuntimeError("[FlyDSL bmm_w8a8] A / B must be 1-byte fp8 storage")
    if a_scale.element_size() != 1 or b_scale.element_size() != 1:
        raise RuntimeError("[FlyDSL bmm_w8a8] a_scale / b_scale must be 1-byte E8M0")

    out_dtype = _OUT_DTYPE_NAME.get(C.dtype)
    if out_dtype is None:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] unsupported out dtype {C.dtype}; expected bf16/f16/f32")

    if N % tile_n != 0:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] N={N} not a multiple of tile_n={tile_n}")
    if K % _WMMA_K != 0 or K % tile_k != 0:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] K={K} must be a multiple of WMMA_K={_WMMA_K} and tile_k={tile_k}")
    if tile_k % group_k != 0:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] tile_k={tile_k} must be a multiple of group_k={group_k}")
    if tile_n % group_n != 0:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] tile_n={tile_n} must be a multiple of group_n={group_n}")

    nb = int(num_buffers)
    if nb not in _SUPPORTED_NUM_BUFFERS:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] num_buffers must be one of {_SUPPORTED_NUM_BUFFERS}, got {nb}")

    split_k = max(1, int(split_k))
    cluster_m = max(1, int(cluster_m))
    cluster_n = max(1, int(cluster_n))
    if split_k > _MAX_SPLIT_K:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] split_k={split_k} exceeds the bf16/f16 atomic-add precision cap of {_MAX_SPLIT_K}")
    if K % (split_k * tile_k) != 0:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] K={K} must be divisible by split_k*tile_k={split_k}*{tile_k}={split_k * tile_k}")
    # Each split-k chunk must hold >= num_buffers K-tiles to fill the pipeline.
    num_k_tiles = (K // split_k) // tile_k
    if num_k_tiles < nb:
        raise RuntimeError(f"[FlyDSL bmm_w8a8] {nb}-buffer pipeline needs >= {nb} K-tiles per split-k chunk, got {num_k_tiles} (K={K}, split_k={split_k}, tile_k={tile_k})")

    exe = _compile_bmm(
        B=Bn,
        N=N,
        K=K,
        group_k=group_k,
        group_n=group_n,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        m_warp=m_warp,
        n_warp=n_warp,
        out_dtype=out_dtype,
        num_buffers=nb,
        waves_per_eu=(None if waves_per_eu <= 0 else waves_per_eu),
        cluster_m=cluster_m,
        cluster_n=cluster_n,
        split_k=split_k,
        preshuffle_b=preshuffle_b,
        layout=layout,
    )

    if split_k > 1:
        C.zero_()  # split-k atomic-accumulates into C

    stream = _fx.Stream(torch.cuda.current_stream(device=A.device))
    _run_compiled(
        exe,
        C,
        A.view(torch.uint8),
        B_mat.view(torch.uint8),
        a_scale.view(torch.uint8),
        b_scale.view(torch.uint8),
        M,
        stream,
    )
    return C

def run_bmm_a8w8_blockscale_bmm_gfx1250(
    A: Tensor,
    B: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    C: Tensor,
    layout: str,
    kernel_name: str,
) -> Tensor:
    cfg = parse_wmma_kernel_name(kernel_name)

    if cfg is None:
        raise ValueError(
            f"[FlyDSL gfx1250 blockscale bmm] unrecognised kernelName: {kernel_name!r}"
        )

    return run_bmm_w8a8_gfx1250(
        C,
        A,
        B,
        a_scale,
        b_scale,
        cfg["tile_m"],
        cfg["tile_n"],
        cfg["tile_k"],
        layout=layout,
        num_buffers=cfg["num_buffers"],
        split_k=cfg["split_k"],
        cluster_m=cfg["cluster_m"],
        cluster_n=cfg["cluster_n"],
        m_warp=cfg["m_warp"],
        n_warp=cfg["n_warp"],
    )
