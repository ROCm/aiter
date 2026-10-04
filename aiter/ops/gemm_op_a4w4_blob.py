# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A4W4 on the tile-blob kernels (``hsa/gfx950/f4f4gemm``): MXFP4 E2M1 for both operands in the compact C0
tile blob the A6W4 kernels read for B (rb*1024 + L*16 per 256x128 tile, +2 guard K tiles), per-1x32 E8M0
scales in the A6W4 scale-tile layout. Same kernarg ABI as ``gemm_a6w4_asm``; only the A tile size differs.
"""

from torch import Tensor

from ..jit.core import compile_ops

_DEFAULT_KERNEL = "aiter_a4w4_blob_stnt_allk"


@compile_ops(
    "module_gemm_a4w4_blob_asm", fc_name="gemm_a4w4_blob_asm", ffi_type="ctypes"
)
def _gemm_a4w4_blob_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str,
    alpha: float,
    bias: Tensor | None,
) -> None: ...


def gemm_a4w4_blob_asm(
    A: Tensor,
    B: Tensor,
    A_scale: Tensor,
    B_scale: Tensor,
    out: Tensor,
    K: int,
    kernelName: str | None = None,
    alpha: float = 1.0,
) -> Tensor:
    """``out[M, N] = A @ B^T`` on packed MXFP4 blobs; ``out`` dims multiples of 256, ``K`` the padded
    multiple of 128 the blobs were packed for. No bias epilogue."""
    if float(alpha) != 1.0:
        raise ValueError("gemm_a4w4_blob supports only alpha=1.0")
    if out.ndim != 2:
        raise ValueError(f"gemm_a4w4_blob_asm expects a 2D output, got {out.ndim}D")
    _gemm_a4w4_blob_asm(
        A, B, A_scale, B_scale, out, K, kernelName or _DEFAULT_KERNEL, alpha, None
    )
    return out
