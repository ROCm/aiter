# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch contract of the gfx950 ring GEMM (``kernels/gemm_a16w16_ring_gfx950.py``)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.gemm_a16w16_gfx950 import GEMM_A16W16_DTYPE_BF16, GEMM_A16W16_DTYPE_FP16
from .kernels.gemm_a16w16_ring_gfx950 import (
    RING_TILE,
    gemm_a16w16_ring_gfx950,
    make_gemm_a16w16_ring_param,
)
from .kernels.tensor_shim import _run_compiled

__all__ = ["gemm_a16w16_ring"]

# The kernel's buffer offsets are 32-bit byte offsets.
_MAX_OPERAND_BYTES = 1 << 31


def gemm_a16w16_ring(
    a: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    block_m: int = RING_TILE["block_m"],
    block_n: int = RING_TILE["block_n"],
    block_k: int = RING_TILE["block_k"],
    stages: int = RING_TILE["stages"],
    m_waves: int = RING_TILE["m_waves"],
    n_waves: int = RING_TILE["n_waves"],
    group_m: int = 1,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """C[M, N] = A[M, K] @ W[N, K].T on gfx950; C has the input dtype.

    A and W are contiguous BF16 or FP16 tensors on one device, 16-byte aligned, each under 2 GiB.
    K must be a multiple of ``block_k`` with at least ``stages + 1`` K tiles and N a multiple of 8;
    M is any size. ``out``, when given, is a contiguous [M, N] tensor of the input dtype on that
    device. ``group_m`` groups M tiles per XCD so the CUs sharing an A row block share an L2
    (1 measured best at 16384x384x5120). Each distinct (N, K, tile) compiles its own kernel.
    """
    if get_gfx_runtime() != "gfx950":
        raise RuntimeError("gemm_a16w16_ring supports gfx950 only")
    if a.dtype not in (torch.bfloat16, torch.float16) or w.dtype != a.dtype:
        raise ValueError("A and W must both be bfloat16 or both float16")
    if a.ndim != 2 or w.ndim != 2 or not a.is_contiguous() or not w.is_contiguous():
        raise ValueError("A and W must be contiguous 2-D tensors")
    if a.device.type != "cuda" or w.device != a.device:
        raise ValueError("A and W must be on one GPU")
    m, k = a.shape
    n = w.shape[0]
    if w.shape[1] != k:
        raise ValueError(f"shape mismatch: A {tuple(a.shape)}, W {tuple(w.shape)}")
    if out is None:
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    elif (
        out.shape != (m, n)
        or out.dtype != a.dtype
        or out.device != a.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous [M, N] tensor of A's dtype on A's device"
        )
    for t in (a, w, out):
        if t.data_ptr() % 16:
            raise ValueError("A, W and out must be 16-byte aligned")
        if t.numel() * t.element_size() >= _MAX_OPERAND_BYTES:
            raise ValueError("A, W and out must each be under 2 GiB")
    if m == 0:
        return out
    in_dtype_id = (
        GEMM_A16W16_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_A16W16_DTYPE_BF16
    )
    param = make_gemm_a16w16_ring_param(
        in_dtype_id, n, k, block_m, block_n, block_k, stages, m_waves, n_waves, group_m
    )
    if stream is None:
        stream = torch.cuda.current_stream(a.device)
    # The kernel reads B as a logical [K, N]: W[N, K] is its column-major form.
    _run_compiled(
        gemm_a16w16_ring_gfx950,
        flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=1),
        flyc.from_dlpack(w.t()).mark_layout_dynamic(leading_dim=0),
        param,
        fx.Stream(stream),
        specialization_key=(param.__cache_signature__(), a.device.index),
    )
    return out
