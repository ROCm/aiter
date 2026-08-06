# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""L2 normalization utilities for chunk_delta_attn kernels."""

import torch
import triton
import triton.language as tl

# Default forward-kernel tile config (MBLOCK=32, num_warps=4): a robust
# choice across the supported head dimensions D in {64, 128, 256, 512}.
_L2NORM_FWD_BT = 32
_L2NORM_FWD_NUM_WARPS = 4


@triton.jit
def l2norm_fwd_kernel1(
    X,
    Y,
    Rstd,
    eps,
    D,
    BD: tl.constexpr,
    STORE_RSTD: tl.constexpr,
):
    """L2 normalize per row, D > 512 (one row per program)."""
    i_t = tl.program_id(0)
    X += i_t * D
    Y += i_t * D
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.rsqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(Y + cols, b_y.to(Y.dtype.element_ty), mask=mask)
    if STORE_RSTD:
        tl.store(Rstd + i_t, b_rstd)


@triton.jit
def l2norm_fwd_kernel(
    X,
    Y,
    Rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    STORE_RSTD: tl.constexpr,
):
    """L2 normalize per row, D <= 512 (BT rows per program)."""
    xoffset = tl.program_id(0) * BT
    row_idx = xoffset + tl.arange(0, BT)[:, None]
    xmask = row_idx < T
    col_idx = tl.arange(0, BD)[None, :]
    cmask = col_idx < D
    mask = xmask & cmask
    x = tl.load(X + col_idx + D * row_idx, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(tl.where(xmask, x * x, 0.0), axis=1)
    rstd = tl.rsqrt(sumsq + eps)
    y = x * rstd[:, None]
    tl.store(Y + col_idx + D * row_idx, y.to(Y.dtype.element_ty), mask=mask)
    if STORE_RSTD:
        row1d = xoffset + tl.arange(0, BT)
        tl.store(Rstd + row1d, rstd, mask=row1d < T)


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
    *,
    need_rstd: bool = False,
):
    """
    Forward pass for L2 normalization.

    Args:
        x: Input tensor of shape ``[..., D]``.
        eps: Numerical-stability constant. Default ``1e-6``.
        output_dtype: Output dtype. ``None`` (default) keeps input dtype.
        need_rstd: If ``True``, also return the per-row reciprocal-std
            tensor. Default ``False`` (inference path).

    Returns:
        ``(y, rstd)`` where ``rstd`` is ``None`` when ``need_rstd=False``.
    """
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if need_rstd:
        rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    else:
        rstd = y  # placeholder; STORE_RSTD=False → never dereferenced

    if D <= 512:
        BT = _L2NORM_FWD_BT
        l2norm_fwd_kernel[(triton.cdiv(T, BT),)](
            x,
            y,
            rstd,
            eps,
            T,
            D,
            BD,
            BT,
            STORE_RSTD=need_rstd,
            num_warps=_L2NORM_FWD_NUM_WARPS,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            rstd,
            eps,
            D,
            BD,
            STORE_RSTD=need_rstd,
        )

    if need_rstd:
        return y.view(x_shape_og), rstd.view(x_shape_og[:-1])
    return y.view(x_shape_og), None
