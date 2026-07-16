# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
L2 normalization utilities for Triton kernels.

This module provides efficient L2 normalization operations implemented in Triton,
supporting both forward and backward passes.
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from ..gated_delta_rule_utils import IS_AMD, autotune_cache_kwargs, input_guard

# Backward-pass autotune config space. Forward kernels deliberately do not
# autotune (see ``l2norm_fwd_kernel`` for the rationale); only the bwd
# kernels still use this list because the bwd path is autograd-only and
# its dispatch overhead is not on a critical inference loop.
BT_LIST = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


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
    """L2 normalize per row, D > 512 (one row per program; no autotune --
    the kernel is too simple to benefit from config sweep, and the host
    dispatch overhead matters here).

    ``STORE_RSTD`` is a compile-time flag: when False the rstd store is
    dead-code-eliminated and the caller may pass any placeholder tensor
    as ``Rstd``.
    """
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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps) for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D

    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


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
    """L2 normalize per row, D <= 512 (``BT`` rows per program).

    Uses direct pointer arithmetic instead of ``tl.make_block_ptr`` because
    on small kernels the block_ptr codegen prologue adds non-trivial
    overhead. ``STORE_RSTD`` is a
    compile-time flag: when False the rstd store is DCE-eliminated and
    the caller may pass any tensor as the ``Rstd`` argument.

    ``BT`` and ``num_warps`` are fixed at the call site (defaults
    ``_L2NORM_FWD_BT=32`` / ``_L2NORM_FWD_NUM_WARPS=4``) rather than
    autotuned: the autotune dispatch + per-NB specialization overhead
    outweighs any tile-size gains for this small kernel.
    """
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


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=["D", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = (
        b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    )
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


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
        need_rstd: If ``True``, also allocate and return the per-row
            reciprocal-std tensor required by ``l2norm_bwd`` for the
            autograd backward path. If ``False`` (default), both the
            rstd allocation and the in-kernel rstd write are skipped.
            ``L2NormFunction.forward`` passes
            ``need_rstd=True`` so autograd users get the correct
            behavior automatically; pure forward inference call sites
            should keep the default.

    Returns:
        ``(y, rstd)`` where ``rstd`` is the per-row reciprocal std when
        ``need_rstd=True`` and ``None`` otherwise.

    .. note::
        The default value of ``need_rstd`` is ``False``. Code that
        previously consumed ``rstd`` directly via ``y, rstd =
        l2norm_fwd(x)`` must now opt in with ``need_rstd=True``, or use
        the higher-level ``l2norm`` / ``L2NormFunction.apply`` wrappers
        which already pass the right flag.
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
        # Placeholder pointer. ``STORE_RSTD=False`` makes the kernel
        # never dereference it, so reusing ``y`` avoids even a 1-elem
        # allocation. (Any in-bounds tensor would work; ``y`` is handy.)
        rstd = y

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


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    """
    Backward pass for L2 normalization.

    Args:
        y (torch.Tensor): Normalized output from forward pass.
        rstd (torch.Tensor): Reciprocal of standard deviation from forward pass.
        dy (torch.Tensor): Gradient w.r.t. output.
        eps (float): Small epsilon for numerical stability. Default: 1e-6.

    Returns:
        torch.Tensor: Gradient w.r.t. input, same shape as y.
    """
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    # allocate output
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_bwd_kernel[grid](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_bwd_kernel1[(T,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            D=D,
            BD=BD,
        )

    return dx.view(y_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        eps=1e-6,
        output_dtype=None,
    ):
        y, rstd = l2norm_fwd(x, eps, output_dtype, need_rstd=True)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy, ctx.eps)
        return dx, None, None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Apply L2 normalization to input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape [..., D].
        eps (float): Small epsilon for numerical stability. Default: 1e-6.
        output_dtype (torch.dtype, optional): Output dtype. If None, uses input dtype.

    Returns:
        torch.Tensor: L2 normalized tensor of same shape as x.
    """
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):
    """
    L2 Normalization module.

    Args:
        eps (float): Small epsilon for numerical stability. Default: 1e-6.
        output_dtype (torch.dtype, optional): Output dtype. If None, uses input dtype.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
