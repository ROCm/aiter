# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""torch.ops.aiter.block_cat_fused custom op.

Fuses cat([leaky_relu(l12[..., :W1], slope), l12[..., W1:] * l4], dim=-1)
into a single op so AOTAutograd dispatches the registered fused
backward (one Triton kernel for both grad_l12 and grad_l4) instead of
the per-half pointwise grad decomposition. Triton kernels handle the
bf16 / fp16 GPU path; eager torch is the CPU / fp32 fallback.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.library import custom_op

_TRITON_KERNEL = None
_TRITON_FW_KERNEL = None


def _build_triton_fw_kernel():
    """Forward kernel: cat_out[i, j] =
    leaky_relu(l12[i, j], slope)        if j <  W1
    l12[i, j] * l4[i, j-W1]             if j >= W1
    """
    global _TRITON_FW_KERNEL
    if _TRITON_FW_KERNEL is not None:
        return _TRITON_FW_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def _block_cat_fused_fw(
        l12_ptr,
        l4_ptr,
        cat_out_ptr,
        xnumel,
        W1: tl.constexpr,
        W12: tl.constexpr,
        W4: tl.constexpr,
        SLOPE: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        xoffset = pid * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        row = xindex // W12
        col = xindex % W12
        is_l1 = col < W1
        col_l4 = col - W1
        l4_idx = row * W4 + col_l4
        l12 = tl.load(l12_ptr + xindex, mask=xmask, other=0.0).to(tl.float32)
        l4 = tl.load(l4_ptr + l4_idx, mask=xmask & (~is_l1), other=0.0).to(tl.float32)
        leaky_v = tl.where(l12 > 0.0, l12, l12 * SLOPE)
        mul_v = l12 * l4
        out = tl.where(is_l1, leaky_v, mul_v)
        tl.store(cat_out_ptr + xindex, out.to(tl.bfloat16), mask=xmask)

    _TRITON_FW_KERNEL = _block_cat_fused_fw
    return _block_cat_fused_fw


def _launch_fw(l12, l4, cat_out, W1, slope):
    import triton

    M, W12 = l12.shape
    W4 = l4.shape[-1]
    assert W12 - W1 == W4, f"W12={W12} W1={W1} W4={W4}"
    xnumel = M * W12
    kernel = _build_triton_fw_kernel()
    # Forward kernel config tuned on MI355X for the workload's per-Block
    # shapes (M ~ 960k, W12 ~ 576). XBLOCK=2048 / warps=4 / stages=1 is
    # roughly 40 percent faster than the previous default
    # (XBLOCK=1024 / warps=4 / stages=2) on those shapes.
    XBLOCK = 2048
    grid = (triton.cdiv(xnumel, XBLOCK),)
    kernel[grid](
        l12,
        l4,
        cat_out,
        xnumel,
        W1,
        W12,
        W4,
        slope,
        XBLOCK,
        num_warps=4,
        num_stages=1,
    )


def _build_triton_kernel():
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def _block_cat_fused_bw(
        grad_cat_ptr,
        l12_out_ptr,
        l4_out_ptr,
        grad_l12_out_ptr,
        grad_l4_out_ptr,
        xnumel,
        W1: tl.constexpr,
        W12: tl.constexpr,
        W4: tl.constexpr,
        SLOPE: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        xoffset = pid * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        row = xindex // W12
        col = xindex % W12
        is_l1 = col < W1
        is_l2 = ~is_l1
        col_l4 = col - W1
        l4_idx = row * W4 + col_l4
        gc = tl.load(grad_cat_ptr + xindex, mask=xmask, other=0.0).to(tl.float32)
        l12 = tl.load(l12_out_ptr + xindex, mask=xmask, other=0.0).to(tl.float32)
        l4 = tl.load(l4_out_ptr + l4_idx, mask=xmask & is_l2, other=0.0).to(tl.float32)
        grad_l1_pre = tl.where(l12 > 0.0, gc, gc * SLOPE)
        grad_l2_pre = gc * l4
        grad_l4_local = gc * l12
        grad_l12 = tl.where(is_l1, grad_l1_pre, grad_l2_pre)
        tl.store(grad_l12_out_ptr + xindex, grad_l12.to(tl.bfloat16), mask=xmask)
        tl.store(
            grad_l4_out_ptr + l4_idx, grad_l4_local.to(tl.bfloat16), mask=xmask & is_l2
        )

    _TRITON_KERNEL = _block_cat_fused_bw
    return _block_cat_fused_bw


def _launch_bw(grad_cat, l12, l4, grad_l12, grad_l4, W1, slope):
    import triton

    M, W12 = l12.shape
    W4 = l4.shape[-1]
    assert W12 - W1 == W4, f"W12={W12} W1={W1} W4={W4}"
    xnumel = M * W12
    kernel = _build_triton_kernel()
    # Backward kernel config tuned on MI355X. XBLOCK=512 with
    # warps=2 / stages=2 hits the bandwidth ceiling for the workload's
    # per-Block shapes; the l4 reads have spatial locality across rows
    # which the L2 picks up.
    XBLOCK = 512
    grid = (triton.cdiv(xnumel, XBLOCK),)
    kernel[grid](
        grad_cat,
        l12,
        l4,
        grad_l12,
        grad_l4,
        xnumel,
        W1,
        W12,
        W4,
        slope,
        XBLOCK,
        num_warps=2,
        num_stages=2,
    )


def _reference_bw(grad_cat, l12, l4, W1, slope):
    grad_l1_part = grad_cat[..., :W1]
    grad_l2l4_part = grad_cat[..., W1:]
    l1_pre = l12[..., :W1]
    l2_pre = l12[..., W1:]
    grad_l1_pre = torch.where(l1_pre > 0, grad_l1_part, grad_l1_part * slope)
    grad_l2_pre = grad_l2l4_part * l4
    grad_l4_out = grad_l2l4_part * l2_pre
    grad_l12_out = torch.cat([grad_l1_pre, grad_l2_pre], dim=-1)
    return grad_l12_out, grad_l4_out


NAMESPACE = "aiter"
OP_NAME = "block_cat_fused"
QUALIFIED = f"{NAMESPACE}::{OP_NAME}"
BW_QUALIFIED = f"{NAMESPACE}::block_cat_fused_bw"


@custom_op(QUALIFIED, mutates_args=())
def block_cat_fused(
    l12: torch.Tensor,
    l4: torch.Tensor,
    slope: float,
    l1_width: int,
) -> torch.Tensor:
    """Forward: cat([leaky_relu(l12[..., :W1], slope), l12[..., W1:] * l4], dim=-1).

    Implementation: single fused Triton kernel on GPU/bf16/fp16 (same HBM
    traffic as Inductor's auto-fused eager pattern, but visible to
    AOTAutograd as a single op so the registered backward fuses
    grad_l12 + grad_l4 production into one kernel as well). Eager torch
    fallback for CPU / fp32 / non-CUDA.
    """
    if l12.is_cuda and l12.dtype in (torch.bfloat16, torch.float16):
        l12_c = l12.contiguous()
        l4_c = l4.contiguous()
        cat_out = torch.empty_like(l12_c)
        _launch_fw(l12_c, l4_c, cat_out, l1_width, slope)
        return cat_out
    l1 = F.leaky_relu(l12[..., :l1_width], slope)
    l2l4 = l12[..., l1_width:] * l4
    return torch.cat([l1, l2l4], dim=-1)


@block_cat_fused.register_fake
def _block_cat_fused_fake(
    l12: torch.Tensor,
    l4: torch.Tensor,
    slope: float,
    l1_width: int,
) -> torch.Tensor:
    return torch.empty_like(l12)


def _setup_context(ctx, inputs, output):
    l12, l4, slope, l1_width = inputs
    ctx.save_for_backward(l12, l4)
    ctx.slope = slope
    ctx.l1_width = l1_width


@custom_op(BW_QUALIFIED, mutates_args=())
def block_cat_fused_bw(
    grad_cat: torch.Tensor,
    l12: torch.Tensor,
    l4: torch.Tensor,
    slope: float,
    l1_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward kernel for block_cat_fused. Returns (grad_l12, grad_l4)."""
    if grad_cat.is_cuda and grad_cat.dtype in (torch.bfloat16, torch.float16):
        grad_l12 = torch.empty_like(l12)
        grad_l4 = torch.empty_like(l4)
        _launch_bw(
            grad_cat.contiguous(),
            l12.contiguous(),
            l4.contiguous(),
            grad_l12,
            grad_l4,
            l1_width,
            slope,
        )
        return grad_l12, grad_l4
    return _reference_bw(grad_cat, l12, l4, l1_width, slope)


@block_cat_fused_bw.register_fake
def _block_cat_fused_bw_fake(grad_cat, l12, l4, slope, l1_width):
    return torch.empty_like(l12), torch.empty_like(l4)


def _backward(ctx, grad_cat):
    l12, l4 = ctx.saved_tensors
    slope = ctx.slope
    W1 = ctx.l1_width
    grad_l12, grad_l4 = torch.ops.aiter.block_cat_fused_bw(grad_cat, l12, l4, slope, W1)
    return grad_l12, grad_l4, None, None


block_cat_fused.register_autograd(_backward, setup_context=_setup_context)


__all__ = [
    "block_cat_fused",
    "QUALIFIED",
    "NAMESPACE",
    "OP_NAME",
    "_launch_fw",
    "_launch_bw",
    "_reference_bw",
    "_build_triton_kernel",
    "_build_triton_fw_kernel",
]
