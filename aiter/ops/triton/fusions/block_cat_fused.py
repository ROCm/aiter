# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
cat([leaky_relu(l12[..., :W1], slope), l12[..., W1:] * l4], dim=-1) as one op
with a fused backward. AOTAutograd sees a single node, so the backward writes
grad_l12 and grad_l4 in one kernel instead of decomposing into per-half
pointwise passes that each round-trip the activation through HBM.
"""

import torch
import torch.nn.functional as F
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.fusions.block_cat_fused import (
    _block_cat_fused_bwd_kernel,
    _block_cat_fused_fwd_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

__all__ = ["block_cat_fused"]

_KERNEL_DTYPES = (torch.float16, torch.bfloat16)


def _check_inputs(l12: torch.Tensor, l4: torch.Tensor, l1_width: int):
    # raise, not assert: under python -O a short l4 would reach the kernel and
    # be read out of bounds
    if l4.device != l12.device:
        raise ValueError("l12 and l4 must be on the same device")
    if l4.dtype != l12.dtype:
        raise ValueError(f"dtype mismatch: {l12.dtype} vs {l4.dtype}")
    if not 0 <= l1_width <= l12.shape[-1]:
        raise ValueError(
            f"l1_width={l1_width} out of range for l12 width {l12.shape[-1]}"
        )
    expected = (*l12.shape[:-1], l12.shape[-1] - l1_width)
    if tuple(l4.shape) != expected:
        raise ValueError(f"l4 shape must be {expected}, got {tuple(l4.shape)}")


def _use_kernel(t: torch.Tensor) -> bool:
    return t.is_cuda and t.dtype in _KERNEL_DTYPES


def _launch(kernel, key, N, W12, l1_width, slope, *tensors):
    config = _get_config(key)
    BLOCK_SIZE_N = config.pop("BLOCK_SIZE_N")
    grid = (triton.cdiv(N, BLOCK_SIZE_N),)
    with torch.cuda.device(tensors[0].device):
        kernel[grid](
            *tensors,
            N,
            W1=l1_width,
            W12=W12,
            W4=W12 - l1_width,
            SLOPE=slope,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            **config,
        )


def _block_cat_fused_fwd_fake(
    l12: torch.Tensor, l4: torch.Tensor, slope: float, l1_width: int
) -> torch.Tensor:
    _check_inputs(l12, l4, l1_width)
    return torch.empty(l12.shape, dtype=l12.dtype, device=l12.device)


@torch_compile_guard(mutates_args=[], gen_fake=_block_cat_fused_fwd_fake)
def block_cat_fused_fwd(
    l12: torch.Tensor, l4: torch.Tensor, slope: float, l1_width: int
) -> torch.Tensor:
    _check_inputs(l12, l4, l1_width)
    _LOGGER.info(
        "BLOCK_CAT_FUSED_FWD: l12=%s l4=%s dtype=%s",
        tuple(l12.shape),
        tuple(l4.shape),
        l12.dtype,
    )
    if not _use_kernel(l12):
        out = torch.cat(
            [F.leaky_relu(l12[..., :l1_width], slope), l12[..., l1_width:] * l4], dim=-1
        )
        return out.contiguous()

    l12 = l12.contiguous()
    l4 = l4.contiguous()
    out = torch.empty(l12.shape, dtype=l12.dtype, device=l12.device)
    if out.numel() > 0:
        _launch(
            _block_cat_fused_fwd_kernel,
            "fwd",
            out.numel(),
            l12.shape[-1],
            l1_width,
            slope,
            l12,
            l4,
            out,
        )
    return out


def _block_cat_fused_bwd_fake(
    grad_out: torch.Tensor,
    l12: torch.Tensor,
    l4: torch.Tensor,
    slope: float,
    l1_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty(l12.shape, dtype=l12.dtype, device=l12.device),
        torch.empty(l4.shape, dtype=l4.dtype, device=l4.device),
    )


@torch_compile_guard(mutates_args=[], gen_fake=_block_cat_fused_bwd_fake)
def block_cat_fused_bwd(
    grad_out: torch.Tensor,
    l12: torch.Tensor,
    l4: torch.Tensor,
    slope: float,
    l1_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _LOGGER.info(
        "BLOCK_CAT_FUSED_BWD: l12=%s l4=%s dtype=%s",
        tuple(l12.shape),
        tuple(l4.shape),
        l12.dtype,
    )
    if not _use_kernel(l12):
        grad_l1 = grad_out[..., :l1_width]
        grad_l2 = grad_out[..., l1_width:]
        grad_l12 = torch.cat(
            [
                torch.where(l12[..., :l1_width] > 0, grad_l1, grad_l1 * slope),
                grad_l2 * l4,
            ],
            dim=-1,
        )
        return grad_l12.contiguous(), (grad_l2 * l12[..., l1_width:]).contiguous()

    # the kernel writes both grads flat, so they have to be contiguous no
    # matter what strides l12 / l4 came in with
    grad_l12 = torch.empty(l12.shape, dtype=l12.dtype, device=l12.device)
    grad_l4 = torch.empty(l4.shape, dtype=l4.dtype, device=l4.device)
    if grad_l12.numel() > 0:
        _launch(
            _block_cat_fused_bwd_kernel,
            "bwd",
            grad_l12.numel(),
            l12.shape[-1],
            l1_width,
            slope,
            grad_out.contiguous(),
            l12.contiguous(),
            l4.contiguous(),
            grad_l12,
            grad_l4,
        )
    return grad_l12, grad_l4


class _BlockCatFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, l12, l4, slope, l1_width):
        ctx.save_for_backward(l12, l4)
        ctx.slope = slope
        ctx.l1_width = l1_width
        return block_cat_fused_fwd(l12, l4, slope, l1_width)

    @staticmethod
    def backward(ctx, grad_out):
        l12, l4 = ctx.saved_tensors
        grad_l12, grad_l4 = block_cat_fused_bwd(
            grad_out, l12, l4, ctx.slope, ctx.l1_width
        )
        return grad_l12, grad_l4, None, None


def block_cat_fused(
    l12: torch.Tensor, l4: torch.Tensor, slope: float, l1_width: int
) -> torch.Tensor:
    """``cat([leaky_relu(l12[..., :W1], slope), l12[..., W1:] * l4], dim=-1)``.

    Args:
        l12: (..., W12) input. Its first ``l1_width`` columns go through
            leaky_relu, the remaining ``W4 = W12 - l1_width`` are multiplied by l4.
        l4: (..., W4) tensor with l12's leading dims, dtype and device.
        slope: leaky_relu negative slope.
        l1_width: W1, the width of the leaky_relu half.

    Returns:
        A new contiguous (..., W12) tensor in l12's dtype. Differentiable: the
        backward returns grad_l12 and grad_l4 from one fused kernel.

    FP16 / BF16 CUDA inputs run the Triton kernels (FP32 math, stored in the
    input dtype) with ``BLOCK_SIZE_N``, ``num_warps`` and ``num_stages`` read from
    the ``fwd`` / ``bwd`` entries of the arch's BLOCK_CAT_FUSED config. FP32 and
    CPU inputs take the eager torch path. Non-contiguous inputs are copied to
    contiguous before the launch.
    """
    return _BlockCatFused.apply(l12, l4, slope, l1_width)
