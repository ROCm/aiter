# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

_REPR_KEYS = ["W1", "W12", "W4", "SLOPE", "BLOCK_SIZE_N"]

_block_cat_fused_fwd_repr = make_kernel_repr("_block_cat_fused_fwd_kernel", _REPR_KEYS)

_block_cat_fused_bwd_repr = make_kernel_repr("_block_cat_fused_bwd_kernel", _REPR_KEYS)


def _get_config(key: str) -> dict:
    """``fwd`` or ``bwd`` launch config."""
    config_dir = resolve_config_dir("fusions", "BLOCK_CAT_FUSED")
    return dict(load_config_json(f"{config_dir}/DEFAULT.json", required=True)[key])


# Both kernels keep the bounds mask even when N divides the block: a mask-free
# variant measured up to 7% slower on gfx950 at the M=960k, W12=576 shapes.


@triton.jit(repr=_block_cat_fused_fwd_repr)
def _block_cat_fused_fwd_kernel(
    l12_ptr,
    l4_ptr,
    out_ptr,
    N,
    W1: tl.constexpr,
    W12: tl.constexpr,
    W4: tl.constexpr,
    SLOPE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """out[m, n] = leaky_relu(l12[m, n]) for n < W1, else l12[m, n] * l4[m, n - W1].

    l12/out are (N // W12, W12) and l4 is (N // W12, W4), all contiguous and
    walked as one flat range of N elements.
    """
    # int64 so N >= 2^31 does not wrap the offsets
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < N
    rows = offs // W12
    cols = offs % W12
    is_l1 = cols < W1
    l4_offs = rows * W4 + cols - W1
    l4_mask = mask & (~is_l1)

    l12 = tl.load(l12_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    l4 = tl.load(l4_ptr + l4_offs, mask=l4_mask, other=0.0).to(tl.float32)
    out = tl.where(is_l1, tl.where(l12 > 0.0, l12, l12 * SLOPE), l12 * l4)
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit(repr=_block_cat_fused_bwd_repr)
def _block_cat_fused_bwd_kernel(
    grad_out_ptr,
    l12_ptr,
    l4_ptr,
    grad_l12_ptr,
    grad_l4_ptr,
    N,
    W1: tl.constexpr,
    W12: tl.constexpr,
    W4: tl.constexpr,
    SLOPE: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """grad_l12 and grad_l4 of the forward above in one pass over grad_out."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = offs < N
    rows = offs // W12
    cols = offs % W12
    is_l1 = cols < W1
    l4_offs = rows * W4 + cols - W1
    l4_mask = mask & (~is_l1)

    grad_out = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    l12 = tl.load(l12_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    l4 = tl.load(l4_ptr + l4_offs, mask=l4_mask, other=0.0).to(tl.float32)
    grad_l12 = tl.where(
        is_l1, tl.where(l12 > 0.0, grad_out, grad_out * SLOPE), grad_out * l4
    )
    tl.store(grad_l12_ptr + offs, grad_l12.to(grad_l12_ptr.dtype.element_ty), mask=mask)
    tl.store(
        grad_l4_ptr + l4_offs,
        (grad_out * l12).to(grad_l4_ptr.dtype.element_ty),
        mask=l4_mask,
    )
