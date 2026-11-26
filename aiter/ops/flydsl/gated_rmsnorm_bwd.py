# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated RMSNorm backward (KDA ``o_norm``): launch wrapper and torch reference."""

import torch

from .kernels.gated_rmsnorm_bwd import (
    EPL,
    create_gated_rmsnorm_bwd_kernel,
    supported_dim,
)
from .kernels.tensor_shim import _run_compiled

BLOCK = 256
PASSES = 4
# large inputs get 8 passes per block while the grid still fills the GPU
PASSES_LARGE = 8
LARGE_ROWS = 131072
_IN_DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


def gated_rmsnorm_bwd(o, gp, w, dy, eps=1e-6):
    """Backward of ``rmsnorm(o) * w * sigmoid(gp)`` over the last dim.

    o, gp, dy: [..., V], read in their own dtype when all three share bf16, fp16
    or fp32 (anything else goes through fp32); w: [V]. V / 8 must be a power of
    two up to 64 (V in 8..512). gp is the gate pre-activation.
    Returns (do, dgp, dw), fp32.
    """
    dim = o.shape[-1]
    if not supported_dim(dim):
        raise ValueError(f"gated rmsnorm row length must be 8 * 2^k <= 512, got {dim}")
    in_dtype = _IN_DTYPES.get(o.dtype)
    if in_dtype is None or gp.dtype != o.dtype or dy.dtype != o.dtype:
        in_dtype = "f32"
    cast = torch.float32 if in_dtype == "f32" else o.dtype
    o_, gp_, dy_ = (t.detach().to(cast).contiguous().view(-1) for t in (o, gp, dy))
    numel = o_.numel()
    if numel * 4 >= 2**31:
        raise ValueError(f"too many elements for 32-bit buffer offsets: {numel}")
    rows = numel // dim
    passes = PASSES_LARGE if rows >= LARGE_ROWS else PASSES
    rows_per_block = BLOCK // (dim // EPL) * passes
    n_blocks = max(1, (rows + rows_per_block - 1) // rows_per_block)
    do = torch.empty(o.shape, device=o.device, dtype=torch.float32)
    dgp = torch.empty_like(do)
    dw_part = torch.empty(n_blocks, dim, device=o.device, dtype=torch.float32)
    launch = create_gated_rmsnorm_bwd_kernel(
        dim, float(eps), in_dtype, BLOCK, passes, o.device.index
    )
    _run_compiled(
        launch,
        o_,
        gp_,
        w.detach().to(cast).contiguous(),
        dy_,
        do.view(-1),
        dgp.view(-1),
        dw_part.view(-1),
        rows,
        n_blocks,
        numel,
    )
    return do, dgp, dw_part.sum(0)


def gated_rmsnorm_bwd_torch(o, gp, w, dy, eps=1e-6):
    """Autograd reference, same contract as ``gated_rmsnorm_bwd``."""
    with torch.enable_grad():
        o, gp, w = (t.detach().float().requires_grad_(True) for t in (o, gp, w))
        out = (
            o
            * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
            * w
            * torch.sigmoid(gp)
        )
        return torch.autograd.grad(out, [o, gp, w], dy.float())
