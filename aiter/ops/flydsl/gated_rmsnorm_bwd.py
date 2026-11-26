# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated RMSNorm backward (KDA ``o_norm``): launch wrapper and torch reference."""

import torch

from .kernels.gated_rmsnorm_bwd import WAVE, create_gated_rmsnorm_bwd_kernel
from .kernels.tensor_shim import _run_compiled

ROW_WAVES = 4
RMSNORM_ROWS_PER_WAVE = 16


def _f32(t):
    return t.detach().to(torch.float32).contiguous().view(-1)


def gated_rmsnorm_bwd(o, gp, w, dy, eps=1e-6):
    """Backward of ``rmsnorm(o) * w * sigmoid(gp)`` over the last dim.

    o, gp, dy: [..., V] (any float dtype, read as fp32); w: [V]; V % 64 == 0.
    gp is the gate pre-activation. Returns (do, dgp, dw), fp32.
    """
    dim = o.shape[-1]
    if dim % WAVE:
        raise ValueError(
            f"gated rmsnorm row length must be a multiple of {WAVE}, got {dim}"
        )
    rows = o.numel() // dim
    n_groups = (rows + RMSNORM_ROWS_PER_WAVE - 1) // RMSNORM_ROWS_PER_WAVE
    do = torch.empty(o.shape, device=o.device, dtype=torch.float32)
    dgp = torch.empty_like(do)
    dw_part = torch.empty(n_groups, dim, device=o.device, dtype=torch.float32)
    launch = create_gated_rmsnorm_bwd_kernel(
        dim, float(eps), ROW_WAVES, RMSNORM_ROWS_PER_WAVE, o.device.index
    )
    _run_compiled(
        launch,
        _f32(o),
        _f32(gp),
        _f32(w),
        _f32(dy),
        do.view(-1),
        dgp.view(-1),
        dw_part.view(-1),
        rows,
        n_groups,
        o.numel(),
        dw_part.numel(),
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
