# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL backward kernels of the Kimi Delta Attention layer (launch wrappers).

Kernels are in ``kernels/kda_layer_bwd.py``. Each wrapper takes and returns fp32
tensors; inputs are made contiguous and flattened, which is what the compiled
handles are specialized for.
"""

import torch

from .kernels.kda_layer_bwd import (
    WAVE,
    create_gate_beta_bwd_kernel,
    create_l2norm_bwd_kernel,
)
from .kernels.tensor_shim import _run_compiled

ROW_WAVES = 4
GATE_BLOCK = 256


def _f32(t):
    return t.detach().to(torch.float32).contiguous().view(-1)


def _check_dim(dim, what):
    if dim % WAVE:
        raise ValueError(f"{what} must be a multiple of {WAVE}, got {dim}")


def l2norm_bwd(x, dy, eps=1e-6):
    """Backward of ``F.normalize(x, dim=-1, eps=eps)``. Returns dx, fp32."""
    dim = x.shape[-1]
    _check_dim(dim, "l2norm row length")
    rows = x.numel() // dim
    dx = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    launch = create_l2norm_bwd_kernel(dim, float(eps), ROW_WAVES, x.device.index)
    _run_compiled(launch, _f32(x), _f32(dy), dx.view(-1), rows, x.numel())
    return dx


def gate_beta_bwd(z, A_log, dg, braw, dbeta):
    """Backward of the F-decay gate and the beta gate.

    z (= f_b(f_a(x)) + dt_bias), dg: [B, T, H, K]; A_log: [H];
    braw (= x @ w_beta^T), dbeta: [B, T, H].
    Returns (dz [B, T, H, K], dA_log [H], db [B, T, H]), fp32. ``dz`` is also the
    gradient of the pre-bias projection; ``ddt_bias = dz.sum((0, 1))``.
    """
    H, K = z.shape[-2:]
    n_gate, n_beta = z.numel(), braw.numel()
    dz = torch.empty(z.shape, device=z.device, dtype=torch.float32)
    dgg = torch.empty_like(dz)
    db = torch.empty(braw.shape, device=z.device, dtype=torch.float32)
    launch = create_gate_beta_bwd_kernel(H, K, GATE_BLOCK, z.device.index)
    _run_compiled(
        launch,
        _f32(z),
        _f32(A_log),
        _f32(dg),
        _f32(braw),
        _f32(dbeta),
        dz.view(-1),
        dgg.view(-1),
        db.view(-1),
        n_gate,
        n_beta,
    )
    return dz, dgg.view(-1, H, K).sum((0, 2)), db
