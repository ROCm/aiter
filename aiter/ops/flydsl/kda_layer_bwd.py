# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels of the Kimi Delta Attention layer (launch wrappers).

``l2norm_bwd`` launches the row kernel in ``kernels/kda_layer_bwd.py`` on flat
contiguous fp32 tensors. ``kda_gate_proj``, ``kda_gate_proj_bwd`` and
``sigmoid_gate_proj`` are the gfx950 bf16 GEMM (``gemm_a16w16``) with a fused
epilogue; they return fp32.
"""

import torch

from .kernels.kda_layer_bwd import (
    WAVE,
    create_l2norm_bwd_kernel,
)
from .kernels.tensor_shim import _run_compiled

ROW_WAVES = 4


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


# The projections below use the gfx950 bf16 GEMM with an element-wise epilogue.
# fp32 output needs the 128x128 tile to fit the C shuffle in LDS.
_PROJ_GEMM_KWARGS = {"block_m": 128, "block_n": 128, "use_half_tile_interleaved": False}


def _proj_gemm(a, w, **kw):
    """a @ w^T with ``w`` an [N, K] weight."""
    return _gemm(a.contiguous(), w.t(), "nt", **kw)


def _gemm(a, b, layout, **kw):
    from .kernels.gemm_a16w16_gfx950 import gemm_a16w16

    return gemm_a16w16(
        a,
        b,
        user_kwargs=dict(_PROJ_GEMM_KWARGS),
        layout=layout,
        out_dtype=torch.float32,
        **kw,
    )


def _gate_params(f_b, dt_bias, A_log, num_heads, lower_bound):
    """fp32 [A_log[:H] | lower_bound | dt_bias] read by the KDA GEMM epilogues."""
    lb = torch.tensor(
        [0.0 if lower_bound is None else float(lower_bound)], device=f_b.device
    )
    return torch.cat([_f32(A_log)[:num_heads], lb, _f32(dt_bias)])


def kda_gate_proj(lr, f_b, dt_bias, A_log, num_heads, lower_bound=None):
    """Decay gate fused in the f_b GEMM, with z = lr @ f_b^T + dt_bias (fp32):
    ``-exp(A_log[h]) * softplus(z)``, or ``lower_bound * sigmoid(exp(A_log[h]) * z)``.

    lr: [M, R]; f_b: [H*K, R]; dt_bias: [H*K]; A_log: [>= H] (first H used).
    Returns fp32 [M, H*K].
    """
    from .kernels.gemm_a16w16_gfx950 import GEMM_EPI_KDA_GATE, GEMM_EPI_KDA_SAFE_GATE

    mode = GEMM_EPI_KDA_GATE if lower_bound is None else GEMM_EPI_KDA_SAFE_GATE
    return _proj_gemm(
        lr.to(f_b.dtype),
        f_b,
        epilogue=mode,
        epi_group=f_b.shape[0] // num_heads,
        a_log=_gate_params(f_b, dt_bias, A_log, num_heads, lower_bound),
    )


def kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, num_heads, dg, lower_bound=None):
    """Recompute z in the f_b GEMM and apply the gate backward.

    dg: [M, H*K]. Returns (dz, per-element dA_log terms), fp32 [M, H*K]; dA_log
    is the per-head sum of the second.
    """
    from .kernels.gemm_a16w16_gfx950 import (
        GEMM_EPI_KDA_GATE_BWD,
        GEMM_EPI_KDA_SAFE_GATE_BWD,
    )

    mode = GEMM_EPI_KDA_GATE_BWD if lower_bound is None else GEMM_EPI_KDA_SAFE_GATE_BWD
    dterm = torch.empty(dg.shape, device=dg.device, dtype=torch.float32)
    dz = _proj_gemm(
        lr.to(f_b.dtype),
        f_b,
        epilogue=mode,
        epi_group=f_b.shape[0] // num_heads,
        a_log=_gate_params(f_b, dt_bias, A_log, num_heads, lower_bound),
        aux=dg.float().contiguous(),
        out2=dterm,
    )
    return dz, dterm


def sigmoid_gate_proj(x, g_proj, aux=None):
    """sigmoid(x @ g_proj^T), times ``aux`` (fp32 [M, N]) when given."""
    from .kernels.gemm_a16w16_gfx950 import GEMM_EPI_SIGMOID, GEMM_EPI_SIGMOID_MUL

    if aux is None:
        return _proj_gemm(x.to(g_proj.dtype), g_proj, epilogue=GEMM_EPI_SIGMOID)
    return _proj_gemm(
        x.to(g_proj.dtype),
        g_proj,
        epilogue=GEMM_EPI_SIGMOID_MUL,
        aux=aux.float().contiguous(),
    )


def gate_proj_bwd_gemms(dgp, x, g_proj):
    """(dgp @ g_proj, dgp^T @ x) in bf16 with fp32 output.

    dgp: [M, H*V]; x: [M, D]; g_proj: [H*V, D]. Shapes the GEMM cannot take
    (a K tail, misaligned strides) fall back to torch for that product.
    """
    dt = g_proj.dtype
    d16 = dgp.to(dt).contiguous()
    x16 = x.to(dt).contiguous()
    try:
        dx = _gemm(d16, g_proj.contiguous(), "nn")
    except (AssertionError, ValueError):
        dx = dgp.float() @ g_proj.float()
    try:
        dw = _gemm(d16.t(), x16, "tn")
    except (AssertionError, ValueError):
        dw = dgp.float().t() @ x.float()
    return dx, dw
