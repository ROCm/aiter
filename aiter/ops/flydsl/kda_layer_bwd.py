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
# Every KDA GEMM uses the full-tile kernel: never the half-tile-interleaved
# (HTI) one, whose tn path is under investigation (#8754).
_PROJ_GEMM_KWARGS = {"block_m": 128, "block_n": 128, "use_half_tile_interleaved": False}


def _proj_gemm(a, w, **kw):
    """a @ w^T with ``w`` an [N, K] weight."""
    return _gemm(a.contiguous(), w.t(), "nt", **kw)


def _gemm(a, b, layout, out=None, **kw):
    from .kernels.gemm_a16w16_gfx950 import gemm_a16w16

    kwargs = dict(_PROJ_GEMM_KWARGS)
    assert not kwargs["use_half_tile_interleaved"]
    return gemm_a16w16(
        a,
        b,
        out=out,
        user_kwargs=kwargs,
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


def _split_lowrank(lr, f_b):
    """[hi | lo] @ [f_b | f_b]^T with lr = hi + lo in 16 bits: the gate's
    pre-activation keeps ~fp32 accuracy (the safe gate's derivative amplifies
    an input rounding by exp(A_log)), for a K of 2R = 256 instead of 128."""
    dt = f_b.dtype
    lr = lr.float()
    hi = lr.to(dt)
    lo = (lr - hi.float()).to(dt)
    return torch.cat([hi, lo], -1), torch.cat([f_b, f_b], -1)


def kda_gate_proj(lr, f_b, dt_bias, A_log, num_heads, lower_bound=None):
    """Decay gate fused in the f_b GEMM, with z = lr @ f_b^T + dt_bias (fp32):
    ``-exp(A_log[h]) * softplus(z)``, or ``lower_bound * sigmoid(exp(A_log[h]) * z)``.

    lr: [M, R]; f_b: [H*K, R]; dt_bias: [H*K]; A_log: [>= H] (first H used).
    Returns fp32 [M, H*K].
    """
    from .kernels.gemm_a16w16_gfx950 import GEMM_EPI_KDA_GATE, GEMM_EPI_KDA_SAFE_GATE

    mode = GEMM_EPI_KDA_GATE if lower_bound is None else GEMM_EPI_KDA_SAFE_GATE
    lr2, f_b2 = _split_lowrank(lr, f_b)
    return _proj_gemm(
        lr2,
        f_b2,
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
    lr2, f_b2 = _split_lowrank(lr, f_b)
    dz = _proj_gemm(
        lr2,
        f_b2,
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


def proj_gemm(a, w, kind, out=None):
    """Plain projection GEMM on the gfx950 bf16 GEMM (fp32 output), with the
    ``_proj_gemm`` operand convention; a shape the GEMM rejects (a K tail,
    misaligned strides) falls back to torch on the same rounded inputs.
    ``out`` (contiguous fp32) receives the result on either path."""
    dt = w.dtype
    a16 = a.to(dt).contiguous()
    w16 = w.to(dt).contiguous()
    try:
        if kind == "fwd":
            return _gemm(a16, w16.t(), "nt", out=out)
        if kind == "dgrad":
            return _gemm(a16, w16, "nn", out=out)
        return _gemm(a16.t(), w16, "tn", out=out)
    except (AssertionError, ValueError):
        a32, w32 = a16.float(), w16.float()
        res = {"fwd": lambda: a32 @ w32.t(), "dgrad": lambda: a32 @ w32}.get(
            kind, lambda: a32.t() @ w32
        )()
        return res if out is None else out.copy_(res)


def gate_proj_bwd_gemms(dgp, x, g_proj):
    """(dgp @ g_proj, dgp^T @ x) in bf16 with fp32 output.

    dgp: [M, H*V]; x: [M, D]; g_proj: [H*V, D]. Shapes the GEMM cannot take
    (a K tail, misaligned strides) fall back to torch for that product.
    """
    # the torch fallback takes the same bf16-rounded dgp/x as the GEMM
    return proj_gemm(dgp, g_proj, "dgrad"), proj_gemm(dgp, x, "wgrad")
