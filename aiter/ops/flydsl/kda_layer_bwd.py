# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL kernels of the Kimi Delta Attention layer (launch wrappers).

``l2norm_bwd`` launches the row kernel in ``kernels/kda_layer_bwd.py`` on flat
contiguous fp32 tensors. ``kda_gate_proj``, ``kda_gate_proj_bwd`` and
``sigmoid_gate_proj`` are the gfx950 bf16 GEMM (``gemm_a16w16``) with a fused
epilogue, and ``gate_proj_bwd_gemms`` the same GEMM without one. Their tiles
come from the tuned ``flydsl_gemm_a16w16_epilogue`` table
(``gemm_epilogue_configs``); ``config=`` overrides it.
"""

import torch

from .kernels.kda_layer_bwd import (
    WAVE,
    create_l2norm_bwd_kernel,
)
from .kernels.tensor_shim import _run_compiled

ROW_WAVES = 4
# Head width (H*K // H) of the KDA epilogues the tuned tables validate against
KDA_HEAD_DIM = 128


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


def _validate(m, n, k, layout, extra):
    from .kernels.gemm_a16w16_gfx950 import (
        GEMM_A16W16_DTYPE_BF16,
        make_gemm_a16w16_param_and_validate,
    )

    def check(kw):
        full = dict(kw)
        full.update(extra)
        full.update(
            a_is_transposed=layout[0] == "t",
            b_is_transposed=layout[1] == "t",
            in_dtype_id=GEMM_A16W16_DTYPE_BF16,
            has_bias=False,
        )
        if k % full["block_k"] and not full["use_half_tile_interleaved"]:
            full["k_tail"] = True
        return make_gemm_a16w16_param_and_validate(m, n, k, full) is not None

    return check


def proj_config(m, n, k, layout, epilogue, out_dtype, dtype=torch.bfloat16):
    """(user_kwargs, "tuned" | "default") of one projection GEMM."""
    from .gemm_epilogue_configs import select_config
    from .kernels.gemm_a16w16_gfx950 import (
        GEMM_A16W16_DTYPE_BF16,
        GEMM_A16W16_DTYPE_FP32,
        GEMM_EPI_KDA_MODES,
    )

    extra = {
        "out_dtype_id": (
            GEMM_A16W16_DTYPE_FP32
            if out_dtype == torch.float32
            else GEMM_A16W16_DTYPE_BF16
        ),
        "epilogue": epilogue,
        "epi_group": KDA_HEAD_DIM if epilogue in GEMM_EPI_KDA_MODES else 1,
        # the gate GEMMs read f_b once per half of the hi/lo split lr
        "b_k_repeat": 2 if epilogue in GEMM_EPI_KDA_MODES else 1,
    }
    return select_config(
        m, n, k, layout, epilogue, dtype, out_dtype, _validate(m, n, k, layout, extra)
    )


def _gemm(a, b, layout, out_dtype=torch.float32, config=None, epilogue=0, **kw):
    """a @ b on the gfx950 GEMM, tiles from ``config`` or the tuned table."""
    from .kernels.gemm_a16w16_gfx950 import gemm_a16w16

    m, k = a.shape
    n = b.shape[1]
    if config is None:
        config, _ = proj_config(m, n, k, layout, epilogue, out_dtype, a.dtype)
    return gemm_a16w16(
        a,
        b,
        user_kwargs=dict(config),
        layout=layout,
        out_dtype=out_dtype,
        epilogue=epilogue,
        **kw,
    )


def _proj_gemm(a, w, **kw):
    """a @ w^T with ``w`` an [N, K] weight."""
    return _gemm(a.contiguous(), w.t(), "nt", **kw)


def _gate_kwargs(f_b, dt_bias, A_log, num_heads, lower_bound):
    return {
        "epi_group": f_b.shape[0] // num_heads,
        "a_log": _f32(A_log),
        "dt_bias": _f32(dt_bias),
        "lower_bound": lower_bound,
    }


def _split_lowrank(lr, f_b):
    """[hi | lo] with lr = hi + lo in 16 bits, for [hi | lo] @ [f_b | f_b]^T: the
    gate's pre-activation keeps ~fp32 accuracy (the safe gate's derivative
    amplifies an input rounding by exp(A_log)), for a K of 2R instead of R. The
    GEMM reads f_b once per half (``b_k_repeat=2``); it is never duplicated."""
    dt = f_b.dtype
    lr = lr.float()
    hi = lr.to(dt)
    lo = (lr - hi.float()).to(dt)
    return torch.cat([hi, lo], -1)


def kda_gate_proj(lr, f_b, dt_bias, A_log, num_heads, lower_bound=None, config=None):
    """Decay gate fused in the f_b GEMM, with z = lr @ f_b^T + dt_bias (fp32):
    ``-exp(A_log[h]) * softplus(z)``, or ``lower_bound * sigmoid(exp(A_log[h]) * z)``.

    lr: [M, R]; f_b: [H*K, R]; dt_bias: [H*K]; A_log: [>= H] (first H used).
    Returns fp32 [M, H*K].
    """
    from .kernels.gemm_a16w16_gfx950 import GEMM_EPI_KDA_GATE, GEMM_EPI_KDA_SAFE_GATE

    mode = GEMM_EPI_KDA_GATE if lower_bound is None else GEMM_EPI_KDA_SAFE_GATE
    return _gemm(
        _split_lowrank(lr, f_b),
        f_b.t(),
        "nt",
        epilogue=mode,
        config=config,
        b_k_repeat=2,
        **_gate_kwargs(f_b, dt_bias, A_log, num_heads, lower_bound),
    )


def kda_gate_proj_bwd(
    lr, f_b, dt_bias, A_log, num_heads, dg, lower_bound=None, config=None
):
    """Recompute z in the f_b GEMM and apply the gate backward.

    dg: [M, H*K]. Returns (dz fp32 [M, H*K], dA_log fp32 [H], d dt_bias fp32
    [H*K]); the kernel sums dA_log's terms and dz per M tile.
    """
    from .kernels.gemm_a16w16_gfx950 import (
        GEMM_EPI_KDA_GATE_BWD,
        GEMM_EPI_KDA_SAFE_GATE_BWD,
    )

    mode = GEMM_EPI_KDA_GATE_BWD if lower_bound is None else GEMM_EPI_KDA_SAFE_GATE_BWD
    a = _split_lowrank(lr, f_b)
    m, n = a.shape[0], f_b.shape[0]
    if config is None:
        config, _ = proj_config(m, n, a.shape[1], "nt", mode, torch.float32, a.dtype)
    rows = (m + config["block_m"] - 1) // config["block_m"]
    colsum = torch.empty((rows, 2, n), device=dg.device, dtype=torch.float32)
    dz = _gemm(
        a,
        f_b.t(),
        "nt",
        epilogue=mode,
        config=config,
        aux=dg.float().contiguous(),
        colsum=colsum,
        b_k_repeat=2,
        **_gate_kwargs(f_b, dt_bias, A_log, num_heads, lower_bound),
    )
    sums = colsum.sum(0)
    return dz, sums[0].view(num_heads, -1).sum(1), sums[1]


def sigmoid_gate_proj(x, g_proj, aux=None, out_dtype=None, config=None):
    """sigmoid(x @ g_proj^T), times ``aux`` (fp32 [M, N]) when given.

    The epilogue runs on the fp32 accumulator; ``out_dtype`` (default: the
    weight dtype, which the output projection reads) only sets the store.
    """
    from .kernels.gemm_a16w16_gfx950 import GEMM_EPI_SIGMOID, GEMM_EPI_SIGMOID_MUL

    out_dtype = g_proj.dtype if out_dtype is None else out_dtype
    kw = {"epilogue": GEMM_EPI_SIGMOID}
    if aux is not None:
        kw = {"epilogue": GEMM_EPI_SIGMOID_MUL, "aux": aux.float().contiguous()}
    return _proj_gemm(
        x.to(g_proj.dtype), g_proj, out_dtype=out_dtype, config=config, **kw
    )


def proj_gemm(a, w, kind):
    """Plain projection GEMM on the gfx950 bf16 GEMM (fp32 output), with the
    ``_proj_gemm`` operand convention; a shape the GEMM rejects (misaligned
    strides) falls back to torch on the same rounded inputs."""
    dt = w.dtype
    a16 = a.to(dt).contiguous()
    w16 = w.to(dt).contiguous()
    try:
        if kind == "fwd":
            return _gemm(a16, w16.t(), "nt")
        if kind == "dgrad":
            return _gemm(a16, w16, "nn")
        return _gemm(a16.t(), w16, "tn")
    except (AssertionError, ValueError):
        a32, w32 = a16.float(), w16.float()
        return {"fwd": lambda: a32 @ w32.t(), "dgrad": lambda: a32 @ w32}.get(
            kind, lambda: a32.t() @ w32
        )()


def gate_proj_bwd_gemms(dgp, x, g_proj, dw_dtype=torch.float32):
    """(dgp @ g_proj fp32, dgp^T @ x in ``dw_dtype``), bf16 inputs.

    dgp: [M, H*V]; x: [M, D]; g_proj: [H*V, D]. Any token count M: the weight
    gradient takes its K tail in the kernel.
    """
    dt = g_proj.dtype
    d16 = dgp.to(dt).contiguous()
    x16 = x.to(dt).contiguous()
    dx = _gemm(d16, g_proj.contiguous(), "nn")
    dw = _gemm(d16.t(), x16, "tn", out_dtype=dw_dtype)
    return dx, dw
