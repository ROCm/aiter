# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test-only PyTorch oracle for the Kimi Delta Attention (KDA) recurrence.

Judges ``flydsl_gdr_decode`` from ``op_tests/test_flydsl_gdr_decode.py`` and the
KDA decode benchmark. It lives here rather than under ``aiter/`` on purpose: it
is not an op anyone should call, and nothing on the inference path may reach it.

Ported from ``naive_recurrent_kda`` in vLLM (``tests/models/kimi_k3/test_kda.py``),
itself from FLA's ``naive.py``. Staying close to that source is what makes the
oracle trustworthy, so keep it obvious by inspection -- f32 throughout, one
timestep per iteration, no fusion, no shape dispatch.

Nothing already in aiter could serve: the Triton references' ``IS_KDA`` branches
change the pointer layout but still compute the scalar gate.
"""

from __future__ import annotations

import torch

# Matches the FlyDSL kernel's in-kernel l2norm and the Triton reference.
_L2NORM_EPS = 1e-6


def l2norm(x: torch.Tensor, eps: float = _L2NORM_EPS) -> torch.Tensor:
    """Normalize over the last axis the way the kernel does it internally."""
    x = x.float()
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_gate(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    g_min: float,
) -> torch.Tensor:
    """KDA's per-channel decay, in log space.

    Args:
        a: ``(B, T, H, K)`` raw gate input.
        A_log: ``(H,)``, per head, so it broadcasts across channels.
        dt_bias: ``(H, K)``, per channel.
        g_min: KDA's lower bound on the decay, -5 in K3.

    Returns:
        Decay in log space, shaped like ``a`` -- callers apply ``exp``.
    """
    A = A_log.float().exp()[:, None]
    return g_min * torch.sigmoid(A * (a.float() + dt_bias.float()))


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Delta-rule recurrence in f32.

    Args:
        q, k, v: ``(B, T, H, K)`` / ``(B, T, H, V)``. Apply :func:`l2norm` first
            if the kernel under test normalizes internally -- this does not,
            matching the vLLM reference.
        g: log-space decay, ``(B, T, H, K)``.
        beta: ``(B, T, H)``, already through sigmoid.
        initial_state: ``(B, H, K, V)``.

    Returns:
        Output in ``v``'s dtype, and the final state when requested.
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * scale

    S = k.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.float()

    o = torch.zeros(B, T, H, V, dtype=torch.float32, device=v.device)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)

    return o.to(dtype), S if output_final_state else None
