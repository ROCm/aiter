# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-PyTorch reference for the Kimi Delta Attention (KDA) recurrence.

Oracle for the per-channel decay work on ``flydsl_gdr_decode``. Nothing in aiter
could serve: the Triton references' ``IS_KDA`` branches change the pointer layout
but still compute the scalar gate.

Ported from ``naive_recurrent_kda`` in vLLM (``tests/models/kimi_k3/test_kda.py``),
itself from FLA's ``naive.py``.
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
    g_min: float | None = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """Log-space decay from the raw gate input.

    Args:
        a: ``(B, T, H, K)`` per-channel, or ``(B, T, H)`` scalar per head.
        A_log: ``(H,)``, always f32, broadcast across channels.
        dt_bias: ``(H, K)`` per-channel, or ``(H,)`` scalar per head.
        g_min: KDA's lower bound, -5. None selects the GDR softplus gate.

    Returns:
        Decay in log space, shaped like ``a`` -- callers apply ``exp``.
    """
    if a.dim() not in (3, 4):
        raise ValueError(
            f"`a` must be (B, T, H) or (B, T, H, K); got {tuple(a.shape)}."
        )
    if A_log.dim() != 1:
        raise ValueError(f"`A_log` must be (H,); got {tuple(A_log.shape)}.")

    # Rank picks the mode: shape alone is ambiguous once H == K.
    per_channel = a.dim() == 4
    num_heads = A_log.shape[0]
    heads = a.shape[-2] if per_channel else a.shape[-1]
    if heads != num_heads:
        raise ValueError(
            f"`a`'s head axis must be {num_heads} to match `A_log`; "
            f"got {tuple(a.shape)}."
        )
    bias_shape = (num_heads, a.shape[-1]) if per_channel else (num_heads,)
    if dt_bias.shape != bias_shape:
        raise ValueError(
            f"`dt_bias` must have shape {bias_shape} for a "
            f"{'per-channel' if per_channel else 'scalar'} gate; "
            f"got {tuple(dt_bias.shape)}."
        )

    x = a.float() + dt_bias.float()
    A = A_log.float().exp()
    if per_channel:
        A = A[:, None]

    if g_min is None:
        beta_x = softplus_beta * x
        softplus_x = torch.where(
            beta_x <= softplus_threshold,
            torch.log1p(torch.exp(beta_x)) / softplus_beta,
            x,
        )
        return -A * softplus_x
    return g_min * torch.sigmoid(A * x)


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
        g: log-space decay, ``(B, T, H, K)`` or ``(B, T, H)``.
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
    if g.dim() == 3:
        g = g[..., None].expand(B, T, H, K)
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
