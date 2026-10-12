# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""Pure-PyTorch reference for Kimi Delta Attention (KDA).

The entry point mirrors the signature of
``aiter.ops.triton.kimi_delta_attn.chunk_kimi_delta_attn`` so a test can drive
both from a single argument dict.

The recurrence is evaluated one token at a time in fp32 -- the definition the
chunked kernels approximate -- so a disagreement points at the blocked
formulation rather than at a second copy of it. It is O(T) Python-level
iterations, so keep the sequence lengths passed here small.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F

__all__ = ["chunk_kda_ref", "kda_gate_ref", "l2norm_ref"]


def l2norm_ref(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize the last dimension in fp32. ``eps`` matches ``l2norm_fwd``."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) + eps)


def kda_gate_ref(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """Per-token log-space forget gate, shape ``[..., H, K]`` in, same out.

    Without ``lower_bound``: ``-exp(A_log) * softplus(g + dt_bias)``.
    With one: ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, which is
    bounded by construction.

    ``F.softplus`` falls back to the identity past ``x = 20``, which is what
    keeps a gate outlier from overflowing fp32 and poisoning the cumsum of
    every later token.
    """
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    A = A_log.float().view(H, 1).exp()
    if lower_bound is None:
        return -A * F.softplus(g)
    return lower_bound * torch.sigmoid(A * g)


def _recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token gated delta rule.

    ``q``/``k`` are ``[B, T, H, K]``, ``v`` is ``[B, T, HV, V]``, ``g`` is the
    per-token log decay ``[B, T, HV, K]`` and ``beta`` is ``[B, T, HV]``. The
    state is K-first ``[B, HV, K, V]``.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H

    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    # GVA: every value head reads the qk head it is grouped under.
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = q.new_zeros(B, T, HV, V)
    for i in range(T):
        q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]
        g_i, beta_i = g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        # Delta rule: write the residual against what the state already holds.
        delta = beta_i[..., None] * (v_i - (k_i[..., None] * S).sum(-2))
        S = S + k_i[..., None] * delta[..., None, :]
        o[:, i] = (q_i[..., None] * S).sum(-2)
    return o, S


def chunk_kda_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    disable_recompute: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference KDA forward pass.

    ``safe_gate``, ``disable_recompute`` and ``chunk_size`` pick a kernel
    schedule rather than a result, so they are accepted and ignored here.

    Returns ``(o, final_state)`` with ``o`` of shape ``[B, T, HV, V]`` in the
    dtype of ``q``, and an fp32 final state laid out like ``initial_state``.
    """
    del safe_gate, disable_recompute, chunk_size

    out_dtype = q.dtype
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q, k = l2norm_ref(q), l2norm_ref(k)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta.float())
    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("`A_log` is required when `use_gate_in_kernel=True`.")
        g = kda_gate_ref(g, A_log, dt_bias, lower_bound)

    h0 = initial_state
    if h0 is not None:
        h0 = h0.float()
        if state_v_first:
            h0 = h0.transpose(-1, -2)

    if cu_seqlens is None:
        o, final_state = _recurrent_kda(q, k, v, g, beta, scale, h0)
    else:
        # Packed sequences: each one carries its own state, so they cannot share
        # a single pass.
        bounds = cu_seqlens.tolist()
        outs, states = [], []
        for n, (bos, eos) in enumerate(itertools.pairwise(bounds)):
            o_n, s_n = _recurrent_kda(
                q[:, bos:eos],
                k[:, bos:eos],
                v[:, bos:eos],
                g[:, bos:eos],
                beta[:, bos:eos],
                scale,
                None if h0 is None else h0[n : n + 1],
            )
            outs.append(o_n)
            states.append(s_n)
        o = torch.cat(outs, dim=1)
        final_state = torch.cat(states, dim=0)

    if not output_final_state:
        final_state = None
    elif state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
    return o.to(out_dtype), final_state
