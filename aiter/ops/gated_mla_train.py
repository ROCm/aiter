# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gated MLA training attention (Kimi-K3 full-attention layers).

Dense attention over per-head keys and values expanded from a shared latent, with no
rotary on the attention path (NoPE) and an elementwise sigmoid output gate. Per token
``t`` and head ``h``, with ``d_nope = q.shape[-1] - k_rot.shape[-1]``::

    k_nope[t, h], v[t, h] = split((kv_latent[t] @ kv_b_weight.T)[h], [d_nope, d_v])
    k[t, h]               = cat(k_nope[t, h], k_rot[t])
    o[t, h]               = softmax(q[t, h] . k[:, h] * softmax_scale) @ v[:, h]
    out[t]                = o[t].flatten() * sigmoid(gate[t])

``out`` is the ``o_proj`` input. The q / kv down-projections, their RMSNorms,
``q_b_proj``, ``g_proj`` and ``o_proj`` stay with the caller.
"""

import torch
import torch.nn.functional as F

from .mha import flash_attn_varlen_func


class _SigmoidGate(torch.autograd.Function):
    """``o * sigmoid(gate)``; backward keeps only the two inputs."""

    @staticmethod
    def forward(ctx, o, gate):
        ctx.save_for_backward(o, gate)
        return (o.float() * torch.sigmoid(gate.float())).to(o.dtype)

    @staticmethod
    def backward(ctx, dout):
        o, gate = ctx.saved_tensors
        s = torch.sigmoid(gate.float())
        dout = dout.float()
        do = dout * s
        dgate = dout * o.float() * s * (1.0 - s)
        return do.to(o.dtype), dgate.to(gate.dtype)


def gated_mla_train(
    q: torch.Tensor,
    kv_latent: torch.Tensor,
    k_rot: torch.Tensor,
    kv_b_weight: torch.Tensor,
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """Gated MLA attention forward; differentiable in every tensor argument.

    q           : [T, H, d_nope + d_rot], the ``q_b_proj`` output.
    kv_latent   : [T, d_latent], the ``kv_a_layernorm`` output.
    k_rot       : [T, d_rot], the companion key from ``kv_a_proj_with_mqa``, shared
                  by all heads.
    kv_b_weight : [H * (d_nope + d_v), d_latent], ``kv_b_proj.weight``; each head's
                  rows are ``[k_nope | v]``.
    gate        : [T, H * d_v], the ``g_proj`` output before the sigmoid.
    cu_seqlens  : [B + 1] int32 packed-sequence boundaries, shared by queries and keys.
    max_seqlen  : longest sequence in the batch.
    softmax_scale : defaults to ``(d_nope + d_rot) ** -0.5``.

    Returns ``[T, H * d_v]`` in ``q.dtype``.
    """
    t, h, d_qk = q.shape
    d_rot = k_rot.shape[-1]
    d_nope = d_qk - d_rot
    if kv_b_weight.shape[0] % h:
        raise ValueError(
            f"kv_b_weight rows {kv_b_weight.shape[0]} are not a multiple of H={h}"
        )
    d_v = kv_b_weight.shape[0] // h - d_nope
    if kv_latent.shape != (t, kv_b_weight.shape[1]):
        raise ValueError(
            f"kv_latent shape {tuple(kv_latent.shape)} != {(t, kv_b_weight.shape[1])}"
        )
    if k_rot.shape != (t, d_rot):
        raise ValueError(f"k_rot shape {tuple(k_rot.shape)} != {(t, d_rot)}")
    if gate.shape != (t, h * d_v):
        raise ValueError(f"gate shape {tuple(gate.shape)} != {(t, h * d_v)}")
    if softmax_scale is None:
        softmax_scale = d_qk**-0.5

    kv = F.linear(kv_latent, kv_b_weight).view(t, h, d_nope + d_v)
    k_nope, v = kv.split([d_nope, d_v], dim=-1)
    k = torch.cat([k_nope, k_rot.unsqueeze(1).expand(t, h, d_rot)], dim=-1)
    o, _ = flash_attn_varlen_func(
        q,
        k,
        v.contiguous(),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
    )
    return _SigmoidGate.apply(o.reshape(t, h * d_v), gate)
