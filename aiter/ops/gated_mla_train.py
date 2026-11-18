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

The forward runs on the gfx950 OPUS D_QK=192 / D_V=128 kernel: it reads ``k_nope`` and
``v`` straight out of the ``kv_b_proj`` output and ``k_rot`` from its own buffer, so K is
never concatenated, and it applies the gate before the O store. Only ``kv_latent`` is kept
for the backward, which recomputes the expansion.
"""

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx
from .mha import _flash_attn_varlen_backward


def _gated_mla_fwd_bf16_opus_fake(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    gate: Tensor,
    out: Tensor,
    lse: Tensor | None,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None:
    return None


# Writes `out` (and `lse`, when given) in place.
@compile_ops(
    "module_fmha_fwd_bf16_opus",
    fc_name="gated_mla_fwd_bf16_opus",
    gen_fake=_gated_mla_fwd_bf16_opus_fake,
)
def _gated_mla_fwd_bf16_opus(
    q: Tensor,
    k_nope: Tensor,
    v: Tensor,
    k_rot: Tensor,
    gate: Tensor,
    out: Tensor,
    lse: Tensor | None,
    seqstart_q: Tensor,
    seqstart_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def _kernel_ready(x: Tensor) -> Tensor:
    """`x` if the kernel's 16-byte row loads can read it as is, else a contiguous copy."""
    aligned = x.stride(-1) == 1 and x.data_ptr() % 16 == 0
    aligned = aligned and all(s % 8 == 0 for s in x.stride()[:-1])
    return x if aligned else x.contiguous()


def _gated_mla_fwd(
    q: Tensor,
    kv: Tensor,
    k_rot: Tensor,
    gate: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    softmax_scale: float,
    causal: bool,
    return_lse: bool,
) -> tuple[Tensor, Tensor | None]:
    """Fused attention + gate on the expanded ``kv`` ``[T, H, d_nope + d_v]``.

    Returns ``out [T, H, d_v]`` and, when ``return_lse``, the ungated attention's
    log-sum-exp ``[H, T]`` (fp32, natural log).
    """
    t, h, d_qk = q.shape
    d_nope = d_qk - k_rot.shape[-1]
    d_v = kv.shape[-1] - d_nope
    out = torch.empty(t, h, d_v, dtype=q.dtype, device=q.device)
    lse = (
        torch.empty(h, t, dtype=torch.float32, device=q.device) if return_lse else None
    )
    _gated_mla_fwd_bf16_opus(
        _kernel_ready(q),
        kv[..., :d_nope],
        kv[..., d_nope:],
        _kernel_ready(k_rot),
        _kernel_ready(gate.reshape(t, h, d_v)),
        out,
        lse,
        cu_seqlens,
        cu_seqlens,
        int(max_seqlen),
        int(max_seqlen),
        bool(causal),
        float(softmax_scale),
    )
    return out, lse


class _GatedMLA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        kv_latent,
        k_rot,
        kv_b_weight,
        gate,
        cu_seqlens,
        max_seqlen,
        scale,
        causal,
    ):
        t, h, _ = q.shape
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        kv = torch.mm(kv_latent, kv_b_weight.t()).view(t, h, -1)
        out, lse = _gated_mla_fwd(
            q,
            kv,
            k_rot,
            gate,
            cu_seqlens,
            max_seqlen,
            scale,
            causal,
            return_lse=any(ctx.needs_input_grad),
        )
        ctx.save_for_backward(
            q, kv_latent, k_rot, kv_b_weight, gate, out, lse, cu_seqlens
        )
        ctx.max_seqlen, ctx.scale, ctx.causal = max_seqlen, scale, causal
        return out.view(t, -1)

    @staticmethod
    def backward(ctx, dout):
        q, kv_latent, k_rot, kv_b_weight, gate, out, lse, cu_seqlens = ctx.saved_tensors
        t, h, d_qk = q.shape
        d_rot = k_rot.shape[-1]
        d_nope = d_qk - d_rot
        kv = torch.mm(kv_latent, kv_b_weight.t()).view(t, h, -1)
        k_nope, v = kv.split([d_nope, kv.shape[-1] - d_nope], dim=-1)
        k = torch.cat([k_nope, k_rot.unsqueeze(1).expand(t, h, d_rot)], dim=-1)
        v = v.contiguous()

        dout = dout.view(out.shape).float()
        out_f = out.float()
        s = torch.sigmoid(gate.reshape(out.shape).float())
        dgate = (dout * out_f * (1.0 - s)).to(gate.dtype)
        do = (dout * s).to(q.dtype)
        # The CK backward forms D = rowsum(dO * O) from the (dO, O) it is given, so it gets
        # O = out / sigmoid(gate): that keeps D = rowsum(dout * out) without saving the
        # ungated O. Where the sigmoid underflows to 0, dO is 0 and O does not matter.
        o = torch.where(s > 0, out_f / s, 0.0).to(q.dtype)

        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            dq,
            dk,
            dv,
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            0.0,
            ctx.scale,
            ctx.causal,
            -1,
            -1,
            None,
            False,
        )
        dkv = torch.cat([dk[..., :d_nope], dv], dim=-1).view(t, -1)
        d_latent = torch.mm(dkv, kv_b_weight)
        d_weight = torch.mm(dkv.t(), kv_latent)
        dk_rot = dk[..., d_nope:].float().sum(1).to(k_rot.dtype)
        return (
            dq,
            d_latent,
            dk_rot,
            d_weight,
            dgate.view_as(gate),
            None,
            None,
            None,
            None,
        )


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

    BF16 on gfx950, with (d_nope, d_rot, d_v) = (128, 64, 128). Returns
    ``[T, H * d_v]`` in ``q.dtype``.
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
    if (d_nope, d_rot, d_v) != (128, 64, 128):
        raise ValueError(
            f"(d_nope, d_rot, d_v) = {(d_nope, d_rot, d_v)}, only (128, 64, 128) is supported"
        )
    if get_gfx() != "gfx950":
        raise NotImplementedError(f"gated_mla_train needs gfx950, got {get_gfx()}")
    if softmax_scale is None:
        softmax_scale = d_qk**-0.5
    return _GatedMLA.apply(
        q,
        kv_latent,
        k_rot,
        kv_b_weight,
        gate,
        cu_seqlens,
        max_seqlen,
        softmax_scale,
        causal,
    )
