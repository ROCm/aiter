# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Numerical reference for the Gated-Residual combine-and-mix operator.

Gated Residual is the gated variant of HyperConnections (arXiv:2409.19606). The
hidden state carries ``hc_count`` parallel residual streams, flattened as
``[tokens, hc_count * stream_dim]`` with the stream index outer and ``stream_dim``
inner. Each transformer sublayer collapses those streams into a single
``stream_dim`` input, runs its block, then injects the block output back into
every stream.

``combine_and_mix`` fuses two adjacent sublayer boundaries: it *combines* a
pending block output back into the residual streams, then runs the *mix* that
produces the next block input. For one token row (``s`` indexes the stream):

    combine     r2[s]    = r[s] + y * (2 * sigmoid(inj[s] / hc_count))
    norm        xn[s]    = rmsnorm(r2[s]) * (1 + norm_weight)      # per stream
    down        lora     = xn @ w_down.T                           # [lowrank]
    activation  lora     = silu(lora / hc_count)
    up          gate     = lora @ w_up.T                           # [hc_count*dim]
    gated mean  x        = mean_s sigmoid(gate[s]) * xn[s]         # [stream_dim]
    injection   inj_next = xn @ w_inject.T                         # [hc_count]

In the shipped model ``w_down`` and ``w_inject`` are one merged projection; this
reference keeps them as separate tensors and reproduces the merge by
concatenation order. ``w_inject`` is ``None`` for the final mixer, which emits no
new injection.

The reference is deliberately pure PyTorch and dependency-free so it can act as
the single shared oracle for every combine-and-mix kernel variant. It follows the
shipped kernels' numerics: the combined residual is rounded to bfloat16 before
normalization, and the GEMM operands are rounded to bfloat16 with float32
accumulation. Pass ``round_bf16=False`` to get the clean float32 math (useful for
cross-checking against a reference module or for error attribution).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "gr_combine",
    "gr_grouped_rmsnorm",
    "gr_mix_body",
    "gr_mix",
    "gr_combine_and_mix",
]


def _bf16_round(t: torch.Tensor, enable: bool) -> torch.Tensor:
    """Round through bfloat16 but stay in float32 for the surrounding math."""
    if not enable:
        return t.float()
    return t.to(torch.bfloat16).float()


def _bf16_matmul(a: torch.Tensor, b_t: torch.Tensor, round_bf16: bool) -> torch.Tensor:
    """``a @ b_t`` emulating a bfloat16 matrix core with float32 accumulation.

    ``a`` and ``b_t`` are rounded to bfloat16 (when enabled) and multiplied in
    float32, which matches an MFMA GEMM that accumulates in float32.
    """
    return _bf16_round(a, round_bf16) @ _bf16_round(b_t, round_bf16)


def gr_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    hc_count: int,
    *,
    round_bf16: bool = True,
) -> torch.Tensor:
    """Inject a pending block output back into every residual stream.

    ``residual`` is ``[tokens, hc_count*stream_dim]``, ``block_output`` is the
    single-stream ``[tokens, stream_dim]`` output to inject, and ``injection`` is
    the raw per-stream logits ``[tokens, hc_count]``. Returns the combined
    residual in float32 (rounded through bfloat16 when ``round_bf16``).
    """
    tokens, hidden = residual.shape
    dim = hidden // hc_count
    res = residual.float().view(tokens, hc_count, dim)
    inject = 2.0 * torch.sigmoid(injection.float() / hc_count)  # [tokens, hc_count]
    out = res + block_output.float().unsqueeze(1) * inject.unsqueeze(-1)
    return _bf16_round(out.reshape(tokens, hidden), round_bf16)


def gr_grouped_rmsnorm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_count: int,
    eps: float,
) -> torch.Tensor:
    """Per-stream RMS normalization with a Gemma-style ``(1 + weight)`` affine.

    Each ``stream_dim``-wide stream is normalized independently. ``norm_weight``
    is either ``[stream_dim]`` (shared across streams) or the full
    ``[hc_count*stream_dim]``.
    """
    tokens, hidden = x.shape
    dim = hidden // hc_count
    grouped = x.float().view(tokens, hc_count, dim)
    variance = grouped.pow(2).mean(dim=-1, keepdim=True)
    normed = (grouped * torch.rsqrt(variance + eps)).reshape(tokens, hidden)
    weight = norm_weight.float()
    if weight.numel() == dim:
        weight = weight.repeat(hc_count)
    elif weight.numel() != hidden:
        raise ValueError(
            f"norm_weight must have {dim} or {hidden} elements, got {weight.numel()}"
        )
    return normed * (1.0 + weight)


def gr_mix_body(
    xn: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor | None,
    hc_count: int,
    *,
    round_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Low-rank gate over the normalized streams.

    ``xn`` is the normalized residual ``[tokens, hc_count*stream_dim]``. Weights
    follow the ``torch.nn.Linear`` convention (``[out, in]``): ``w_down`` is
    ``[lowrank, hc_count*stream_dim]``, ``w_up`` is
    ``[hc_count*stream_dim, lowrank]``, and ``w_inject`` (or ``None``) is
    ``[hc_count, hc_count*stream_dim]``. Returns the single-stream block input
    ``[tokens, stream_dim]`` and the next injection logits ``[tokens, hc_count]``
    (``None`` when ``w_inject`` is ``None``).
    """
    tokens, hidden = xn.shape
    dim = hidden // hc_count
    xn_q = _bf16_round(xn, round_bf16)

    lora = _bf16_matmul(xn_q, w_down.t(), round_bf16)  # [tokens, lowrank]
    inj_next = (
        _bf16_matmul(xn_q, w_inject.t(), round_bf16) if w_inject is not None else None
    )

    lora = F.silu(lora / hc_count)
    gate = _bf16_matmul(lora, w_up.t(), round_bf16)  # [tokens, hidden]

    gated = torch.sigmoid(gate).view(tokens, hc_count, dim) * xn_q.view(
        tokens, hc_count, dim
    )
    block_input = gated.mean(dim=1)  # [tokens, stream_dim]
    return block_input, inj_next


def gr_mix(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor | None,
    hc_count: int,
    eps: float,
    *,
    round_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the mix pipeline on an already-combined residual (no pending combine)."""
    xn = gr_grouped_rmsnorm(residual, norm_weight, hc_count, eps)
    xn = _bf16_round(xn, round_bf16)
    return gr_mix_body(xn, w_down, w_up, w_inject, hc_count, round_bf16=round_bf16)


def gr_combine_and_mix(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor | None,
    hc_count: int,
    eps: float,
    *,
    round_bf16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Full operator: combine a pending block output, then produce the next mix.

    Returns ``(r2, block_input, inj_next)`` where ``r2`` is the updated
    multi-stream residual, ``block_input`` is the next block's single-stream
    input, and ``inj_next`` is the next injection logits (``None`` for the final
    mixer). All returned tensors are float32; callers cast to the storage dtype.
    """
    r2 = gr_combine(residual, block_output, injection, hc_count, round_bf16=round_bf16)
    xn = gr_grouped_rmsnorm(r2, norm_weight, hc_count, eps)
    xn = _bf16_round(xn, round_bf16)
    block_input, inj_next = gr_mix_body(
        xn, w_down, w_up, w_inject, hc_count, round_bf16=round_bf16
    )
    return r2, block_input, inj_next
