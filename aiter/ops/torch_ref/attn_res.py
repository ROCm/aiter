# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-PyTorch reference for ``flydsl_attn_res``.

Mirrors the ``attn_res_gate`` contract: the candidates are the ``B`` rows of
``block_residual`` plus the prefix, whose FP32 sum with ``add_hidden`` [and
``add_hidden2``] feeds scoring and mixing unrounded. Scores use RMS-normalized
candidates against the pre-folded ``score_weight``; values stay raw.

Scoring, mixing, and output RMSNorm accumulate in ``dtype`` (default FP32).
``prefix_out`` and ``block_out`` are BF16, as the kernel stores them. ``y``
stays in ``dtype``, or becomes a per-token ``(fp8, scale)`` pair of that
``dtype`` result when ``out_quant_dtype`` is set. ``bf16_intermediates``
rounds the prefix candidate and the pre-quant ``y`` to BF16. Inputs are never
mutated.
"""

import torch


def per_token_fp8_quant(
    x: torch.Tensor, fp8_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply by reciprocals, not divide, to match the kernel's per-token quant."""
    x = x.float()
    scale = x.abs().amax(-1, keepdim=True) * (1.0 / torch.finfo(fp8_dtype).max)
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    return (x * inv_scale).to(fp8_dtype), scale


def attn_res(
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-6,
    add_hidden: torch.Tensor | None = None,
    add_hidden2: torch.Tensor | None = None,
    *,
    output_rms_weight: torch.Tensor | None = None,
    output_rms_eps: float = 1e-6,
    scale: float = 1.0,
    close_block: bool = False,
    out_quant_dtype: torch.dtype | None = None,
    bf16_intermediates: bool = False,
    dtype: torch.dtype = torch.float32,
):
    """Return ``(y, prefix_out)`` or ``(y, prefix_out, block_out)`` like the kernel."""
    if add_hidden is not None:
        summed = prefix.float() + add_hidden.float()
        if add_hidden2 is not None:
            summed = summed + add_hidden2.float()
        prefix_out = summed.to(prefix.dtype)
        candidate = (prefix_out if bf16_intermediates else summed).to(dtype)
    else:
        candidate = prefix.to(dtype)
        prefix_out = prefix

    values = torch.cat((block_residual.to(dtype), candidate.unsqueeze(-2)), dim=-2)
    rstd = torch.rsqrt((values * values).mean(-1) + eps)
    logits = rstd * (values * score_weight.to(dtype)).sum(-1) * scale
    probs = logits.softmax(dim=-1)
    y = (probs.unsqueeze(-1) * values).sum(-2)
    if output_rms_weight is not None:
        y_rstd = torch.rsqrt((y * y).mean(-1, keepdim=True) + output_rms_eps)
        y = y * y_rstd * output_rms_weight.to(dtype)
    if out_quant_dtype is not None:
        if bf16_intermediates:
            y = y.to(torch.bfloat16).to(dtype)
        y = per_token_fp8_quant(y, out_quant_dtype)

    if not close_block:
        return y, prefix_out
    block_out = torch.cat((block_residual, prefix_out.unsqueeze(-2)), dim=-2)
    return y, prefix_out, block_out
