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

The kernel accepts ``w_down`` and ``w_inject`` as one merged projection. This
reference keeps them separate and preserves their concatenation order.
``w_inject`` is ``None`` for the final mixer, which emits no new injection.

The pure-PyTorch implementation is the shared oracle for each kernel variant.
It matches kernel numerics by rounding the combined residual and GEMM operands
to bfloat16 while accumulating GEMMs in float32. Pass ``round_bf16=False`` to
evaluate the operator entirely in float32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "gr_combine",
    "gr_combine_and_mix",
    "gr_grouped_rmsnorm",
    "gr_mix",
    "gr_mix_body",
    "seam_combine_only_ref",
    "seam_deferred_partials_ref",
    "seam_ref_deferred",
    "seam_ref_fp64",
    "seam_ref_operational",
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


# ---------------------------------------------------------------------------
# Finished-gate seam references.
#
# The seam contract differs from the raw-logit references above: ``gate_prev`` is
# the finished FP32 gate ``2*sigmoid(logit/hc_count)`` of shape ``[tokens, hc_count]``
# and the next gate is returned finished, in FP32. ``y`` and ``gate_prev`` are both
# ``None`` for a no-combine (mix-only) call, and ``w_inject`` is ``None`` for no
# next gate. Returns ``(residual_new BF16, mixed BF16, gate_next FP32 | None)``;
# without a combine ``residual_new`` is ``residual`` itself.
# ``norm_weight`` is ``[stream_dim]`` or ``[hc_count*stream_dim]``.
# ---------------------------------------------------------------------------


def _seam_norm_weight(norm_weight: torch.Tensor, hc_count: int, dim: int, dtype):
    w = norm_weight.to(dtype)
    if w.numel() == dim:
        w = w.repeat(hc_count)
    return w.view(hc_count, dim)


def seam_combine_only_ref(
    y: torch.Tensor, residual: torch.Tensor, gate_prev: torch.Tensor
) -> torch.Tensor:
    """BF16(residual + gate_prev * y) with one FP32 rounding (FMA), then BF16.

    The product of an FP32 gate and a BF16 value has at most 32 significant bits,
    so it is exact in FP64; the FP64 sum rounded to FP32 emulates the FP32 FMA.
    """
    tokens, hidden = residual.shape
    hc_count = gate_prev.shape[1]
    dim = hidden // hc_count
    a = gate_prev.double().unsqueeze(-1) * y.double().unsqueeze(1)
    out = residual.double().view(tokens, hc_count, dim) + a
    return out.float().bfloat16().reshape(tokens, hidden)


def seam_ref_fp64(
    y,
    residual,
    gate_prev,
    norm_weight,
    w_down,
    w_up,
    w_inject,
    eps,
    hc_count=4,
):
    """FP64 mathematical oracle. Only R' and the gate's X are rounded to BF16.

    Mirrors the SGLang seam oracle: the combined residual is stored as BF16, the
    projections see unrounded normalized X, and the next gate sees BF16(X).
    """
    f = torch.float64
    tokens, hidden = residual.shape
    dim = hidden // hc_count
    r = seam_combine_only_ref(y, residual, gate_prev) if y is not None else residual
    x = r.to(f).view(tokens, hc_count, dim)
    rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    xn = x * rs * (1.0 + _seam_norm_weight(norm_weight, hc_count, dim, f))
    xn = xn.view(tokens, hidden)
    d = xn @ w_down.to(f).t() / hc_count
    g = torch.sigmoid((d * torch.sigmoid(d)) @ w_up.to(f).t())
    mixed = (g.view(tokens, hc_count, dim) * xn.view(tokens, hc_count, dim)).mean(1)
    gate_next = None
    if w_inject is not None:
        gate_next = 2.0 * torch.sigmoid(
            xn.bfloat16().to(f) @ w_inject.to(f).t() / hc_count
        )
    return r, mixed, gate_next


def seam_ref_operational(
    y,
    residual,
    gate_prev,
    norm_weight,
    w_down,
    w_up,
    w_inject,
    eps,
    hc_count=4,
):
    """Explicit-rounding oracle matching the kernel's storage points.

    BF16 combine via FP32 FMA; BF16 normalized X; FP32 down accumulation; BF16
    SiLU bottleneck; FP32 up logits, sigmoid and branch mean over BF16 X; BF16
    mixed output; next gate 2*sigmoid(FP32_dot(BF16(X), W_inject)/hc_count) in FP32.
    """
    tokens, hidden = residual.shape
    dim = hidden // hc_count
    r = seam_combine_only_ref(y, residual, gate_prev) if y is not None else residual
    x = r.float().view(tokens, hc_count, dim)
    rs = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    xn = x * rs * (1.0 + _seam_norm_weight(norm_weight, hc_count, dim, torch.float32))
    xn = xn.bfloat16().float().view(tokens, hidden)

    # Keep these GEMMs in true FP32 regardless of the global TF32 setting.
    d = torch.matmul(xn, w_down.float().t()) / hc_count
    bott = F.silu(d).bfloat16().float()
    logits = torch.matmul(bott, w_up.float().t())
    g = torch.sigmoid(logits).view(tokens, hc_count, dim)
    mixed = (g * xn.view(tokens, hc_count, dim)).mean(1).bfloat16()
    gate_next = None
    if w_inject is not None:
        gate_next = 2.0 * torch.sigmoid(
            torch.matmul(xn, w_inject.float().t()) / hc_count
        )
    return r, mixed, gate_next


def seam_deferred_partials_ref(
    y, residual, gate_prev, norm_weight, w_down, hc_count=4, chunk_width=256
):
    """Operational deferred operands and FP32 dots, without RMS or /HC.

    Returns R' [M,H], dots [HC,HS/chunk,M,320], and FP64 sums of
    squares [HC,HS/chunk,M]. Each chunk stays within one branch.
    """
    tokens, hidden = residual.shape
    dim = hidden // hc_count
    if chunk_width <= 0 or dim % chunk_width:
        raise ValueError("chunk_width must divide the branch width")
    if w_down.shape != (320, hidden):
        raise ValueError("deferred partials require down width 320")
    r = seam_combine_only_ref(y, residual, gate_prev) if y is not None else residual
    chunks = dim // chunk_width
    x = r.float().view(tokens, hc_count, dim)
    onepw = 1.0 + _seam_norm_weight(norm_weight, hc_count, dim, torch.float32)
    z = (x * onepw).bfloat16().float().view(tokens, hc_count, chunks, chunk_width)
    wd = w_down.float().view(320, hc_count, chunks, chunk_width)
    dots = torch.stack(
        [
            torch.stack([z[:, c, k] @ wd[:, c, k].t() for k in range(chunks)])
            for c in range(hc_count)
        ]
    )
    squares = r.double().view(tokens, hc_count, chunks, chunk_width).square().sum(-1)
    return r, dots, squares.permute(1, 2, 0).contiguous()


def seam_ref_deferred(y, residual, gate_prev, norm_weight, w_down, w_up, w_inject, eps):
    """Chunk-reduced down with BF16 affine operands and post-dot stream RMS.

    RMS uses the per-stream mean over 2560 channels, as in seam_ref_fp64.
    Gates and K2 retain BF16((R' * rrms) * (1+w)), not the deferred operand.
    """
    r, dots, _ = seam_deferred_partials_ref(y, residual, gate_prev, norm_weight, w_down)
    rows = residual.shape[0]
    # Match chunk owners: eight squares per lane, then XOR offsets 1,2,4
    # within each 64-channel panel, followed by four panel sums per chunk.
    squares = r.float().view(rows, 4, 10, 4, 8, 8).square()
    panel = torch.zeros_like(squares[..., 0])
    for i in range(8):
        panel = panel + squares[..., i]
    lanes = torch.arange(8, device=r.device)
    for shift in (1, 2, 4):
        panel = panel + panel[..., lanes ^ shift]
    chunk = torch.zeros_like(panel[..., 0, 0])
    for i in range(4):
        chunk = chunk + panel[..., i, 0]
    total = torch.zeros_like(chunk[..., 0])
    for i in range(10):
        total = total + chunk[..., i]
    rrms = 1.0 / torch.sqrt(total * (1.0 / 2560) + eps)
    down = (dots.sum(1).permute(1, 0, 2) * rrms.unsqueeze(-1)).sum(1) / 4
    bott = F.silu(down).bfloat16().float()
    x = (
        (
            (r.float().view(rows, 4, 2560) * rrms.unsqueeze(-1))
            * (1 + norm_weight.float().view(4, 2560))
        )
        .bfloat16()
        .float()
        .view(rows, 10240)
    )
    logits = bott @ w_up.float().t()
    mixed = (
        (torch.sigmoid(logits).view(rows, 4, 2560) * x.view(rows, 4, 2560))
        .mean(1)
        .bfloat16()
    )
    gate = None if w_inject is None else 2 * torch.sigmoid(x @ w_inject.float().t() / 4)
    return r, mixed, gate
