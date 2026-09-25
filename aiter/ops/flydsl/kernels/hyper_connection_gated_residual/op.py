# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two-stage Gated-Residual ``combine_and_mix`` op (SILOTIGER-1042).

The shipped, fully-fused two-stage kernel: ``K1`` = combine + grouped-RMSNorm +
down GEMM (:func:`~.k1.flydsl_k1_combine_norm_down`), ``K2`` = silu + up GEMM +
gated mean (:func:`~.k2.flydsl_up_gate_mix_norm`), with ``xn`` re-formed from the
stored ``r2`` inside K2 so it never touches HBM. Collapses today's 5 launches to
2 (modulo K1's split-K/decouple reduce).

Three ticket entry points -- ``combine`` (write-only ``R2``), ``mix`` (no pending
combine), and ``combine_and_mix`` -- all backed by the fused path. Only the fused
two-stage ships here; the earlier composed / interim fused-K1 variants are not
part of this package.
"""

from __future__ import annotations

import torch

import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .common import arch_name, merge_down_inject, split_down_inject
from .k1 import (
    DECODE_MAX_M,
    _build_combine_rms,
    flydsl_k1_combine_norm_down,
    flydsl_k1k2_skinny_decode,
)
from .k2 import flydsl_up_gate_mix_norm


def _merged_pad(width: int) -> int:
    """Round a down-output width up to the pipeline's 64-wide N block."""
    return ((width + 63) // 64) * 64


def merge_gr_two_stage_weight(
    w_down: torch.Tensor,
    w_inject: torch.Tensor | None,
    hc_count: int,
) -> torch.Tensor:
    """Pack the down (+ inject) projection into the merged weight K1 wants.

    ``w_inject=None`` is the final mixer (``Nd=r``, no inject columns): the merged
    weight is just ``w_down`` zero-padded to a 64-multiple. A model stores this
    once and passes it as ``w_down_merged`` to keep the merge off the timed path.
    """
    lowrank = w_down.shape[0]
    if w_inject is None:
        n_pad = _merged_pad(lowrank)
        merged = torch.zeros(
            n_pad, w_down.shape[1], dtype=w_down.dtype, device=w_down.device
        )
        merged[:lowrank] = w_down
        return merged
    return merge_down_inject(w_down, w_inject, _merged_pad(lowrank + w_inject.shape[0]))


def fold_norm_weight(
    w_down_merged: torch.Tensor, norm_weight: torch.Tensor, hc_count: int
) -> torch.Tensor:
    """Bake grouped-RMSNorm ``(1 + w)`` into each column of the merged down weight.

    With this fold K1 skips the per-element ``(1 + w)`` multiply (``fold_w=True``);
    a static weight transform a model applies once. ``norm_weight`` is ``[stream_dim]``
    (shared across streams) or the full ``[hidden]``.
    """
    hidden = w_down_merged.shape[1]
    w = norm_weight.reshape(-1).float()
    w_full = w.repeat(hidden // w.numel()) if w.numel() != hidden else w
    return (w_down_merged.float() * (1.0 + w_full)[None, :]).to(w_down_merged.dtype).contiguous()


def _resolve_merged(w_down, w_inject, w_down_merged, norm_weight, hc_count, fold_w):
    """Merged down weight; a caller-supplied ``w_down_merged`` is used as-is.

    CONTRACT: when ``fold_w=True`` the merged weight MUST already have ``(1+w)``
    folded in -- K1's down and the skinny decode both skip the affine and assume it
    is baked into the weight. A caller passing ``fold_w=True`` with a *non-folded*
    ``w_down_merged`` gets silently wrong output (the fold state cannot be detected
    from the tensor). Build it with :func:`fold_norm_weight`, or pass
    ``w_down_merged=None`` and let this fold it.
    """
    if w_down_merged is not None:
        return w_down_merged  # caller owns the fold state (see fold_norm_weight)
    merged = merge_gr_two_stage_weight(w_down, w_inject, hc_count)
    return fold_norm_weight(merged, norm_weight, hc_count) if fold_w else merged


def _k1_then_k2(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_up: torch.Tensor,
    w_down_merged: torch.Tensor,
    lowrank: int,
    hc_count: int,
    eps: float,
    need_inj: bool,
    stream: torch.cuda.Stream | None,
    fold_w: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Fully-fused two-stage: K1 (combine+norm+down) then K2 (up + gated mean),
    with ``xn`` re-formed from ``r2`` inside K2.

    K1 emits ``(r2, packed)`` plus the per-stream ``rrms``; K2 rebuilds
    ``xn = r2*rrms*(1+w)`` on the fly -- so neither ``xn`` nor an interim
    norm-rebuild launch ever exists.
    """
    tokens = residual.shape[0]
    # Decode M (non-gfx950): a fully-skinny MMA-free GEMV two-stage beats the
    # padding-MMA path at small M (no 64-row pad; xn still never materialized).
    # Gated to fold_w (the GEMV down assumes the folded weight), non-gfx950
    # (gfx950 has the async-LDS pipe + native tail), and M<=DECODE_MAX_M.
    if fold_w and 1 <= tokens <= DECODE_MAX_M:
        arch = arch_name(residual.device)
        if arch != "gfx950":
            return flydsl_k1k2_skinny_decode(
                residual, block_output, injection, norm_weight, w_up,
                w_down_merged, lowrank, hc_count, eps, need_inj, stream=stream,
            )
    # Low-M tail path: when tokens is not a tile multiple, run the GEMM stages
    # over a padded row count P while the combine prologue reads only the true
    # tokens rows. All K1->K2 bridge buffers (r2, rrms, packed) live at P; the
    # final views slice back to [:tokens]. Replaces a caller-side pad-memcpy.
    PAD = 64
    pad_tokens = ((tokens + PAD - 1) // PAD) * PAD
    gemm_pad = None if pad_tokens == tokens else pad_tokens
    rrms = residual.new_empty((pad_tokens, hc_count), dtype=torch.float32)
    r2, packed = flydsl_k1_combine_norm_down(
        residual, block_output, injection, norm_weight, w_down_merged,
        lowrank, hc_count, eps, rrms_out=rrms, fold_w=fold_w, gemm_pad=gemm_pad,
        stream=stream,
    )
    if need_inj:
        lora, inj = split_down_inject(packed, lowrank, hc_count)
        lora, inj_next = lora.contiguous(), inj.contiguous()
    else:
        lora, inj_next = packed[:, :lowrank].contiguous(), None
    block_input = flydsl_up_gate_mix_norm(
        lora, r2, rrms, norm_weight, w_up, hc_count, stream=stream
    )
    if gemm_pad is not None:
        r2, block_input = r2[:tokens], block_input[:tokens]
        inj_next = inj_next[:tokens] if inj_next is not None else None
    return r2, block_input, inj_next


def flydsl_gr_two_stage_combine_and_mix(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor | None,
    hc_count: int,
    eps: float = 1e-6,
    w_down_merged: torch.Tensor | None = None,
    fold_w: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Combine a pending block output, then mix (the fused two-stage kernel).

    Returns ``(r2, block_input, inj_next)`` (``inj_next`` is ``None`` on the final
    mixer, ``w_inject=None``). Matches :func:`.reference.gr_combine_and_mix`.

    ``fold_w=True`` bakes ``(1 + norm_weight)`` into the down weight so K1 skips
    the per-element affine (a small win); pass a pre-folded ``w_down_merged`` (see
    :func:`fold_norm_weight`) to keep the fold off the timed path, or let this
    build it. ``fold_w=False`` (default) applies the affine inside K1.
    """
    lowrank = w_down.shape[0]
    merged = _resolve_merged(w_down, w_inject, w_down_merged, norm_weight, hc_count, fold_w)
    return _k1_then_k2(
        residual, block_output, injection, norm_weight, w_up, merged,
        lowrank, hc_count, eps, w_inject is not None, stream, fold_w=fold_w,
    )


def flydsl_gr_two_stage_mix(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    w_inject: torch.Tensor | None,
    hc_count: int,
    eps: float = 1e-6,
    w_down_merged: torch.Tensor | None = None,
    fold_w: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """No pending combine: normalize ``residual`` directly, then mix.

    Returns ``(residual, block_input, inj_next)`` -- ``r2`` is ``residual``
    unchanged (1042: "mix skips steps 1 and 2"). Runs the fused two-stage with a
    zero block output (so ``r2 == residual`` up to the bf16 store).
    """
    lowrank = w_down.shape[0]
    tokens, hidden = residual.shape
    stream_dim = hidden // hc_count
    zero_y = residual.new_zeros((tokens, stream_dim))
    zero_inj = residual.new_zeros((tokens, hc_count))
    merged = _resolve_merged(w_down, w_inject, w_down_merged, norm_weight, hc_count, fold_w)
    _r2, block_input, inj_next = _k1_then_k2(
        residual, zero_y, zero_inj, norm_weight, w_up, merged,
        lowrank, hc_count, eps, w_inject is not None, stream, fold_w=fold_w,
    )
    return residual, block_input, inj_next


def flydsl_gr_two_stage_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    norm_weight: torch.Tensor,
    hc_count: int,
    eps: float = 1e-6,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Write-only combine: inject the pending block output, return ``R2``.

    ``R2`` is the delayed residual write vLLM flushes on PLE / deepstack / PP
    handoff. Runs K1's combine+RMS prologue and keeps only ``r2`` (``norm_weight``
    is unused here -- the combine does not normalize its output).
    """
    tokens, hidden = residual.shape
    stream_dim = hidden // hc_count
    if stream is None:
        stream = torch.cuda.current_stream()
    r2 = torch.empty_like(residual)
    rrms = torch.empty(tokens, hc_count, dtype=torch.float32, device=residual.device)
    launch = _build_combine_rms(hc_count, stream_dim, float(eps))
    _run_compiled(launch, residual, block_output, injection, r2, rrms, fx.Stream(stream))
    return r2
