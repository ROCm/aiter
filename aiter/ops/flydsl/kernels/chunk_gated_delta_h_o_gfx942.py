# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GDN K5+K6 fused forward — gfx942 (CDNA3 / MI300X).

This module will host the *fused* inter-chunk state-scan (K5) + inter/intra-chunk
output (K6) FlyDSL kernel. See ``docs/gdn_k5k6_fusion_plan.md`` for the design.

Phase 0 (current): this file provides only a **placeholder** orchestrator,
``chunk_gated_delta_h_o_placeholder``, which runs the existing separate K5 (FlyDSL)
and K6 (Triton) kernels back-to-back and writes ``o`` in place. It exists so the
host wrapper, correctness tests, and benchmark harness can all be built and
verified against a known-correct implementation *before* the real fused kernel is
written.

Phase 1 will replace the placeholder with ``compile_chunk_gated_delta_h_o_gfx942``,
a single-dispatch FlyDSL kernel derived from
``chunk_gated_delta_h_gfx942.compile_chunk_gated_delta_h_gfx942`` plus the minimal
GEMM3/GEMM4 additions from the Triton K6 kernel.

Unlike ``chunk_gated_delta_h_gfx942`` (a ``torch``-free kernel-compile module), this
placeholder deliberately depends on ``torch`` and on the two existing wrappers: it
is scaffolding, not a kernel. The Phase 1 kernel-compile function added here will be
kept ``torch``-free, matching the sibling module's layering.
"""

from __future__ import annotations

import torch


def chunk_gated_delta_h_o_placeholder(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    cu_seqlens: torch.LongTensor | None,
    state_dtype: torch.dtype | None,
    use_exp2: bool,
    num_decodes: int,
    num_decode_tokens: int,
    prefill_metadata,
    variant: str | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Unfused placeholder: run FlyDSL K5 then Triton K6, writing ``o`` in place.

    Returns ``(o, final_state)``. This mirrors the tensor contract the fused
    kernel will honour: ``o`` in token-major ``[B, T, H, V]`` and the optional
    ``final_state`` in ``[N, H, V, K]``. ``h`` and ``v_new`` are materialized to
    HBM here (exactly the traffic the fused kernel is meant to eliminate) — that
    is expected and is the baseline the fusion is measured against.

    Args match ``chunk_gated_delta_rule_fwd_h_o_flydsl``; see that wrapper's
    docstring for the layout of each tensor.
    """
    # Local imports keep this module import-safe when FlyDSL/Triton K5/K6 are not
    # both available (e.g. Triton-only bench builds import the wrapper lazily).
    from ..linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_flydsl,
    )
    from ...triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk,
    )

    # -- K5: FlyDSL inter-chunk state scan --------------------------------------
    # Produces h [B, NT, H, V, K] and v_new [B, H, T_flat, V]; both are consumed
    # by K6 below. save_new_value=True is required (K6 reads v_new as its ``v``).
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
        variant=variant,
    )

    # -- K6: Triton inter/intra-chunk output ------------------------------------
    # K6 gates only on scalar ``g``; the per-channel ``gk`` (KDA) path already
    # folds its decay into h/v_new inside K5, so K6 sees g=None there. This
    # matches the production pipeline (chunk.py passes g_cumsum, and the KDA
    # entry point routes gk through K5 only).
    chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        o=o,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )

    return o, final_state
