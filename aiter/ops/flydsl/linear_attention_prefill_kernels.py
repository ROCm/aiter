# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end FlyDSL Linear Attention Prefill APIs (gated delta rule).

This module exposes ``flydsl_gdr_prefill``, a drop-in replacement for the
in-tree Triton ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk``
where the K5 hidden-state recurrence is executed by the FlyDSL kernel
(``chunk_gated_delta_rule_fwd_h_flydsl``) and the rest of the chunk pipeline
(K1+K2 fused cumsum/dot-kkt, K3+K4 fused solve-tril/recompute-w-u, K6 output)
re-uses the existing Triton implementations.
"""

from __future__ import annotations

import torch

from .kernels.chunk_gated_delta_h import chunk_gated_delta_rule_fwd_h_flydsl

from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
    chunk_fwd_o_opt_vk,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_cumsum_kkt import (
    fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_solve_tril_recompute import (
    fused_solve_tril_recompute_w_u,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils.l2norm import (
    l2norm_fwd,
)

__all__ = [
    "flydsl_gdr_prefill",
]


def flydsl_gdr_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """End-to-end GDN forward where K5 runs on FlyDSL.

    Signature is identical to
    ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` so that
    the two can be used interchangeably as drop-in backends.

    Pipeline (matches ``chunk_gated_delta_rule_fwd_opt_vk``):

      * K1+K2 fused : ``fused_chunk_local_cumsum_scaled_dot_kkt_fwd``  (Triton)
      * K3+K4 fused : ``fused_solve_tril_recompute_w_u``               (Triton)
      * **K5**      : ``chunk_gated_delta_rule_fwd_h_flydsl``          (FlyDSL)
      * K6          : ``chunk_fwd_o_opt_vk``                           (Triton)

    Args:
        q: queries ``[B, T, H, K]``.
        k: keys ``[B, T, Hg, K]`` (GQA: ``Hg`` may be smaller than ``H``).
        v: values ``[B, T, H, V]``.
        g: log-decays ``[B, T, H]`` (raw, will be cumsum'd by K1).
        beta: betas ``[B, T, H]``.
        scale: attention scale; default ``1 / sqrt(K)``.
        initial_state: optional ``[N, H, V, K]`` (VK layout).
        output_final_state: whether to return the final state.
        use_qk_l2norm_in_kernel: apply L2 normalization to ``q`` and ``k``
            before the chunk pipeline.
        cu_seqlens: ``[N+1]`` cumulative sequence lengths for varlen mode.

    Returns:
        ``(o, final_state)`` where ``o`` is shape ``[B, T, H, V]`` and
        ``final_state`` is ``[N, H, V, K]`` (or ``None``).
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the "
                f"number of input sequences, i.e., {len(cu_seqlens) - 1} "
                f"rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    # -- K1+K2 (Triton) : g_cumsum, A_raw ----------------------------------
    g_cumsum, A_raw = fused_chunk_local_cumsum_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    # -- K3+K4 (Triton) : w (head-major), u (head-major) -------------------
    w, u = fused_solve_tril_recompute_w_u(
        A_raw=A_raw,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
    )

    # -- K5 (FlyDSL) : h, v_new, final_state -------------------------------
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # -- K6 (Triton) : o = chunk_fwd_o_opt_vk(q, k, v_new, h, g_cumsum) ----
    o = chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    return o.to(q.dtype), final_state
