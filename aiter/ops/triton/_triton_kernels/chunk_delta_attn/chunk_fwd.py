# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Top-level forward function for chunk_delta_attn.

Pipeline:
  1. Gate cumsum:  apply fused A_log / dt_bias / softplus (or sigmoid) gating
                   and chunk-local prefix sum to produce gk (in log2 space).
  2. Intra-chunk:  compute Aqk, Akk inverse, and auxiliary W/U/KG tensors.
  3. Inter-chunk:  update recurrent hidden state H via chunk_gated_delta_rule_fwd_h.
  4. Output:       compute final output via chunk_gla_fwd_o (gated q * h + A * v).
"""

import torch

from aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_delta_attn_utils import (
    RCP_LN2,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.flash_kda import (
    CHUNK_DELTA_ATTN_USE_FLASH_KDA,
    FLASH_KDA_CHUNK,
    flash_kda_fwd,
    flash_kda_supported,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.gate import beta_sigmoid_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn.gla_output import (
    chunk_gla_fwd_o,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.intra_attn import (
    chunk_delta_attn_fwd_intra,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.utils.cumsum import (
    chunk_gate_cumsum,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.utils.index import (
    prepare_chunk_indices,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.utils.l2norm import l2norm_fwd
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)

# What an unset `chunk_size` falls back to when the FlashKDA path cannot serve
# the call.
_DEFAULT_CHUNK_SIZE = 64


def chunk_delta_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    state_v_first: bool = False,
) -> tuple:
    """
    Forward pass for chunk_delta_attn.

    Args:
        q:                  Query      ``[B, T, H, K]``.
        k:                  Key        ``[B, T, H, K]``.
        v:                  Value      ``[B, T, HV, V]``.
        g:                  Gate input ``[B, T, HV, K]``.
                            If ``use_gate_in_kernel=True`` this is raw gate
                            (A_log / softplus / sigmoid applied inside the kernel).
                            If ``use_gate_in_kernel=False`` this is a pre-computed
                            gate (only chunk-local cumsum is applied).
        beta:               Beta gate  ``[B, T, HV]``. Raw logits if
                            ``use_beta_sigmoid_in_kernel=True``, post-sigmoid otherwise.
        scale:              Attention scale (1 / sqrt(K) or similar).
        initial_state:      Initial recurrent state ``[N, HV, K, V]`` or None
                            (``[N, HV, V, K]`` when ``state_v_first=True``).
        output_final_state: Whether to return the final recurrent state.
        cu_seqlens:         Cumulative sequence lengths for variable-length mode.
        chunk_indices:      Pre-computed chunk index pairs (computed if None).
        chunk_size:         Chunk size BT, either 32 or 64. ``None`` lets this
                            function choose: 32 when that lets the FlashKDA path
                            serve the call, 64 otherwise.
        safe_gate:          Use the sub-chunk intra kernel (more stable at boundaries).
        lower_bound:        If set, use sigmoid gating; else softplus gating.
        use_gate_in_kernel: If True, fuse A_log / dt_bias into the gate cumsum.
        A_log:              Per-head log-scale ``[HV]`` (needed when use_gate_in_kernel).
        dt_bias:            Per-head dt bias ``[HV * K]`` (optional).
        disable_recompute:  If True, store QG/KG/W/U for reuse (not freed early).
        use_qk_l2norm_in_kernel: If True, apply L2 normalization to q and k.
        use_beta_sigmoid_in_kernel: If True, apply sigmoid to beta.
        state_v_first:      Store the recurrent state V-first (``[V, K]``) instead
                            of the default ``[K, V]``. Matches fla's option of the
                            same name.

    Returns:
        (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg)
          o           ``[B, T, HV, V]``
          final_state ``[N, HV, K, V]`` (or ``[N, HV, V, K]``) or None
          g_cumsum    ``[B, T, HV, K]`` (gate in log2 space)
          Aqk         ``[B, T, HV, BT]``
          Akk         ``[B, T, HV, BT]``
          w, u, qg, kg or None depending on disable_recompute
    """
    # ------------------------------------------------------------------
    # Fast path — two-kernel FlashKDA split
    # ------------------------------------------------------------------
    # Returns None for every intermediate: the fused kernels never materialize
    # g_cumsum / Aqk / Akk / w / u, so this can only serve callers that discard
    # them, which is why `disable_recompute` gates it too.
    #
    # The dispatch and an unset `chunk_size` are decided together because the
    # only reason to prefer 32 is that it is this path's entry ticket: inside
    # the default pipeline 64 is faster at every shape measured, so resolving
    # the two apart risks landing on the default pipeline at 32, the slowest
    # combination of the three.
    use_flash_kda = (
        CHUNK_DELTA_ATTN_USE_FLASH_KDA
        and not disable_recompute
        and flash_kda_supported(
            q=q,
            v=v,
            chunk_size=FLASH_KDA_CHUNK if chunk_size is None else chunk_size,
            safe_gate=safe_gate,
            use_gate_in_kernel=use_gate_in_kernel,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            lower_bound=lower_bound,
            A_log=A_log,
        )
    )
    if chunk_size is None:
        chunk_size = FLASH_KDA_CHUNK if use_flash_kda else _DEFAULT_CHUNK_SIZE

    if chunk_size not in (32, 64):
        raise ValueError(
            f"`chunk_size` must be either 32 or 64 for chunk_delta_attn, got {chunk_size}."
        )

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    if use_flash_kda:
        o, final_state = flash_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            lower_bound=lower_bound,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        return o, final_state, None, None, None, None, None, None, None

    # ------------------------------------------------------------------
    # Step 0 — Optional QK L2 normalization (matches FLA API)
    # ------------------------------------------------------------------
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    if use_beta_sigmoid_in_kernel:
        beta = beta_sigmoid_fwd(beta)

    # ------------------------------------------------------------------
    # Step 1 — Gate cumsum
    # ------------------------------------------------------------------
    if use_gate_in_kernel:
        assert A_log is not None, "A_log required when use_gate_in_kernel=True"
        g_cumsum = chunk_gate_cumsum(
            g=g,
            A_log=A_log,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            dt_bias=dt_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
            chunk_local_cumsum,
        )

        g_cumsum = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    # ------------------------------------------------------------------
    # Step 2 — Intra-chunk (sub_chunk + inter_solve + recompute_w_u)
    # ------------------------------------------------------------------
    w, u, qg, kg, Aqk, Akk = chunk_delta_attn_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g_cumsum,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
    )

    # ------------------------------------------------------------------
    # Step 3 — Inter-chunk hidden state update
    # ------------------------------------------------------------------
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        transpose_state=state_v_first,
        use_exp2=True,
    )

    # ------------------------------------------------------------------
    # Step 4 — Output
    # ------------------------------------------------------------------
    o = chunk_gla_fwd_o(
        q=q,
        v=v_new,
        g=g_cumsum,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )

    if not disable_recompute:
        w, u, qg, kg, v_new = None, None, None, None, None
        h = None

    return o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg
