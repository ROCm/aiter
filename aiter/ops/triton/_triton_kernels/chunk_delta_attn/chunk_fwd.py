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

import importlib
import os

import torch
import triton

from ..gated_delta_rule.prefill.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .gate import beta_sigmoid_fwd
from .gla_output import chunk_gla_fwd_o
from .intra_attn import chunk_delta_attn_fwd_intra, chunk_delta_attn_fwd_intra_opt
from .utils import (
    RCP_LN2,
    chunk_gate_cumsum,
    l2norm_fwd,
    prepare_chunk_indices,
)
from .wy_fast import recompute_w_u_fwd

# ---------------------------------------------------------------------------
# Gluon kernel imports (AMD CDNA4 only, graceful fallback)
# ---------------------------------------------------------------------------
_IS_AMD_CDNA4: bool = torch.cuda.is_available() and "gfx95" in getattr(
    torch.cuda.get_device_properties(0), "gcnArchName", ""
)


def _import_gluon_kernels(module: str, *names: str):
    """Import Gluon kernels by name, yielding None for each when unavailable.

    Unavailable covers a non-CDNA4 device, a Triton build without Gluon (the
    kernel module fails to import) and a kernel that has been renamed/removed.
    """
    if not _IS_AMD_CDNA4:
        return (None,) * len(names)
    try:
        mod = importlib.import_module(
            f"aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.{module}"
        )
        return tuple(getattr(mod, name) for name in names)
    except (ImportError, AttributeError):
        return (None,) * len(names)


(_gluon_o_kernel,) = _import_gluon_kernels("gla_output", "chunk_gla_fwd_kernel_o_gluon")
_gluon_wu_smallh_kernel, _gluon_wu_persist_kernel = _import_gluon_kernels(
    "wy_fast",
    "recompute_w_u_fwd_kda_kernel_gluon_small_h",
    "recompute_w_u_fwd_kda_kernel_persistent_gluon",
)
(_gluon_dh_kernel,) = _import_gluon_kernels(
    "chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64_gluon"
)


# ---------------------------------------------------------------------------
# Gluon dispatch helpers — each returns the result or None on fallback
# ---------------------------------------------------------------------------


def _gluon_gla_fwd_o(q, v, g, A, h, scale, chunk_size):
    """Gluon dispatch for gla_o.  Returns None when conditions are not met."""
    B, T, H, K = q.shape
    HV, V, BT = v.shape[2], v.shape[-1], chunk_size
    if not (K == 128 and V == 128 and BT == 64):
        return None
    NT = triton.cdiv(T, BT)
    o = torch.zeros_like(v)
    BK_gluon, BV_gluon = 64, 128
    if HV >= 64:
        bh = B * HV
        target_blocks = max(1, -(-256 // bh))
        npb = max(1, NT // target_blocks)
        while npb > 1 and NT % npb != 0:
            npb -= 1
        BLOCK_T = npb * BT
        BUFFERED = 2
        load_cache, store_cache = ".cg", ".cg"
    else:
        BLOCK_T = BT
        BUFFERED = 1
        load_cache, store_cache = ".cs", ".cs"
    if T % BLOCK_T != 0:
        return None
    grid_g = (T // BLOCK_T, B * HV, triton.cdiv(V, BV_gluon))
    _gluon_o_kernel[grid_g](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        BLOCK_T=BLOCK_T,
        BK=BK_gluon,
        BV=BV_gluon,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
        BUFFERED=BUFFERED,
        LOAD_CACHE=load_cache,
        STORE_CACHE=store_cache,
        num_warps=4,
    )
    return o


def _gluon_recompute_w_u(k, v, beta, A, gk):
    """Gluon dispatch for recompute_w_u.  Returns None when conditions are not met."""
    B, T, H, K = k.shape
    V, HV = v.shape[-1], v.shape[2]
    BT = A.shape[-1]
    if not (K == 128 and V == 128 and BT == 64):
        return None
    NT = triton.cdiv(T, BT)
    w = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    u = torch.empty_like(v)
    kg = torch.empty(B, T, HV, K, device=k.device, dtype=k.dtype)
    if HV >= 64 and T % 2048 == 0:
        BLOCK_T = 2048
        grid_g = (T // BLOCK_T, B * HV)
        _gluon_wu_persist_kernel[grid_g](
            k=k,
            kg=kg,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            gk=gk,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            BLOCK_T=BLOCK_T,
            LOAD_CACHE=".cs",
            STORE_CACHE=".cs",
            num_warps=4,
        )
        return w, u, None, kg
    if T % BT == 0:
        grid_g = (NT, B * HV)
        _gluon_wu_smallh_kernel[grid_g](
            k=k,
            kg=kg,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            gk=gk,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=BT,
            LOAD_CACHE=".cs",
            STORE_CACHE=".cs",
            num_warps=4,
        )
        return w, u, None, kg
    return None


def _gluon_delta_h(k, w, u, gk, chunk_size, output_final_state):
    """Gluon dispatch for chunk_delta_h.  Returns None when conditions are not met."""
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size
    if not (8 <= H <= 64 and K == 128 and not output_final_state):
        return None
    NT = triton.cdiv(T, BT)
    h = k.new_empty(B, NT, H, K, V)
    v_new = torch.empty_like(u)
    final_state = None
    BV_gluon = 32
    grid_g = (B * H, triton.cdiv(V, BV_gluon))
    _gluon_dh_kernel[grid_g](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gk,
        h=h,
        h0=None,
        ht=None,
        cu_seqlens=None,
        chunk_offsets=None,
        T=T,
        H=H,
        HV=H,
        K=K,
        V=V,
        BT=BT,
        BV=BV_gluon,
        USE_G=False,
        USE_GK=True,
        USE_INITIAL_STATE=False,
        STORE_FINAL_STATE=False,
        SAVE_NEW_VALUE=True,
        USE_EXP2=True,
        TRANSPOSE_STATE=False,
        IS_VARLEN=False,
        LOAD_CACHE="",
        STORE_CACHE="",
        num_warps=2,
        num_stages=2,
        waves_per_eu=1,
    )
    return h, v_new, final_state


# ---------------------------------------------------------------------------
# Top-level forward
# ---------------------------------------------------------------------------


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
    chunk_size: int = 64,
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
        chunk_size:         Chunk size BT, either 32 or 64 (default 64).
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
    if chunk_size not in (32, 64):
        raise ValueError(
            f"`chunk_size` must be either 32 or 64 for chunk_delta_attn, got {chunk_size}."
        )

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    # ------------------------------------------------------------------
    # Step 0 — Optional QK L2 normalization (matches FLA API)
    # ------------------------------------------------------------------
    # Centralized dispatch flags — read once, decide all branches here
    use_gluon = os.environ.get("DA_USE_GLUON", "0") == "1"
    triton_opt = os.environ.get("DA_TRITON_OPT", "1") == "1"

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q, use_persistent=triton_opt)
        k, _ = l2norm_fwd(k, use_persistent=triton_opt)

    if use_beta_sigmoid_in_kernel:
        beta = beta_sigmoid_fwd(beta)
    # Gluon requires no varlen, no initial_state for some kernels
    gluon_eligible = use_gluon and cu_seqlens is None

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
            cache_input=".cs" if triton_opt else "",
            cache_output=".wt" if triton_opt else "",
        )
    else:
        from ..gated_delta_rule.utils import chunk_local_cumsum

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
    _intra_fn = (
        chunk_delta_attn_fwd_intra_opt if triton_opt else chunk_delta_attn_fwd_intra
    )
    # Skip Triton recompute_w_u when Gluon WU will handle it
    _skip_wu = (
        gluon_eligible and _gluon_wu_smallh_kernel is not None and not disable_recompute
    )
    w, u, qg, kg, Aqk, Akk = _intra_fn(
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
        skip_recompute_wu=_skip_wu,
    )

    # Gluon recompute_w_u (or Triton fallback if gluon returns None)
    if _skip_wu:
        gluon_wu = _gluon_recompute_w_u(k, v, beta, Akk, g_cumsum)
        if gluon_wu is not None:
            w, u, qg, kg = gluon_wu
        else:
            # Gluon WU conditions not met, fall back to Triton
            w, u, qg, kg = recompute_w_u_fwd(
                k=k,
                v=v,
                beta=beta,
                A=Akk,
                gk=g_cumsum,
                q=q if disable_recompute else None,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )

    # ------------------------------------------------------------------
    # Step 3 — Inter-chunk hidden state update
    # ------------------------------------------------------------------
    h_result = None
    if gluon_eligible and _gluon_dh_kernel is not None and initial_state is None:
        h_result = _gluon_delta_h(
            kg,
            w,
            u,
            g_cumsum,
            chunk_size,
            output_final_state,
        )
    if h_result is None:
        h_result = chunk_gated_delta_rule_fwd_h(
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
        )
    h, v_new, final_state = h_result

    # ------------------------------------------------------------------
    # Step 4 — Output
    # ------------------------------------------------------------------
    o = None
    if gluon_eligible and _gluon_o_kernel is not None:
        o = _gluon_gla_fwd_o(q, v_new, g_cumsum, Aqk, h, scale, chunk_size)
    if o is None:
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
