# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Chunk-based gated delta rule forward computation.

This module implements the chunk-based parallel computation for the gated delta rule.
Note: Only forward pass is implemented. Backward pass is not supported in aiter.
"""

from collections.abc import Sequence

import torch

from ..utils import (
    GatedDeltaRulePrefillMetadata,
    build_gated_delta_rule_prefill_metadata,
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    recompute_w_u_fwd,
    solve_tril,
)
from .chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h_opt,
    chunk_gated_delta_rule_fwd_h_opt_vk,
)
from .chunk_o import chunk_fwd_o, chunk_fwd_o_opt, chunk_fwd_o_opt_vk
from .fused_cumsum_kkt import fused_chunk_local_cumsum_scaled_dot_kkt_fwd
from .fused_solve_tril_recompute import fused_solve_tril_recompute_w_u

_SUPPORTED_GFX12_ARCHS = frozenset({"gfx1200", "gfx1201"})


def _is_unsupported_gfx12_runtime(device: torch.device) -> bool:
    try:
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", "")
        arch = arch.split(":")[0] if arch else ""
        return arch.startswith("gfx12") and arch not in _SUPPORTED_GFX12_ARCHS
    except Exception:  # noqa: BLE001
        return False


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """
    Chunk gated delta rule forward computation (Forward only).

    This function implements chunk-based parallel computation for the gated delta rule,
    combining all necessary steps for efficient sequence processing.

    Note: This implementation only supports forward pass. Backward pass is not available.

    Args:
        q: Query tensor of shape [B, T, H, K]
        k: Key tensor of shape [B, T, H, K]
        v: Value tensor of shape [B, T, H, V]
        g: Gate tensor (in log space) of shape [B, T, H]
        beta: Beta parameter tensor of shape [B, T, H]
        scale: Scaling factor for queries
        initial_state: Initial hidden state of shape [N, H, K, V]
        output_final_state: Whether to output the final state
        cu_seqlens: Cumulative sequence lengths for variable-length inputs (optional) [N+1]

    Returns:
        tuple: (g, o, A, final_state) where:
            - g: Cumulative gate values [B, T, H]
            - o: Output tensor [B, T, H, V]
            - A: WY representation matrix
            - final_state: Final hidden state [N, H, K, V] if output_final_state=True, else None
    """
    # Step 1: Compute local cumulative sum of gates
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    # Step 2: Compute WY representation
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    # Step 3: Compute hidden states
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Step 4: Compute output
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    return g, o, A, final_state


def chunk_gated_delta_rule_fwd_opt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
):
    """
    Optimized chunk gated delta rule forward computation (Forward only).

    This function implements an optimized chunk-based parallel computation for
    the gated delta rule, using fused kernels and transposed intermediate layouts
    to reduce global memory round-trips.

    Note: This implementation only supports forward pass. Backward pass is not available.

    Args:
        q: Query tensor of shape [B, T, Hg, K]
        k: Key tensor of shape [B, T, Hg, K]
        v: Value tensor of shape [B, T, H, V]
        g: Gate tensor (in log space, pre-cumsum) of shape [B, T, H]
        beta: Beta parameter tensor of shape [B, T, H]
        scale: Scaling factor for queries
        initial_state: Optional initial hidden state of shape [N, H, K, V]
        output_final_state: Whether to output the final state
        cu_seqlens: Cumulative sequence lengths for variable-length inputs (optional) [N+1]

    Returns:
        tuple: (g_cumsum, o, final_state) where:
            - g_cumsum: Cumulative gate values [B, H, T]
            - o: Output tensor [B, T, H, V]
            - final_state: Final hidden state [N, H, K, V] if output_final_state=True, else None
    """
    # Step 1: Compute fused local cumulative sum of gates and KKT
    g_cumsum, A_raw = fused_chunk_local_cumsum_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        use_exp2=False,
    )

    # Step 2: Compute fused triangular solve and recompute w, u
    # w, u are already in [B, H, T, K/V] head-major contiguous layout
    w, u = fused_solve_tril_recompute_w_u(
        A_raw=A_raw,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        use_exp2=False,
    )

    # k5_opt / k6_opt index g with token-major [B, T, H] strides, but the fused
    # k12 returns g_cumsum head-major [B, H, T]. Convert to token-major here.
    g_cumsum_tok = g_cumsum.transpose(1, 2).contiguous()

    # Step 3: Compute hidden states
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_opt(
        k=k,
        w=w,
        u=u,
        g=g_cumsum_tok,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Step 4: Compute output
    o = chunk_fwd_o_opt(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum_tok,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    return g_cumsum, o, final_state


def chunk_gated_delta_rule_fwd_opt_vk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    use_chunk_hip: bool = False,
    use_chunk_flydsl: bool = False,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    o: torch.Tensor | None = None,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    seq_lens_cpu: Sequence[int] | None = None,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    initial_state_indices: torch.Tensor | None = None,
    inplace_final_state: bool | None = None,
    snapshot_dtype: torch.dtype | None = None,
):
    """
    Optimized chunk gated delta rule forward with h layout [V, K].

    Uses the same fused kernels as opt, but with transposed
    h layout [V, K] instead of [K, V].

    When use_chunk_hip=True, hidden state computation uses a HIP kernel
    instead of Triton. When use_chunk_flydsl=True, hidden state computation
    uses the FlyDSL kernel. The two flags are mutually exclusive.

    Args:
        q: [B, T, Hg, K]
        k: [B, T, Hg, K]
        v: [B, T, H, V]
        g: [B, T, H] — raw gate (pre-cumsum)
        beta: [B, T, H]
        scale: float
        initial_state: optional [N, H, V, K] — note transposed h layout
        output_final_state: bool
        cu_seqlens: [N+1] optional
        use_chunk_hip: bool — use HIP kernel for hidden state (K5)
        use_chunk_flydsl: bool — use FlyDSL kernel for hidden state (K5)
        state_dtype: optional initial/final state dtype (`fp32` or `bf16`),
            supported by both the HIP and Triton hidden-state paths
        use_exp2: bool — use exp2 instead of exp for gate computation
        o: optional pre-allocated [B, T, H, V] output buffer (written in
            place by K6). If None, a fresh buffer is allocated.
        num_decodes / num_decode_tokens: skip a leading decode-only prefix in
            the ORIGINAL cu_seqlens (data tensors are expected pre-sliced).
            Threaded into every stage; the cached prologue helpers
            (prepare_chunk_indices / prepare_chunk_offsets /
            prepare_rebased_cu_seqlens) key on the original cu_seqlens identity,
            so chunk-index / offset builds stay cache-warm across forwards
            (no per-forward .tolist() D2H).
        seq_lens_cpu: Host-resident original sequence lengths. Builds one
            shared K1--K6 schedule without device readback.
        prefill_metadata: Prebuilt reusable host/device schedule. This is the
            preferred path when multiple layers process the same batch.
        initial_state_indices: Optional ``[N]`` state-pool slot indices. When
            provided, K5 gathers from and writes back to ``initial_state`` in
            place. Supported by the HIP and Triton VK K5 paths only.
        inplace_final_state: Controls in-place K5 state-pool write-back. It
            defaults to ``True`` when ``initial_state_indices`` is provided.
        snapshot_dtype: optional temporary chunk snapshot dtype (`fp32` or
            `bf16`). Defaults to `k.dtype` and is independent of state_dtype.

    Returns:
        tuple: (g_cumsum, o, final_state) where:
            - g_cumsum: [B, H, T]
            - o: [B, T, H, V]
            - final_state: [N, H, V, K] if output_final_state=True, else None
    """
    if use_chunk_hip and use_chunk_flydsl:
        raise ValueError(
            "use_chunk_hip and use_chunk_flydsl are mutually exclusive; "
            "set at most one."
        )
    if use_chunk_flydsl and (
        initial_state_indices is not None or inplace_final_state is True
    ):
        raise ValueError(
            "Indexed state pools and in-place final-state write-back are not "
            "supported by the FlyDSL K5 path."
        )
    if cu_seqlens is None:
        if seq_lens_cpu is not None or prefill_metadata is not None:
            raise ValueError(
                "`seq_lens_cpu` and `prefill_metadata` require `cu_seqlens`."
            )
    elif prefill_metadata is None and seq_lens_cpu is not None:
        prefill_metadata = build_gated_delta_rule_prefill_metadata(
            seq_lens_cpu,
            cu_seqlens=cu_seqlens,
            chunk_size=64,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )

    if use_chunk_hip and (
        _is_unsupported_gfx12_runtime(q.device)
        or (num_decodes > 0 and prefill_metadata is None)
    ):
        use_chunk_hip = False
    if use_chunk_flydsl:
        if _is_unsupported_gfx12_runtime(q.device):
            use_chunk_flydsl = False
        elif k.dtype != torch.bfloat16 or k.shape[-1] != 128 or v.shape[-1] != 128:
            raise ValueError(
                "use_chunk_flydsl requires bfloat16 inputs with K=128 and V=128; "
                f"got dtype={k.dtype}, K={k.shape[-1]}, V={v.shape[-1]}."
            )

    g_cumsum, A_raw = fused_chunk_local_cumsum_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )

    w, u = fused_solve_tril_recompute_w_u(
        A_raw=A_raw,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )

    if use_chunk_hip:
        from aiter.ops.chunk_gated_delta_rule_fwd_h import (
            chunk_gated_delta_rule_fwd_h_hip_fn,
        )

        h, v_new, final_state = chunk_gated_delta_rule_fwd_h_hip_fn(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_dtype=state_dtype,
            snapshot_dtype=snapshot_dtype,
            use_exp2=use_exp2,
            g_head_major=True,
            prefill_metadata=prefill_metadata,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            initial_state_indices=initial_state_indices,
            inplace_final_state=inplace_final_state,
        )
    elif use_chunk_flydsl:
        if snapshot_dtype is not None and snapshot_dtype != k.dtype:
            raise ValueError(
                "FlyDSL K5 does not support overriding `snapshot_dtype`; "
                "omit it or pass `k.dtype`."
            )
        # ``g_cumsum`` from K1+K2 is 3-D head-major [B, H, T] (same tensor the
        # HIP path above passes with g_head_major=True). The FlyDSL wrapper now
        # mirrors the HIP g-layout contract (default token-major), so pass
        # g_head_major=True explicitly to select the head-major layout. The
        # wrapper accepts ``use_exp2`` as a kwarg and pre-scales ``gk``
        # internally.
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip,
        )

        h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            state_dtype=state_dtype,
            use_exp2=use_exp2,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            g_head_major=True,
            prefill_metadata=prefill_metadata,
        )
    else:
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h_opt_vk(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_exp2=use_exp2,
            state_dtype=state_dtype,
            snapshot_dtype=snapshot_dtype,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            initial_state_indices=initial_state_indices,
            inplace_final_state=inplace_final_state,
            prefill_metadata=prefill_metadata,
        )

    if o is None:
        # Output matches v's [B, T, H, V] layout.
        o = v.new_empty(v.shape)

    o = chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        o=o,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )

    return g_cumsum, o, final_state
