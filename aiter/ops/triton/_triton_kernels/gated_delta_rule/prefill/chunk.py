# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Chunk-based gated delta rule forward computation.

This module implements the chunk-based parallel computation for the gated delta rule.
Note: Only forward pass is implemented. Backward pass is not supported in aiter.
"""

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from ..utils import (
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    recompute_w_u_fwd,
    solve_tril,
)


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
