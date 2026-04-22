# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Linear Attention Prefill APIs (chunk_gated_delta_h)."""

from __future__ import annotations

import torch

from .kernels.chunk_gated_delta_h import chunk_gated_delta_rule_fwd_h_flydsl

__all__ = [
    "flydsl_gdr_prefill",
]


def flydsl_gdr_prefill(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    wu_contiguous: bool = True,
    BV: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """FlyDSL K5: chunk gated delta rule forward hidden-state recurrence.

    Args:
        k: Key tensor [B, T, Hg, K].
        w: Weight tensor. If wu_contiguous: [B, H, T_flat, K]; else [B, T, H, K].
        u: Value tensor. If wu_contiguous: [B, H, T_flat, V]; else [B, T, H, V].
        g: Cumulative gate tensor [T_total, H] (float32), or None.
        initial_state: Initial hidden state [N, H, V, K] (float32), or None.
        output_final_state: Whether to output the final hidden state.
        chunk_size: Chunk size (default 64).
        save_new_value: Whether to save v_new for downstream kernels.
        cu_seqlens: Cumulative sequence lengths for variable-length batching.
        wu_contiguous: Whether w/u are in contiguous [B, H, T, dim] layout.
        BV: Explicit BV tile size (0 = autotune).

    Returns:
        (h, v_new, final_state) where:
          h: Hidden state snapshots [B, NT, H, V, K] (bf16).
          v_new: Delta-corrected values [B, H, T_flat, V] (bf16), or None.
          final_state: Final hidden state [N, H, V, K] (float32), or None.
    """
    return chunk_gated_delta_rule_fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        wu_contiguous=wu_contiguous,
        BV=BV,
    )
