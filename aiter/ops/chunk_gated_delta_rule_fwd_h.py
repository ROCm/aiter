# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Python wrapper for the HIP overlap_2 BV kernel for chunk_gated_delta_rule_fwd_h.

This kernel is specialized for K=128, V=128, bf16 inputs, and supports
adaptive BV tile sizes (16, 32, 64) for AMD CDNA3 GPUs.
"""

from typing import Optional

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_chunk_gdr_fwd_h"


@compile_ops(MD_NAME)
def chunk_gated_delta_rule_fwd_h_hip(
    k: Tensor,
    w: Tensor,
    u: Tensor,
    g: Tensor,
    initial_state: Tensor,
    cu_seqlens: Tensor,
    chunk_offsets: Tensor,
    selected_bv: int,
    has_initial_state: bool,
    output_final_state: bool,
    save_new_value: bool,
) -> list[Tensor]:
    ...


def chunk_gated_delta_rule_fwd_h_hip_fn(
    k: Tensor,
    w: Tensor,
    u: Tensor,
    g: Tensor,
    initial_state: Optional[Tensor] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[Tensor] = None,
    selected_bv: int = 64,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    HIP overlap_2 BV kernel for chunk_gated_delta_rule_fwd_h.

    Drop-in replacement for the Triton chunk_gated_delta_rule_fwd_h_opt
    when K=128, V=128, bf16 on AMD GPUs.

    Triton pipeline layout -> HIP kernel layout:
      k: [B, T, Hg, K] -> [1, Hg, B*T, K]
      w: [B, H, T, K]  -> [1, H, B*T, K]
      u: [B, H, T, V]  -> [1, H, B*T, V]
      g: [B, T, H]     -> [B*T, H]
    """
    assert chunk_size == 64, "HIP overlap_2 kernel is specialized for chunk_size=64."
    assert k.shape[-1] == 128, "HIP overlap_2 kernel is specialized for K=128."
    assert u.shape[-1] == 128, "HIP overlap_2 kernel is specialized for V=128."

    from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
        prepare_chunk_offsets,
    )

    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = B * T
    is_varlen = cu_seqlens is not None

    _has_initial_state = initial_state is not None

    if not is_varlen:
        cu_seqlens_int32 = torch.arange(
            0, T_flat + 1, T, dtype=torch.int32, device=k.device
        )
    else:
        cu_seqlens_int32 = cu_seqlens.to(torch.int32)

    N = cu_seqlens_int32.shape[0] - 1
    _chunk_offsets = prepare_chunk_offsets(cu_seqlens_int32.to(torch.int64), chunk_size)
    chunk_offsets_int32 = _chunk_offsets.to(torch.int32)

    if is_varlen:
        assert B == 1, "Varlen mode expects B=1 (flattened input)."
        k_hip = k.permute(2, 0, 1, 3).reshape(1, Hg, T_flat, K).contiguous()
        w_hip = w.contiguous()
        u_hip = u.contiguous()
        g_hip = g.reshape(T_flat, H).contiguous()
    else:
        # k: [B, T, Hg, K] -> [Hg, B, T, K] -> [1, Hg, B*T, K]
        k_hip = k.permute(2, 0, 1, 3).reshape(1, Hg, T_flat, K).contiguous()
        # w: [B, H, T, K] -> [H, B, T, K] -> [1, H, B*T, K]
        w_hip = w.permute(1, 0, 2, 3).reshape(1, H, T_flat, K).contiguous()
        # u: [B, H, T, V] -> [H, B, T, V] -> [1, H, B*T, V]
        u_hip = u.permute(1, 0, 2, 3).reshape(1, H, T_flat, V).contiguous()
        # g: [B, T, H] -> [B*T, H]
        g_hip = g.reshape(T_flat, H).contiguous()

    _initial_state = (
        initial_state.to(torch.float32)
        if initial_state is not None
        else torch.empty(0, device=k.device, dtype=torch.float32)
    )

    import triton

    NT = triton.cdiv(T, chunk_size)

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_hip(
        k_hip, w_hip, u_hip, g_hip,
        _initial_state,
        cu_seqlens_int32,
        chunk_offsets_int32,
        selected_bv,
        _has_initial_state,
        output_final_state,
        save_new_value,
    )

    if not is_varlen:
        # h: [1, total_chunks, H, K, V] -> [B, NT, H, K, V]
        h = h.view(B, NT, H, K, V)
        # v_new: [1, H, B*T, V] -> [H, B, T, V] -> [B, H, T, V]
        if save_new_value:
            v_new = v_new.view(H, B, T, V).permute(1, 0, 2, 3).contiguous()
        else:
            v_new = None
    else:
        if not save_new_value:
            v_new = None

    if not output_final_state:
        final_state = None

    return h, v_new, final_state
