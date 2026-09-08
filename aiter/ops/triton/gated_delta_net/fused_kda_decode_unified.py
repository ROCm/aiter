# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified fused KDA decode: conv1d + recurrence + gated RMSNorm.

Python wrapper for the unified kernel. Mode is auto-selected from args:
  - state_indices only             → normal decode
  - state_indices + num_accepted   → DSpark spec decode
  - slot_idx + buf_k/u/g           → ReplaySSM
  - slot_idx + buf_k/u/g + conv_state_indices → DSpark + ReplaySSM
"""

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_conv_recurrent_norm_unified import (
    _fused_kda_decode_unified_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def fused_kda_decode_unified(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    out_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    head_dim: int,
    num_local_heads: int,
    lower_bound: float,
    # Non-replay: state indices
    state_indices: torch.Tensor | None = None,
    # Spec decode
    num_accepted_tokens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    # ReplaySSM
    buf_k: torch.Tensor | None = None,
    buf_u: torch.Tensor | None = None,
    buf_g: torch.Tensor | None = None,
    write_pos: torch.Tensor | None = None,
    slot_idx: torch.Tensor | None = None,
    max_query_len: int = 1,
) -> torch.Tensor:
    """Unified fused KDA decode.

    Fuses conv1d + delta-rule recurrence + gated RMSNorm into one kernel.
    Supports four modes via optional arguments:

    Normal decode (default):
        Pass ``state_indices`` (1D [B] int32).

    DSpark spec decode:
        Pass ``state_indices`` (2D [B, num_spec+1]),
        ``num_accepted_tokens`` ([B] int32), ``conv_state_indices`` ([B] int32).

    ReplaySSM:
        Pass ``buf_k/buf_u/buf_g`` (ring buffers), ``write_pos`` ([N_slots]),
        ``slot_idx`` ([B]), ``max_query_len`` (for flush threshold).

    DSpark + ReplaySSM:
        Pass all of the above.
    """
    T = mixed_qkv.shape[0]
    K = V = head_dim
    H = num_local_heads
    lp = H * K
    batch = cu_seqlens.shape[0] - 1
    device = mixed_qkv.device

    use_replay = buf_k is not None
    is_spec = (num_accepted_tokens is not None) or (
        use_replay and conv_state_indices is not None
    )

    out = torch.empty(T, lp, dtype=torch.bfloat16, device=device)

    # Conv weight strides
    if conv_weight.dim() == 3:
        W_val = conv_weight.shape[1]
        s_cw_group = conv_weight.stride(0)
        s_cw_width = conv_weight.stride(1)
        s_cw_ch = conv_weight.stride(2)
    else:
        W_val = conv_weight.shape[-1]
        s_cw_group = lp * conv_weight.stride(0)
        s_cw_width = conv_weight.stride(1)
        s_cw_ch = conv_weight.stride(0)

    STATE_LEN = conv_state.shape[2] if is_spec else W_val - 1
    s_beta_tok = beta.stride(1) if beta.dim() == 3 else beta.stride(0)
    s_og_tok = out_gate.stride(0)

    # State indices strides
    if state_indices is not None and state_indices.ndim == 2:
        s_idx_seq = state_indices.stride(0)
        s_idx_tok = state_indices.stride(1)
    elif state_indices is not None:
        s_idx_seq = state_indices.stride(0)
        s_idx_tok = 1
    else:
        s_idx_seq = 1
        s_idx_tok = 1

    # Replay params
    if use_replay:
        CAP = buf_k.shape[2]
        BH = max(16, triton.next_power_of_2(CAP - max_query_len))
        s_bk_slot, s_bk_hv, s_bk_pos = buf_k.stride(0), buf_k.stride(1), buf_k.stride(2)
        s_bu_slot, s_bu_hv, s_bu_pos = buf_u.stride(0), buf_u.stride(1), buf_u.stride(2)
        s_bg_slot, s_bg_hv, s_bg_pos = buf_g.stride(0), buf_g.stride(1), buf_g.stride(2)
    else:
        CAP, BH = 1, 16
        s_bk_slot = s_bk_hv = s_bk_pos = 1
        s_bu_slot = s_bu_hv = s_bu_pos = 1
        s_bg_slot = s_bg_hv = s_bg_pos = 1

    # Dummy tensors for unused pointers
    dummy = torch.empty(1, device=device)

    grid = (batch, H)
    _fused_kda_decode_unified_kernel[grid](
        mixed_qkv,
        conv_weight,
        conv_state,
        gate,
        beta,
        A_log,
        dt_bias,
        state,
        state_indices if state_indices is not None else dummy,
        cu_seqlens,
        norm_weight,
        out_gate,
        out,
        num_accepted_tokens if num_accepted_tokens is not None else dummy,
        conv_state_indices if conv_state_indices is not None else dummy,
        buf_k if buf_k is not None else dummy,
        buf_u if buf_u is not None else dummy,
        buf_g if buf_g is not None else dummy,
        write_pos if write_pos is not None else dummy,
        slot_idx if slot_idx is not None else dummy,
        lower_bound,
        norm_eps,
        K**-0.5,
        T,
        H=H,
        K=K,
        V=V,
        W=W_val,
        STATE_LEN=STATE_LEN,
        CAP=CAP,
        T_MAX=max_query_len,
        BH=BH,
        USE_REPLAY=use_replay,
        IS_SPEC=is_spec,
        stride_x_tok=mixed_qkv.stride(0),
        stride_cw_group=s_cw_group,
        stride_cw_width=s_cw_width,
        stride_cw_ch=s_cw_ch,
        stride_cs_slot=conv_state.stride(0),
        stride_cs_dim=conv_state.stride(1),
        stride_cs_pos=conv_state.stride(2),
        stride_beta_tok=s_beta_tok,
        stride_og_tok=s_og_tok,
        stride_state_slot=state.stride(0),
        stride_indices_seq=s_idx_seq,
        stride_indices_tok=s_idx_tok,
        stride_bufk_slot=s_bk_slot,
        stride_bufk_hv=s_bk_hv,
        stride_bufk_pos=s_bk_pos,
        stride_bufu_slot=s_bu_slot,
        stride_bufu_hv=s_bu_hv,
        stride_bufu_pos=s_bu_pos,
        stride_bufg_slot=s_bg_slot,
        stride_bufg_hv=s_bg_hv,
        stride_bufg_pos=s_bg_pos,
        num_warps=2 if get_arch() == "gfx942" else 4,
    )
    return out
