# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode: conv1d + delta-rule recurrence + gated RMSNorm.

Single Triton kernel launch. Fuses three separate operations into one,
eliminating q/k/v intermediate HBM traffic and kernel launch overhead.
"""

import torch

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_conv_recurrent_norm import (
    fused_conv_recurrent_norm_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def fused_kda_decode(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    out_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    head_dim: int,
    num_local_heads: int,
    lower_bound: float,
) -> torch.Tensor:
    """Fused KDA decode: conv1d + recurrence + gated RMSNorm.

    Args:
        mixed_qkv: [B, 3*lp] bf16, may be strided (sliced from in_proj output).
        conv_state: [N, 3*lp, W-1] bf16, transposed view of conv cache.
        conv_weight: [3*lp, W] or [3, W, lp], fp32 conv1d weights.
        gate: [1, B, H, K] bf16, KDA decay gate (raw logits).
        beta: [1, B, H] bf16, write strength (raw logits), may be strided.
        out_gate: [B, H*K] bf16, output gate for RMSNorm, may be strided.
        A_log: [H] fp32, per-head decay parameter.
        dt_bias: [H*K] fp32, per-channel bias.
        ssm_state: [N, H, V, K] fp32, delta-rule state matrices.
        ssm_state_indices: [B] int32, batch-to-slot mapping.
        cu_seqlens: [B+1] int64, cumulative sequence lengths.
        norm_weight: [K] fp32, RMSNorm weight.
        norm_eps: float, RMSNorm epsilon.
        head_dim: int, K = V = head_dim.
        num_local_heads: int, H = num_local_heads.
        lower_bound: float, KDA gate lower bound (typically -5.0).

    Returns:
        out: [B, H*K] bf16, final output after RMSNorm + gate.
    """
    T = mixed_qkv.shape[0]
    K = V = head_dim
    H = num_local_heads
    lp = H * K
    batch = cu_seqlens.shape[0] - 1

    out = torch.empty(T, lp, dtype=torch.bfloat16, device=mixed_qkv.device)

    # Conv weight strides: support [3*lp, W] and [3, W, lp]
    if conv_weight.dim() == 3:
        W = conv_weight.shape[1]
        stride_cw_group = conv_weight.stride(0)
        stride_cw_width = conv_weight.stride(1)
        stride_cw_ch = conv_weight.stride(2)
    else:
        W = conv_weight.shape[-1]
        stride_cw_group = lp * conv_weight.stride(0)
        stride_cw_width = conv_weight.stride(1)
        stride_cw_ch = conv_weight.stride(0)

    stride_beta_tok = beta.stride(1) if beta.dim() == 3 else beta.stride(0)
    stride_og_tok = out_gate.stride(0)

    grid = (batch, H)
    fused_conv_recurrent_norm_kernel[grid](
        mixed_qkv,
        conv_weight,
        conv_state,
        gate,
        beta,
        A_log,
        dt_bias,
        ssm_state,
        ssm_state_indices,
        cu_seqlens,
        norm_weight,
        out_gate,
        out,
        lower_bound,
        norm_eps,
        K**-0.5,
        T,
        H=H,
        K=K,
        V=V,
        W=W,
        stride_x_tok=mixed_qkv.stride(0),
        stride_cw_group=stride_cw_group,
        stride_cw_width=stride_cw_width,
        stride_cw_ch=stride_cw_ch,
        stride_cs_slot=conv_state.stride(0),
        stride_cs_dim=conv_state.stride(1),
        stride_cs_pos=conv_state.stride(2),
        stride_beta_tok=stride_beta_tok,
        stride_og_tok=stride_og_tok,
        stride_ssm_slot=ssm_state.stride(0),
        num_warps=2 if get_arch() == "gfx942" else 4,
    )
    return out
