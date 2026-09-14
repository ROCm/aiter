# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next GDN (gated delta-net) prefill, fused end-to-end at the AITER
Python level.

AITER ships every *piece* of the GDN prefill but no single entry that chains
them across the shared conv/recurrent state pools:

  1. ``causal_conv1d_split_qkv_hip_fn``     -- causal conv1d (silu) that splits
                                               the packed ``[Q|K|V]`` channels,
                                               mutating ``conv_state`` in place.
  2. ``fused_gdn_gating_and_sigmoid``        -- ``g = -exp(A_log)*softplus(a+dt_bias)``
                                               and ``beta = sigmoid(b)``.
  3. ``chunk_gated_delta_rule``              -- the chunked gated delta recurrence
                                               (q/k L2-normed in-kernel), reading
                                               and advancing ``delta_state``.
  4. ``gated_rmsnorm_fp8_group_quant``       -- per-head gated RMSNorm followed by
                                               fp8 group (size-128) quantization.

This module provides the missing fused interface, mirroring the Artemis Gluon
``gdn_prefill_group_fp8_quant`` signature so it is a drop-in for the same call
site and a fair A/B baseline for it:

    (bf16_out, conv_state, delta_state, fp8_out, scales) = gdn_prefill_group_fp8_quant(
        projected_qkvz, projected_ba, conv_state, delta_state,
        cache_indices, cu_seqlens, has_initial_state,
        conv_weight, conv_bias, a_log, dt_bias, norm_weight,
        scale=head_k_dim ** -0.5, eps=norm.eps,
    )

Unlike the Gluon mega-kernel this is *composed* -- several kernel launches with
no cross-kernel fusion -- so it is not expected to match single-kernel perf. Its
value is a correct, Triton-3.7-safe, aiter-native fused entry point (and the
thing SIKL routes to). Semantics match sglang's ``Qwen3GatedDeltaNet`` prefill
path (``srt/layers/attention/linear/gdn_backend.py``).

Constraints (inherited from the tail kernel): ``head_k_dim == head_v_dim == 128``
and group size 128.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter.ops.causal_conv1d_fwd_split_qkv import causal_conv1d_split_qkv_hip_fn
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
from aiter.ops.triton._triton_kernels.gated_delta_rule.fused_qkvzba_split import (
    fused_qkvzba_split_reshape_cat_prefill,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_gdn_gating_prefill import (
    fused_gdn_gating_and_sigmoid,
)
from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_gated_delta_rule

__all__ = ["gdn_prefill_group_fp8_quant", "FP8PrecisionConfig", "FP8_E4M3_FN"]

_HEAD_DIM = 128  # gated_rmsnorm_fp8_group_quant supports only head_dim == 128.
_GROUP_SIZE = 128


@dataclass(frozen=True)
class FP8PrecisionConfig:
    """Mirror of the Artemis fp8 precision descriptor so a caller can pass the
    same object to either backend. Only ``dtype`` steers the aiter tail kernel;
    the group-quant clamp (``group_quant_max``) is applied inside it."""

    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)


def gdn_prefill_group_fp8_quant(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float = 1.0e-6,
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
    return_bf16: bool = True,
    store_state_transposed: bool = False,
) -> tuple[
    torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fused GDN prefill: conv1d -> gating -> chunk gated-delta -> gated RMSNorm
    + fp8 group-quant.

    Args:
        projected_qkvz: ``[T, D_qkvz]`` in-projection output (packed q/k/v/z).
        projected_ba: ``[T, 2*H_v]`` in-projection output (packed b/a).
        conv_state: ``[num_lines, conv_dim, width-1]`` bf16, advanced in place.
        delta_state: ``[num_lines, H_v, K, V]`` fp32 recurrent pool; the rows
            named by ``cache_indices`` are read as the initial state and
            overwritten with the final state.
        cache_indices: ``[N]`` int32 per-sequence pool slots.
        cu_seqlens: ``[N+1]`` int32 cumulative sequence lengths (query_start_loc).
        has_initial_state: ``[N]`` bool; whether each sequence continues a prior
            conv window.
        conv_weight: ``[conv_dim, 4]`` bf16 depthwise conv weights.
        conv_bias: ``[conv_dim]`` bf16 or None.
        a_log, dt_bias: ``[H_v]`` gating parameters.
        norm_weight: ``[128]`` gated-RMSNorm weight.
        scale: query scale for the delta rule (typically ``head_k_dim ** -0.5``).
        eps: RMSNorm epsilon.
        precision_config: fp8 output dtype descriptor.
        return_bf16: if True, also return the bf16 reconstruction of the fp8
            output (dequantized); if False the first tuple slot is None (the
            Gluon kernel emits it for free -- the composed path does not, so
            skipping keeps the A/B fair).
        store_state_transposed: how the advanced recurrent state is written back
            to ``delta_state``. aiter's delta rule is native ``[N, H, K, V]``
            (default, pairs with aiter's GDN decode). The Artemis Gluon kernel
            stores the transpose ``[N, H, V, K]``; set True to match its pool
            convention when this op replaces the Gluon prefill in place (so the
            downstream decode reader sees the layout it expects). The two are
            numerically identical up to the last-two-axis transpose.

    Returns:
        ``(bf16_out, conv_state, delta_state, fp8_out, scales)`` where
        ``fp8_out`` is ``[T, H_v*128]`` and ``scales`` is ``[T, H_v]`` fp32.
    """
    if projected_qkvz.dtype != torch.bfloat16:
        raise TypeError("projected_qkvz must be bfloat16.")
    device = projected_qkvz.device
    seq_len = projected_qkvz.shape[0]

    # --- Infer head geometry from shapes (matches the Gluon kernel, which is
    # likewise specialized purely by signature). head_k == head_v == 128.
    num_v_heads = projected_ba.shape[1] // 2
    conv_dim = conv_weight.shape[0]
    value_dim = num_v_heads * _HEAD_DIM
    key_dim = (conv_dim - value_dim) // 2
    num_k_heads = key_dim // _HEAD_DIM
    if key_dim * 2 + value_dim != conv_dim:
        raise ValueError(
            f"conv_dim={conv_dim} is inconsistent with num_v_heads={num_v_heads} "
            f"(head_dim={_HEAD_DIM}); expected 2*key_dim + value_dim."
        )
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num_v_heads={num_v_heads} must be a multiple of num_k_heads={num_k_heads}."
        )

    # --- 1. Split the packed projections into conv input + z/b/a (fused).
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_prefill(
        projected_qkvz, projected_ba, num_k_heads, num_v_heads, _HEAD_DIM, _HEAD_DIM
    )

    # --- 2. Causal conv1d with silu; splits Q|K|V and advances conv_state.
    # The HIP kernel wants x laid out as [conv_dim, T].
    q, k, v = causal_conv1d_split_qkv_hip_fn(
        mixed_qkv.transpose(0, 1).contiguous(),
        conv_weight,
        conv_bias,
        conv_state,
        cu_seqlens,
        key_dim,
        value_dim,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    # --- Reshape to [B=1, T, H, head] and expand GQA q/k up to the value heads
    # (each value head in a group shares its group's q/k). The delta-rule state
    # pool is sized for H_v, so all of q/k/v/g/beta must speak H_v.
    q = q.view(1, seq_len, num_k_heads, _HEAD_DIM)
    k = k.view(1, seq_len, num_k_heads, _HEAD_DIM)
    v = v.view(1, seq_len, num_v_heads, _HEAD_DIM)
    if num_v_heads != num_k_heads:
        groups = num_v_heads // num_k_heads
        q = q.repeat_interleave(groups, dim=2)
        k = k.repeat_interleave(groups, dim=2)

    # --- 3. Gating: g (log-space decay) and beta (sigmoid gate), per value head.
    g, beta = fused_gdn_gating_and_sigmoid(a_log, a, b, dt_bias)
    g = g.view(1, seq_len, num_v_heads)
    beta = beta.view(1, seq_len, num_v_heads)

    # --- 4. Chunked gated delta recurrence over the packed sequence. Gather the
    # per-sequence initial states, run, and scatter the final states back.
    initial_state = delta_state.index_select(0, cache_indices.to(torch.long))
    core_attn_out, final_state = chunk_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    if store_state_transposed:
        final_state = final_state.transpose(-1, -2).contiguous()
    delta_state.index_copy_(
        0, cache_indices.to(torch.long), final_state.to(delta_state.dtype)
    )

    # --- 5. Gated RMSNorm + fp8 group quantization tail.
    core = core_attn_out.reshape(seq_len, num_v_heads, _HEAD_DIM)
    fp8_out = torch.empty(
        (seq_len, num_v_heads * _HEAD_DIM), dtype=precision_config.dtype, device=device
    )
    scales = torch.empty((seq_len, num_v_heads), dtype=torch.float32, device=device)
    gated_rmsnorm_fp8_group_quant(
        fp8_out, scales, core, z, norm_weight, eps, _GROUP_SIZE, False
    )

    # --- 6. Optional bf16 reconstruction of the quantized output.
    bf16_out = None
    if return_bf16:
        bf16_out = (
            fp8_out.view(seq_len, num_v_heads, _HEAD_DIM).to(torch.float32)
            * scales[:, :, None]
        ).to(torch.bfloat16).reshape(seq_len, num_v_heads * _HEAD_DIM)

    return bf16_out, conv_state, delta_state, fp8_out, scales
