# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused KDA decode: conv1d + delta-rule recurrence + gated RMSNorm.

Single Triton kernel launch. Fuses three separate operations into one,
eliminating q/k/v intermediate HBM traffic and kernel launch overhead.
"""

import functools

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_net.decode.fused_conv_recurrent_norm import (
    fused_conv_recurrent_norm_kernel,
    fused_kda_spec_finalize_kernel,
    fused_kda_spec_parallel_v_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

# An arch without a published BV stays on the generic kernel.
_KERNEL_FALLBACK = triton.Config({})


@functools.cache
def _launch_config(kernel_name: str) -> triton.Config:
    """Read-only launch config, resolved once per kernel per process.

    Decode runs this per step, and an arch with no published entry warns on
    every miss, so the uncached call would log once per launch.
    """
    return get_tuned_kernel_config(
        "attention", "FUSED_KDA_DECODE", kernel_name, _KERNEL_FALLBACK
    )


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
    num_accepted_tokens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused KDA decode: conv1d + recurrence + gated RMSNorm.

    T is the number of tokens (B for single-token decode, up to B * S for
    speculative decode, possibly padded past that by FULL decode graphs).

    Args:
        mixed_qkv: [T, 3*lp] bf16, may be strided (sliced from in_proj output).
        conv_state: [N, 3*lp, STATE_LEN] bf16, transposed view of conv cache.
            STATE_LEN is W-1 for normal decode; speculative decode needs
            STATE_LEN >= S + W - 2.
        conv_weight: [3*lp, W] or [3, W, lp], bf16 or fp32 conv1d weights.
        gate: [1, T, H, K] bf16, KDA decay gate (raw logits).
        beta: [1, T, H] bf16, write strength (raw logits), may be strided.
        out_gate: [T, H*K] bf16, output gate for RMSNorm, may be strided.
        A_log: [H] fp32, per-head decay parameter.
        dt_bias: [H*K] fp32, per-channel bias.
        ssm_state: [N, H, V, K] fp32, delta-rule state matrices.
        ssm_state_indices: [B] for normal decode or [B, S] for spec decode.
            Normal decode skips negative slots (PAD_SLOT_ID); speculative
            decode also skips slot 0 (vLLM NULL_BLOCK_ID).
        cu_seqlens: [B+1] int64, cumulative sequence lengths. For speculative
            decode every sequence spans at most S tokens (the width of
            ssm_state_indices); tokens past S are not processed.
        norm_weight: [K] bf16 or fp32 RMSNorm weight.
        norm_eps: float, RMSNorm epsilon.
        head_dim: int, K = V = head_dim.
        num_local_heads: int, H = num_local_heads.
        lower_bound: float, KDA gate lower bound (typically -5.0).
        num_accepted_tokens: [B] int32 accepted-token counts for spec decode.
        conv_state_indices: [B] int32 convolution-cache slots for spec decode,
            may be strided (vLLM passes ssm_state_indices[:, 0]).
        out: Optional preallocated contiguous [T, H*K] bf16 output.

    Returns:
        out: [T, H*K] bf16, final output after RMSNorm + gate.
    """
    T = mixed_qkv.shape[0]
    K = V = head_dim
    H = num_local_heads
    lp = H * K
    batch = cu_seqlens.shape[0] - 1

    is_spec_decoding = num_accepted_tokens is not None
    if is_spec_decoding:
        if ssm_state_indices.ndim != 2 or conv_state_indices is None:
            raise ValueError(
                "Spec decode requires 2-D ssm_state_indices and conv_state_indices"
            )
        if ssm_state_indices.shape[1] < 2:
            raise ValueError(
                "Spec decode requires at least 2 state-index entries per sequence, "
                f"got {ssm_state_indices.shape[1]}"
            )
        if ssm_state_indices.shape[0] < batch:
            raise ValueError(
                "ssm_state_indices must cover every sequence: "
                f"got {ssm_state_indices.shape[0]} rows for batch {batch}"
            )
        for name, vec in (
            ("num_accepted_tokens", num_accepted_tokens),
            ("conv_state_indices", conv_state_indices),
        ):
            if vec.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {tuple(vec.shape)}")
            if vec.shape[0] < batch:
                raise ValueError(
                    f"{name} must cover every sequence: got {vec.shape[0]} "
                    f"entries for batch {batch}"
                )
        # conv_state_indices carries a stride because the natural caller slices
        # it as ssm_state_indices[:, 0], stride S not 1.
        stride_cidx = conv_state_indices.stride(0)
        # Stride 0 (an expanded tensor) would give every sequence the first
        # sequence's conv slot, so they would all share one live cache entry.
        if batch > 1 and stride_cidx <= 0:
            raise ValueError(
                f"conv_state_indices must have a positive stride, got "
                f"{stride_cidx} (an expanded or broadcast tensor?)"
            )
        # num_accepted_tokens is read packed, so a strided one would silently
        # pick up the wrong sequence rather than fail.
        if batch > 1 and num_accepted_tokens.stride(0) != 1:
            raise ValueError(
                f"num_accepted_tokens must be contiguous, got stride "
                f"{num_accepted_tokens.stride(0)}"
            )
        stride_indices_seq = ssm_state_indices.stride(0)
        stride_indices_tok = ssm_state_indices.stride(1)
        if ssm_state_indices.shape[1] > 1 and stride_indices_tok <= 0:
            raise ValueError(
                "ssm_state_indices must have a positive token stride, got "
                f"{stride_indices_tok}"
            )
    else:
        stride_indices_seq = ssm_state_indices.stride(0)
        stride_indices_tok = 1
        # Unused off the speculative path; the pointers alias ssm_state_indices.
        stride_cidx = 1
    if batch > 1 and stride_indices_seq <= 0:
        raise ValueError(
            "ssm_state_indices must have a positive sequence stride, got "
            f"{stride_indices_seq}"
        )
    # A single sequence only forms i_n == 0, so normalize a zero stride before
    # passing it to kernels that assume positive strides.
    if batch <= 1:
        stride_cidx = max(stride_cidx, 1)
        stride_indices_seq = max(stride_indices_seq, 1)
    num_accepted_tokens_arg = (
        num_accepted_tokens if num_accepted_tokens is not None else ssm_state_indices
    )
    conv_state_indices_arg = (
        conv_state_indices if conv_state_indices is not None else ssm_state_indices
    )

    if out is None:
        allocator = torch.zeros if is_spec_decoding else torch.empty
        out = allocator(T, lp, dtype=torch.bfloat16, device=mixed_qkv.device)
    elif out.shape != (T, lp):
        raise ValueError(f"Expected out shape {(T, lp)}, got {tuple(out.shape)}")
    elif out.dtype != torch.bfloat16:
        raise ValueError(f"out must have dtype torch.bfloat16, got {out.dtype}")
    elif out.device != mixed_qkv.device:
        raise ValueError(f"out must be on device {mixed_qkv.device}, got {out.device}")
    elif not out.is_contiguous():
        # The kernels address out as tok * (H * V) + channel.
        raise ValueError("out must be contiguous")

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

    if is_spec_decoding and W != 4:
        # Both speculative paths keep exactly W - 1 = 3 history taps in
        # registers.
        raise NotImplementedError(
            f"fused_kda_decode speculative decode supports W == 4, got W={W}"
        )

    stride_beta_tok = beta.stride(1) if beta.dim() == 3 else beta.stride(0)
    stride_og_tok = out_gate.stride(0)
    if stride_beta_tok <= 0:
        raise ValueError(
            f"beta must have a positive token stride, got {stride_beta_tok}"
        )
    if stride_og_tok <= 0:
        raise ValueError(
            f"out_gate must have a positive token stride, got {stride_og_tok}"
        )

    # Tokens per speculative sequence: num_speculative_tokens + 1. H and this
    # width are not specialized: the parallel kernel indexes heads from
    # program_id(1) and clamps its token loop to SPEC_LEN.
    spec_tokens = ssm_state_indices.shape[1] if is_spec_decoding else 0
    if is_spec_decoding:
        required_conv_state_len = spec_tokens + W - 2
        if conv_state.shape[2] < required_conv_state_len:
            raise ValueError(
                "Spec decode requires conv_state.shape[2] >= "
                f"{required_conv_state_len}, got {conv_state.shape[2]}"
            )
    # BV is a compile-time tile, so resolve it before the V % BV test below.
    parallel_cfg = None
    block_v = 0
    min_spec_tokens = 0
    min_batch_for_width2 = 0
    if is_spec_decoding and get_arch() == "gfx950":
        cfg = _launch_config("fused_kda_spec_parallel_v_kernel")
        # An unpublished tile drops to the generic kernel rather than raising.
        bv = cfg.kwargs.get("BV")
        if bv is not None:
            # tl.arange needs a power of two.
            if not isinstance(bv, int) or bv <= 0 or bv & (bv - 1):
                raise ValueError(
                    f"BV must be a positive power of two, got {bv!r} in the "
                    "fused_kda_decode config"
                )
            parallel_cfg = cfg
            block_v = bv
            min_spec_tokens = cfg.kwargs["MIN_SPEC_TOKENS"]
            min_batch_for_width2 = cfg.kwargs["MIN_BATCH_FOR_WIDTH2"]
    # W == 4 is structural: the recurrence keeps exactly W - 1 = 3 conv history
    # taps in registers.
    use_parallel_spec = (
        parallel_cfg is not None
        and K == 128
        and V % block_v == 0
        and W == 4
        and (spec_tokens >= min_spec_tokens or batch >= min_batch_for_width2)
    )
    if use_parallel_spec:
        # cu_seqlens stays on device. A host read would sync and break CUDA
        # graph capture. Both kernels clamp each sequence to SPEC_LEN, so a
        # packed batch with a short sequence still takes this path, and a span
        # longer than the index row drops the surplus tokens.
        finalize_cfg = _launch_config("fused_kda_spec_finalize_kernel")
        conv_carry = torch.empty(
            batch,
            3 * H * K,
            2,
            dtype=torch.bfloat16,
            device=mixed_qkv.device,
        )
        fused_kda_spec_parallel_v_kernel[(batch, H, V // block_v)](
            mixed_qkv,
            conv_weight,
            conv_state,
            conv_carry,
            gate,
            beta,
            A_log,
            dt_bias,
            ssm_state,
            ssm_state_indices,
            num_accepted_tokens,
            conv_state_indices,
            cu_seqlens,
            out,
            lower_bound,
            K**-0.5,
            H=H,
            K=K,
            V=V,
            W=W,
            BV=block_v,
            SPEC_LEN=spec_tokens,
            stride_x_tok=mixed_qkv.stride(0),
            stride_cw_group=stride_cw_group,
            stride_cw_width=stride_cw_width,
            stride_cw_ch=stride_cw_ch,
            stride_cs_slot=conv_state.stride(0),
            stride_cs_dim=conv_state.stride(1),
            stride_cs_pos=conv_state.stride(2),
            stride_beta_tok=stride_beta_tok,
            stride_ssm_slot=ssm_state.stride(0),
            stride_indices_seq=stride_indices_seq,
            stride_indices_tok=stride_indices_tok,
            stride_cidx=stride_cidx,
            num_warps=parallel_cfg.num_warps,
            num_stages=parallel_cfg.num_stages,
        )
        fused_kda_spec_finalize_kernel[(batch, H, spec_tokens)](
            mixed_qkv,
            conv_state,
            conv_carry,
            ssm_state_indices,
            num_accepted_tokens,
            conv_state_indices,
            cu_seqlens,
            norm_weight,
            out_gate,
            out,
            norm_eps,
            H=H,
            K=K,
            V=V,
            W=W,
            SPEC_LEN=spec_tokens,
            stride_x_tok=mixed_qkv.stride(0),
            stride_cs_slot=conv_state.stride(0),
            stride_cs_dim=conv_state.stride(1),
            stride_cs_pos=conv_state.stride(2),
            stride_og_tok=stride_og_tok,
            stride_indices_seq=stride_indices_seq,
            stride_indices_tok=stride_indices_tok,
            stride_cidx=stride_cidx,
            num_warps=finalize_cfg.num_warps,
            num_stages=finalize_cfg.num_stages,
        )
        return out

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
        num_accepted_tokens_arg,
        conv_state_indices_arg,
        cu_seqlens,
        norm_weight,
        out_gate,
        out,
        lower_bound,
        norm_eps,
        K**-0.5,
        H=H,
        K=K,
        V=V,
        W=W,
        STATE_LEN=conv_state.shape[2],
        SHIFT_LEN_P2=triton.next_power_of_2(max(conv_state.shape[2] - 1, 1)),
        IS_SPEC_DECODING=is_spec_decoding,
        SPEC_LEN=spec_tokens if is_spec_decoding else 1,
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
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_cidx=stride_cidx,
        num_warps=2 if get_arch() == "gfx942" else 4,
    )
    return out
