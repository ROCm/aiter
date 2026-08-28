# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from flash-linear-attention / vLLM (see _triton_kernels copy).

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_rearrange_sigmoid_gdr import (
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel,
)


def fused_rearrange_sigmoid_gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Triton sigmoid-gated delta rule over packed QKV (decode-oriented).
    """
    expected_shape = (qkv.shape[0], key_dim * 2 + value_dim)
    assert (
        qkv.shape == expected_shape
    ), f"expect qkv to be in shape {expected_shape}, got {qkv.shape}"
    if scale is None:
        scale = head_k_dim**-0.5
    else:
        assert scale > 0, "scale must be positive"

    B = 1
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if inplace_final_state and ssm_state_indices is None:
        raise ValueError(
            "ssm_state_indices is required when inplace_final_state=True "
            "(kernel indexes final state slots per token)."
        )

    o = (
        core_attn_out[: NK * B * T * HV * V].view(NK, B, T, HV, V)
        if core_attn_out is not None
        else qkv.new_empty(NK, B, T, HV, V)
    )
    if inplace_final_state:
        if initial_state is None:
            raise ValueError("initial_state is required when inplace_final_state=True")
        final_state = initial_state
    else:
        st_dtype = initial_state.dtype if initial_state is not None else qkv.dtype
        final_state = qkv.new_empty(T, HV, V, K, dtype=st_dtype)

    stride_init_state_token = (
        int(initial_state.stride(0)) if initial_state is not None else 0
    )
    stride_final_state_token = int(final_state.stride(0))

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    stride_qkv_l, stride_qkv_hd = qkv.stride()

    grid = (NK, NV, N * HV)
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        qkv=qkv,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_qkv_l=stride_qkv_l,
        stride_qkv_hd=stride_qkv_hd,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
