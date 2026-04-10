# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton.gated_delta_net import fused_rearrange_sigmoid_gated_delta_rule

cuda_ok = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device required"
)


def _softplus(x: torch.Tensor, beta: float, threshold: float) -> torch.Tensor:
    return torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
        x,
    )


def ref_fused_rearrange_sigmoid_gdr(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float,
    threshold: float,
    scale: float,
    initial_state: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float reference for the H == HV (no GQA) decode path (B=1, one sequence)."""
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    assert H == HV, "reference only implements H == HV"
    B = 1
    o = torch.empty(B, T, HV, V, dtype=torch.float32, device=qkv.device)
    h_state = torch.zeros(HV, V, K, dtype=torch.float32, device=qkv.device)
    if initial_state is not None:
        h_state = initial_state[0].to(torch.float32).clone()

    for t in range(T):
        row = qkv[t]
        for hv in range(HV):
            q_vec = row[hv * K : (hv + 1) * K].float()
            k_vec = row[H * K + hv * K : H * K + (hv + 1) * K].float()
            v_vec = row[2 * H * K + hv * V : 2 * H * K + (hv + 1) * V].float()
            b_gate = b[t, hv].float()
            x = a[t, hv].float() + dt_bias[hv].float()
            sp = _softplus(x, beta, threshold)
            g = -torch.exp(A_log[hv].float()) * sp
            beta_out = torch.sigmoid(b_gate)
            if use_qk_l2norm_in_kernel:
                q_vec = q_vec * torch.rsqrt((q_vec * q_vec).sum() + 1e-6)
                k_vec = k_vec * torch.rsqrt((k_vec * k_vec).sum() + 1e-6)
            q_vec = q_vec * scale
            h_sub = h_state[hv]
            h_sub = h_sub * torch.exp(g)
            v_adj = v_vec - (h_sub * k_vec.unsqueeze(0)).sum(dim=-1)
            v_adj = v_adj * beta_out
            h_sub = h_sub + v_adj.unsqueeze(-1) * k_vec.unsqueeze(0)
            out_vec = (h_sub * q_vec.unsqueeze(0)).sum(dim=-1)
            o[0, t, hv] = out_vec
            h_state[hv] = h_sub
    return o, h_state.unsqueeze(0)


@cuda_ok
def test_fused_rearrange_sigmoid_gdr_basic():
    device = "cuda"
    torch.manual_seed(0)
    T, K, V = 8, 16, 16
    H = HV = 4
    key_dim = H * K
    value_dim = HV * V
    qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.02
    a = torch.randn(T, HV, device=device, dtype=torch.bfloat16) * 0.1
    b_gate = torch.randn(T, HV, device=device, dtype=torch.bfloat16) * 0.1
    dt_bias = torch.randn(HV, device=device, dtype=torch.bfloat16) * 0.01
    initial = torch.randn(1, HV, V, K, device=device, dtype=torch.bfloat16)

    o_ref, h_ref = ref_fused_rearrange_sigmoid_gdr(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        1.0,
        20.0,
        K**-0.5,
        initial,
        False,
    )

    core = torch.empty(1 * 1 * T * HV * V, device=device, dtype=torch.bfloat16)
    o_tr, h_tr = fused_rearrange_sigmoid_gated_delta_rule(
        A_log,
        a,
        b_gate,
        dt_bias,
        qkv,
        key_dim,
        value_dim,
        K,
        V,
        beta=1.0,
        threshold=20.0,
        scale=K**-0.5,
        initial_state=initial,
        inplace_final_state=False,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False,
        is_kda=False,
        core_attn_out=core,
    )
    torch.testing.assert_close(o_tr.float(), o_ref, rtol=0.05, atol=0.1)
    torch.testing.assert_close(h_tr[-1].float(), h_ref[0], rtol=0.05, atol=0.1)
