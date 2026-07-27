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
    """Float reference for decode path (B=1, one sequence), including GQA (HV >= H)."""
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    if HV % H != 0:
        raise ValueError(f"reference expects HV divisible by H, got H={H}, HV={HV}")
    group = HV // H
    B = 1
    o = torch.empty(B, T, HV, V, dtype=torch.float32, device=qkv.device)
    h_state = torch.zeros(HV, V, K, dtype=torch.float32, device=qkv.device)
    if initial_state is not None:
        h_state = initial_state[0].to(torch.float32).clone()

    for t in range(T):
        row = qkv[t]
        for hv in range(HV):
            i_h = hv // group
            q_vec = row[i_h * K : (i_h + 1) * K].float()
            k_vec = row[H * K + i_h * K : H * K + (i_h + 1) * K].float()
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


# Shapes aligned with ``test_gated_delta_rule.test_fused_recurrent``; dtypes are
# half-precision only — long packed ``T`` with float32 activations tends to blow
# up the recurrent reference / kernel without tighter dynamic-range clamps.
# Each row ends with ``use_qk_l2norm_in_kernel`` (True for stable long-T sweep).
# One small bf16 row uses False to cover the no–L2-norm path (replaces former ``basic``).
_FUSED_GDR_SWEEP = [
    (63, 1, 1, 64, 1, 1, torch.float16, True),
    (500, 4, 4, 60, 1, 1, torch.float16, True),
    (1000, 2, 8, 128, 1, 0.1, torch.float16, True),
    (1024, 2, 2, 128, 0.1, 1, torch.float16, True),
    (1024, 3, 3, 128, 1, 10, torch.float16, True),
    (2048, 4, 4, 64, 0.1, 1, torch.float16, True),
    (1024, 4, 4, 128, 1, 0.1, torch.float16, True),
    (1024, 4, 8, 128, 1, 10, torch.float16, True),
    (1024, 4, 4, 128, 1, 0.1, torch.bfloat16, True),
    (1024, 4, 8, 128, 1, 1, torch.bfloat16, True),
    (2048, 4, 8, 64, 0.1, 1, torch.bfloat16, True),
    (8, 4, 4, 16, 16**-0.5, 1, torch.bfloat16, False),
]


@cuda_ok
@pytest.mark.parametrize(
    (
        "T",
        "H",
        "HV",
        "D",
        "scale",
        "gate_logit_normalizer",
        "dtype",
        "use_qk_l2norm_in_kernel",
    ),
    [
        pytest.param(
            *row,
            id="T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}-l2{}".format(*row),
        )
        for row in _FUSED_GDR_SWEEP
    ],
)
def test_fused_rearrange_sigmoid_gdr_sweep(
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
    use_qk_l2norm_in_kernel: bool,
):
    """Shape/dtype sweep aligned with ``test_gated_delta_rule.test_fused_recurrent``."""
    if HV % H != 0:
        pytest.skip("reference/kernel GQA mapping needs HV divisible by H")
    device = "cuda"
    K = V = D
    key_dim = H * K
    value_dim = HV * V

    if use_qk_l2norm_in_kernel:
        torch.manual_seed(42)
        qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype) * 0.05
        A_log = (
            torch.randn(HV, device=device, dtype=torch.float32).clamp(-2.0, 0.5) * 0.02
        )
        a = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(-1.0, 1.0)
        a = a / gate_logit_normalizer
        b_gate = (torch.randn(T, HV, device=device, dtype=dtype) * 0.05).clamp(
            -1.0, 1.0
        )
        dt_bias = (torch.randn(HV, device=device, dtype=dtype) * 0.005).clamp(-0.5, 0.5)
        initial = torch.randn(1, HV, V, K, device=device, dtype=dtype) * 0.05
    else:
        torch.manual_seed(0)
        qkv = torch.randn(T, key_dim * 2 + value_dim, device=device, dtype=dtype)
        A_log = torch.randn(HV, device=device, dtype=torch.float32) * 0.02
        a = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
        a = a / gate_logit_normalizer
        b_gate = torch.randn(T, HV, device=device, dtype=dtype) * 0.1
        dt_bias = torch.randn(HV, device=device, dtype=dtype) * 0.01
        initial = torch.randn(1, HV, V, K, device=device, dtype=dtype)

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
        scale,
        initial,
        use_qk_l2norm_in_kernel,
    )

    core = torch.empty(1 * 1 * T * HV * V, device=device, dtype=dtype)
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
        scale=scale,
        initial_state=initial,
        inplace_final_state=False,
        cu_seqlens=None,
        ssm_state_indices=None,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        is_kda=False,
        core_attn_out=core,
    )

    if dtype == torch.bfloat16:
        rtol, atol = 0.05, 0.1
    elif dtype == torch.float16:
        rtol, atol = 0.03, 0.08
    else:
        rtol, atol = 0.02, 0.05

    if use_qk_l2norm_in_kernel:
        assert torch.isfinite(o_tr.float()).all(), "non-finite Triton output"
        assert torch.isfinite(h_tr.float()).all(), "non-finite Triton final_state"
    torch.testing.assert_close(o_tr.float(), o_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(h_tr[-1].float(), h_ref[0], rtol=rtol, atol=atol)
