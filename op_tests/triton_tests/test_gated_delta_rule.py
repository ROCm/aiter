# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import pytest
import torch
import torch.nn.functional as F
from einops import repeat

# from aiter.ops.triton.gated_delta_net import (
#     fused_recurrent_gated_delta_rule,
#     chunk_gated_delta_rule,
# )
from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    assert_close,
    device,
)


def fused_sigmoid_gating_delta_rule_update_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,  # beta logits (will apply sigmoid internally)
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, b, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, b, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = torch.sigmoid(b[:, :, i])  # Apply sigmoid to logits
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "gate_logit_normalizer", "dtype"),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-HV{}-D{}-scale{}-gate_logit_normalizer{}-{}".format(*test),
        )
        for test in [
            (1, 63, 1, 1, 64, 1, 1, torch.float),
            (2, 500, 4, 4, 60, 1, 1, torch.float),
            (2, 1000, 2, 8, 128, 1, 0.1, torch.float),
            (3, 1024, 2, 2, 128, 0.1, 1, torch.float),
            (4, 1024, 3, 3, 128, 1, 10, torch.float),
            (4, 2048, 4, 4, 64, 0.1, 1, torch.float),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.float16),
            (2, 1024, 4, 8, 128, 1, 10, torch.float16),
            (2, 1024, 4, 4, 128, 1, 0.1, torch.bfloat16),
            (2, 1024, 4, 8, 128, 1, 1, torch.bfloat16),
            (4, 2048, 4, 8, 64, 0.1, 1, torch.bfloat16),
        ]
    ],
)
def test_fused_sigmoid_gating_delta_rule_update(
    B: int,
    T: int,
    H: int,
    HV: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
):
    """Test fused sigmoid gating delta rule update kernel."""
    torch.manual_seed(42)

    # Create input tensors
    q = torch.randn(B, T, H, D, dtype=torch.float32)
    k = torch.randn(B, T, H, D, dtype=torch.float32)
    v = torch.randn(B, T, HV, D, dtype=dtype)
    b = torch.randn(B, T, HV, dtype=dtype)  # beta logits

    # Gating parameters
    A_log = torch.randn(HV, dtype=torch.float32)
    a = torch.randn(B, T, HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    softplus_beta = 1.0
    softplus_threshold = 20.0

    # Initial state
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32)
    initial_state_indices = torch.arange(B, dtype=torch.long)

    # Move to device
    q, k, v, b, A_log, a, dt_bias, h0, initial_state_indices = map(
        lambda x: x.to(device),
        (q, k, v, b, A_log, a, dt_bias, h0, initial_state_indices),
    )

    # Compute reference using recurrent implementation
    # Compute g = -exp(A_log) * softplus(a + dt_bias)
    # This is already in log-space (negative values for decay)
    x = a + dt_bias[None, None, :]
    softplus_x = F.softplus(x, beta=softplus_beta, threshold=softplus_threshold)
    g = -torch.exp(A_log[None, None, :]) * softplus_x

    # Expand q and k to match HV heads for reference implementation
    q_expanded = F.normalize(
        repeat(q.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
    ).to(dtype)
    k_expanded = F.normalize(
        repeat(k.clone(), "b t h d -> b t (h g) d", g=HV // H), p=2, dim=-1
    ).to(dtype)

    ref, _ = fused_sigmoid_gating_delta_rule_update_ref(
        q=q_expanded,
        k=k_expanded,
        v=v.clone(),
        b=b.clone(),  # Pass beta logits directly, sigmoid will be applied internally
        g=g.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    # Compute using fused kernel
    tri = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        b=b.clone(),
        initial_state_source=h0.clone(),
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=True,
    )

    # Use higher tolerance for bfloat16 due to lower precision
    tol = 0.005 if dtype == torch.bfloat16 else 0.002
    assert_close("o", ref, tri, tol)
