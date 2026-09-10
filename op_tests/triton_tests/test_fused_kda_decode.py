# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for fused KDA decode kernel (conv1d + recurrence + gated RMSNorm)."""

import pytest
import torch
from einops import rearrange

from aiter.ops.triton.gated_delta_net.causal_conv1d_decode import (
    causal_conv1d_update_split_qkv,
)
from aiter.ops.triton.gated_delta_net.fused_kda_decode import fused_kda_decode

device = "cuda"
ATOL = 0.05


def _ref_kda_decode(
    mixed_qkv,
    conv_state,
    conv_weight,
    gate,
    beta,
    out_gate,
    A_log,
    dt_bias,
    ssm_state,
    ssm_state_indices,
    cu_seqlens,
    norm_weight,
    norm_eps,
    head_dim,
    num_local_heads,
    lower_bound,
):
    """Reference: 3 separate aiter kernel calls."""
    T = mixed_qkv.shape[0]
    lp = num_local_heads * head_dim
    D = head_dim
    H = num_local_heads

    # Kernel 1: conv1d
    q, k, v = causal_conv1d_update_split_qkv(
        mixed_qkv,
        conv_state,
        conv_weight,
        lp,
        lp,
        bias=None,
        activation="silu",
        conv_state_indices=ssm_state_indices,
        use_gluon=False,
    )

    # Kernel 2: recurrence (aiter uses softplus gating, not KDA lower-bound)
    # aiter's API: fused_sigmoid_gating_delta_rule_update(A_log, a, dt_bias,
    #   softplus_beta, softplus_threshold, q, k, v, b, state, indices, ...)
    # This uses softplus gating, not KDA's lower-bounded sigmoid.
    # For a proper reference we need the KDA variant.
    # Use a simple PyTorch reference instead.

    q = rearrange(q, "t (h d) -> 1 t h d", d=D)
    k = rearrange(k, "t (h d) -> 1 t h d", d=D)
    v = rearrange(v, "t (h d) -> 1 t h d", d=D)

    # QK L2 norm
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-6) * (D**-0.5)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

    # Per-token recurrence (PyTorch reference)
    out = torch.empty(T, H, D, dtype=torch.float32, device=device)
    for t_idx in range(T):
        slot = ssm_state_indices[t_idx].item()
        if slot < 0:
            out[t_idx] = 0
            continue
        for h in range(H):
            qt = q[0, t_idx, h]  # [D]
            kt = k[0, t_idx, h]  # [D]
            vt = v[0, t_idx, h]  # [D]
            state = ssm_state[slot, h].float()  # [D, D]

            # Gate
            a_val = gate[0, t_idx, h].float()  # [D]
            dt_val = dt_bias[h * D : (h + 1) * D].float()
            A_val = A_log[h].float()
            g = lower_bound * torch.sigmoid(torch.exp(A_val) * (a_val + dt_val))

            # Beta
            beta_val = torch.sigmoid(beta[0, t_idx, h].float())

            # Decay
            state = state * torch.exp(g[None, :])

            # Delta rule
            dot = (state * kt[None, :]).sum(dim=1)
            delta_v = (vt - dot) * beta_val
            state = state + delta_v[:, None] * kt[None, :]

            # Output
            o_val = (state * qt[None, :]).sum(dim=1)
            out[t_idx, h] = o_val
            ssm_state[slot, h] = state.to(ssm_state.dtype)

    # Round to bf16 (match kernel behavior)
    out_bf16 = out.to(torch.bfloat16).float()

    # RMSNorm + gate
    sumsq = (out_bf16**2).sum(dim=-1, keepdim=True)
    rstd = torch.rsqrt(sumsq / D + norm_eps)
    out_gate_3d = rearrange(out_gate[:T], "t (h d) -> t h d", d=D).float()
    normed = (
        out_bf16
        * rstd
        * norm_weight.float()[None, None, :]
        * torch.sigmoid(out_gate_3d)
    )

    return rearrange(normed.to(torch.bfloat16), "t h d -> t (h d)")


def _make_inputs(batch, Hloc, D, W=4, dtype=torch.bfloat16):
    lp = Hloc * D
    num_slots = batch + 2
    return {
        "mixed_qkv": torch.randn(batch, 3 * lp, dtype=dtype, device=device),
        "conv_weight": torch.randn(3 * lp, W, dtype=dtype, device=device) * 0.1,
        "conv_state": torch.randn(num_slots, 3 * lp, W - 1, dtype=dtype, device=device)
        * 0.1,
        "gate": torch.randn(1, batch, Hloc, D, dtype=dtype, device=device) * 0.5,
        "beta": torch.randn(1, batch, Hloc, dtype=dtype, device=device),
        "out_gate": torch.randn(batch, lp, dtype=dtype, device=device),
        "A_log": torch.randn(Hloc, dtype=dtype, device=device) * 0.1,
        "dt_bias": torch.randn(lp, dtype=dtype, device=device) * 0.1,
        "ssm_state": torch.randn(
            num_slots, Hloc, D, D, dtype=torch.float32, device=device
        )
        * 0.01,
        "norm_weight": torch.ones(D, dtype=dtype, device=device),
        "ssm_state_indices": torch.arange(batch, dtype=torch.int32, device=device),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int64, device=device),
    }


@pytest.mark.parametrize("batch", [1, 4, 32, 64])
@pytest.mark.parametrize("Hloc", [2, 8])
@pytest.mark.parametrize("D", [128])
def test_fused_kda_decode_correctness(batch, Hloc, D):
    """Fused kernel output matches PyTorch reference."""
    torch.manual_seed(42)
    inp = _make_inputs(batch, Hloc, D)

    ref = _ref_kda_decode(
        inp["mixed_qkv"],
        inp["conv_state"].clone(),
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["ssm_state"].clone(),
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
    )

    out = fused_kda_decode(
        inp["mixed_qkv"],
        inp["conv_state"].clone(),
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["ssm_state"].clone(),
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
    )

    torch.testing.assert_close(out.float(), ref.float(), atol=ATOL, rtol=0.02)


def test_fused_kda_decode_determinism():
    """Same input produces identical output across multiple runs."""
    torch.manual_seed(42)
    batch, Hloc, D = 128, 8, 128
    inp = _make_inputs(batch, Hloc, D)

    results = []
    for _ in range(5):
        out = fused_kda_decode(
            inp["mixed_qkv"],
            inp["conv_state"].clone(),
            inp["conv_weight"],
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            inp["ssm_state"].clone(),
            inp["ssm_state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            D,
            Hloc,
            -5.0,
        )
        results.append(out.clone())

    for i in range(1, len(results)):
        assert torch.equal(results[0], results[i]), (
            f"Run 0 vs run {i}: max diff = "
            f"{(results[0].float() - results[i].float()).abs().max().item()}"
        )


def test_fused_kda_decode_pad_slot():
    """PAD_SLOT_ID (-1) sequences are skipped without modifying state."""
    torch.manual_seed(42)
    batch, Hloc, D = 1, 8, 128
    inp = _make_inputs(batch, Hloc, D)

    ssm_before = inp["ssm_state"].clone()
    conv_before = inp["conv_state"].clone()
    inp["ssm_state_indices"] = torch.tensor([-1], dtype=torch.int32, device=device)

    fused_kda_decode(
        inp["mixed_qkv"],
        inp["conv_state"],
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["ssm_state"],
        inp["ssm_state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
    )

    assert torch.equal(
        inp["ssm_state"], ssm_before
    ), "SSM state modified for PAD_SLOT_ID"
    assert torch.equal(
        inp["conv_state"], conv_before
    ), "Conv state modified for PAD_SLOT_ID"
