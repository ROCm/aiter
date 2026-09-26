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
        # Slot 0 is vLLM's NULL_BLOCK_ID. The kernel skips it, so real cache
        # slots in this test start at 1. num_slots is batch + 2.
        "ssm_state_indices": torch.arange(
            1, batch + 1, dtype=torch.int32, device=device
        ),
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
        conv_state = inp["conv_state"].clone()
        ssm_state = inp["ssm_state"].clone()
        out = fused_kda_decode(
            inp["mixed_qkv"],
            conv_state,
            inp["conv_weight"],
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            ssm_state,
            inp["ssm_state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            D,
            Hloc,
            -5.0,
        )
        results.append((out.clone(), conv_state, ssm_state))

    for i in range(1, len(results)):
        for name, expected, actual in zip(
            ("output", "conv_state", "ssm_state"), results[0], results[i]
        ):
            assert torch.equal(expected, actual), (
                f"{name}, run 0 vs run {i}: max diff = "
                f"{(expected.float() - actual.float()).abs().max().item()}"
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


def test_fused_kda_decode_skips_null_slot():
    """vLLM NULL_BLOCK_ID (0) is skipped without modifying state or output."""
    torch.manual_seed(42)
    batch, Hloc, D = 1, 2, 128
    inp = _make_inputs(batch, Hloc, D)

    ssm_before = inp["ssm_state"].clone()
    conv_before = inp["conv_state"].clone()
    inp["ssm_state_indices"] = torch.zeros(batch, dtype=torch.int32, device=device)
    out = torch.zeros(batch, Hloc * D, dtype=torch.bfloat16, device=device)

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
        out=out,
    )

    assert torch.equal(inp["ssm_state"], ssm_before), "SSM state modified for NULL slot"
    assert torch.equal(
        inp["conv_state"], conv_before
    ), "Conv state modified for NULL slot"
    assert torch.equal(out, torch.zeros_like(out)), "Output written for NULL slot"


SPEC_D = 128
SPEC_W = 4


class _SpecKernelSpy:
    """Record grids the parallel spec kernel is launched with.

    Numerical agreement alone cannot tell the two paths apart, since the generic
    fallback computes the same result. Without this the suite passes while the
    optimized kernel never runs.
    """

    def __init__(self, module):
        self._module = module
        self._name = "fused_kda_spec_parallel_v_kernel"
        self.grids = []

    def __enter__(self):
        self._orig = getattr(self._module, self._name)
        grids = self.grids
        orig = self._orig

        class _Launcher:
            def __getitem__(self, grid):
                grids.append(grid)
                return orig[grid]

        setattr(self._module, self._name, _Launcher())
        return self

    def __exit__(self, *exc):
        setattr(self._module, self._name, self._orig)
        return False


def _ref_spec_conv1d_step(x_qkv, conv_state, conv_weight, state_len):
    dim = x_qkv.shape[0]
    width = conv_weight.shape[1]
    out = torch.zeros(dim, dtype=torch.float32, device=device)
    for j in range(width - 1):
        idx = state_len - width + 1 + j
        out += conv_state[:, idx].float() * conv_weight[:, j].float()
    out += x_qkv.float() * conv_weight[:, width - 1].float()
    out = out * torch.sigmoid(out)
    conv_state[:, :-1] = conv_state[:, 1:].clone()
    conv_state[:, -1] = x_qkv
    return out.to(torch.bfloat16)


def _ref_spec_kda_step(q_h, k_h, v_h, gate_h, dt_h, A_h, beta_val, h_state, K):
    q_h = q_h * torch.rsqrt(torch.sum(q_h * q_h) + 1e-6) * (K**-0.5)
    k_h = k_h * torch.rsqrt(torch.sum(k_h * k_h) + 1e-6)
    g = -5.0 * torch.sigmoid(torch.exp(A_h) * (gate_h + dt_h))
    b_beta = torch.sigmoid(beta_val)
    h_state = h_state * torch.exp(g[None, :])
    dot = (h_state * k_h[None, :]).sum(1)
    u = (v_h - dot) * b_beta
    h_state = h_state + u[:, None] * k_h[None, :]
    o = (h_state * q_h[None, :]).sum(1)
    return o, h_state


def _ref_spec_decode(
    mixed_qkv,
    conv_state,
    conv_weight,
    gate,
    beta,
    out_gate,
    A_log,
    dt_bias,
    state,
    cu_seqlens,
    norm_weight,
    norm_eps,
    H,
    state_indices=None,
    num_accepted_tokens=None,
    conv_state_indices=None,
    full_spec_sequence=False,
):
    T = mixed_qkv.shape[0]
    K = V = SPEC_D
    lp = H * K
    batch = cu_seqlens.shape[0] - 1
    state_len = conv_state.shape[2]
    is_spec = num_accepted_tokens is not None
    out = torch.empty(T, lp, dtype=torch.bfloat16, device=device)

    for n in range(batch):
        bos, eos = cu_seqlens[n].item(), cu_seqlens[n + 1].item()
        if eos - bos == 0:
            continue

        if is_spec:
            i_start = max(num_accepted_tokens[n].item() - 1, 0)
            s_idx = state_indices[n, i_start].item()
            conv_slot_idx = conv_state_indices[n].item()
            b_h = state[s_idx].clone().float()
            if full_spec_sequence:
                spec_conv_history = conv_state[
                    conv_slot_idx, :, i_start : i_start + SPEC_W - 1
                ].clone()
                spec_next_conv_state = torch.cat(
                    (
                        spec_conv_history[:, 1:],
                        mixed_qkv[bos:eos].transpose(0, 1),
                    ),
                    dim=1,
                )
        else:
            s_idx = state_indices[n].item()
            conv_slot_idx = s_idx
            b_h = state[s_idx].clone().float()

        for t in range(eos - bos):
            tok = bos + t
            qkv_out = _ref_spec_conv1d_step(
                mixed_qkv[tok],
                (
                    spec_conv_history
                    if is_spec and full_spec_sequence
                    else conv_state[conv_slot_idx]
                ),
                conv_weight,
                SPEC_W - 1 if is_spec and full_spec_sequence else state_len,
            )
            q = qkv_out[:lp].reshape(H, K)
            k = qkv_out[lp : 2 * lp].reshape(H, K)
            v = qkv_out[2 * lp :].reshape(H, V)

            per_head = torch.empty(H, V, dtype=torch.float32, device=device)
            for hh in range(H):
                o_h, b_h[hh] = _ref_spec_kda_step(
                    q[hh].float(),
                    k[hh].float(),
                    v[hh].float(),
                    gate[0, tok, hh].float(),
                    dt_bias[hh * K : (hh + 1) * K].float(),
                    A_log[hh].float(),
                    beta[0, tok, hh].float(),
                    b_h[hh],
                    K,
                )
                per_head[hh] = o_h

            if is_spec:
                fidx = state_indices[n, t].item()
                if fidx >= 0:
                    state[fidx] = b_h.to(state.dtype)
            else:
                state[s_idx] = b_h.to(state.dtype)

            for hh in range(H):
                o_bf16 = per_head[hh].bfloat16().float()
                sumsq = (o_bf16 * o_bf16).sum()
                rstd = torch.rsqrt(sumsq / V + norm_eps)
                w = norm_weight.float()
                og = out_gate[tok, hh * V : (hh + 1) * V].float()
                out[tok, hh * V : (hh + 1) * V] = (
                    o_bf16 * rstd * w * torch.sigmoid(og)
                ).bfloat16()

        if is_spec and full_spec_sequence:
            conv_state[conv_slot_idx] = spec_next_conv_state

    return out


def _make_spec_inputs(
    batch,
    Hloc,
    num_spec=0,
    full_spec_sequence=False,
    normal_seq_len=1,
):
    lp = Hloc * SPEC_D
    seq_len = 1 + num_spec if num_spec > 0 and full_spec_sequence else normal_seq_len
    total_tokens = batch * seq_len
    state_len = (SPEC_W - 1 + num_spec) if num_spec > 0 else (SPEC_W - 1)
    num_slots = batch + num_spec * batch + 4
    torch.manual_seed(42)
    dtype = torch.bfloat16
    inp = {
        "mixed_qkv": torch.randn(total_tokens, 3 * lp, dtype=dtype, device=device)
        * 0.1,
        "conv_weight": torch.randn(3 * lp, SPEC_W, dtype=dtype, device=device) * 0.1,
        "conv_state": torch.randn(
            num_slots, 3 * lp, state_len, dtype=dtype, device=device
        )
        * 0.1,
        "gate": torch.randn(1, total_tokens, Hloc, SPEC_D, dtype=dtype, device=device)
        * 0.5,
        "beta": torch.randn(1, total_tokens, Hloc, dtype=dtype, device=device),
        "out_gate": torch.randn(total_tokens, lp, dtype=dtype, device=device),
        "A_log": torch.randn(Hloc, dtype=dtype, device=device) * 0.1,
        "dt_bias": torch.randn(lp, dtype=dtype, device=device) * 0.1,
        "state": torch.randn(
            num_slots, Hloc, SPEC_D, SPEC_D, dtype=torch.float32, device=device
        )
        * 0.01,
        "norm_weight": torch.ones(SPEC_D, dtype=dtype, device=device),
        "cu_seqlens": torch.arange(
            0, total_tokens + 1, seq_len, dtype=torch.int64, device=device
        ),
    }
    if num_spec > 0:
        inp["state_indices"] = torch.arange(
            1, batch * (1 + num_spec) + 1, dtype=torch.int32, device=device
        ).reshape(batch, 1 + num_spec)
        inp["num_accepted_tokens"] = torch.ones(batch, dtype=torch.int32, device=device)
        inp["conv_state_indices"] = torch.arange(
            1, batch + 1, dtype=torch.int32, device=device
        )
    else:
        inp["state_indices"] = torch.arange(
            1, batch + 1, dtype=torch.int32, device=device
        )
    return inp


def test_fused_normal_decode_multi_token():
    batch, Hloc, seq_len = 2, 2, 4
    inp = _make_spec_inputs(batch, Hloc, normal_seq_len=seq_len)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_spec_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        Hloc,
        state_indices=inp["state_indices"],
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = fused_kda_decode(
        inp["mixed_qkv"],
        fused_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        SPEC_D,
        Hloc,
        -5.0,
    )
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=0.05, rtol=0.02)


@pytest.mark.parametrize("weight_layout", ["channels_width", "group_width_channels"])
@pytest.mark.parametrize(
    "batch,Hloc,num_spec",
    [
        (4, 2, 7),
        # Hloc=12 is Kimi-K3 at TP8 (96 KDA heads). The gate used to require
        # Hloc==2, so serving silently ran the generic fallback kernel.
        (1, 12, 7),
        (2, 12, 7),
        (4, 12, 3),
        (2, 4, 1),
        (1, 12, 15),
    ],
)
def test_optimized_fused_spec_decode(weight_layout, batch, Hloc, num_spec):
    import aiter.ops.triton.gated_delta_net.fused_kda_decode as fkd
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip("parallel spec kernel is only dispatched on gfx950")

    spec_tokens = num_spec + 1
    inp = _make_spec_inputs(batch, Hloc, num_spec=num_spec, full_spec_sequence=True)
    inp["num_accepted_tokens"] = torch.tensor(
        [1 + (i * 3) % spec_tokens for i in range(batch)],
        dtype=torch.int32,
        device=device,
    )
    fused_weight = inp["conv_weight"]
    if weight_layout == "group_width_channels":
        fused_weight = fused_weight.reshape(3, Hloc * SPEC_D, SPEC_W).transpose(1, 2)
        fused_weight = fused_weight.contiguous()

    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_spec_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        Hloc,
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
        full_spec_sequence=True,
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    dispatched = _SpecKernelSpy(fkd)
    with dispatched:
        out = fused_kda_decode(
            inp["mixed_qkv"],
            fused_cs,
            fused_weight,
            inp["gate"],
            inp["beta"],
            inp["out_gate"],
            inp["A_log"],
            inp["dt_bias"],
            fused_ss,
            inp["state_indices"],
            inp["cu_seqlens"],
            inp["norm_weight"],
            1e-6,
            SPEC_D,
            Hloc,
            -5.0,
            num_accepted_tokens=inp["num_accepted_tokens"],
            conv_state_indices=inp["conv_state_indices"],
        )
    assert dispatched.grids, (
        "parallel spec kernel was not dispatched; fused_kda_decode silently fell "
        "back to fused_conv_recurrent_norm_kernel"
    )
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=0.05, rtol=0.02)


def test_optimized_fused_spec_decode_uses_real_cu_seqlens_with_padded_tokens():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip("parallel spec-7 kernel is only dispatched on gfx950")

    batch, Hloc, num_spec = 1, 2, 7
    inp = _make_spec_inputs(batch, Hloc, num_spec=num_spec, full_spec_sequence=True)
    inp["num_accepted_tokens"] = torch.tensor([4], dtype=torch.int32, device=device)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_spec_decode(
        inp["mixed_qkv"],
        ref_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        ref_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        Hloc,
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
        full_spec_sequence=True,
    )

    pad = 8
    mixed_qkv = torch.cat(
        [inp["mixed_qkv"], torch.zeros_like(inp["mixed_qkv"][:pad])], dim=0
    )
    gate = torch.cat([inp["gate"], torch.zeros_like(inp["gate"][:, :pad])], dim=1)
    beta = torch.cat([inp["beta"], torch.zeros_like(inp["beta"][:, :pad])], dim=1)
    out_gate = torch.cat(
        [inp["out_gate"], torch.zeros_like(inp["out_gate"][:pad])], dim=0
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = torch.zeros(
        mixed_qkv.shape[0], Hloc * SPEC_D, dtype=torch.bfloat16, device=device
    )
    fused_kda_decode(
        mixed_qkv,
        fused_cs,
        inp["conv_weight"],
        gate,
        beta,
        out_gate,
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        SPEC_D,
        Hloc,
        -5.0,
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
        out=out,
    )
    torch.testing.assert_close(out[:8], ref, atol=0.15, rtol=0.1)
    torch.testing.assert_close(out[8:], torch.zeros_like(out[8:]), atol=0, rtol=0)
    torch.testing.assert_close(fused_cs, ref_cs, atol=0, rtol=0)
    torch.testing.assert_close(fused_ss, ref_ss, atol=0.05, rtol=0.02)


def test_optimized_fused_spec_decode_skips_vllm_null_block():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if get_arch() != "gfx950":
        pytest.skip("parallel spec-7 kernel is only dispatched on gfx950")

    inp = _make_spec_inputs(1, 2, num_spec=7, full_spec_sequence=True)
    inp["state_indices"].zero_()
    inp["conv_state_indices"].zero_()
    before_cs, before_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = torch.zeros(8, 2 * SPEC_D, dtype=torch.bfloat16, device=device)
    fused_kda_decode(
        inp["mixed_qkv"],
        inp["conv_state"],
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        inp["state"],
        inp["state_indices"],
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        SPEC_D,
        2,
        -5.0,
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
        out=out,
    )
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    torch.testing.assert_close(inp["conv_state"], before_cs, atol=0, rtol=0)
    torch.testing.assert_close(inp["state"], before_ss, atol=0, rtol=0)
