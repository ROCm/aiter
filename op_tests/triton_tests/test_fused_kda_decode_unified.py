# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for unified fused KDA decode kernel (all 4 modes).

Compares fused kernel against a pure-Python reference for:
  1. Normal decode (USE_REPLAY=False, IS_SPEC=False)
  2. Spec decode (USE_REPLAY=False, IS_SPEC=True)
  3. ReplaySSM (USE_REPLAY=True, IS_SPEC=False)
  4. Flush checkpoint (USE_REPLAY=True, write_pos near CAP)
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")

D = 128
W = 4
DTYPE = torch.bfloat16
DEVICE = "cuda"


def _ref_conv1d_step(x_qkv, conv_state, conv_weight, state_len):
    dim = x_qkv.shape[0]
    width = conv_weight.shape[1]
    out = torch.zeros(dim, dtype=torch.float32, device=DEVICE)
    for j in range(width - 1):
        idx = state_len - width + 1 + j
        out += conv_state[:, idx].float() * conv_weight[:, j].float()
    out += x_qkv.float() * conv_weight[:, width - 1].float()
    out = out * torch.sigmoid(out)
    conv_state[:, :-1] = conv_state[:, 1:].clone()
    conv_state[:, -1] = x_qkv
    return out.to(DTYPE)


def _ref_kda_step(q_h, k_h, v_h, gate_h, dt_h, A_h, beta_val, h_state, K, lower_bound):
    q_h = q_h * torch.rsqrt(torch.sum(q_h * q_h) + 1e-6) * (K**-0.5)
    k_h = k_h * torch.rsqrt(torch.sum(k_h * k_h) + 1e-6)
    g = lower_bound * torch.sigmoid(torch.exp(A_h) * (gate_h + dt_h))
    b_beta = torch.sigmoid(beta_val)
    h_state = h_state * torch.exp(g[None, :])
    dot = (h_state * k_h[None, :]).sum(1)
    u = (v_h - dot) * b_beta
    h_state = h_state + u[:, None] * k_h[None, :]
    o = (h_state * q_h[None, :]).sum(1)
    return o, h_state, k_h, u, g


def _ref_decode(
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
    lower_bound,
    state_indices=None,
    num_accepted_tokens=None,
    conv_state_indices=None,
    is_replay=False,
    buf_k=None,
    buf_u=None,
    buf_g=None,
    write_pos=None,
    slot_idx=None,
):
    T = mixed_qkv.shape[0]
    K = V = D
    lp = H * K
    batch = cu_seqlens.shape[0] - 1
    state_len = conv_state.shape[2]
    is_spec = num_accepted_tokens is not None
    out = torch.empty(T, lp, dtype=DTYPE, device=DEVICE)

    for n in range(batch):
        bos, eos = cu_seqlens[n].item(), cu_seqlens[n + 1].item()
        if eos - bos == 0:
            continue

        if is_replay:
            slot = slot_idx[n].item()
            conv_slot_idx = (
                conv_state_indices[n].item() if conv_state_indices is not None else slot
            )
            b_h = state[slot].clone().float()
            h_cur = max(write_pos[slot].item(), 0)
            cap = buf_k.shape[2]
            do_flush = h_cur + 2 * (eos - bos) > cap
            base = 0 if do_flush else h_cur
            if h_cur > 0:
                for hh in range(H):
                    rec_g = buf_g[slot, hh, :h_cur].float()
                    c = torch.cumsum(rec_g, dim=0)
                    ctot = rec_g.sum(dim=0)
                    w_decay = torch.exp(ctot[None, :] - c)
                    rec_k = buf_k[slot, hh, :h_cur].float()
                    rec_u = buf_u[slot, hh, :h_cur].float()
                    kw = rec_k * w_decay
                    b_h[hh] = b_h[hh] * torch.exp(ctot)[None, :] + rec_u.T @ kw
            if do_flush:
                state[slot] = b_h.to(state.dtype)
        elif is_spec:
            i_start = max(num_accepted_tokens[n].item() - 1, 0)
            s_idx = state_indices[n, i_start].item()
            conv_slot_idx = conv_state_indices[n].item()
            b_h = state[s_idx].clone().float()
        else:
            s_idx = state_indices[n].item()
            conv_slot_idx = s_idx
            b_h = state[s_idx].clone().float()

        for t in range(eos - bos):
            tok = bos + t
            qkv_out = _ref_conv1d_step(
                mixed_qkv[tok], conv_state[conv_slot_idx], conv_weight, state_len
            )
            q = qkv_out[:lp].reshape(H, K)
            k = qkv_out[lp : 2 * lp].reshape(H, K)
            v = qkv_out[2 * lp :].reshape(H, V)

            per_head = torch.empty(H, V, dtype=torch.float32, device=DEVICE)
            for hh in range(H):
                o_h, b_h[hh], k_out, u_out, g_out = _ref_kda_step(
                    q[hh].float(),
                    k[hh].float(),
                    v[hh].float(),
                    gate[0, tok, hh].float(),
                    dt_bias[hh * K : (hh + 1) * K].float(),
                    A_log[hh].float(),
                    beta[0, tok, hh].float(),
                    b_h[hh],
                    K,
                    lower_bound,
                )
                per_head[hh] = o_h
                if is_replay:
                    pos = base + t
                    buf_k[slot, hh, pos] = k_out.to(DTYPE)
                    buf_u[slot, hh, pos] = u_out.to(DTYPE)
                    buf_g[slot, hh, pos] = g_out.to(DTYPE)

            if is_replay:
                pass
            elif is_spec:
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

    return out


def _make_inputs(batch, Hloc, num_spec=0, replay=False, cap=64, write_pos_val=0):
    lp = Hloc * D
    state_len = (W - 1 + num_spec) if num_spec > 0 else (W - 1)
    num_slots = batch + num_spec * batch + 4
    torch.manual_seed(42)
    inp = {
        "mixed_qkv": torch.randn(batch, 3 * lp, dtype=DTYPE, device=DEVICE) * 0.1,
        "conv_weight": torch.randn(3 * lp, W, dtype=DTYPE, device=DEVICE) * 0.1,
        "conv_state": torch.randn(
            num_slots, 3 * lp, state_len, dtype=DTYPE, device=DEVICE
        )
        * 0.1,
        "gate": torch.randn(1, batch, Hloc, D, dtype=DTYPE, device=DEVICE) * 0.5,
        "beta": torch.randn(1, batch, Hloc, dtype=DTYPE, device=DEVICE),
        "out_gate": torch.randn(batch, lp, dtype=DTYPE, device=DEVICE),
        "A_log": torch.randn(Hloc, dtype=DTYPE, device=DEVICE) * 0.1,
        "dt_bias": torch.randn(lp, dtype=DTYPE, device=DEVICE) * 0.1,
        "state": torch.randn(num_slots, Hloc, D, D, dtype=torch.float32, device=DEVICE)
        * 0.01,
        "norm_weight": torch.ones(D, dtype=DTYPE, device=DEVICE),
        "cu_seqlens": torch.arange(batch + 1, dtype=torch.int64, device=DEVICE),
    }
    if num_spec > 0:
        inp["state_indices"] = torch.arange(
            batch * (1 + num_spec), dtype=torch.int32, device=DEVICE
        ).reshape(batch, 1 + num_spec)
        inp["num_accepted_tokens"] = torch.ones(batch, dtype=torch.int32, device=DEVICE)
        inp["conv_state_indices"] = torch.arange(
            batch, dtype=torch.int32, device=DEVICE
        )
    else:
        inp["state_indices"] = torch.arange(batch, dtype=torch.int32, device=DEVICE)

    if replay:
        inp["buf_k"] = (
            torch.randn(num_slots, Hloc, cap, D, dtype=DTYPE, device=DEVICE) * 0.01
        )
        inp["buf_u"] = (
            torch.randn(num_slots, Hloc, cap, D, dtype=DTYPE, device=DEVICE) * 0.01
        )
        inp["buf_g"] = (
            torch.randn(num_slots, Hloc, cap, D, dtype=DTYPE, device=DEVICE) * 0.01
        )
        inp["write_pos"] = torch.full(
            (num_slots,), write_pos_val, dtype=torch.int32, device=DEVICE
        )
        inp["slot_idx"] = torch.arange(batch, dtype=torch.int32, device=DEVICE)
    return inp


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("batch,Hloc", [(1, 12), (4, 12), (8, 4)])
def test_normal_decode(batch, Hloc):
    from aiter.ops.triton.gated_delta_net.fused_kda_decode_unified import (
        fused_kda_decode_unified,
    )

    inp = _make_inputs(batch, Hloc)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_decode(
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
        -5.0,
        state_indices=inp["state_indices"],
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = fused_kda_decode_unified(
        inp["mixed_qkv"],
        fused_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
        state_indices=inp["state_indices"],
    )
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)


@pytest.mark.parametrize("batch,Hloc,num_spec", [(1, 12, 3), (2, 4, 3)])
def test_spec_decode(batch, Hloc, num_spec):
    from aiter.ops.triton.gated_delta_net.fused_kda_decode_unified import (
        fused_kda_decode_unified,
    )

    inp = _make_inputs(batch, Hloc, num_spec=num_spec)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref = _ref_decode(
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
        -5.0,
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = fused_kda_decode_unified(
        inp["mixed_qkv"],
        fused_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
        state_indices=inp["state_indices"],
        num_accepted_tokens=inp["num_accepted_tokens"],
        conv_state_indices=inp["conv_state_indices"],
    )
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)


@pytest.mark.parametrize("batch,Hloc", [(1, 12), (2, 4)])
def test_replay_no_history(batch, Hloc):
    from aiter.ops.triton.gated_delta_net.fused_kda_decode_unified import (
        fused_kda_decode_unified,
    )

    inp = _make_inputs(batch, Hloc, replay=True, cap=64, write_pos_val=0)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref_bk, ref_bu, ref_bg = (
        inp["buf_k"].clone(),
        inp["buf_u"].clone(),
        inp["buf_g"].clone(),
    )
    ref = _ref_decode(
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
        -5.0,
        is_replay=True,
        buf_k=ref_bk,
        buf_u=ref_bu,
        buf_g=ref_bg,
        write_pos=inp["write_pos"].clone(),
        slot_idx=inp["slot_idx"],
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = fused_kda_decode_unified(
        inp["mixed_qkv"],
        fused_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
        buf_k=inp["buf_k"].clone(),
        buf_u=inp["buf_u"].clone(),
        buf_g=inp["buf_g"].clone(),
        write_pos=inp["write_pos"].clone(),
        slot_idx=inp["slot_idx"],
        max_query_len=4,
    )
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)


@pytest.mark.parametrize("batch,Hloc,wp", [(1, 4, 3), (2, 4, 5)])
def test_replay_with_history(batch, Hloc, wp):
    from aiter.ops.triton.gated_delta_net.fused_kda_decode_unified import (
        fused_kda_decode_unified,
    )

    inp = _make_inputs(batch, Hloc, replay=True, cap=64, write_pos_val=wp)
    ref_cs, ref_ss = inp["conv_state"].clone(), inp["state"].clone()
    ref_bk, ref_bu, ref_bg = (
        inp["buf_k"].clone(),
        inp["buf_u"].clone(),
        inp["buf_g"].clone(),
    )
    ref = _ref_decode(
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
        -5.0,
        is_replay=True,
        buf_k=ref_bk,
        buf_u=ref_bu,
        buf_g=ref_bg,
        write_pos=inp["write_pos"].clone(),
        slot_idx=inp["slot_idx"],
    )
    fused_cs, fused_ss = inp["conv_state"].clone(), inp["state"].clone()
    out = fused_kda_decode_unified(
        inp["mixed_qkv"],
        fused_cs,
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
        buf_k=inp["buf_k"].clone(),
        buf_u=inp["buf_u"].clone(),
        buf_g=inp["buf_g"].clone(),
        write_pos=inp["write_pos"].clone(),
        slot_idx=inp["slot_idx"],
        max_query_len=4,
    )
    torch.testing.assert_close(out, ref, atol=0.15, rtol=0.1)


@pytest.mark.parametrize("batch,Hloc", [(1, 4), (2, 4)])
def test_flush_checkpoint(batch, Hloc):
    from aiter.ops.triton.gated_delta_net.fused_kda_decode_unified import (
        fused_kda_decode_unified,
    )

    cap = 32
    wp = cap - 1
    inp = _make_inputs(batch, Hloc, replay=True, cap=cap, write_pos_val=wp)
    fused_ss = inp["state"].clone()
    ckpt_before = fused_ss[:batch].clone()
    fused_kda_decode_unified(
        inp["mixed_qkv"],
        inp["conv_state"].clone(),
        inp["conv_weight"],
        inp["gate"],
        inp["beta"],
        inp["out_gate"],
        inp["A_log"],
        inp["dt_bias"],
        fused_ss,
        inp["cu_seqlens"],
        inp["norm_weight"],
        1e-6,
        D,
        Hloc,
        -5.0,
        buf_k=inp["buf_k"].clone(),
        buf_u=inp["buf_u"].clone(),
        buf_g=inp["buf_g"].clone(),
        write_pos=inp["write_pos"].clone(),
        slot_idx=inp["slot_idx"],
        max_query_len=4,
    )
    ckpt_after = fused_ss[:batch]
    changed = (ckpt_after - ckpt_before).abs().max().item()
    assert changed > 0, "Checkpoint should be updated on flush"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
