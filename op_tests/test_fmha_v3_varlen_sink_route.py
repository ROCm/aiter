# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Varlen FMHA must not send a set attention sink to the v3 ASM kernel.

fmha_v3_varlen_fwd has no sink argument and the launcher hardcodes a null
sink. On gfx942/gfx950, hdim 128 and hdim 192 used to take that path and
silently drop sink_ptr. Those calls belong on CK mha_varlen_fwd. Calls with
no sink stay on v3, except gfx950 hdim 192, which the OPUS kernel claims first.
"""

import pytest
import torch

import aiter
import aiter.ops.mha as mha
from aiter.jit.utils.chip_info import get_gfx


def _ref_varlen(q, k, v, cu_q, cu_k, causal, sink, scale):
    """fp32 reference. sink is a per-Q-head logit in the scaled-score domain."""
    outs = []
    hq = q.shape[1]
    rep = hq // k.shape[1]
    for b in range(cu_q.numel() - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        qb = q[qs:qe].float()
        kb = k[ks:ke].float().repeat_interleave(rep, dim=1)
        vb = v[ks:ke].float().repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * scale
        sq, sk = scores.shape[-2], scores.shape[-1]
        if causal:
            q_idx = torch.arange(sq, device=q.device)[:, None]
            k_idx = torch.arange(sk, device=q.device)[None, :]
            scores = scores.masked_fill(k_idx > (q_idx + (sk - sq)), float("-inf"))
        if sink is not None:
            virt = sink.float().view(hq, 1, 1).expand(hq, sq, 1)
            scores = torch.cat([scores, virt], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        if sink is not None:
            probs = probs[..., :-1]
        outs.append(torch.einsum("hqk,khd->qhd", probs, vb))
    return torch.cat(outs, dim=0)


def _expected_route(use_sink, hdim_q, gfx):
    if use_sink:
        return "ck"
    # gfx950 OPUS varlen claims hdim 192 before v3, and it rejects a sink.
    if hdim_q == 192 and gfx == "gfx950":
        return "opus"
    return "v3"


@pytest.mark.parametrize(
    "hdim_q,hdim_v,nhead_q,nhead_k",
    [
        (128, 128, 8, 8),
        (128, 128, 8, 2),
        (128, 128, 32, 4),  # North Mini GQA ratio, 32/4
        (128, 128, 32, 2),  # Command A Plus GQA ratio, 128/8
        (192, 128, 4, 4),
        (192, 128, 4, 2),
    ],
)
@pytest.mark.parametrize("use_sink", [False, True])
def test_varlen_fmha_v3_sink_routing(hdim_q, hdim_v, nhead_q, nhead_k, use_sink):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device required")
    gfx = get_gfx()
    if gfx not in ("gfx942", "gfx950"):
        pytest.skip(f"v3 varlen sink routing is gfx942/gfx950, got {gfx}")

    device = "cuda"
    seqlens_q = [24, 40]
    seqlens_k = [48, 64]
    torch.manual_seed(0)
    cu_q_vals = [0]
    cu_k_vals = [0]
    for sq, sk in zip(seqlens_q, seqlens_k):
        cu_q_vals.append(cu_q_vals[-1] + sq)
        cu_k_vals.append(cu_k_vals[-1] + sk)
    cu_q = torch.tensor(cu_q_vals, device=device, dtype=torch.int32)
    cu_k = torch.tensor(cu_k_vals, device=device, dtype=torch.int32)
    q = torch.randn(cu_q_vals[-1], nhead_q, hdim_q, device=device, dtype=torch.bfloat16)
    k = torch.randn(cu_k_vals[-1], nhead_k, hdim_q, device=device, dtype=torch.bfloat16)
    v = torch.randn(cu_k_vals[-1], nhead_k, hdim_v, device=device, dtype=torch.bfloat16)
    scale = hdim_q**-0.5
    sink = None
    if use_sink:
        sink = torch.linspace(-1.5, 2.5, nhead_q, device=device, dtype=torch.float32)

    calls = []
    real_v3 = mha.fmha_v3_varlen_fwd
    real_ck = mha.mha_varlen_fwd

    def wrap_v3(*args, **kwargs):
        calls.append("v3")
        return real_v3(*args, **kwargs)

    def wrap_ck(*args, **kwargs):
        calls.append("ck")
        return real_ck(*args, **kwargs)

    mha.fmha_v3_varlen_fwd = wrap_v3
    mha.mha_varlen_fwd = wrap_ck
    try:
        out = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            min_seqlen_q=1,
            dropout_p=0.0,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            sink_ptr=sink,
        )
        torch.cuda.synchronize()
    finally:
        mha.fmha_v3_varlen_fwd = real_v3
        mha.mha_varlen_fwd = real_ck

    route = calls[-1] if calls else "opus"
    expect = _expected_route(use_sink, hdim_q, gfx)
    assert route == expect, f"route {route}, expected {expect}"

    ref_sink = _ref_varlen(q, k, v, cu_q, cu_k, True, sink, scale)
    ref_nosink = _ref_varlen(q, k, v, cu_q, cu_k, True, None, scale)
    err = (out.float() - ref_sink).abs().max().item()
    err_nosink = (out.float() - ref_nosink).abs().max().item()
    assert err < 0.02, f"err vs reference {err}"
    if use_sink:
        assert err_nosink > 0.05, f"sink did not move the output ({err_nosink})"
