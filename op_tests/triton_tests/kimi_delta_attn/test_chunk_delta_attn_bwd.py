# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for the chunk_delta_attn backward pass, through ``chunk_kimi_delta_attn``.

Gradients are checked against autograd through the token-by-token fp32
reference in ``op_tests/triton_tests/utils/kda_ref.py``.
"""

import os

os.environ.setdefault("AITER_TRITON_ONLY", "1")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.kimi_delta_attn import chunk_fwd
from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
from op_tests.triton_tests.utils.kda_ref import chunk_kda_ref

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="KDA needs a GPU."
)

# RMS error relative to the reference's RMS. bf16 inputs land at ~3-5e-3.
_ERR_RATIO = 1e-2

_FLASH = {
    "use_qk_l2norm_in_kernel": True,
    "use_gate_in_kernel": True,
    "use_beta_sigmoid_in_kernel": True,
    "safe_gate": True,
    "lower_bound": -5.0,
}
_SOFTPLUS = {
    "use_qk_l2norm_in_kernel": True,
    "use_gate_in_kernel": True,
    "use_beta_sigmoid_in_kernel": True,
}


def _err_ratio(ref, x):
    d = (x.float() - ref.float()).flatten()
    ref_rms = ref.float().flatten().square().mean().sqrt()
    return (d.square().mean().sqrt() / (ref_rms + 1e-8)).item()


def _inputs(B, T, H, HV, K, V, varlen, state, precomputed_gate, seed=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def r(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=gen)

    Bx = 1 if varlen else B
    x = {
        "q": r(Bx, T, H, K),
        "k": r(Bx, T, H, K),
        "v": r(Bx, T, HV, V),
        "g": r(Bx, T, HV, K),
        "beta": r(Bx, T, HV, dtype=torch.float32),
    }
    if precomputed_gate:
        # A precomputed gate is a log-space decay, so it must be negative.
        x["g"] = F.logsigmoid(x["g"].float() + 3).to(torch.bfloat16)
    else:
        x["A_log"] = r(HV, dtype=torch.float32)
        x["dt_bias"] = r(HV * K, dtype=torch.float32)
    cu = None
    if varlen:
        cuts = torch.randint(
            1, T, (B - 1,), generator=torch.Generator().manual_seed(seed)
        )
        cu = torch.tensor(
            [0] + sorted(cuts.tolist()) + [T], dtype=torch.int32, device="cuda"
        )
    if state != "none":
        shape = (B, HV, V, K) if state == "h0_vfirst" else (B, HV, K, V)
        x["initial_state"] = r(*shape, dtype=torch.float32) * 0.1
    return x, cu


_CASES = {
    # name: (B, T, H, HV, K, V, varlen, state, opts)
    "flash": (2, 256, 2, 2, 128, 128, False, "h0", _FLASH),
    "flash_varlen_vfirst": (3, 300, 2, 2, 128, 128, True, "h0_vfirst", _FLASH),
    "safe_chunk64": (2, 200, 2, 2, 128, 128, False, "h0", dict(_FLASH, chunk_size=64)),
    "softplus_gva_varlen": (3, 250, 2, 4, 64, 64, True, "h0", _SOFTPLUS),
    "softplus_chunk32_k256": (
        1,
        130,
        2,
        2,
        256,
        128,
        False,
        "none",
        dict(_SOFTPLUS, chunk_size=32),
    ),
    "precomputed_gate": (
        2,
        150,
        2,
        2,
        128,
        128,
        False,
        "h0_vfirst",
        {"use_qk_l2norm_in_kernel": True, "use_beta_sigmoid_in_kernel": True},
    ),
    # Enough states that fwd_h and dhu take their wide tiles and launch apart.
    "wide_recurrences": (4, 128, 32, 32, 128, 128, False, "h0", _FLASH),
    # Kimi-K3: 96 heads x 128, a head count that is not a power of two.
    "k3_96_heads": (1, 200, 96, 96, 128, 128, False, "h0", _FLASH),
    "k3_96_heads_varlen": (2, 260, 96, 96, 128, 128, True, "h0", _SOFTPLUS),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_backward(case):
    """Gradients of every input match autograd through the fp32 recurrence."""
    B, T, H, HV, K, V, varlen, state, opts = _CASES[case]
    opts = dict(opts, state_v_first=state == "h0_vfirst")
    x, cu = _inputs(
        B,
        T,
        H,
        HV,
        K,
        V,
        varlen,
        state,
        precomputed_gate="use_gate_in_kernel" not in opts,
    )
    leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
    gen = torch.Generator(device="cuda").manual_seed(7)
    o, s = chunk_kimi_delta_attn(
        **leaves, cu_seqlens=cu, output_final_state=True, **opts
    )
    do = torch.randn(o.shape, device="cuda", dtype=torch.float32, generator=gen)
    ds = torch.randn(s.shape, device="cuda", dtype=torch.float32, generator=gen) * 0.1
    ((o.float() * do).sum() + (s.float() * ds).sum()).backward()

    ref = {n: t.detach().float().requires_grad_(True) for n, t in x.items()}
    ro, rs = chunk_kda_ref(
        **ref, cu_seqlens=cu, output_final_state=True, scale=K**-0.5, **opts
    )
    ((ro * do).sum() + (rs * ds).sum()).backward()
    for n in x:
        grad, ref_grad = leaves[n].grad, ref[n].grad
        assert grad is not None and grad.dtype == x[n].dtype, n
        assert torch.isfinite(grad).all(), n
        if n == "A_log":
            # A sum over every token of one head that largely cancels: bound the
            # error by the size of the terms, not of the (small) sum.
            gate_in = x["g"].float() + x["dt_bias"].float().view(HV, K)
            terms = (ref["g"].grad * gate_in).abs().sum((0, 1, 3))
            assert ((grad.float() - ref_grad).abs() <= 1e-3 * terms).all(), n
        else:
            assert _err_ratio(ref_grad, grad) < _ERR_RATIO, n


def test_backward_flash_matches_default(monkeypatch):
    """The backward does not depend on which path ran the forward."""
    x, cu = _inputs(2, 300, 2, 2, 128, 128, True, "h0", precomputed_gate=False)
    grads = []
    for flash in (True, False):
        monkeypatch.setattr(chunk_fwd, "AITER_FDA_ENABLE", flash)
        leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
        o, _ = chunk_kimi_delta_attn(**leaves, cu_seqlens=cu, **_FLASH)
        o.float().square().sum().backward()
        grads.append({n: t.grad for n, t in leaves.items()})
    for n in x:
        assert _err_ratio(grads[1][n], grads[0][n]) < _ERR_RATIO, n


def test_backward_partial_requires_grad():
    """Only the inputs that require grad get one; the forward is unchanged."""
    x, _ = _inputs(1, 128, 2, 2, 128, 128, False, "h0", precomputed_gate=False)
    o_ref, s_ref = chunk_kimi_delta_attn(**x, output_final_state=True, **_FLASH)
    v = x["v"].detach().clone().requires_grad_(True)
    o, s = chunk_kimi_delta_attn(**dict(x, v=v), output_final_state=True, **_FLASH)
    assert torch.equal(o, o_ref) and torch.equal(s, s_ref)
    o.float().sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert all(not t.requires_grad for t in x.values())


def test_backward_rejects_inference_only_args():
    x, _ = _inputs(1, 64, 2, 2, 128, 128, False, "none", precomputed_gate=False)
    x["q"].requires_grad_(True)
    with pytest.raises(ValueError, match="inference-only"):
        chunk_kimi_delta_attn(**x, out=torch.empty_like(x["v"]), **_FLASH)


def test_backward_matches_fla():
    """Gradients agree with fla's chunk_kda on the same call."""
    fla_kda = pytest.importorskip("fla.ops.kda")
    x, cu = _inputs(3, 400, 4, 4, 128, 128, True, "h0", precomputed_gate=False)
    opts = dict(_FLASH, chunk_size=64)
    grads = []
    for fn in (chunk_kimi_delta_attn, fla_kda.chunk_kda):
        kw = dict(opts)
        if fn is fla_kda.chunk_kda:
            del kw["chunk_size"]
        leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
        o, s = fn(**leaves, cu_seqlens=cu, output_final_state=True, **kw)
        (o.float().square().sum() + s.float().sum()).backward()
        grads.append({n: t.grad for n, t in leaves.items()})
    for n in x:
        assert _err_ratio(grads[1][n], grads[0][n]) < _ERR_RATIO, n


def _flydsl_vs_triton(x, cu, opts, final_state=True):
    """Gradients of every input from the Triton and FlyDSL backwards, same call."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    grads = []
    for backend in ("triton", "flydsl"):
        leaves = {n: t.detach().clone().requires_grad_(True) for n, t in x.items()}
        o, s = chunk_kimi_delta_attn(
            **leaves,
            cu_seqlens=cu,
            output_final_state=final_state,
            bwd_backend=backend,
            **opts,
        )
        if not grads:
            do = torch.randn(o.shape, device="cuda", generator=gen)
            ds = (
                torch.randn(s.shape, device="cuda", generator=gen)
                if final_state
                else None
            )
        loss = (o.float() * do).sum()
        if final_state:
            loss = loss + (s.float() * ds).sum()
        loss.backward()
        grads.append({n: t.grad for n, t in leaves.items()})
    for n in x:
        assert torch.isfinite(grads[1][n]).all(), n
        assert _err_ratio(grads[0][n], grads[1][n]) < _ERR_RATIO, n


@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("T", [100, 1000])
@pytest.mark.parametrize("state", ["none", "h0", "h0_vfirst"])
def test_backward_flydsl_matches_triton(state, T, varlen):
    """bwd_backend="flydsl" agrees with the Triton backward, dh0 and dht included.

    T=100 is one full chunk and a 36-token tail, as in #8739's layer test;
    varlen packs 3 sequences of random lengths, so sequence edges fall inside
    chunks and every sequence carries its own state.
    """
    pytest.importorskip("flydsl")
    x, cu = _inputs(3, T, 96, 96, 128, 128, varlen, state, precomputed_gate=False)
    opts = dict(_FLASH, chunk_size=64, state_v_first=state == "h0_vfirst")
    _flydsl_vs_triton(x, cu, opts)


# Lumen#35 timing shapes (all safe_gate), at the issue's heads and at K3's 96:
# (sequences, tokens per sequence, heads, varlen).
_LUMEN_SHAPES = [
    (1, 8192, 32, False),
    (1, 8192, 96, False),
    (2, 4096, 16, False),
    (2, 4096, 96, False),
    (1, 32768, 16, False),
    (1, 32768, 96, False),
    (8, 2048, 16, True),
    (8, 2048, 96, True),
]


@pytest.mark.parametrize("B,T,H,varlen", _LUMEN_SHAPES)
def test_backward_flydsl_lumen_shapes(B, T, H, varlen):
    """FlyDSL agrees with Triton on the Lumen#35 shapes, varlen 8x2048 included."""
    pytest.importorskip("flydsl")
    if varlen:
        x, _ = _inputs(1, B * T, H, H, 128, 128, False, "none", precomputed_gate=False)
        cu = torch.arange(0, B * T + 1, T, dtype=torch.int32, device="cuda")
    else:
        x, cu = _inputs(B, T, H, H, 128, 128, False, "none", precomputed_gate=False)
    _flydsl_vs_triton(x, cu, dict(_FLASH, chunk_size=64), final_state=False)


@pytest.mark.parametrize("l2norm", [False, True])
@pytest.mark.parametrize("gate", [False, "softplus", "safe"])
@pytest.mark.parametrize("sigmoid", [False, True])
def test_chunk_bwd_contract_flydsl(l2norm, gate, sigmoid):
    """The #8739 7-tuple adapter matches the Triton backward for every prologue fusion."""
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.kda_chunk_bwd import chunk_kda_bwd_flydsl
    from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_bwd import (
        chunk_delta_attn_bwd,
    )

    x, _ = _inputs(1, 1000, 96, 96, 128, 128, False, "none", precomputed_gate=not gate)
    if not l2norm:
        # Unnormalized keys make the delta rule diverge over 1000 tokens.
        x["q"], x["k"] = (F.normalize(x[n].float(), dim=-1).bfloat16() for n in "qk")
    if not sigmoid:
        x["beta"] = torch.sigmoid(x["beta"])
    gen = torch.Generator(device="cuda").manual_seed(3)
    do = torch.randn(x["v"].shape, device="cuda", generator=gen).to(torch.bfloat16)
    kw = {
        "use_qk_l2norm_in_kernel": l2norm,
        "use_gate_in_kernel": bool(gate),
        "A_log": x.get("A_log"),
        "dt_bias": x.get("dt_bias"),
        "use_beta_sigmoid_in_kernel": sigmoid,
    }
    if gate == "safe":
        # K3's safe gate: g = -5 * sigmoid(exp(A_log) * (g + dt_bias)).
        kw.update(lower_bound=-5.0, safe_gate=True)
    args = (x["q"], x["k"], x["v"], x["g"], x["beta"], 128**-0.5)
    got = chunk_kda_bwd_flydsl(do, *args, 64, **kw)
    ref = chunk_delta_attn_bwd(do, None, *args, chunk_size=64, **kw)[:7]
    assert len(got) == 7
    for name, a, b in zip(("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt"), got, ref):
        if b is None:
            assert a is None, name
            continue
        assert torch.isfinite(a).all(), name
        assert _err_ratio(b, a) < _ERR_RATIO, name
