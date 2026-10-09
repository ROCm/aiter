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
    do = torch.randn(o.shape, device="cuda", generator=gen)
    ds = torch.randn(s.shape, device="cuda", generator=gen) * 0.1
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
