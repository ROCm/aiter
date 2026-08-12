# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness tests for the FlyDSL fused GDN K5+K6 forward.

Verifies ``chunk_gated_delta_rule_fwd_h_o_flydsl`` (inter-chunk state scan fused
with the inter/intra-chunk output) against a known-correct reference:

    reference o  =  Triton K6( q, k, v_new_ref, h_ref, g )

where ``h_ref`` / ``v_new_ref`` come from the pure-PyTorch K5 reference
(``ref_chunk_gated_delta_rule_fwd_h``). This decouples the fused output check
from the FlyDSL K5 kernel itself (whose correctness is covered by
``test_flydsl_linear_attention_prefill.py``).

Two levels:
  * ``test_fused_unit``     — call the fused wrapper directly.
  * ``test_fused_pipeline`` — call the end-to-end pipeline with
    ``use_chunk_flydsl_fused=True`` and compare to the Triton-only baseline.

Phase 0: the fused wrapper delegates to an unfused placeholder (K5 then K6);
these tests are the gate that keeps the real Phase-1 kernel honest.
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL fused K5+K6 tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_o_flydsl,
    )
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk,
    )
    from op_tests.flydsl_tests.test_flydsl_linear_attention_prefill import (
        ref_chunk_gated_delta_rule_fwd_h,
    )
except ImportError as exc:  # pragma: no cover
    pytest.skip(
        f"Unable to import FlyDSL fused K5+K6 dependencies: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")

_RMSE_TOL = 5e-2  # same tolerance as the K5 suite


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rmse_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).pow(2).mean().sqrt()
    denom = b.float().pow(2).mean().sqrt() + 1e-8
    return (diff / denom).item()


def _make_cu_seqlens(seq_lens: list[int], device="cuda") -> torch.Tensor:
    bounds = [0]
    for length in seq_lens:
        bounds.append(bounds[-1] + length)
    return torch.tensor(bounds, dtype=torch.int32, device=device)


def _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate, *, device="cuda"):
    """Build fused-forward inputs.

    Returns a dict with both token-major (reference) and head-major (kernel)
    layouts of w/u, plus q/k, the chosen gate tensor, h0, and cu_seqlens.
    """
    dtype = torch.bfloat16
    B = 1
    N = len(seq_lens)
    cu = _make_cu_seqlens(seq_lens, device) if N > 1 else None

    q = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(B, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    w_tm = torch.randn(B, T_flat, H, K, dtype=dtype, device=device) * 0.1
    u_tm = torch.randn(B, T_flat, H, V, dtype=dtype, device=device) * 0.1
    w_hm = w_tm.permute(0, 2, 1, 3).contiguous()
    u_hm = u_tm.permute(0, 2, 1, 3).contiguous()

    g = gk = None
    if gate == "g":
        g = (
            torch.randn(H, T_flat, dtype=torch.float32, device=device).abs() * -0.5
        ).cumsum(dim=1).contiguous()
    else:  # "gk"
        gk = (
            torch.randn(T_flat, H, K, dtype=torch.float32, device=device).abs() * -0.1
        ).cumsum(dim=0).contiguous()

    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01

    return {
        "q": q, "k": k, "w_tm": w_tm, "u_tm": u_tm, "w_hm": w_hm, "u_hm": u_hm,
        "g": g, "gk": gk, "h0": h0, "cu": cu, "N": N,
        "H": H, "Hg": Hg, "K": K, "V": V, "T_flat": T_flat,
    }


def _reference_o(inp, *, scale, use_exp2):
    """reference o via pure-PyTorch K5 ref then Triton K6.

    K5 ref produces token-major v_new [B, T, H, V] and h [B, NT, H, V, K];
    Triton K6 expects head-major v [B, H, T, V], so permute v_new. The K6 gate
    is scalar-``g`` only (KDA folds gk into K5), so pass g through and gk=None.
    """
    h_ref, v_new_ref, _ = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"], w=inp["w_tm"], u=inp["u_tm"],
        g=inp["g"], gk=inp["gk"], initial_state=inp["h0"],
        output_final_state=False, cu_seqlens=inp["cu"],
    )
    # token-major [B, T, H, V] -> head-major [B, H, T, V]
    v_hm = v_new_ref.permute(0, 2, 1, 3).contiguous().to(inp["u_tm"].dtype)
    o = inp["u_tm"].new_empty(inp["u_tm"].shape)  # [B, T, H, V]
    chunk_fwd_o_opt_vk(
        q=inp["q"], k=inp["k"], v=v_hm, o=o, h=h_ref.to(inp["u_tm"].dtype),
        g=inp["g"], scale=scale, cu_seqlens=inp["cu"], use_exp2=use_exp2,
    )
    return o


# --------------------------------------------------------------------------- #
# Unit test: fused wrapper vs reference
# --------------------------------------------------------------------------- #
# BV in {16, 32, 64} are all supported (Phase 2 Lever 2 enabled bv64 by aliasing
# lds_A onto the dead lds_h region). ``None`` exercises the auto BV heuristic.
@pytest.mark.parametrize("gate", ["g", "gk"])
@pytest.mark.parametrize(
    "H,Hg",
    [(12, 12), (24, 24), (4, 2)],  # MHA (KDA), MHA (KDA TP4), GQA (GDN)
)
@pytest.mark.parametrize("seq_lens", [[512], [512, 512], [640, 384, 512]])
@pytest.mark.parametrize("variant", ["bv16", "bv32", "bv64", None])
def test_fused_unit(gate, H, Hg, seq_lens, variant):
    """Fused wrapper output matches pure-PyTorch K5 ref + Triton K6."""
    K = V = 128
    T_flat = sum(seq_lens)
    scale = K ** -0.5
    use_exp2 = False  # inputs are natural-log gates (see K5 bench rationale)

    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, gate)

    o_ref = _reference_o(inp, scale=scale, use_exp2=use_exp2)

    o_fused, _ = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"], k=inp["k"], w=inp["w_hm"], u=inp["u_hm"],
        g=inp["g"], gk=inp["gk"], scale=scale,
        initial_state=inp["h0"], output_final_state=False,
        cu_seqlens=inp["cu"], use_exp2=use_exp2, variant=variant,
    )

    ratio = _rmse_ratio(o_fused, o_ref)
    assert ratio < _RMSE_TOL, (
        f"fused o mismatch: rmse_ratio={ratio:.3e} "
        f"(gate={gate} H={H} Hg={Hg} seqs={seq_lens} variant={variant})"
    )


def test_fused_final_state():
    """output_final_state=True returns a final state matching the K5 ref."""
    H = Hg = 12
    K = V = 128
    seq_lens = [512, 512]
    T_flat = sum(seq_lens)
    scale = K ** -0.5
    inp = _make_inputs(H, Hg, K, V, T_flat, seq_lens, "g")

    _, fs_ref, = ref_chunk_gated_delta_rule_fwd_h(
        k=inp["k"], w=inp["w_tm"], u=inp["u_tm"], g=inp["g"], gk=inp["gk"],
        initial_state=inp["h0"], output_final_state=True, cu_seqlens=inp["cu"],
    )[0::2]

    _, fs_fused = chunk_gated_delta_rule_fwd_h_o_flydsl(
        q=inp["q"], k=inp["k"], w=inp["w_hm"], u=inp["u_hm"],
        g=inp["g"], scale=scale, initial_state=inp["h0"],
        output_final_state=True, cu_seqlens=inp["cu"], use_exp2=False,
    )
    assert fs_fused is not None
    ratio = _rmse_ratio(fs_fused, fs_ref)
    assert ratio < _RMSE_TOL, f"final_state mismatch: rmse_ratio={ratio:.3e}"


# --------------------------------------------------------------------------- #
# Pipeline test: use_chunk_flydsl_fused vs Triton-only baseline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,Hg", [(4, 2), (8, 4)])
@pytest.mark.parametrize("seq_lens", [[512], [512, 512]])
def test_fused_pipeline(H, Hg, seq_lens):
    """End-to-end pipeline with use_chunk_flydsl_fused matches Triton baseline."""
    from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
        chunk_gated_delta_rule_fwd_opt_vk,
    )

    K = V = 128
    T_flat = sum(seq_lens)
    N = len(seq_lens)
    device = "cuda"
    dtype = torch.bfloat16
    cu = _make_cu_seqlens(seq_lens, device) if N > 1 else None
    scale = K ** -0.5

    q = torch.randn(1, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    k = torch.randn(1, T_flat, Hg, K, dtype=dtype, device=device) * 0.1
    v = torch.randn(1, T_flat, H, V, dtype=dtype, device=device) * 0.1
    g = torch.randn(1, T_flat, H, dtype=torch.float32, device=device) * -0.5
    beta = torch.randn(1, T_flat, H, dtype=torch.float32, device=device).sigmoid()
    h0 = torch.randn(N, H, V, K, dtype=torch.float32, device=device) * 0.01

    common = dict(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        initial_state=h0, output_final_state=True, cu_seqlens=cu,
    )
    _, o_base, fs_base = chunk_gated_delta_rule_fwd_opt_vk(**common)
    _, o_fused, fs_fused = chunk_gated_delta_rule_fwd_opt_vk(
        use_chunk_flydsl_fused=True, **common
    )

    ratio_o = _rmse_ratio(o_fused, o_base)
    assert ratio_o < _RMSE_TOL, f"pipeline o mismatch: rmse_ratio={ratio_o:.3e}"
    ratio_fs = _rmse_ratio(fs_fused, fs_base)
    assert ratio_fs < _RMSE_TOL, f"pipeline final_state mismatch: {ratio_fs:.3e}"
