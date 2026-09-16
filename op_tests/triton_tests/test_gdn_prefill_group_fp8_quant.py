# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the gdn_prefill_group_fp8_quant dispatcher.

The op runs the fused single-launch Gluon kernel when it covers the call and the
four-kernel composition otherwise. These tests assert the dispatch is wired
correctly: on a covered call the ``"auto"`` default is bit-identical to forcing
``"gluon"`` and agrees with ``"compose"`` to fp8 tolerance; on an uncovered call
``"gluon"`` raises and ``"auto"`` falls back to (bit-identical) ``"compose"``.
The state layout (``store_state_transposed``) must be honored on both backends.

Covered calls need the gfx950 Gluon kernel (Triton >= 3.8); the covered-shape
test skips when it is unavailable.
"""

import pytest
import torch

from aiter.gdn_prefill_group_fp8_quant import gdn_prefill_group_fp8_quant

device = "cuda"
HEAD = 128


def _make_inputs(num_k_heads, num_v_heads, batch, seqlen, seed=0):
    torch.manual_seed(seed)
    m = batch * seqlen
    key_dim, value_dim = num_k_heads * HEAD, num_v_heads * HEAD
    conv_dim = 2 * key_dim + value_dim
    d_qkvz = 2 * key_dim + 2 * value_dim
    g = lambda *s: torch.randn(*s, dtype=torch.bfloat16, device=device) * 0.1
    inp = dict(
        projected_qkvz=g(m, d_qkvz),
        projected_ba=g(m, 2 * num_v_heads),
        conv_weight=g(conv_dim, 4),
        conv_bias=g(conv_dim),
        a_log=torch.randn(num_v_heads, device=device) * 0.1,
        dt_bias=torch.randn(num_v_heads, dtype=torch.bfloat16, device=device) * 0.1,
        norm_weight=torch.ones(HEAD, dtype=torch.bfloat16, device=device),
        cache_indices=torch.arange(batch, dtype=torch.int32, device=device),
        cu_seqlens=torch.arange(0, m + 1, seqlen, dtype=torch.int32, device=device),
        has_initial_state=torch.ones(batch, dtype=torch.bool, device=device),
    )
    cs0 = torch.randn(batch, conv_dim, 3, dtype=torch.bfloat16, device=device) * 0.1
    ds0 = torch.randn(batch, num_v_heads, HEAD, HEAD, dtype=torch.float32, device=device) * 0.1
    return inp, cs0, ds0


def _call(inp, cs0, ds0, backend, transposed):
    cs, ds = cs0.clone(), ds0.clone()
    out = gdn_prefill_group_fp8_quant(
        inp["projected_qkvz"], inp["projected_ba"], cs, ds,
        inp["cache_indices"], inp["cu_seqlens"], inp["has_initial_state"],
        inp["conv_weight"], inp["conv_bias"], inp["a_log"], inp["dt_bias"],
        inp["norm_weight"], scale=HEAD ** -0.5, eps=1e-6,
        return_bf16=True, store_state_transposed=transposed, backend=backend,
    )
    return out, ds


def _dequant(fp8, scales, num_v_heads):
    t, hd = fp8.shape
    return (fp8.reshape(t, num_v_heads, hd // num_v_heads).float() * scales[:, :, None]).reshape(t, hd)


def _relmae(a, b):
    a, b = a.float(), b.float()
    return (a - b).abs().mean() / b.abs().mean().clamp_min(1e-6)


def _gluon_covers(inp):
    from aiter.ops.triton.gated_delta_net import fused_gdn_prefill_qkvz_supported

    ok, _ = fused_gdn_prefill_qkvz_supported(
        inp["projected_qkvz"], inp["projected_ba"],
        torch.empty(inp["cache_indices"].numel(), inp["conv_weight"].shape[0], 3,
                    dtype=torch.bfloat16, device=device),
        torch.empty(inp["cache_indices"].numel(), inp["projected_ba"].shape[1] // 2,
                    HEAD, HEAD, dtype=torch.float32, device=device),
        inp["cache_indices"], inp["cu_seqlens"], inp["has_initial_state"],
        inp["conv_weight"], inp["conv_bias"], torch.float8_e4m3fn,
    )
    return ok


@pytest.mark.parametrize("transposed", [False, True])
def test_dispatch_covered_matches(transposed):
    """On a covered shape: auto == gluon (bit-identical) and gluon ~= compose."""
    inp, cs0, ds0 = _make_inputs(4, 8, 2, 1024)  # tokens=2048, covered
    if not _gluon_covers(inp):
        pytest.skip("fused Gluon prefill kernel unavailable (needs gfx950, Triton >= 3.8)")

    (comp, comp_ds) = _call(inp, cs0, ds0, "compose", transposed)
    (glu, glu_ds) = _call(inp, cs0, ds0, "gluon", transposed)
    (aut, aut_ds) = _call(inp, cs0, ds0, "auto", transposed)

    # auto must resolve to the same kernel as gluon on a covered call.
    assert torch.equal(aut[3].view(torch.uint8), glu[3].view(torch.uint8))
    assert torch.equal(aut[4], glu[4])
    assert torch.equal(aut_ds, glu_ds)

    # gluon vs the independent composition: fp8 e4m3 tolerance.
    act_g = _dequant(glu[3], glu[4].float(), 8)
    act_c = _dequant(comp[3], comp[4].float(), 8)
    assert _relmae(act_g, act_c) < 0.10
    # state is in the same layout on both paths (transpose-back handles False).
    assert _relmae(glu_ds, comp_ds) < 0.10


def test_uncovered_falls_back():
    """On an uncovered shape: gluon raises; auto falls back to compose exactly."""
    inp, cs0, ds0 = _make_inputs(4, 8, 1, 512)  # tokens=512 < 1024, uncovered

    with pytest.raises((ValueError, RuntimeError)):
        _call(inp, cs0, ds0, "gluon", False)

    (aut, aut_ds) = _call(inp, cs0, ds0, "auto", False)
    (comp, comp_ds) = _call(inp, cs0, ds0, "compose", False)
    assert torch.equal(aut[3].view(torch.uint8), comp[3].view(torch.uint8))
    assert torch.equal(aut[4], comp[4])
    assert torch.equal(aut_ds, comp_ds)


def test_invalid_backend_rejected():
    inp, cs0, ds0 = _make_inputs(4, 8, 1, 1024)
    with pytest.raises(ValueError):
        _call(inp, cs0, ds0, "nope", False)
