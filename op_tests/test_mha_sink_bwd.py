# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Tests for mha_bwd / mha_varlen_bwd with sink gradient support.
#
# The sink_bwd feature adds two arguments to mha_bwd:
#   sink   : [batch, nhead] float32  – per-batch-per-head log-space sink score
#   d_sink : [nhead]        float32  – accumulator for the sink gradient (output)
#
# Reference formula (derived from kernel block_fmha_bwd_dot_do_o.hpp):
#   D[b, h, q]     = sum_j(dout[b, q, h, j] * out[b, q, h, j]) * p_undrop
#   P_sink[b, h, q] = exp(sink[b, h] - lse_fwd[b, h, q])
#   d_sink[h]      = sum_{b, q} (-P_sink[b, h, q] * D[b, h, q])

import pytest
import torch

import aiter
from aiter import dtypes, mha_bwd, mha_fwd, mha_varlen_bwd


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_qkvo(batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, hdim_v, dtype, device):
    """Return (q, k, v, dout) in BSHD layout, requires_grad=True."""
    q    = torch.randn(batch, seqlen_q, nhead,   hdim,   device=device, dtype=dtype).requires_grad_(True)
    k    = torch.randn(batch, seqlen_k, nhead_k, hdim,   device=device, dtype=dtype).requires_grad_(True)
    v    = torch.randn(batch, seqlen_k, nhead_k, hdim_v, device=device, dtype=dtype).requires_grad_(True)
    dout = torch.randn(batch, seqlen_q, nhead,   hdim_v, device=device, dtype=dtype)
    return q, k, v, dout


def run_fwd(q, k, v, softmax_scale, causal):
    """Run mha_fwd and return (out, lse)."""
    out, lse, _, _ = mha_fwd(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=False,
    )
    return out, lse


def reference_d_sink(dout, out, lse, sink, p_undrop=1.0):
    """
    Pure-PyTorch reference for d_sink.

    dout : [B, Sq, H, Dv]
    out  : [B, Sq, H, Dv]
    lse  : [B, H, Sq]       (forward LSE without sink)
    sink : [B, H]
    returns d_sink : [H]
    """
    # D[b, q, h] = sum_j(dout * out) * p_undrop  ->  shape [B, Sq, H]
    D_bsh = (dout.float() * out.float()).sum(dim=-1) * p_undrop  # [B, Sq, H]
    # reorder to [B, H, Sq] to align with lse
    D_bhs = D_bsh.permute(0, 2, 1)                               # [B, H, Sq]

    # P_sink[b, h, q] = exp(sink[b, h] - lse[b, h, q])
    sink_bhs = sink.unsqueeze(-1)                                 # [B, H, 1]
    p_sink = torch.exp(sink_bhs.float() - lse.float())           # [B, H, Sq]

    # d_sink[h] = sum_{b, q} (-P_sink * D)
    d_sink = (-p_sink * D_bhs).sum(dim=(0, 2))                   # [H]
    return d_sink.float()


# ---------------------------------------------------------------------------
# parametrize
# ---------------------------------------------------------------------------

DTYPES   = [dtypes.fp16, dtypes.bf16]
CAUSALS  = [False, True]
CONFIGS  = [
    # (batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim)
    (2, 128, 128, 4, 4, 64),
    (1, 64,  64,  6, 2, 128),
]


@pytest.mark.parametrize("causal",  CAUSALS)
@pytest.mark.parametrize("dtype",   DTYPES)
@pytest.mark.parametrize("batch,seqlen_q,seqlen_k,nhead,nhead_k,hdim", CONFIGS)
def test_mha_bwd_sink_dsink(batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, dtype, causal):
    """
    Verify that mha_bwd correctly accumulates d_sink.

    Strategy
    --------
    1. Run mha_fwd to obtain (out, lse).
    2. Create a random sink tensor in log-space [30, 60] and a zero d_sink buffer.
    3. Call mha_bwd with sink/d_sink.
    4. Compare the kernel d_sink with the PyTorch reference.
    """
    device = torch.device("cuda")
    hdim_v = hdim
    softmax_scale = hdim ** -0.5

    q, k, v, dout = make_qkvo(
        batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, hdim_v, dtype, device
    )

    # --- forward ---
    out, lse = run_fwd(q.detach(), k.detach(), v.detach(), softmax_scale, causal)

    # --- sink tensors ---
    # sink: [batch, nhead], uniform in [30, 60] in log-space
    sink = torch.empty(batch, nhead, device=device, dtype=torch.float32).uniform_(30.0, 60.0)
    d_sink = torch.zeros(nhead, device=device, dtype=torch.float32)

    # --- backward ---
    dq, dk, dv, softmax_d = mha_bwd(
        dout, q.detach(), k.detach(), v.detach(), out, lse,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        deterministic=False,
        sink=sink,
        d_sink=d_sink,
    )

    # d_sink must have been written (non-zero for non-trivial inputs)
    assert d_sink.abs().max() > 0, "d_sink was not updated by mha_bwd"

    # --- reference ---
    d_sink_ref = reference_d_sink(dout, out, lse, sink)

    # Tolerances: fp16/bf16 are noisy; use relatively loose absolute tolerance
    # because sink values are large (exp() amplifies small differences)
    rtol = 0.02
    atol = 0.5   # absolute tolerance in float units for d_sink
    torch.testing.assert_close(
        d_sink, d_sink_ref,
        rtol=rtol, atol=atol,
        msg=f"d_sink mismatch for dtype={dtype}, causal={causal}, "
            f"B={batch}, Sq={seqlen_q}, H={nhead}"
    )


@pytest.mark.parametrize("causal",  CAUSALS)
@pytest.mark.parametrize("dtype",   DTYPES)
@pytest.mark.parametrize("batch,seqlen_q,seqlen_k,nhead,nhead_k,hdim", CONFIGS)
def test_mha_bwd_with_sink_dq_dk_dv(batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, dtype, causal):
    """
    Verify that passing sink/d_sink does not corrupt the dQ, dK, dV outputs.

    We compare mha_bwd with sink=None (baseline) against mha_bwd with a
    near-zero sink (small values so the rescaling is negligible).
    The gradients should be numerically close.
    """
    device = torch.device("cuda")
    hdim_v = hdim
    softmax_scale = hdim ** -0.5

    q, k, v, dout = make_qkvo(
        batch, seqlen_q, seqlen_k, nhead, nhead_k, hdim, hdim_v, dtype, device
    )

    # --- forward ---
    out, lse = run_fwd(q.detach(), k.detach(), v.detach(), softmax_scale, causal)

    common_bwd_args = dict(
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=0 if causal else -1,
        deterministic=False,
    )

    # baseline: no sink
    dq_base, dk_base, dv_base, _ = mha_bwd(
        dout, q.detach(), k.detach(), v.detach(), out, lse,
        **common_bwd_args,
    )

    # with sink = very negative values → exp(sink - lse) ≈ 0 → no effect
    sink_small = torch.full((batch, nhead), -1000.0, device=device, dtype=torch.float32)
    d_sink     = torch.zeros(nhead, device=device, dtype=torch.float32)

    dq_sink, dk_sink, dv_sink, _ = mha_bwd(
        dout, q.detach(), k.detach(), v.detach(), out, lse,
        **common_bwd_args,
        sink=sink_small,
        d_sink=d_sink,
    )

    # With negligible sink, gradients should match the no-sink baseline
    rtol, atol = (0.01, 0.01) if dtype == dtypes.fp16 else (0.02, 0.02)
    torch.testing.assert_close(dq_sink, dq_base, rtol=rtol, atol=atol, msg="dQ mismatch with small sink")
    torch.testing.assert_close(dk_sink, dk_base, rtol=rtol, atol=atol, msg="dK mismatch with small sink")
    torch.testing.assert_close(dv_sink, dv_base, rtol=rtol, atol=atol, msg="dV mismatch with small sink")


@pytest.mark.parametrize("dtype", DTYPES)
def test_mha_bwd_sink_null_gives_same_as_no_sink(dtype):
    """Passing sink=None must give identical output to omitting sink entirely."""
    device = torch.device("cuda")
    batch, seqlen, nhead, hdim = 2, 64, 4, 64
    softmax_scale = hdim ** -0.5

    q, k, v, dout = make_qkvo(batch, seqlen, seqlen, nhead, nhead, hdim, hdim, dtype, device)
    out, lse = run_fwd(q.detach(), k.detach(), v.detach(), softmax_scale, False)

    common = dict(
        dropout_p=0.0, softmax_scale=softmax_scale,
        is_causal=False, window_size_left=-1, window_size_right=-1,
        deterministic=False,
    )

    dq1, dk1, dv1, d1 = mha_bwd(dout, q.detach(), k.detach(), v.detach(), out, lse, **common)
    dq2, dk2, dv2, d2 = mha_bwd(dout, q.detach(), k.detach(), v.detach(), out, lse, **common,
                                 sink=None, d_sink=None)

    torch.testing.assert_close(dq1, dq2, msg="dQ differs with sink=None vs omitted")
    torch.testing.assert_close(dk1, dk2, msg="dK differs with sink=None vs omitted")
    torch.testing.assert_close(dv1, dv2, msg="dV differs with sink=None vs omitted")
    torch.testing.assert_close(d1,  d2,  msg="softmax_d differs with sink=None vs omitted")


@pytest.mark.parametrize("dtype", DTYPES)
def test_mha_varlen_bwd_sink_dsink(dtype):
    """
    Smoke test: mha_varlen_bwd with sink/d_sink produces finite, non-zero d_sink
    and doesn't corrupt dQ/dK/dV shapes.

    In group (varlen) mode the CK kernel expects:
      lse:  [nhead, total_q]   (not the batch-mode [batch, nhead, seqlen])
      sink: [batch, nhead]     (one log-space score per batch-head pair)
    We derive these from a batch-mode forward pass.
    """
    device = torch.device("cuda")
    batch, seqlen, nhead, hdim = 2, 64, 4, 64
    hdim_v = hdim
    softmax_scale = hdim ** -0.5

    # build equal-length varlen inputs (no padding)
    cu_seqlens_q = torch.tensor([0, seqlen, seqlen * 2], device=device, dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_q.clone()
    total_q = seqlen * batch
    total_k = seqlen * batch

    q    = torch.randn(total_q, nhead, hdim,   device=device, dtype=dtype)
    k    = torch.randn(total_k, nhead, hdim,   device=device, dtype=dtype)
    v    = torch.randn(total_k, nhead, hdim_v, device=device, dtype=dtype)
    dout = torch.randn(total_q, nhead, hdim_v, device=device, dtype=dtype)

    # forward (batch mode) → convert outputs to group-mode shapes
    q_b = q.view(batch, seqlen, nhead, hdim)
    k_b = k.view(batch, seqlen, nhead, hdim)
    v_b = v.view(batch, seqlen, nhead, hdim_v)
    out_b, lse_b = run_fwd(q_b, k_b, v_b, softmax_scale, causal=False)

    out = out_b.view(total_q, nhead, hdim_v)

    # lse for group mode: [nhead, total_q]
    # lse_b is [batch, nhead, seqlen]; permute to [nhead, batch, seqlen] then flatten
    lse = lse_b.permute(1, 0, 2).reshape(nhead, total_q).contiguous()

    # sink: [batch, nhead], moderate log-space values
    sink   = torch.empty(batch, nhead, device=device, dtype=torch.float32).uniform_(30.0, 60.0)
    d_sink = torch.zeros(nhead, device=device, dtype=torch.float32)

    dq, dk, dv, _ = mha_varlen_bwd(
        dout, q, k, v, out, lse,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        zero_tensors=False,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        deterministic=False,
        sink=sink,
        d_sink=d_sink,
    )

    assert torch.isfinite(d_sink).all(), f"d_sink contains non-finite values: {d_sink}"
    assert d_sink.abs().max() > 0, "mha_varlen_bwd did not update d_sink"
    assert dq.shape == q.shape
    assert dk.shape == k.shape
    assert dv.shape == v.shape
