# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for chunk_delta_attn forward pass and constituent kernels.

"""

import os

os.environ.setdefault("AITER_TRITON_ONLY", "1")
os.environ.setdefault("AITER_USE_SYSTEM_TRITON", "1")

import math

import pytest
import torch
import torch.nn.functional as F

# Attempt to import the FLA reference; skip if unavailable.
# Install flash-linear-attention (`pip install -e /path/to/fla`) to enable
# the consistency tests against the upstream reference.
try:
    from fla.ops.kda.chunk import chunk_kda

    HAS_FLA = True
except (ImportError, ModuleNotFoundError):
    HAS_FLA = False

from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_delta_attn_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn.gate import beta_sigmoid_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn.utils import l2norm_fwd
from aiter.ops.triton._triton_kernels.chunk_delta_attn.utils.cumsum import (
    chunk_gate_cumsum,
)

device = "cuda"
dtype = torch.bfloat16


def make_inputs(B, T, H, HV, K, V, seed=42):
    torch.manual_seed(seed)
    scale = 1.0 / math.sqrt(K)
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    g = torch.randn(B, T, HV, K, device=device, dtype=dtype) * 0.1
    beta = torch.randn(B, T, HV, device=device, dtype=dtype)
    A_log = torch.randn(HV, device=device, dtype=torch.float32).abs() * 0.5
    dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1
    return q, k, v, g, beta, A_log, dt_bias, scale


def _ref_gate_cumsum_softplus(g, A_log, chunk_size, scale=None, dt_bias=None):
    """Pure-torch reference: -exp(A) * softplus(g + bias), then chunk cumsum."""
    B, T, H, S = g.shape
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, S)
    gate = -A_log.exp().view(H, 1) * F.softplus(g.view(B * T, H, S)).view(B, T, H, S)
    out = torch.zeros_like(gate)
    for b in range(B):
        for h in range(H):
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                chunk = gate[b, t_start:t_end, h, :]
                out[b, t_start:t_end, h, :] = torch.cumsum(chunk, dim=0)
    if scale is not None:
        out = out * scale
    return out


def _ref_gate_cumsum_sigmoid(
    g, A_log, chunk_size, lower_bound, scale=None, dt_bias=None
):
    """Pure-torch reference: lower_bound * sigmoid(exp(A) * (g + bias)), then chunk cumsum."""
    B, T, H, S = g.shape
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, S)
    gate = lower_bound * torch.sigmoid(A_log.exp().view(H, 1) * g.view(B, T, H, S))
    out = torch.zeros_like(gate)
    for b in range(B):
        for h in range(H):
            for t_start in range(0, T, chunk_size):
                t_end = min(t_start + chunk_size, T)
                chunk = gate[b, t_start:t_end, h, :]
                out[b, t_start:t_end, h, :] = torch.cumsum(chunk, dim=0)
    if scale is not None:
        out = out * scale
    return out


@pytest.fixture(params=[(1, 64, 4, 32), (2, 128, 8, 64)])
def cumsum_inputs(request):
    B, T, H, S = request.param
    torch.manual_seed(42)
    g = torch.randn(B, T, H, S, device=device, dtype=dtype)
    A_log = torch.randn(H, device=device, dtype=torch.float32).abs() * 0.5
    return g, A_log, B, T, H, S


class TestChunkGateCumsum:

    def test_softplus_no_bias(self, cumsum_inputs):
        g, A_log, B, T, H, S = cumsum_inputs
        ref = _ref_gate_cumsum_softplus(g, A_log, 32)
        out = chunk_gate_cumsum(g, A_log, chunk_size=32, output_dtype=torch.float32)
        assert out.shape == (B, T, H, S)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_softplus_with_bias(self, cumsum_inputs):
        g, A_log, _B, _T, H, S = cumsum_inputs
        dt_bias = torch.randn(H * S, device=device, dtype=torch.float32) * 0.1
        ref = _ref_gate_cumsum_softplus(g, A_log, 32, dt_bias=dt_bias)
        out = chunk_gate_cumsum(
            g, A_log, chunk_size=32, dt_bias=dt_bias, output_dtype=torch.float32
        )
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_softplus_with_scale(self, cumsum_inputs):
        g, A_log, *_ = cumsum_inputs
        scale = 1.4426950408889634  # RCP_LN2
        ref = _ref_gate_cumsum_softplus(g, A_log, 32, scale=scale)
        out = chunk_gate_cumsum(
            g, A_log, chunk_size=32, scale=scale, output_dtype=torch.float32
        )
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_sigmoid_lower_bound(self, cumsum_inputs):
        g, A_log, *_ = cumsum_inputs
        lower_bound = -5.0
        ref = _ref_gate_cumsum_sigmoid(g, A_log, 32, lower_bound)
        out = chunk_gate_cumsum(
            g, A_log, chunk_size=32, lower_bound=lower_bound, output_dtype=torch.float32
        )
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("chunk_size", [32, 64])
    def test_chunk_size_variants(self, chunk_size):
        B, T, H, S = 1, 128, 4, 32
        torch.manual_seed(0)
        g = torch.randn(B, T, H, S, device=device, dtype=dtype)
        A_log = torch.randn(H, device=device, dtype=torch.float32).abs() * 0.5
        ref = _ref_gate_cumsum_softplus(g, A_log, chunk_size)
        out = chunk_gate_cumsum(
            g, A_log, chunk_size=chunk_size, output_dtype=torch.float32
        )
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_output_dtype_bfloat16(self, cumsum_inputs):
        g, A_log, *_ = cumsum_inputs
        out = chunk_gate_cumsum(g, A_log, chunk_size=32, output_dtype=torch.bfloat16)
        assert out.dtype == torch.bfloat16

    def test_output_dtype_float32(self, cumsum_inputs):
        g, A_log, *_ = cumsum_inputs
        out = chunk_gate_cumsum(g, A_log, chunk_size=32, output_dtype=torch.float32)
        assert out.dtype == torch.float32

    def test_triton_opt_disabled(self, cumsum_inputs, monkeypatch):
        """Non-optimised Triton kernel path also produces correct results."""
        monkeypatch.setenv("DA_TRITON_OPT", "0")
        g, A_log, *_ = cumsum_inputs
        ref = _ref_gate_cumsum_softplus(g, A_log, 32)
        out = chunk_gate_cumsum(g, A_log, chunk_size=32, output_dtype=torch.float32)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


class TestChunkDeltaAttnFwdSmoke:

    @pytest.mark.parametrize(
        "B,T,H,HV,K,V",
        [
            (1, 64, 4, 4, 64, 64),
            (2, 128, 8, 8, 64, 64),
            (1, 64, 2, 4, 64, 64),  # GVA: HV > H
            (1, 64, 64, 64, 128, 128),  # K=V=128, H=64
        ],
    )
    def test_output_shape(self, B, T, H, HV, K, V):
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        beta = beta_sigmoid_fwd(beta)
        o, final_state, *_ = chunk_delta_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=True,
            chunk_size=64,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )
        assert o.shape == (B, T, HV, V)
        assert final_state is not None
        assert final_state.shape == (B, HV, K, V)
        assert not torch.isnan(o).any()

    def test_no_gate_in_kernel(self):
        B, T, H, HV, K, V = 1, 64, 4, 4, 64, 64
        q, k, v, g, beta, _A_log, _dt_bias, scale = make_inputs(B, T, H, HV, K, V)
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        beta = beta_sigmoid_fwd(beta)
        o, _, *_ = chunk_delta_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            chunk_size=64,
            use_gate_in_kernel=False,
        )
        assert o.shape == (B, T, HV, V)
        assert not torch.isnan(o).any()

    def test_with_initial_state(self):
        B, T, H, HV, K, V = 1, 64, 4, 4, 64, 64
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        beta = beta_sigmoid_fwd(beta)
        h0 = torch.zeros(B, HV, K, V, device=device, dtype=torch.float32)
        o, fs, *_ = chunk_delta_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=h0,
            output_final_state=True,
            chunk_size=64,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )
        assert o.shape == (B, T, HV, V)
        assert fs.shape == (B, HV, K, V)


class TestChunkSize32:
    """The operator is chunking-invariant, so BT=32 and BT=64 must agree.

    Guards the inter_solve kernel, which used to load the Akk22/Akk33 diagonal
    blocks unconditionally and silently pulled data from the neighbouring chunk
    when NC < 3.
    """

    @pytest.mark.parametrize(
        "B,T,H,HV,K,V",
        [
            (1, 64, 4, 4, 64, 64),
            (2, 128, 8, 8, 64, 64),
            (1, 256, 4, 8, 128, 128),
        ],
    )
    @pytest.mark.parametrize("safe_gate", [False, True])
    def test_matches_chunk_size_64(self, B, T, H, HV, K, V, safe_gate):
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)

        def run(chunk_size):
            q2, _ = l2norm_fwd(q.clone())
            k2, _ = l2norm_fwd(k.clone())
            beta2 = beta_sigmoid_fwd(beta.clone())
            o, fs, *_ = chunk_delta_attn_fwd(
                q=q2,
                k=k2,
                v=v.clone(),
                g=g.clone(),
                beta=beta2,
                scale=scale,
                initial_state=None,
                output_final_state=True,
                chunk_size=chunk_size,
                safe_gate=safe_gate,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=-5.0,
            )
            return o.float(), fs.float()

        o32, s32 = run(32)
        o64, s64 = run(64)
        assert not torch.isnan(o32).any()
        torch.testing.assert_close(o32, o64, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(s32, s64, atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("safe_gate", [False, True])
    def test_varlen_matches_chunk_size_64(self, safe_gate):
        B, T, H, HV, K, V = 1, 384, 4, 4, 64, 64
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)
        cu_seqlens = torch.tensor([0, 96, 224, 384], device=device, dtype=torch.int32)

        def run(chunk_size):
            q2, _ = l2norm_fwd(q.clone())
            k2, _ = l2norm_fwd(k.clone())
            beta2 = beta_sigmoid_fwd(beta.clone())
            o, *_ = chunk_delta_attn_fwd(
                q=q2,
                k=k2,
                v=v.clone(),
                g=g.clone(),
                beta=beta2,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                chunk_size=chunk_size,
                safe_gate=safe_gate,
                cu_seqlens=cu_seqlens,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=-5.0,
            )
            return o.float()

        torch.testing.assert_close(run(32), run(64), atol=2e-2, rtol=2e-2)

    def test_unsupported_chunk_size_rejected(self):
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(1, 64, 4, 4, 64, 64)
        with pytest.raises(ValueError, match="32 or 64"):
            chunk_delta_attn_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=None,
                output_final_state=False,
                chunk_size=48,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=-5.0,
            )


class TestStateVFirst:
    """``state_v_first`` only changes how the recurrent state is laid out.

    The kernel accumulates the same registers either way and only the epilogue
    store (and the h0 load) is re-strided, so the results must agree bitwise --
    not just within tolerance.
    """

    @pytest.mark.parametrize("with_initial_state", [False, True])
    @pytest.mark.parametrize("varlen", [False, True])
    def test_matches_k_first_layout(self, with_initial_state, varlen):
        B, T, H, HV, K, V = (1, 384, 4, 4, 64, 64) if varlen else (2, 256, 4, 4, 64, 64)
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)
        cu_seqlens = (
            torch.tensor([0, 96, 224, 384], device=device, dtype=torch.int32)
            if varlen
            else None
        )
        N = B if cu_seqlens is None else len(cu_seqlens) - 1

        h0 = None
        if with_initial_state:
            torch.manual_seed(7)
            h0 = torch.randn(N, HV, K, V, device=device, dtype=torch.float32)

        def run(state_v_first):
            # h0 is expressed in whichever layout the call expects; both describe
            # the same mathematical state.
            init = None
            if h0 is not None:
                init = (
                    h0.transpose(-1, -2).contiguous() if state_v_first else h0.clone()
                )
            o, fs, *_ = chunk_delta_attn_fwd(
                q=q.clone(),
                k=k.clone(),
                v=v.clone(),
                g=g.clone(),
                beta=beta.clone(),
                scale=scale,
                initial_state=init,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_size=64,
                safe_gate=True,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=-5.0,
                use_qk_l2norm_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                state_v_first=state_v_first,
            )
            return o, fs

        o_kv, fs_kv = run(False)
        o_vk, fs_vk = run(True)

        assert fs_kv.shape == (N, HV, K, V)
        assert fs_vk.shape == (N, HV, V, K)
        assert not torch.isnan(fs_vk).any()
        torch.testing.assert_close(o_vk, o_kv, atol=0, rtol=0)
        torch.testing.assert_close(fs_vk, fs_kv.transpose(-1, -2), atol=0, rtol=0)

    def test_rejects_mismatched_initial_state_shape(self):
        B, T, H, HV, K, V = 1, 128, 4, 4, 64, 64
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)
        # [N, HV, K, V] is the K-first layout, rejected when state_v_first is on
        # (it is square here only if K == V, so use a non-square head dim pair).
        h0 = torch.zeros(B, HV, K, V + 64, device=device, dtype=torch.float32)
        with pytest.raises(ValueError, match="initial_state"):
            chunk_delta_attn_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                chunk_size=64,
                use_gate_in_kernel=True,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=-5.0,
            )


@pytest.mark.skipif(not HAS_FLA, reason="flash-linear-attention not available")
class TestChunkDeltaAttnFwdVsFLA:

    @pytest.mark.parametrize(
        "B,T,H,K,V",
        [
            (1, 64, 4, 64, 64),
            (2, 128, 8, 64, 64),
            (1, 64, 64, 128, 128),
        ],
    )
    def test_output_matches_fla(self, B, T, H, K, V):
        HV = H
        q, k, v, g, beta, A_log, dt_bias, scale = make_inputs(B, T, H, HV, K, V)

        o_ref, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=-5.0,
        )

        q2, _ = l2norm_fwd(q.clone())
        k2, _ = l2norm_fwd(k.clone())
        beta2 = beta_sigmoid_fwd(beta.clone())
        o_ait, *_ = chunk_delta_attn_fwd(
            q=q2,
            k=k2,
            v=v.clone(),
            g=g.clone(),
            beta=beta2,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            chunk_size=64,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=-5.0,
        )

        torch.testing.assert_close(
            o_ait.float(),
            o_ref.float(),
            atol=1e-2,
            rtol=1e-2,
            msg=f"Output mismatch for B={B} T={T} H={H} K={K} V={V}",
        )
