# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FlyDSL KDA chunkwise prefill.

``flydsl_chunk_kda`` is checked against ``fla.ops.kda.chunk_kda`` on the exact
argument set Kimi-K3 uses: packed varlen input, raw gate and beta, a V-first
recurrent state, and an in-kernel q/k L2 norm.

Usage:
    pytest -sv op_tests/flydsl_tests/test_flydsl_kda.py
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip("flydsl is not installed.", allow_module_level=True)

fla_kda = pytest.importorskip("fla.ops.kda")

from aiter.ops.flydsl import flydsl_chunk_kda, flydsl_kda_supported

DK = DV = 128
LOWER_BOUND = -5.0
DEV = "cuda"

if not flydsl_kda_supported(DK, DV, torch.bfloat16, torch.device(DEV)):
    pytest.skip(
        "FlyDSL KDA needs gfx950 with 128-wide bf16 heads.", allow_module_level=True
    )


def _build(seqlens, num_heads, has_h0, saturated, seed):
    torch.manual_seed(seed)
    total, num_seqs = sum(seqlens), len(seqlens)
    shape_k = (1, total, num_heads, DK)
    q = torch.randn(shape_k, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(shape_k, device=DEV, dtype=torch.bfloat16)
    v = torch.randn((1, total, num_heads, DV), device=DEV, dtype=torch.bfloat16)
    # 30.0 saturates the sigmoid, pinning every gate channel at the lower bound
    # -- the worst-case decay a KDA chunk can accumulate.
    g = (
        torch.full(shape_k, 30.0, device=DEV, dtype=torch.bfloat16)
        if saturated
        else torch.randn(shape_k, device=DEV, dtype=torch.bfloat16)
    )
    # Kimi-K3 slices beta out of a wider fused projection, so it reaches the op
    # with a row stride of the fused width rather than num_heads.
    wide = torch.randn(total, 4 * num_heads, device=DEV, dtype=torch.float32)
    beta = wide[:, num_heads : 2 * num_heads].unsqueeze(0)
    A_log = torch.randn(num_heads, device=DEV, dtype=torch.float32)
    dt_bias = torch.randn(num_heads * DK, device=DEV, dtype=torch.float32)
    h0 = (
        torch.randn(num_seqs, num_heads, DV, DK, device=DEV, dtype=torch.float32)
        if has_h0
        else None
    )
    cu = torch.tensor(
        [0, *torch.tensor(seqlens).cumsum(0).tolist()], device=DEV, dtype=torch.int32
    )
    return q, k, v, g, beta, A_log, dt_bias, h0, cu


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize(
    "seqlens",
    [
        [1024] * 10,  # a typical packed prefill step
        [1024],
        [32],  # a single chunk, the shortest legal prefill
        [1000] * 8,  # lengths that are not chunk aligned
        [900, 1024, 1000, 1024],  # ragged
        [2048] * 5,
    ],
)
@pytest.mark.parametrize("has_h0", [True, False])
@pytest.mark.parametrize("saturated", [False, True])
@pytest.mark.parametrize("num_heads", [12])
def test_matches_fla(seqlens, has_h0, saturated, num_heads):
    q, k, v, g, beta, A_log, dt_bias, h0, cu = _build(
        seqlens, num_heads, has_h0, saturated, seed=0
    )
    common = {
        "A_log": A_log,
        "dt_bias": dt_bias,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": LOWER_BOUND,
    }
    o_ref, ht_ref = fla_kda.chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta.float(),
        initial_state=None if h0 is None else h0.clone(),
        cu_seqlens=cu.long(),
        safe_gate=True,
        state_v_first=True,
        disable_recompute=True,
        **common,
    )
    result = flydsl_chunk_kda(
        q,
        k,
        v,
        g,
        beta.float(),
        cu_seqlens=cu,
        max_seqlen=max(seqlens),
        initial_state=None if h0 is None else h0.clone(),
        state_v_first=True,
        **common,
    )
    assert result is not None, "expected FlyDSL to support this shape"
    o, ht = result

    assert not torch.isnan(o).any(), "output has NaNs"
    assert not torch.isnan(ht).any(), "final state has NaNs"
    assert o.shape == o_ref.shape
    assert ht.shape == ht_ref.shape
    # Both sides run bf16 MFMA/MMA with different chunk sizes and accumulation
    # orders, so agreement is bounded by bf16 rounding rather than by the math.
    assert _rel(o, o_ref) < 2e-2
    assert _rel(ht, ht_ref) < 2e-2


def test_raw_bf16_beta_matches_fp32_beta():
    """The pack kernel widens beta before the sigmoid, so a bf16 beta must give
    the same write strength as pre-casting it to fp32 on the host."""
    seqlens = [1024, 1024]
    q, k, v, g, beta, A_log, dt_bias, h0, cu = _build(seqlens, 12, True, False, seed=3)
    beta_bf16 = beta.to(torch.bfloat16)
    common = {
        "cu_seqlens": cu,
        "max_seqlen": max(seqlens),
        "A_log": A_log,
        "dt_bias": dt_bias,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": LOWER_BOUND,
        "state_v_first": True,
    }
    o_wide, ht_wide = flydsl_chunk_kda(
        q, k, v, g, beta_bf16.float(), initial_state=h0.clone(), **common
    )
    o_raw, ht_raw = flydsl_chunk_kda(
        q, k, v, g, beta_bf16, initial_state=h0.clone(), **common
    )
    assert torch.equal(o_wide, o_raw)
    assert torch.equal(ht_wide, ht_raw)


def test_writes_into_supplied_out():
    seqlens = [512, 512]
    q, k, v, g, beta, A_log, dt_bias, h0, cu = _build(seqlens, 12, True, False, seed=1)
    out = torch.empty(sum(seqlens), 12, DV, device=DEV, dtype=torch.bfloat16)
    result = flydsl_chunk_kda(
        q,
        k,
        v,
        g,
        beta.float(),
        cu_seqlens=cu,
        max_seqlen=max(seqlens),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
        out=out,
    )
    assert result is not None
    o, _ = result
    assert o.data_ptr() == out.data_ptr()


@pytest.mark.parametrize(
    "seqlens,supported",
    [
        ([1024] * 4, True),
        ([2048] * 4, True),  # wide enough to hide the serial chunk walk
        ([2048], False),  # same length, too narrow
        ([4096] * 2, False),  # past the walk's crossover at any width
        ([8192], False),
    ],
)
def test_declines_shapes_it_would_run_slower_than_fla(seqlens, supported):
    """Past ~2K tokens/seq the serial chunk walk stops paying off, so those
    shapes report unsupported even though the kernel would compute them correctly."""
    q, k, v, g, beta, A_log, dt_bias, h0, cu = _build(seqlens, 12, True, False, seed=4)
    result = flydsl_chunk_kda(
        q,
        k,
        v,
        g,
        beta.float(),
        cu_seqlens=cu,
        max_seqlen=max(seqlens),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
    )
    assert (result is not None) == supported


def test_lopsided_batch_falls_back():
    """A batch that would pad far past its token count reports unsupported."""
    seqlens = [37, 128, 4096, 55]
    q, k, v, g, beta, A_log, dt_bias, h0, cu = _build(seqlens, 12, True, False, seed=2)
    result = flydsl_chunk_kda(
        q,
        k,
        v,
        g,
        beta.float(),
        cu_seqlens=cu,
        max_seqlen=max(seqlens),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=LOWER_BOUND,
        state_v_first=True,
    )
    assert result is None
