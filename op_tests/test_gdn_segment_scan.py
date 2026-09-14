# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness coverage for the context-parallel GDN K5 path."""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_opt,
    gdn_prepare_fwd_flydsl,
)
from aiter.ops.prefill_batch_metadata import (
    build_gated_delta_rule_prefill_metadata,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill import (
    gdn_segment_scan as segment_scan_mod,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.gdn_segment_scan import (
    gdn_segment_scan_fwd,
)

pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 kernel")

H, HG, K, V = 16, 8, 128, 128


def _prepare(lengths: tuple[int, ...]):
    tokens = sum(lengths)
    torch.manual_seed(0)
    k = torch.randn(1, tokens, HG, K, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, tokens, H, V, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
    v = torch.nn.functional.normalize(v.float(), dim=-1).to(torch.bfloat16)
    g_raw = torch.full((1, tokens, H), -0.05, device="cuda")
    beta = torch.full((1, tokens, H), 0.7, device="cuda")
    w, u, g = gdn_prepare_fwd_flydsl(k=k, v=v, g=g_raw, beta=beta, use_exp2=True)
    offsets = (0, *tuple(sum(lengths[: i + 1]) for i in range(len(lengths))))
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    metadata = build_gated_delta_rule_prefill_metadata(
        lengths, cu_seqlens=cu, chunk_size=64
    )
    pool = torch.randn(8, H, V, K, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(len(lengths), dtype=torch.int32, device="cuda")
    return k, w, u, g, cu, metadata, pool, indices


def _stock(k, w, u, g, cu, metadata, pool, indices):
    return chunk_gated_delta_rule_fwd_h_flydsl_opt(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=pool.clone(),
        output_final_state=True,
        use_exp2=True,
        g_head_major=True,
        cu_seqlens=cu,
        prefill_metadata=metadata,
        initial_state_indices=indices,
    )


def test_gdn_segment_scan_matches_flydsl():
    k, w, u, g, _cu, _metadata, pool, _indices = _prepare((1024,))
    h0 = pool[:1].clone()
    expected = chunk_gated_delta_rule_fwd_h_flydsl_opt(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        use_exp2=True,
        g_head_major=True,
    )
    actual = gdn_segment_scan_fwd(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=h0,
        output_final_state=True,
        seq_lens=(1024,),
        snapshot_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        chunks_per_segment=4,
    )
    torch.testing.assert_close(actual[0], expected[0], atol=0.032, rtol=0.05)
    torch.testing.assert_close(actual[1], expected[1], atol=0.016, rtol=0.05)
    torch.testing.assert_close(actual[2], expected[2], atol=6e-4, rtol=0.05)


def test_gdn_segment_scan_packed_n2_matches_flydsl():
    k, w, u, g, cu, metadata, pool, indices = _prepare((768, 512))
    expected = _stock(k, w, u, g, cu, metadata, pool, indices)
    actual = gdn_segment_scan_fwd(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=pool.clone(),
        output_final_state=True,
        seq_lens=(768, 512),
        state_indices=indices,
        inplace_final_state=True,
        snapshot_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        chunks_per_segment=4,
    )
    torch.testing.assert_close(actual[0], expected[0], atol=0.04, rtol=0.05)
    torch.testing.assert_close(actual[1], expected[1], atol=0.04, rtol=0.05)


def test_wrapper_dispatches_n1_and_skips_n4(monkeypatch):
    calls: list[tuple[int, ...]] = []
    original = segment_scan_mod.gdn_segment_scan_fwd

    def _counting_fwd(**kwargs):
        calls.append(tuple(kwargs["seq_lens"]))
        return original(**kwargs)

    monkeypatch.setattr(segment_scan_mod, "gdn_segment_scan_fwd", _counting_fwd)
    monkeypatch.setenv("AITER_GDN_K5_SEGMENT_SCAN", "0")
    k, w, u, g, cu, metadata, pool, indices = _prepare((1024,))
    first = _stock(k, w, u, g, cu, metadata, pool, indices)
    second = _stock(k, w, u, g, cu, metadata, pool, indices)
    assert calls == []
    torch.testing.assert_close(first[0], second[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(first[1], second[1], atol=0.0, rtol=0.0)

    monkeypatch.setenv("AITER_GDN_K5_SEGMENT_SCAN", "1")
    monkeypatch.setenv("AITER_GDN_K5_SEGMENT_MIN_TOTAL_CHUNKS", "8")
    monkeypatch.setenv("AITER_GDN_K5_SEGMENT_CHUNKS", "4")
    dispatched = _stock(k, w, u, g, cu, metadata, pool, indices)
    assert calls == [(1024,)]
    torch.testing.assert_close(dispatched[0], first[0], atol=0.04, rtol=0.05)
    torch.testing.assert_close(dispatched[1], first[1], atol=0.04, rtol=0.05)

    calls.clear()
    packed = _prepare((256, 256, 256, 256))
    skipped = _stock(*packed)
    monkeypatch.setenv("AITER_GDN_K5_SEGMENT_SCAN", "0")
    stock = _stock(*packed)
    assert calls == []
    torch.testing.assert_close(skipped[0], stock[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(skipped[1], stock[1], atol=0.0, rtol=0.0)
