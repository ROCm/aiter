# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL HERD routing tests for the supported model profiles."""

import pytest
import torch
import triton

import aiter.ops.topk as topk_mod
from aiter.ops.flydsl.herd_topk import herd_topk_gating
from aiter.ops.triton.moe.moe_routing.minunique import keepk_sort0
from aiter.ops.triton.moe.moe_routing.topk import topk as triton_topk
from aiter.ops.triton.utils._triton.arch_info import get_arch

_DSV4_EXPERTS = 384
_DSV4_TOPK = 6


def _skip_if_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("FlyDSL HERD routing requires a GPU")
    if get_arch() != "gfx950":
        pytest.skip("DeepSeek-V4 TP8 FlyDSL HERD is specialized for gfx950")


def _inputs(tokens: int):
    torch.manual_seed(2)
    logits = torch.randn(
        (tokens, _DSV4_EXPERTS), dtype=torch.bfloat16, device="cuda"
    )
    bias = (
        torch.randn(_DSV4_EXPERTS, dtype=torch.float32, device="cuda") * 0.1
    )
    return logits, bias


def _triton_herd(logits, bias, topk, score_func, renorm, route_scale):
    tokens = logits.shape[0]
    experts = logits.shape[1]
    hist_block_m = 32
    popularity = torch.zeros(experts, dtype=torch.int32, device=logits.device)
    candidate_values, candidate_ids, _ = triton_topk(
        logits,
        topk + 1,
        apply_softmax=False,
        score_mode=score_func,
        bias=bias,
        renorm=False,
        HIST_BLOCK_M=hist_block_m,
        pop_out=popularity,
    )
    num_blocks = triton.cdiv(tokens, hist_block_m)
    hist = torch.zeros(experts, dtype=torch.int32, device=logits.device)
    partials = torch.zeros(
        (num_blocks, experts), dtype=torch.int32, device=logits.device
    )
    return keepk_sort0(
        candidate_values,
        candidate_ids,
        popularity,
        hist,
        partials,
        experts,
        topk,
        apply_softmax=False,
        HIST_BLOCK_M=hist_block_m,
        apply_renorm=renorm,
        routed_scaling_factor=route_scale,
    )


@pytest.mark.parametrize("tokens", [16, 32, 64, 128])
@pytest.mark.parametrize("renorm", [False, True])
def test_dsv4_herd_matches_triton(tokens, renorm):
    _skip_if_unsupported()
    logits, bias = _inputs(tokens)
    route_scale = 2.5
    ref_weights, ref_ids = _triton_herd(
        logits,
        bias,
        _DSV4_TOPK,
        "sqrtsoftplus",
        renorm,
        route_scale,
    )

    weights = torch.empty(
        (tokens, _DSV4_TOPK), dtype=torch.float32, device="cuda"
    )
    ids = torch.empty((tokens, _DSV4_TOPK), dtype=torch.int32, device="cuda")
    herd_topk_gating(
        weights, ids, logits, bias, renorm, route_scale, "sqrtsoftplus"
    )

    assert torch.equal(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(
        weights, ref_weights.float(), atol=2e-3, rtol=2e-3
    )
    assert torch.all(ids[:, 1:] > ids[:, :-1])
    if renorm:
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.full((tokens,), route_scale, device="cuda"),
            atol=5e-3,
            rtol=0,
        )


@pytest.mark.parametrize("tokens", [16, 32, 64, 128])
def test_dsv4_herd_reduces_active_experts(monkeypatch, tokens):
    _skip_if_unsupported()
    logits, bias = _inputs(tokens)

    def run():
        weights = torch.empty(
            (tokens, _DSV4_TOPK), dtype=torch.float32, device="cuda"
        )
        ids = torch.empty(
            (tokens, _DSV4_TOPK), dtype=torch.int32, device="cuda"
        )
        topk_mod.topk_gating(
            weights,
            ids,
            logits,
            bias,
            need_renorm=True,
            routed_scaling_factor=2.5,
            score_func="sqrtsoftplus",
        )
        return weights, ids

    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", False)
    _, baseline_ids = run()
    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", True)
    monkeypatch.setattr(topk_mod, "_FLYDSL_HERD_MIN_M", 16)
    monkeypatch.setattr(topk_mod, "_FLYDSL_HERD_MAX_M", 128)
    weights, herd_ids = run()

    assert herd_ids.numel() == tokens * _DSV4_TOPK
    assert int(torch.unique(herd_ids).numel()) < int(torch.unique(baseline_ids).numel())
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.full((tokens,), 2.5, device="cuda"),
        atol=5e-3,
        rtol=0,
    )


def test_dsv4_herd_window_falls_back(monkeypatch):
    _skip_if_unsupported()
    tokens = 8
    logits, bias = _inputs(tokens)

    def run():
        weights = torch.empty(
            (tokens, _DSV4_TOPK), dtype=torch.float32, device="cuda"
        )
        ids = torch.empty(
            (tokens, _DSV4_TOPK), dtype=torch.int32, device="cuda"
        )
        topk_mod.topk_gating(weights, ids, logits, bias, True, 2.5, "sqrtsoftplus")
        return weights, ids

    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", False)
    ref_weights, ref_ids = run()
    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", True)
    monkeypatch.setattr(topk_mod, "_FLYDSL_HERD_MIN_M", 16)
    monkeypatch.setattr(topk_mod, "_FLYDSL_HERD_MAX_M", 128)
    got_weights, got_ids = run()

    assert torch.equal(got_ids, ref_ids)
    assert torch.equal(got_weights, ref_weights)


def test_dsv4_herd_negative_ties_match_triton():
    """Packed tie keys must still favor larger IDs for negative scores."""
    _skip_if_unsupported()
    tokens = 16
    logits = torch.full(
        (tokens, _DSV4_EXPERTS),
        -100.0,
        dtype=torch.bfloat16,
        device="cuda",
    )
    bias = torch.full(
        (_DSV4_EXPERTS,), -1.0, dtype=torch.float32, device="cuda"
    )
    ref_weights, ref_ids = _triton_herd(
        logits, bias, _DSV4_TOPK, "sqrtsoftplus", True, 2.5
    )

    weights = torch.empty(
        (tokens, _DSV4_TOPK), dtype=torch.float32, device="cuda"
    )
    ids = torch.empty((tokens, _DSV4_TOPK), dtype=torch.int32, device="cuda")
    herd_topk_gating(
        weights, ids, logits, bias, True, 2.5, "sqrtsoftplus"
    )

    assert torch.equal(ids, ref_ids.to(torch.int32))
    assert torch.equal(weights, ref_weights.float())


def test_dsv4_herd_dispatch_rejects_fp16(monkeypatch):
    _skip_if_unsupported()
    tokens = 16
    logits = torch.randn(
        (tokens, _DSV4_EXPERTS), dtype=torch.float16, device="cuda"
    )
    bias = torch.randn(_DSV4_EXPERTS, dtype=torch.float32, device="cuda")
    weights = torch.empty(
        (tokens, _DSV4_TOPK), dtype=torch.float32, device="cuda"
    )
    ids = torch.empty((tokens, _DSV4_TOPK), dtype=torch.int32, device="cuda")

    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", True)
    assert not topk_mod._use_flydsl_herd(
        weights, ids, logits, bias, "sqrtsoftplus"
    )


def test_dsv4_herd_dispatch_requires_bias(monkeypatch):
    _skip_if_unsupported()
    tokens = 16
    logits, _ = _inputs(tokens)
    weights = torch.empty(
        (tokens, _DSV4_TOPK), dtype=torch.float32, device="cuda"
    )
    ids = torch.empty((tokens, _DSV4_TOPK), dtype=torch.int32, device="cuda")
    empty_bias = torch.empty(0, dtype=torch.float32, device="cuda")

    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", True)
    assert not topk_mod._use_flydsl_herd(
        weights, ids, logits, empty_bias, "sqrtsoftplus"
    )


@pytest.mark.parametrize(
    "experts,topk,route_scale,row_strided",
    [
        (896, 16, 1.0, True),
        (128, 4, 2.0, False),
    ],
    ids=["kimi_k3", "minimax_m3"],
)
@pytest.mark.parametrize("tokens", [16, 32, 64, 128])
def test_sigmoid_herd_matches_triton(
    experts, topk, route_scale, row_strided, tokens
):
    _skip_if_unsupported()
    torch.manual_seed(3)
    logits = torch.randn(
        (tokens, experts + (8 if row_strided else 0)),
        dtype=torch.float32,
        device="cuda",
    )[:, :experts]
    assert logits.stride(1) == 1
    assert logits.is_contiguous() != row_strided
    bias = torch.randn(experts, dtype=torch.float32, device="cuda") * 0.1
    ref_weights, ref_ids = _triton_herd(
        logits, bias, topk, "sigmoid", True, route_scale
    )
    weights = torch.empty((tokens, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((tokens, topk), dtype=torch.int32, device="cuda")

    herd_topk_gating(
        weights, ids, logits, bias, True, route_scale, "sigmoid"
    )

    assert torch.equal(ids, ref_ids.to(torch.int32))
    torch.testing.assert_close(weights, ref_weights.float(), atol=2e-6, rtol=2e-6)
    assert torch.all(ids[:, 1:] > ids[:, :-1])
    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.full((tokens,), route_scale, device="cuda"),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.parametrize(
    "experts,topk,route_scale",
    [(896, 16, 1.0), (128, 4, 2.0)],
    ids=["kimi_k3", "minimax_m3"],
)
def test_sigmoid_herd_grouped_topk_dispatch_reduces_active_experts(
    monkeypatch, experts, topk, route_scale
):
    _skip_if_unsupported()
    tokens = 64
    torch.manual_seed(4)
    logits = torch.randn((tokens, experts), dtype=torch.float32, device="cuda")
    bias = torch.randn(experts, dtype=torch.float32, device="cuda") * 0.1

    # MiniMax writes routed Top-4 into the leading view of its E129/K5 buffer;
    # keeping that strided output here also covers the shared-expert contract.
    total_topk = topk + (experts == 128)
    weights = torch.empty((tokens, total_topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((tokens, total_topk), dtype=torch.int32, device="cuda")
    routed_weights = weights[:, :topk]
    routed_ids = ids[:, :topk]
    if total_topk != topk:
        weights[:, topk].fill_(1.0)
        ids[:, topk].fill_(experts)

    monkeypatch.setattr(topk_mod, "_FLYDSL_USE_HERD", True)
    monkeypatch.setattr(topk_mod, "_FLYDSL_HERD_MIN_M", 16)
    monkeypatch.setattr(topk_mod, "_FLYDSL_HERD_MAX_M", 128)
    topk_mod.biased_grouped_topk(
        logits,
        bias,
        routed_weights,
        routed_ids,
        num_expert_group=1,
        topk_group=1,
        need_renorm=True,
        routed_scaling_factor=route_scale,
    )

    baseline_ids = torch.topk(logits.sigmoid() + bias, topk, dim=1).indices
    assert int(torch.unique(routed_ids).numel()) < int(
        torch.unique(baseline_ids).numel()
    )
    torch.testing.assert_close(
        routed_weights.sum(dim=1),
        torch.full((tokens,), route_scale, device="cuda"),
        atol=2e-6,
        rtol=2e-6,
    )
    if total_topk != topk:
        assert torch.all(ids[:, topk] == experts)
        assert torch.all(weights[:, topk] == 1.0)
