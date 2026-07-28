# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.latent_moe_tail import (
    latent_moe_tail,
    supports_latent_moe_tail,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 latent-MoE local-tail specialization requires gfx950",
)

LATENT_DIM = 3584
HIDDEN_DIM = 7168
EPSILON = 1.0e-6


def _inputs(seed: int = 20260728):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    routed = torch.randn((1, LATENT_DIM), generator=generator).bfloat16().cuda()
    shared = torch.randn((1, HIDDEN_DIM), generator=generator).bfloat16().cuda()
    rms_weight = torch.randn(LATENT_DIM, generator=generator).bfloat16().cuda()
    up_weight = (
        torch.randn((HIDDEN_DIM, LATENT_DIM), generator=generator, dtype=torch.float32)
        .mul_(LATENT_DIM**-0.5)
        .bfloat16()
        .cuda()
    )
    return routed, shared, rms_weight, up_weight


def _oracle(routed, shared, rms_weight, up_weight):
    inverse_rms = torch.rsqrt(
        routed.float().square().mean(dim=-1, keepdim=True) + EPSILON
    )
    normalized = (routed.float() * inverse_rms * rms_weight.float()).bfloat16()
    projected = torch.mm(normalized.float(), up_weight.float().T).bfloat16()
    return (projected.float() + shared.float()).bfloat16()


@pytest.mark.parametrize("seed", [1, 17, 20260728])
def test_latent_moe_tail_matches_explicit_fp32_oracle(seed):
    routed, shared, rms_weight, up_weight = _inputs(seed)
    routed_before = routed.clone()
    shared_before = shared.clone()

    actual = latent_moe_tail(routed, shared, rms_weight, up_weight, EPSILON)
    expected = _oracle(routed, shared, rms_weight, up_weight)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.015625)
    torch.testing.assert_close(routed, routed_before, rtol=0, atol=0)
    torch.testing.assert_close(shared, shared_before, rtol=0, atol=0)


def test_latent_moe_tail_support_predicate_is_narrow():
    routed, shared, rms_weight, up_weight = _inputs()
    noncontiguous = torch.empty((LATENT_DIM, 2), dtype=torch.bfloat16, device="cuda")[
        :, 0
    ].unsqueeze(0)

    assert supports_latent_moe_tail(routed, shared, rms_weight, up_weight, EPSILON)
    assert not supports_latent_moe_tail(
        routed.expand(8, -1),
        shared.expand(8, -1),
        rms_weight,
        up_weight,
        EPSILON,
    )
    assert not supports_latent_moe_tail(
        routed.expand(16, -1),
        shared.expand(16, -1),
        rms_weight,
        up_weight,
        EPSILON,
    )
    assert not supports_latent_moe_tail(
        noncontiguous, shared, rms_weight, up_weight, EPSILON
    )
    assert not supports_latent_moe_tail(
        routed, shared, rms_weight, up_weight.float(), EPSILON
    )
    assert not supports_latent_moe_tail(
        routed, shared, rms_weight, up_weight, float("nan")
    )
    assert not supports_latent_moe_tail(
        routed, shared, rms_weight, up_weight, float("inf")
    )
    assert not supports_latent_moe_tail(routed, shared, rms_weight, up_weight, 0.0)


def test_latent_moe_tail_rejects_noncontiguous_input():
    _, shared, rms_weight, up_weight = _inputs()
    routed = torch.empty((LATENT_DIM, 2), dtype=torch.bfloat16, device="cuda")[
        :, 0
    ].unsqueeze(0)

    with pytest.raises(NotImplementedError, match="requires contiguous gfx950"):
        latent_moe_tail(routed, shared, rms_weight, up_weight, EPSILON)


def test_latent_moe_tail_graph_capture_and_output_reuse():
    routed, shared, rms_weight, up_weight = _inputs()
    out = torch.empty_like(shared)
    graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(graph):
        actual = latent_moe_tail(
            routed,
            shared,
            rms_weight,
            up_weight,
            EPSILON,
            out=out,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert actual is out
    torch.testing.assert_close(
        actual,
        _oracle(routed, shared, rms_weight, up_weight),
        rtol=0.01,
        atol=0.015625,
    )
