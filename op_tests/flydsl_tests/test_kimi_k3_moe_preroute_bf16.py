# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kimi_k3_moe_preroute_bf16 import (
    is_kimi_k3_moe_preroute_bf16_available,
    kimi_k3_moe_preroute_bf16,
    supports_kimi_k3_moe_preroute_bf16,
)
from aiter.ops.flydsl.utils import is_flydsl_available


def _relative_rmse(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    reference = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return (error / reference).item()


def test_support_predicate_fails_closed_on_cpu():
    hidden = torch.empty((1, 7168), dtype=torch.bfloat16)
    routed_weight = torch.empty((3584, 7168), dtype=torch.bfloat16)
    shared_gate_up_weight = torch.empty((1536, 7168), dtype=torch.bfloat16)
    shared_down_weight = torch.empty((7168, 768), dtype=torch.bfloat16)

    assert not supports_kimi_k3_moe_preroute_bf16(
        hidden,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a GPU runtime",
)
def test_backend_availability_matches_flydsl_and_architecture():
    assert is_kimi_k3_moe_preroute_bf16_available() == (
        is_flydsl_available() and get_gfx_runtime() == "gfx950"
    )


@pytest.mark.parametrize(
    ("situ_beta", "situ_linear_beta"),
    [
        (0.0, 25.0),
        (4.0, -1.0),
        (float("nan"), 25.0),
    ],
)
def test_preroute_rejects_invalid_situ_parameters(
    situ_beta: float,
    situ_linear_beta: float,
):
    tensor = torch.empty((1,), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="finite and positive"):
        kimi_k3_moe_preroute_bf16(
            tensor,
            tensor,
            tensor,
            tensor,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_flydsl_available()
    or get_gfx_runtime() != "gfx950",
    reason="requires FlyDSL on gfx950",
)
def test_kimi_k3_preroute_bf16_matches_reference():
    torch.manual_seed(20260729)
    device = torch.device("cuda")
    hidden = torch.randn(
        (1, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    routed_weight = torch.randn(
        (3584, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_gate_up_weight = torch.randn(
        (1536, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_down_weight = torch.randn(
        (7168, 768),
        device=device,
        dtype=torch.bfloat16,
    )
    original_hidden = hidden.clone()

    routed, shared = kimi_k3_moe_preroute_bf16(
        hidden,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )

    routed_reference = F.linear(
        hidden.float(),
        routed_weight.float(),
    ).to(torch.bfloat16)
    gate_up_reference = F.linear(
        hidden.float(),
        shared_gate_up_weight.float(),
    ).to(torch.bfloat16)
    gate, up = gate_up_reference.float().chunk(2, dim=-1)
    activated = (
        4.0
        * torch.tanh(gate / 4.0)
        * torch.sigmoid(gate)
        * 25.0
        * torch.tanh(up / 25.0)
    ).to(torch.bfloat16)
    shared_reference = F.linear(
        activated.float(),
        shared_down_weight.float(),
    ).to(torch.bfloat16)

    assert _relative_rmse(routed, routed_reference) < 2e-4
    assert _relative_rmse(shared, shared_reference) < 2e-4
    torch.testing.assert_close(
        hidden,
        original_hidden,
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_flydsl_available()
    or get_gfx_runtime() != "gfx950",
    reason="requires FlyDSL on gfx950",
)
def test_preroute_support_predicate_rejects_contract_families():
    device = torch.device("cuda")
    hidden = torch.empty((1, 7168), device=device, dtype=torch.bfloat16)
    routed_weight = torch.empty(
        (3584, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_gate_up_weight = torch.empty(
        (1536, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_down_weight = torch.empty(
        (7168, 768),
        device=device,
        dtype=torch.bfloat16,
    )

    def supports(
        hidden_arg=hidden,
        routed_weight_arg=routed_weight,
        shared_gate_up_weight_arg=shared_gate_up_weight,
        shared_down_weight_arg=shared_down_weight,
    ):
        return supports_kimi_k3_moe_preroute_bf16(
            hidden_arg,
            routed_weight_arg,
            shared_gate_up_weight_arg,
            shared_down_weight_arg,
        )

    assert supports()
    assert not supports(hidden_arg=hidden.expand(2, -1))
    assert not supports(hidden_arg=hidden.half())
    assert not supports(routed_weight_arg=routed_weight.transpose(0, 1))
    assert not supports(shared_gate_up_weight_arg=shared_gate_up_weight.transpose(0, 1))
    assert not supports(shared_down_weight_arg=shared_down_weight.transpose(0, 1))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_flydsl_available()
    or get_gfx_runtime() != "gfx950",
    reason="requires FlyDSL on gfx950",
)
def test_kimi_k3_preroute_bf16_graph_capture_and_replay():
    device = torch.device("cuda")
    hidden = torch.randn((1, 7168), device=device, dtype=torch.bfloat16)
    routed_weight = torch.zeros(
        (3584, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_gate_up_weight = torch.zeros(
        (1536, 7168),
        device=device,
        dtype=torch.bfloat16,
    )
    shared_down_weight = torch.zeros(
        (7168, 768),
        device=device,
        dtype=torch.bfloat16,
    )

    for _ in range(2):
        kimi_k3_moe_preroute_bf16(
            hidden,
            routed_weight,
            shared_gate_up_weight,
            shared_down_weight,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        routed, shared = kimi_k3_moe_preroute_bf16(
            hidden,
            routed_weight,
            shared_gate_up_weight,
            shared_down_weight,
        )
    graph.replay()
    expected_routed = routed.clone()
    expected_shared = shared.clone()
    graph.replay()

    torch.testing.assert_close(routed, expected_routed, atol=0, rtol=0)
    torch.testing.assert_close(shared, expected_shared, atol=0, rtol=0)
