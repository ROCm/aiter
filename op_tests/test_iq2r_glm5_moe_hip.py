# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Diagnostic gfx950 coverage for Flash and plain GLM-5.3 IQ2R MoE.

The tensors in this test are synthetic and use uniform importance.  They prove
the production expert/top-k geometries and GLM SwiGLU ABIs, not O0 model quality.
"""

from types import SimpleNamespace

import pytest
import torch

from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.iq2r import (
    iq2r_encode_device,
    iq2r_materialize_device,
    iq2r_swiglu_out,
)
from aiter.ops.iq2r_encoder import iq2r_initial_codebook
from aiter.ops.iq2r_format import IQ2RMetadata

_TOPK = 8
_INTERMEDIATE = 2048

_CASES = (
    SimpleNamespace(
        name="flash",
        experts=288,
        hidden=4096,
        swiglu_limit=10.0,
        seed=0x5310,
    ),
    SimpleNamespace(
        name="plain",
        experts=256,
        hidden=6144,
        swiglu_limit=0.0,
        seed=0x5330,
    ),
)


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


pytestmark = pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")


def _dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scales.float() - 127.0)
    return (values.float().reshape(values.shape[0], -1, 32) * scale[..., None]).reshape(
        values.shape
    )


def _relative_metrics(actual: torch.Tensor, expected: torch.Tensor):
    actual = actual.float()
    expected = expected.float()
    relative_rmse = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1), expected.reshape(1, -1), dim=-1
    )[0]
    return relative_rmse, cosine


@pytest.fixture(scope="module", params=_CASES, ids=lambda case: case.name)
def glm53_synthetic_iq2r(request):
    case = request.param
    gate_metadata = IQ2RMetadata(logical_n=2 * _INTERMEDIATE, logical_k=case.hidden)
    down_metadata = IQ2RMetadata(logical_n=case.hidden, logical_k=_INTERMEDIATE)
    importance = torch.ones((case.hidden,), dtype=torch.float32, device="cuda")
    down_importance = importance[:_INTERMEDIATE].contiguous()
    codebook = iq2r_initial_codebook("cuda")

    def encode(n: int, k: int, seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = (
            torch.randn((n, k), generator=generator, device="cuda") * 0.01
        ).contiguous()
        return iq2r_encode_device(
            weight,
            importance if k == case.hidden else down_importance,
            codebook,
            exponent_radius=8,
        )

    gate_data_0, gate_auxiliary_0 = encode(2 * _INTERMEDIATE, case.hidden, case.seed)
    gate_data_last, gate_auxiliary_last = encode(
        2 * _INTERMEDIATE, case.hidden, case.seed + 1
    )
    down_data_0, down_auxiliary_0 = encode(case.hidden, _INTERMEDIATE, case.seed + 0x10)
    down_data_last, down_auxiliary_last = encode(
        case.hidden, _INTERMEDIATE, case.seed + 0x11
    )

    gate_data = gate_data_0.expand(case.experts, -1).contiguous()
    gate_auxiliary = gate_auxiliary_0.expand(case.experts, -1).contiguous()
    down_data = down_data_0.expand(case.experts, -1).contiguous()
    down_auxiliary = down_auxiliary_0.expand(case.experts, -1).contiguous()
    gate_data[-1].copy_(gate_data_last)
    gate_auxiliary[-1].copy_(gate_auxiliary_last)
    down_data[-1].copy_(down_data_last)
    down_auxiliary[-1].copy_(down_auxiliary_last)

    return SimpleNamespace(
        case=case,
        gate_metadata=gate_metadata,
        down_metadata=down_metadata,
        gate_data=gate_data,
        gate_auxiliary=gate_auxiliary,
        down_data=down_data,
        down_auxiliary=down_auxiliary,
        gate_weights={
            False: iq2r_materialize_device(
                gate_data, gate_auxiliary, gate_metadata, expert_index=0
            ),
            True: iq2r_materialize_device(
                gate_data,
                gate_auxiliary,
                gate_metadata,
                expert_index=case.experts - 1,
            ),
        },
        down_weights={
            False: iq2r_materialize_device(
                down_data, down_auxiliary, down_metadata, expert_index=0
            ),
            True: iq2r_materialize_device(
                down_data,
                down_auxiliary,
                down_metadata,
                expert_index=case.experts - 1,
            ),
        },
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_glm53_swiglu_uses_adjacent_gate_up_and_model_parameters(case):
    generator = torch.Generator(device="cuda").manual_seed(0x53A6)
    gate_up = (
        torch.randn((7, 2 * _INTERMEDIATE), generator=generator, device="cuda") * 12.0
    ).to(torch.bfloat16)
    actual = torch.empty((7, _INTERMEDIATE), dtype=torch.bfloat16, device="cuda")

    iq2r_swiglu_out(
        gate_up,
        actual,
        limit=case.swiglu_limit,
        alpha=1.0,
        up_offset=0.0,
    )

    gate = gate_up[:, 0::2].float()
    up = gate_up[:, 1::2].float()
    if case.swiglu_limit > 0:
        gate = gate.clamp(max=case.swiglu_limit)
        up = up.clamp(min=-case.swiglu_limit, max=case.swiglu_limit)
    expected = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_glm53_full_moe_matches_materialized_oracle(glm53_synthetic_iq2r, tokens):
    fixture = glm53_synthetic_iq2r
    case = fixture.case
    generator = torch.Generator(device="cuda").manual_seed(0x5300 + tokens)
    hidden = (
        torch.randn((tokens, case.hidden), generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    expert_row = torch.tensor(
        [0, 17, 65, 127, 160, 223, case.experts - 2, case.experts - 1],
        dtype=torch.int32,
        device="cuda",
    )
    topk_ids = torch.stack(
        [torch.roll(expert_row, shifts=token) for token in range(tokens)]
    ).contiguous()
    topk_weights = torch.softmax(
        torch.randn(
            (tokens, _TOPK), generator=generator, dtype=torch.float32, device="cuda"
        ),
        dim=-1,
    ).contiguous()
    workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        _TOPK,
        device="cuda",
        max_experts=case.experts,
        hidden_size=case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    output = torch.empty_like(hidden)

    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data,
        fixture.gate_auxiliary,
        fixture.down_data,
        fixture.down_auxiliary,
        topk_weights,
        topk_ids,
        output,
        gate_up_metadata=fixture.gate_metadata,
        down_metadata=fixture.down_metadata,
        gate_up_tile_n=128,
        down_tile_n=128,
        gate_up_bias=None,
        down_bias=None,
        workspace=workspace,
        swiglu_limit=case.swiglu_limit,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )

    routes = tokens * _TOPK
    sorted_ids = workspace.sorted_expert_ids[:routes]
    if routes <= 16:
        torch.testing.assert_close(
            workspace.gather_indices[:routes],
            torch.arange(routes, dtype=torch.int32, device="cuda"),
            rtol=0,
            atol=0,
        )
        assert int(workspace.task_count.item()) == routes
    else:
        flat_ids = topk_ids.reshape(-1)
        expected_gather = torch.argsort(flat_ids, stable=True).to(torch.int32)
        torch.testing.assert_close(
            workspace.gather_indices[:routes], expected_gather, rtol=0, atol=0
        )
        assert int(workspace.task_count.item()) == _TOPK

    route_input = _dequant_mxfp8(
        workspace.route_input_fp8[:routes], workspace.route_input_scales[:routes]
    )
    expected_gate_up = torch.empty_like(workspace.gate_up[:routes])
    for is_last_expert in (False, True):
        selected = (
            (sorted_ids == case.experts - 1)
            if is_last_expert
            else (sorted_ids != case.experts - 1)
        )
        expected_gate_up[selected] = (
            route_input[selected] @ fixture.gate_weights[is_last_expert].T
        ).to(torch.bfloat16)
    gate_rmse, gate_cosine = _relative_metrics(
        workspace.gate_up[:routes], expected_gate_up
    )
    assert gate_rmse.item() < 0.01
    assert gate_cosine.item() > 0.999

    gate = workspace.gate_up[:routes, 0::2].float()
    up = workspace.gate_up[:routes, 1::2].float()
    if case.swiglu_limit > 0:
        gate = gate.clamp(max=case.swiglu_limit)
        up = up.clamp(min=-case.swiglu_limit, max=case.swiglu_limit)
    expected_activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    activated = _dequant_mxfp8(
        workspace.intermediate_fp8[:routes],
        workspace.intermediate_scales[:routes],
    )
    torch.testing.assert_close(
        activated, expected_activated.float(), rtol=0.2, atol=0.25
    )

    expected_route_output = torch.empty_like(workspace.route_output[:routes])
    for is_last_expert in (False, True):
        selected = (
            (sorted_ids == case.experts - 1)
            if is_last_expert
            else (sorted_ids != case.experts - 1)
        )
        expected_route_output[selected] = (
            activated[selected] @ fixture.down_weights[is_last_expert].T
        ).to(torch.bfloat16)
    original_route_order = expected_route_output[
        workspace.scatter_indices[:routes].long()
    ]
    expected = (
        (
            original_route_order.reshape(tokens, _TOPK, case.hidden).float()
            * topk_weights[..., None]
        )
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    relative_rmse, cosine = _relative_metrics(output, expected)
    assert relative_rmse.item() < 0.003
    assert cosine.item() > 0.999
    assert torch.isfinite(output).all()


def test_plain_glm53_nonlocal_ep_routes_contribute_exact_zero(
    glm53_synthetic_iq2r,
):
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the EP4 production target is plain GLM-5.3")

    case = fixture.case
    generator = torch.Generator(device="cuda").manual_seed(0x53E4)
    hidden = (
        torch.randn((1, case.hidden), generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, -1, case.experts - 1, -1, 17, -1, 65, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.softmax(
        torch.randn((1, _TOPK), generator=generator, device="cuda"), dim=-1
    ).contiguous()
    # Production graph capture sizes the shared workspace for the configured
    # batch ceiling, even when the live decode batch is one token. Keep that
    # large-capacity/small-live-work shape in the regression.
    workspace = IQ2RMoeWorkspace.allocate(
        256,
        _TOPK,
        device="cuda",
        max_experts=case.experts,
        hidden_size=case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    output = torch.empty_like(hidden)

    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data,
        fixture.gate_auxiliary,
        fixture.down_data,
        fixture.down_auxiliary,
        topk_weights,
        topk_ids,
        output,
        gate_up_metadata=fixture.gate_metadata,
        down_metadata=fixture.down_metadata,
        gate_up_tile_n=128,
        down_tile_n=128,
        gate_up_bias=None,
        down_bias=None,
        workspace=workspace,
        swiglu_limit=case.swiglu_limit,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )

    valid = topk_ids[0] >= 0
    expected_routes = []
    route_input = _dequant_mxfp8(
        workspace.route_input_fp8[:_TOPK], workspace.route_input_scales[:_TOPK]
    )
    for route, expert in enumerate(topk_ids[0].tolist()):
        if expert < 0:
            expected_routes.append(torch.zeros(case.hidden, device="cuda"))
            continue
        gate_weight = fixture.gate_weights[expert == case.experts - 1]
        down_weight = fixture.down_weights[expert == case.experts - 1]
        gate_up = (route_input[route].float() @ gate_weight.T).to(torch.bfloat16)
        gate = gate_up[0::2].float()
        up = gate_up[1::2].float()
        activated = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
        expected_routes.append((activated.float() @ down_weight.T).to(torch.bfloat16))
    expected_routes = torch.stack(expected_routes)
    expected = (
        (expected_routes.float() * topk_weights[0, :, None])
        .sum(dim=0, keepdim=True)
        .to(torch.bfloat16)
    )

    sorted_invalid = workspace.sorted_expert_ids[:_TOPK] < 0
    assert int(sorted_invalid.sum().item()) == int((~valid).sum().item())
    assert torch.count_nonzero(workspace.route_output[:_TOPK][sorted_invalid]) == 0
    torch.testing.assert_close(output, expected, rtol=0.003, atol=0.02)
    assert torch.isfinite(output).all()


def test_plain_glm53_ep4_production_prefill_shape_is_safe(
    glm53_synthetic_iq2r,
):
    """Exercise the sorted-task path at ATOM's initial TP4 prefill budget."""
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the EP4 production target is plain GLM-5.3")

    tokens = 2048
    local_experts = 64
    generator = torch.Generator(device="cuda").manual_seed(0x53E42048)
    hidden = torch.randn(
        (tokens, fixture.case.hidden), generator=generator, device="cuda"
    ).to(torch.bfloat16)
    local_ids = torch.arange(tokens, dtype=torch.int32, device="cuda") % local_experts
    topk_ids = torch.stack(
        (
            local_ids,
            torch.full_like(local_ids, -1),
            (local_ids + 17) % local_experts,
            torch.full_like(local_ids, -1),
            torch.full_like(local_ids, -1),
            torch.full_like(local_ids, -1),
            torch.full_like(local_ids, -1),
            torch.full_like(local_ids, -1),
        ),
        dim=1,
    ).contiguous()
    topk_weights = torch.softmax(
        torch.randn(
            (tokens, _TOPK), generator=generator, device="cuda", dtype=torch.float32
        ),
        dim=-1,
    ).contiguous()
    workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        _TOPK,
        device="cuda",
        max_experts=local_experts,
        hidden_size=fixture.case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    output = torch.empty_like(hidden)

    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data[:local_experts],
        fixture.gate_auxiliary[:local_experts],
        fixture.down_data[:local_experts],
        fixture.down_auxiliary[:local_experts],
        topk_weights,
        topk_ids,
        output,
        gate_up_metadata=fixture.gate_metadata,
        down_metadata=fixture.down_metadata,
        gate_up_tile_n=128,
        down_tile_n=128,
        gate_up_bias=None,
        down_bias=None,
        workspace=workspace,
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )
    torch.cuda.synchronize()

    routes = tokens * _TOPK
    invalid = workspace.sorted_expert_ids[:routes] < 0
    assert int(invalid.sum().item()) == tokens * 6
    assert torch.count_nonzero(workspace.route_output[:routes][invalid]) == 0
    assert torch.isfinite(output).all()
