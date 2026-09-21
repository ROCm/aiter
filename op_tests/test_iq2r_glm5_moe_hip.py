# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Diagnostic gfx950 coverage for the GLM-5.3 IQ2R MoE contract.

The tensors in this test are synthetic and use uniform importance.  They prove
the 288-expert/top-8 runtime geometry and GLM SwiGLU ABI, not O0 model quality.
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

_EXPERTS = 288
_TOPK = 8
_HIDDEN = 4096
_INTERMEDIATE = 2048


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


@pytest.fixture(scope="module")
def glm53_synthetic_iq2r():
    gate_metadata = IQ2RMetadata(logical_n=2 * _INTERMEDIATE, logical_k=_HIDDEN)
    down_metadata = IQ2RMetadata(logical_n=_HIDDEN, logical_k=_INTERMEDIATE)
    importance = torch.ones((_HIDDEN,), dtype=torch.float32, device="cuda")
    down_importance = importance[:_INTERMEDIATE].contiguous()
    codebook = iq2r_initial_codebook("cuda")

    def encode(n: int, k: int, seed: int):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        weight = (
            torch.randn((n, k), generator=generator, device="cuda") * 0.01
        ).contiguous()
        return iq2r_encode_device(
            weight,
            importance if k == _HIDDEN else down_importance,
            codebook,
            exponent_radius=8,
        )

    gate_data_0, gate_auxiliary_0 = encode(2 * _INTERMEDIATE, _HIDDEN, 0x5310)
    gate_data_287, gate_auxiliary_287 = encode(2 * _INTERMEDIATE, _HIDDEN, 0x5311)
    down_data_0, down_auxiliary_0 = encode(_HIDDEN, _INTERMEDIATE, 0x5320)
    down_data_287, down_auxiliary_287 = encode(_HIDDEN, _INTERMEDIATE, 0x5321)

    gate_data = gate_data_0.expand(_EXPERTS, -1).contiguous()
    gate_auxiliary = gate_auxiliary_0.expand(_EXPERTS, -1).contiguous()
    down_data = down_data_0.expand(_EXPERTS, -1).contiguous()
    down_auxiliary = down_auxiliary_0.expand(_EXPERTS, -1).contiguous()
    gate_data[287].copy_(gate_data_287)
    gate_auxiliary[287].copy_(gate_auxiliary_287)
    down_data[287].copy_(down_data_287)
    down_auxiliary[287].copy_(down_auxiliary_287)

    return SimpleNamespace(
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
                gate_data, gate_auxiliary, gate_metadata, expert_index=287
            ),
        },
        down_weights={
            False: iq2r_materialize_device(
                down_data, down_auxiliary, down_metadata, expert_index=0
            ),
            True: iq2r_materialize_device(
                down_data, down_auxiliary, down_metadata, expert_index=287
            ),
        },
    )


def test_glm53_swiglu_uses_adjacent_gate_up_and_model_parameters():
    generator = torch.Generator(device="cuda").manual_seed(0x53A6)
    gate_up = (
        torch.randn((7, 2 * _INTERMEDIATE), generator=generator, device="cuda") * 12.0
    ).to(torch.bfloat16)
    actual = torch.empty((7, _INTERMEDIATE), dtype=torch.bfloat16, device="cuda")

    iq2r_swiglu_out(gate_up, actual, limit=10.0, alpha=1.0, up_offset=0.0)

    gate = gate_up[:, 0::2].float().clamp(max=10.0)
    up = gate_up[:, 1::2].float().clamp(min=-10.0, max=10.0)
    expected = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_glm53_full_moe_matches_materialized_oracle(glm53_synthetic_iq2r, tokens):
    fixture = glm53_synthetic_iq2r
    generator = torch.Generator(device="cuda").manual_seed(0x5300 + tokens)
    hidden = (
        torch.randn((tokens, _HIDDEN), generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    expert_row = torch.tensor(
        [0, 17, 65, 127, 160, 223, 271, 287],
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
        max_experts=_EXPERTS,
        hidden_size=_HIDDEN,
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
        swiglu_limit=10.0,
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
        selected = (sorted_ids == 287) if is_last_expert else (sorted_ids != 287)
        expected_gate_up[selected] = (
            route_input[selected] @ fixture.gate_weights[is_last_expert].T
        ).to(torch.bfloat16)
    gate_rmse, gate_cosine = _relative_metrics(
        workspace.gate_up[:routes], expected_gate_up
    )
    assert gate_rmse.item() < 0.01
    assert gate_cosine.item() > 0.999

    gate = workspace.gate_up[:routes, 0::2].float().clamp(max=10.0)
    up = workspace.gate_up[:routes, 1::2].float().clamp(min=-10.0, max=10.0)
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
        selected = (sorted_ids == 287) if is_last_expert else (sorted_ids != 287)
        expected_route_output[selected] = (
            activated[selected] @ fixture.down_weights[is_last_expert].T
        ).to(torch.bfloat16)
    original_route_order = expected_route_output[
        workspace.scatter_indices[:routes].long()
    ]
    expected = (
        (
            original_route_order.reshape(tokens, _TOPK, _HIDDEN).float()
            * topk_weights[..., None]
        )
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    relative_rmse, cosine = _relative_metrics(output, expected)
    assert relative_rmse.item() < 0.003
    assert cosine.item() > 0.999
    assert torch.isfinite(output).all()
