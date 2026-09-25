# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Diagnostic gfx950 coverage for Flash and plain GLM-5.3 IQ2R MoE.

The tensors in this test are synthetic and use uniform importance.  They prove
the production expert/top-k geometries and GLM SwiGLU ABIs, not O0 model quality.
"""

from types import SimpleNamespace

import pytest
import torch

from aiter import biased_grouped_topk
from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.iq2r import (
    iq2r_encode_device,
    iq2r_materialize_device,
    iq2r_route_gather_quant_out,
    iq2r_route_scatter_quant_out,
    iq2r_route_sort_tasks_out,
    iq2r_swiglu_out,
    iq2r_task_capacity,
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


@pytest.mark.parametrize("routes", [257, 512, 2048, 2304])
def test_route_sort_supports_glm53_fused_shared_expert(routes):
    expert_count = 257
    task_rows = 32
    generator = torch.Generator(device="cuda").manual_seed(0xE039 + routes)
    expert_ids = torch.randint(
        expert_count,
        (routes,),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
    # Guarantee that the appended shared-expert bucket is exercised even at
    # the smallest route count.
    expert_ids[-1] = 256
    sorted_ids = torch.empty_like(expert_ids)
    gather = torch.empty_like(expert_ids)
    scatter = torch.empty_like(expert_ids)
    capacity = iq2r_task_capacity(routes, expert_count, task_rows)
    tasks = torch.empty((capacity, 3), dtype=torch.int32, device="cuda")
    task_count = torch.empty((1,), dtype=torch.int32, device="cuda")

    iq2r_route_sort_tasks_out(
        expert_ids,
        sorted_ids,
        gather,
        scatter,
        tasks,
        task_count,
        expert_count=expert_count,
        task_rows=task_rows,
    )

    torch.testing.assert_close(sorted_ids, expert_ids[gather.long()], rtol=0, atol=0)
    torch.testing.assert_close(
        sorted_ids, torch.sort(expert_ids).values, rtol=0, atol=0
    )
    torch.testing.assert_close(
        gather[scatter.long()],
        torch.arange(routes, dtype=torch.int32, device="cuda"),
        rtol=0,
        atol=0,
    )
    counts = torch.bincount(expert_ids, minlength=expert_count)
    expected_task_count = int(((counts + task_rows - 1) // task_rows).sum().item())
    assert int(task_count.item()) == expected_task_count
    assert bool((tasks[:expected_task_count, 2] == 256).any())


def test_route_scatter_quant_supports_glm53_fused_topk9():
    tokens = 4
    topk = 9
    hidden_size = 6144
    routes = tokens * topk
    generator = torch.Generator(device="cuda").manual_seed(0xE048)
    hidden = torch.randn(
        (tokens, hidden_size),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    routed_ids = torch.randint(
        256,
        (tokens, topk - 1),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
    expert_ids = torch.cat(
        (
            routed_ids,
            torch.full((tokens, 1), 256, dtype=torch.int32, device="cuda"),
        ),
        dim=1,
    ).reshape(-1)
    sorted_ids = torch.empty_like(expert_ids)
    gather = torch.empty_like(expert_ids)
    scatter = torch.empty_like(expert_ids)
    task_rows = 16
    tasks = torch.empty(
        (iq2r_task_capacity(routes, 257, task_rows), 3),
        dtype=torch.int32,
        device="cuda",
    )
    task_count = torch.empty((1,), dtype=torch.int32, device="cuda")
    iq2r_route_sort_tasks_out(
        expert_ids,
        sorted_ids,
        gather,
        scatter,
        tasks,
        task_count,
        expert_count=257,
        task_rows=task_rows,
    )

    expected = torch.empty(
        (routes, hidden_size), dtype=torch.float8_e4m3fn, device="cuda"
    )
    expected_scales = torch.empty(
        (routes, hidden_size // 32), dtype=torch.uint8, device="cuda"
    )
    actual = torch.empty_like(expected)
    actual_scales = torch.empty_like(expected_scales)
    iq2r_route_gather_quant_out(hidden, gather, expected, expected_scales, topk=topk)
    iq2r_route_scatter_quant_out(hidden, scatter, actual, actual_scales, topk=topk)

    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)


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


@pytest.mark.parametrize("tokens", [1, 2, 4, 16, 64, 256])
def test_plain_glm53_fused_shared_expert_matches_separate_iq2r(
    glm53_synthetic_iq2r, tokens
):
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the fused shared-expert contract targets plain GLM-5.3")

    generator = torch.Generator(device="cuda").manual_seed(0x53257 + tokens)
    hidden = (
        torch.randn((tokens, fixture.case.hidden), generator=generator, device="cuda")
        * 0.2
    ).to(torch.bfloat16)
    routed_ids = torch.stack(
        [
            torch.roll(
                torch.tensor(
                    [0, 17, 65, 127, 160, 223, 254, 255],
                    dtype=torch.int32,
                    device="cuda",
                ),
                shifts=token,
            )
            for token in range(tokens)
        ]
    ).contiguous()
    routed_weights = torch.softmax(
        torch.randn(
            (tokens, _TOPK), generator=generator, dtype=torch.float32, device="cuda"
        ),
        dim=-1,
    ).contiguous()
    shared_ids = torch.full((tokens, 1), 256, dtype=torch.int32, device="cuda")
    shared_weights = torch.ones((tokens, 1), dtype=torch.float32, device="cuda")
    fused_ids = torch.cat((routed_ids, shared_ids), dim=1).contiguous()
    fused_weights = torch.cat((routed_weights, shared_weights), dim=1).contiguous()

    gate_data = torch.cat((fixture.gate_data, fixture.gate_data[-1:]), dim=0)
    gate_auxiliary = torch.cat(
        (fixture.gate_auxiliary, fixture.gate_auxiliary[-1:]), dim=0
    )
    down_data = torch.cat((fixture.down_data, fixture.down_data[-1:]), dim=0)
    down_auxiliary = torch.cat(
        (fixture.down_auxiliary, fixture.down_auxiliary[-1:]), dim=0
    )

    routed_workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        _TOPK,
        device="cuda",
        max_experts=256,
        hidden_size=fixture.case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    shared_workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        1,
        device="cuda",
        max_experts=1,
        hidden_size=fixture.case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    fused_workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        9,
        device="cuda",
        max_experts=257,
        hidden_size=fixture.case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    routed_output = torch.empty_like(hidden)
    shared_output = torch.empty_like(hidden)
    fused_output = torch.empty_like(hidden)
    common = {
        "gate_up_metadata": fixture.gate_metadata,
        "down_metadata": fixture.down_metadata,
        "gate_up_tile_n": 128,
        "down_tile_n": 128,
        "gate_up_bias": None,
        "down_bias": None,
        "swiglu_limit": 0.0,
        "swiglu_alpha": 1.0,
        "swiglu_up_offset": 0.0,
    }

    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data,
        fixture.gate_auxiliary,
        fixture.down_data,
        fixture.down_auxiliary,
        routed_weights,
        routed_ids,
        routed_output,
        workspace=routed_workspace,
        global_expert_count=256,
        **common,
    )
    iq2r_fused_moe_out(
        hidden,
        gate_data[-1:],
        gate_auxiliary[-1:],
        down_data[-1:],
        down_auxiliary[-1:],
        shared_weights,
        torch.zeros_like(shared_ids),
        shared_output,
        workspace=shared_workspace,
        global_expert_count=1,
        **common,
    )
    iq2r_fused_moe_out(
        hidden,
        gate_data,
        gate_auxiliary,
        down_data,
        down_auxiliary,
        fused_weights,
        fused_ids,
        fused_output,
        workspace=fused_workspace,
        global_expert_count=257,
        **common,
    )

    expected = (routed_output.float() + shared_output.float()).to(torch.bfloat16)
    torch.testing.assert_close(fused_output, expected, rtol=0.004, atol=0.03)
    assert torch.isfinite(fused_output).all()


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_plain_glm53_fused_route_reduce_add_matches_bf16_add(
    glm53_synthetic_iq2r, tokens
):
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the specialized fused reduction/add targets plain GLM-5.3")
    generator = torch.Generator(device="cuda").manual_seed(0x53AD + tokens)
    hidden = torch.randn(
        (tokens, fixture.case.hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    shared = torch.randn(
        hidden.shape,
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    topk_ids = torch.randint(
        0,
        fixture.case.experts,
        (tokens, _TOPK),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
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
        max_experts=fixture.case.experts,
        hidden_size=fixture.case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    routed = torch.empty_like(hidden)
    combined = torch.empty_like(hidden)

    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data,
        fixture.gate_auxiliary,
        fixture.down_data,
        fixture.down_auxiliary,
        topk_weights,
        topk_ids,
        routed,
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
    expected = routed + shared
    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data,
        fixture.gate_auxiliary,
        fixture.down_data,
        fixture.down_auxiliary,
        topk_weights,
        topk_ids,
        combined,
        gate_up_metadata=fixture.gate_metadata,
        down_metadata=fixture.down_metadata,
        gate_up_tile_n=128,
        down_tile_n=128,
        gate_up_bias=None,
        down_bias=None,
        workspace=workspace,
        shared_output=shared,
        swiglu_limit=0.0,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )

    torch.testing.assert_close(combined, expected, rtol=0, atol=0)


def test_plain_glm53_fused_route_reduce_add_validates_optional_inputs(
    glm53_synthetic_iq2r,
):
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the specialized fused reduction/add targets plain GLM-5.3")
    hidden = torch.zeros((1, fixture.case.hidden), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.zeros((1, _TOPK), dtype=torch.int32, device="cuda")
    topk_weights = torch.full(
        (1, _TOPK), 1.0 / _TOPK, dtype=torch.float32, device="cuda"
    )
    output = torch.empty_like(hidden)
    workspace = IQ2RMoeWorkspace.allocate(
        1,
        _TOPK,
        device="cuda",
        max_experts=fixture.case.experts,
        hidden_size=fixture.case.hidden,
        intermediate_size=_INTERMEDIATE,
    )

    def run(*, shared_output=None, pre_reduce_stream=None):
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
            shared_output=shared_output,
            pre_reduce_stream=pre_reduce_stream,
            swiglu_limit=0.0,
            swiglu_alpha=1.0,
            swiglu_up_offset=0.0,
        )

    with pytest.raises(ValueError, match="pre_reduce_stream requires shared_output"):
        run(pre_reduce_stream=torch.cuda.Stream())
    with pytest.raises(ValueError, match="shared_output must be contiguous BF16"):
        run(shared_output=torch.empty_like(hidden, dtype=torch.float32))
    with pytest.raises(ValueError, match="shared_output must be contiguous BF16"):
        run(shared_output=torch.empty((1, fixture.case.hidden + 1), device="cuda"))


@pytest.mark.parametrize(
    "layout",
    ["contiguous", "interleaved", "arbitrary"],
)
def test_plain_glm53_nonlocal_ep_routes_contribute_exact_zero(
    glm53_synthetic_iq2r, layout
):
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the EP4 production target is plain GLM-5.3")

    case = fixture.case
    generator = torch.Generator(device="cuda").manual_seed(0x53E4)
    hidden = (
        torch.randn((1, case.hidden), generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    local_experts = 32
    topk_ids = torch.tensor(
        [[0, 32, 31, 63, 17, 127, 191, case.experts - 1]],
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
        max_experts=local_experts,
        hidden_size=case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    workspace.route_output.fill_(torch.nan)
    output = torch.empty_like(hidden)
    if layout == "contiguous":
        expert_start = 0
        expert_stride = 1
        owned_ids = torch.arange(local_experts, device="cuda")
        expert_map = None
    elif layout == "interleaved":
        expert_start = 7
        expert_stride = 8
        owned_ids = expert_start + torch.arange(local_experts, device="cuda") * 8
        expert_map = None
    else:
        expert_start = 0
        expert_stride = 1
        selected = {0, 17, 31, 32, 63, 127, 191, 255}
        owned = [255, 17, 63, 191]
        owned.extend(i for i in range(case.experts) if i not in selected)
        owned_ids = torch.tensor(owned[:local_experts], device="cuda")
        expert_map = torch.full((case.experts,), -1, dtype=torch.int32, device="cuda")
        expert_map[owned_ids] = torch.arange(
            local_experts, dtype=torch.int32, device="cuda"
        )

    iq2r_fused_moe_out(
        hidden,
        fixture.gate_data[owned_ids].contiguous(),
        fixture.gate_auxiliary[owned_ids].contiguous(),
        fixture.down_data[owned_ids].contiguous(),
        fixture.down_auxiliary[owned_ids].contiguous(),
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
        expert_map=expert_map,
        expert_start=expert_start,
        expert_stride=expert_stride,
        global_expert_count=case.experts,
        swiglu_limit=case.swiglu_limit,
        swiglu_alpha=1.0,
        swiglu_up_offset=0.0,
    )

    if expert_map is None:
        offsets = topk_ids[0] - expert_start
        local_ids = torch.where(
            (offsets >= 0)
            & (offsets % expert_stride == 0)
            & (offsets // expert_stride < local_experts),
            offsets // expert_stride,
            -1,
        ).to(torch.int32)
    else:
        local_ids = expert_map[topk_ids[0].long()]
    valid = local_ids >= 0
    expected_routes = []
    route_input = _dequant_mxfp8(
        workspace.route_input_fp8[:_TOPK], workspace.route_input_scales[:_TOPK]
    )
    for route, expert in enumerate(topk_ids[0].tolist()):
        if local_ids[route] < 0:
            expected_routes.append(torch.zeros(case.hidden, device="cuda"))
            continue
        sorted_route = int(workspace.scatter_indices[route].item())
        gate_weight = fixture.gate_weights[expert == case.experts - 1]
        down_weight = fixture.down_weights[expert == case.experts - 1]
        gate_up = (route_input[sorted_route].float() @ gate_weight.T).to(torch.bfloat16)
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
    assert torch.all(workspace.scatter_indices[:_TOPK][~valid] == -1)
    assert int(workspace.task_count.item()) == int(valid.sum().item())
    torch.testing.assert_close(output, expected, rtol=0.003, atol=0.02)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_plain_glm53_fused_router_matches_split_ep_path(glm53_synthetic_iq2r, tokens):
    fixture = glm53_synthetic_iq2r
    if fixture.case.name != "plain":
        pytest.skip("the fused biased-sigmoid router targets plain GLM-5.3")

    case = fixture.case
    local_experts = 32
    generator = torch.Generator(device="cuda").manual_seed(0x53F000 + tokens)
    hidden = (
        torch.randn((tokens, case.hidden), generator=generator, device="cuda") * 0.2
    ).to(torch.bfloat16)
    router_logits = torch.randn(
        (tokens, case.experts),
        generator=generator,
        dtype=torch.float32,
        device="cuda",
    )
    correction_bias = torch.zeros(case.experts, dtype=torch.float32, device="cuda")
    favored = torch.tensor([0, 1, 2, 3, 32, 33, 34, 35], device="cuda")
    correction_bias[favored] = torch.linspace(10.0, 9.0, 8, device="cuda")

    expected_weights = torch.empty((tokens, _TOPK), dtype=torch.float32, device="cuda")
    expected_ids = torch.empty((tokens, _TOPK), dtype=torch.int32, device="cuda")
    biased_grouped_topk(
        router_logits,
        correction_bias,
        expected_weights,
        expected_ids,
        1,
        1,
        True,
        2.5,
    )

    expert_map = torch.full((case.experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[:local_experts] = torch.arange(
        local_experts, dtype=torch.int32, device="cuda"
    )
    local_weights = (
        fixture.gate_data[:local_experts].contiguous(),
        fixture.gate_auxiliary[:local_experts].contiguous(),
        fixture.down_data[:local_experts].contiguous(),
        fixture.down_auxiliary[:local_experts].contiguous(),
    )
    split_workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        _TOPK,
        device="cuda",
        max_experts=local_experts,
        hidden_size=case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    fused_workspace = IQ2RMoeWorkspace.allocate(
        tokens,
        _TOPK,
        device="cuda",
        max_experts=local_experts,
        hidden_size=case.hidden,
        intermediate_size=_INTERMEDIATE,
    )
    split_output = torch.empty_like(hidden)
    fused_output = torch.empty_like(hidden)

    common = {
        "gate_up_metadata": fixture.gate_metadata,
        "down_metadata": fixture.down_metadata,
        "gate_up_tile_n": 128,
        "down_tile_n": 128,
        "gate_up_bias": None,
        "down_bias": None,
        "expert_map": expert_map,
        "global_expert_count": case.experts,
        "swiglu_limit": case.swiglu_limit,
        "swiglu_alpha": 1.0,
        "swiglu_up_offset": 0.0,
    }
    iq2r_fused_moe_out(
        hidden,
        *local_weights,
        expected_weights,
        expected_ids,
        split_output,
        workspace=split_workspace,
        **common,
    )
    iq2r_fused_moe_out(
        hidden,
        *local_weights,
        fused_workspace.topk_weights[:tokens],
        fused_workspace.topk_ids[:tokens],
        fused_output,
        workspace=fused_workspace,
        router_logits=router_logits,
        router_bias=correction_bias,
        router_scoring_func="sigmoid",
        router_routed_scaling_factor=2.5,
        **common,
    )

    expected_order = torch.argsort(expected_ids, dim=1)
    actual_order = torch.argsort(fused_workspace.topk_ids[:tokens], dim=1)
    torch.testing.assert_close(
        torch.gather(fused_workspace.topk_ids[:tokens], 1, actual_order),
        torch.gather(expected_ids, 1, expected_order),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        torch.gather(fused_workspace.topk_weights[:tokens], 1, actual_order),
        torch.gather(expected_weights, 1, expected_order),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(fused_output, split_output, rtol=0.003, atol=0.02)


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
