# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from aiter import QuantType, rmsnorm2d_fwd_with_add, topk_softmax
from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.iq2r_moe import IQ2RMoeWorkspace, iq2r_fused_moe_out
from aiter.ops.iq2r import (
    iq2r_materialize_device,
    iq2r_route_direct_gather_quant_out,
    iq2r_route_gather_indexed_out,
    iq2r_route_gather_quant_out,
    iq2r_route_sort_tasks_out,
    iq2r_route_topk_direct_gather_quant_out,
    iq2r_route_topk_sort_gather_quant_out,
    iq2r_swiglu_out,
    iq2r_swiglu_quant_out,
    iq2r_task_capacity,
)
from aiter.ops.iq2r_encoder import iq2r_dequantize_mxfp4

_FIXTURE = Path("/models/openai/gpt-oss-120b-o0-e132-profile")
_SOURCE = Path("/models/openai/gpt-oss-120b")


def _has_gfx950_fixture() -> bool:
    if (
        not torch.cuda.is_available()
        or not (_FIXTURE / "config.json").is_file()
        or not (_SOURCE / "model.safetensors.index.json").is_file()
    ):
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName


pytestmark = pytest.mark.skipif(
    not _has_gfx950_fixture(),
    reason="requires gfx950 and the compiled GPT-OSS O0 fixture",
)


@pytest.fixture(scope="module")
def first_expert():
    return load_iq2r_layer_checkpoint(
        _FIXTURE, 0, expert_start=0, expert_count=1, device="cuda"
    )


@pytest.fixture(scope="module")
def source_first_expert():
    with (_SOURCE / "model.safetensors.index.json").open(encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = "model.layers.0.mlp.experts."

    def load(name: str) -> torch.Tensor:
        key = prefix + name
        with safe_open(
            _SOURCE / weight_map[key], framework="pt", device="cpu"
        ) as handle:
            return handle.get_slice(key)[0:1].contiguous().to("cuda")

    gate_up = iq2r_dequantize_mxfp4(
        load("gate_up_proj_blocks"),
        load("gate_up_proj_scales"),
        output_dtype=torch.float32,
    )[0]
    down = iq2r_dequantize_mxfp4(
        load("down_proj_blocks"),
        load("down_proj_scales"),
        output_dtype=torch.float32,
    )[0]
    return {
        "gate_up": gate_up,
        "gate_up_bias": load("gate_up_proj_bias")[0],
        "down": down,
        "down_bias": load("down_proj_bias")[0],
    }


def _inputs(tokens: int, *, seed: int = 0x120B):
    generator = torch.Generator(device="cuda").manual_seed(seed + tokens)
    hidden = (torch.randn((tokens, 2880), generator=generator, device="cuda") * 0.2).to(
        torch.bfloat16
    )
    ids = torch.zeros((tokens, 4), dtype=torch.int32, device="cuda")
    logits = torch.randn((tokens, 4), generator=generator, device="cuda")
    weights = torch.softmax(logits, dim=-1).float().contiguous()
    return hidden, weights, ids


def _dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = torch.exp2(scales.float() - 127.0)
    return (values.float().reshape(values.shape[0], -1, 32) * scale[..., None]).reshape(
        values.shape
    )


def _quantize_mxfp8_reference(values: torch.Tensor):
    """Canonical row-major MXFP8/E8M0 reference for 32-value blocks."""

    groups = values.float().reshape(values.shape[0], -1, 32)
    amax = groups.abs().amax(dim=-1).clamp_min(1.0e-10)
    scale_exponents = torch.ceil(torch.log2(amax / 448.0))
    scales = (scale_exponents + 127).to(torch.uint8)
    quantized = (groups / torch.exp2(scale_exponents)[..., None]).to(
        torch.float8_e4m3fn
    )
    return quantized.reshape(values.shape), scales


def _unpack_tile16_scales(
    scales: torch.Tensor, rows: int, groups_per_row: int
) -> torch.Tensor:
    return scales.permute(1, 3, 0, 2).reshape(-1, scales.shape[0] * 4)[
        :rows, :groups_per_row
    ]


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


def _run(first_expert, hidden, topk_weights, topk_ids, output, workspace):
    iq2r_fused_moe_out(
        hidden,
        first_expert.gate_up_data,
        first_expert.gate_up_auxiliary,
        first_expert.down_data,
        first_expert.down_auxiliary,
        topk_weights,
        topk_ids,
        output,
        gate_up_metadata=first_expert.gate_up_metadata,
        down_metadata=first_expert.down_metadata,
        gate_up_tile_n=first_expert.gate_up_tile_n,
        down_tile_n=first_expert.down_tile_n,
        gate_up_bias=first_expert.gate_up_bias,
        down_bias=first_expert.down_bias,
        workspace=workspace,
    )


def test_quant_identity_is_distinct():
    assert QuantType.iq2r_2bit != QuantType.per_1x32


def test_compiled_o0_preserves_native_gpt_oss_gate_up_rows(
    first_expert, source_first_expert
):
    gate_up = iq2r_materialize_device(
        first_expert.gate_up_data,
        first_expert.gate_up_auxiliary,
        first_expert.gate_up_metadata,
    )
    down = iq2r_materialize_device(
        first_expert.down_data,
        first_expert.down_auxiliary,
        first_expert.down_metadata,
    )
    gate_error, gate_cosine = _relative_metrics(gate_up, source_first_expert["gate_up"])
    down_error, down_cosine = _relative_metrics(down, source_first_expert["down"])

    # Applying the split-half -> adjacent helper to this checkpoint would be a
    # second permutation: native GPT-OSS MXFP4 is adjacent already.
    source_gate_up = source_first_expert["gate_up"]
    wrongly_reinterleaved = torch.stack(
        (source_gate_up[:2880], source_gate_up[2880:]), dim=1
    ).flatten(0, 1)
    _, wrong_cosine = _relative_metrics(gate_up, wrongly_reinterleaved)

    assert gate_error.item() < 0.40
    assert gate_cosine.item() > 0.93
    assert down_error.item() < 0.46
    assert down_cosine.item() > 0.90
    assert wrong_cosine.item() < 0.05
    torch.testing.assert_close(
        first_expert.gate_up_bias[0], source_first_expert["gate_up_bias"]
    )
    torch.testing.assert_close(
        first_expert.down_bias[0], source_first_expert["down_bias"]
    )


def test_full_o0_pipeline_tracks_source_mxfp4_expert(first_expert, source_first_expert):
    hidden, topk_weights, topk_ids = _inputs(2, seed=0x706)
    workspace = IQ2RMoeWorkspace.allocate(
        2, 4, device="cuda", max_experts=1, task_rows=64
    )
    output = torch.empty_like(hidden)
    _run(first_expert, hidden, topk_weights, topk_ids, output, workspace)

    gate_up = (
        hidden.float() @ source_first_expert["gate_up"].T
        + source_first_expert["gate_up_bias"].float()
    ).to(torch.bfloat16)
    gate = gate_up[:, 0::2].float().clamp(max=7.0)
    up = gate_up[:, 1::2].float().clamp(min=-7.0, max=7.0)
    activated = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(torch.bfloat16)
    expected = (
        activated.float() @ source_first_expert["down"].T
        + source_first_expert["down_bias"].float()
    ).to(torch.bfloat16)

    relative_rmse, cosine = _relative_metrics(output, expected)
    assert relative_rmse.item() < 0.25
    assert cosine.item() > 0.97


@pytest.mark.parametrize("tokens", [1, 2, 4, 5, 8, 16])
def test_fused_route_reduce_add_rmsnorm_matches_unfused(first_expert, tokens):
    hidden, topk_weights, topk_ids = _inputs(tokens, seed=0xA11D)
    workspace = IQ2RMoeWorkspace.allocate(
        tokens, 4, device="cuda", max_experts=1, task_rows=16
    )
    residual = torch.randn_like(hidden)
    norm_weight = torch.randn((2880,), dtype=torch.bfloat16, device="cuda")

    moe_output = torch.empty_like(hidden)
    expected_output = torch.empty_like(hidden)
    expected_residual = torch.empty_like(hidden)
    _run(first_expert, hidden, topk_weights, topk_ids, moe_output, workspace)
    rmsnorm2d_fwd_with_add(
        expected_output,
        moe_output,
        residual,
        expected_residual,
        norm_weight,
        1e-5,
    )

    fused_output = torch.empty_like(hidden)
    fused_residual = torch.empty_like(hidden)
    iq2r_fused_moe_out(
        hidden,
        first_expert.gate_up_data,
        first_expert.gate_up_auxiliary,
        first_expert.down_data,
        first_expert.down_auxiliary,
        topk_weights,
        topk_ids,
        fused_output,
        gate_up_metadata=first_expert.gate_up_metadata,
        down_metadata=first_expert.down_metadata,
        gate_up_tile_n=first_expert.gate_up_tile_n,
        down_tile_n=first_expert.down_tile_n,
        gate_up_bias=first_expert.gate_up_bias,
        down_bias=first_expert.down_bias,
        workspace=workspace,
        residual=residual,
        norm_weight=norm_weight,
        residual_out=fused_residual,
        norm_epsilon=1e-5,
        norm_block_size=1024,
    )

    torch.testing.assert_close(fused_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(fused_output, expected_output, rtol=0, atol=0.004)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16])
def test_tile16_activation_scales_match_row_major_pipeline(first_expert, tokens):
    hidden, topk_weights, topk_ids = _inputs(tokens, seed=0x1600)
    row_workspace = IQ2RMoeWorkspace.allocate(
        tokens, 4, device="cuda", max_experts=1, scale_layout="row_major"
    )
    tiled_workspace = IQ2RMoeWorkspace.allocate(
        tokens, 4, device="cuda", max_experts=1, scale_layout="tile16"
    )
    row_output = torch.empty_like(hidden)
    tiled_output = torch.empty_like(hidden)

    _run(
        first_expert,
        hidden,
        topk_weights,
        topk_ids,
        row_output,
        row_workspace,
    )
    _run(
        first_expert,
        hidden,
        topk_weights,
        topk_ids,
        tiled_output,
        tiled_workspace,
    )

    routes = tokens * 4
    torch.testing.assert_close(tiled_output, row_output, rtol=0, atol=0)
    torch.testing.assert_close(
        tiled_workspace.route_input_fp8[:routes].view(torch.uint8),
        row_workspace.route_input_fp8[:routes].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _unpack_tile16_scales(tiled_workspace.route_input_scales, routes, 90),
        row_workspace.route_input_scales[:routes],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        tiled_workspace.intermediate_fp8[:routes].view(torch.uint8),
        row_workspace.intermediate_fp8[:routes].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _unpack_tile16_scales(tiled_workspace.intermediate_scales, routes, 90),
        row_workspace.intermediate_scales[:routes],
        rtol=0,
        atol=0,
    )


def test_stable_route_sort_and_task_construction():
    expert_ids = torch.tensor(
        [3, 1, 3, 0, 1, 2, 3, 2, 0, 1, 3, 0],
        dtype=torch.int32,
        device="cuda",
    )
    routes = expert_ids.numel()
    sorted_ids = torch.empty_like(expert_ids)
    gather = torch.empty_like(expert_ids)
    scatter = torch.empty_like(expert_ids)
    capacity = iq2r_task_capacity(routes, 4, 16)
    tasks = torch.empty((capacity, 3), dtype=torch.int32, device="cuda")
    task_count = torch.empty((1,), dtype=torch.int32, device="cuda")
    iq2r_route_sort_tasks_out(
        expert_ids,
        sorted_ids,
        gather,
        scatter,
        tasks,
        task_count,
        expert_count=4,
        task_rows=16,
    )
    expected_gather = torch.argsort(expert_ids, stable=True)
    torch.testing.assert_close(gather, expected_gather.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(sorted_ids, expert_ids[gather.long()], rtol=0, atol=0)
    torch.testing.assert_close(
        gather[scatter.long()],
        torch.arange(routes, dtype=torch.int32, device="cuda"),
    )
    count = int(task_count.item())
    assert count == 4
    assert tasks[:count].cpu().tolist() == [
        [0, 3, 0],
        [3, 3, 1],
        [6, 2, 2],
        [8, 4, 3],
    ]


@pytest.mark.parametrize("routes", [17, 32, 64, 128, 256, 257])
def test_route_sort_boundaries(routes):
    expert_count = 128
    task_rows = 16
    generator = torch.Generator(device="cuda").manual_seed(0xC400 + routes)
    expert_ids = torch.randint(
        expert_count,
        (routes,),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
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

    # The general route path intentionally uses unordered shared-memory
    # cursors; only expert grouping and the gather/scatter inverse are part of
    # the MoE execution contract.
    torch.testing.assert_close(sorted_ids, expert_ids[gather.long()], rtol=0, atol=0)
    torch.testing.assert_close(
        sorted_ids,
        torch.sort(expert_ids).values,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gather[scatter.long()],
        torch.arange(routes, dtype=torch.int32, device="cuda"),
        rtol=0,
        atol=0,
    )

    counts = torch.bincount(expert_ids.cpu(), minlength=expert_count).tolist()
    expected_tasks = []
    sorted_begin = 0
    for expert, expert_rows in enumerate(counts):
        for local in range(0, expert_rows, task_rows):
            expected_tasks.append(
                [sorted_begin + local, min(task_rows, expert_rows - local), expert]
            )
        sorted_begin += expert_rows
    count = int(task_count.item())
    assert count == len(expected_tasks)
    assert tasks[:count].cpu().tolist() == expected_tasks


def test_direct_low_m_route_gather_quant_builds_identity_tasks():
    hidden, _, _ = _inputs(4, seed=0xC4)
    expert_ids = torch.tensor(
        [3, 1, 7, 9, 1, 4, 12, 17, 2, 8, 7, 6, 5, 11, 10, 0],
        dtype=torch.int32,
        device="cuda",
    )
    routes = expert_ids.numel()
    sorted_ids = torch.empty_like(expert_ids)
    gather = torch.empty_like(expert_ids)
    scatter = torch.empty_like(expert_ids)
    tasks = torch.empty((routes, 3), dtype=torch.int32, device="cuda")
    task_count = torch.empty((1,), dtype=torch.int32, device="cuda")
    output = torch.empty((routes, 2880), dtype=torch.float8_e4m3fn, device="cuda")
    scales = torch.empty((routes, 90), dtype=torch.uint8, device="cuda")

    iq2r_route_direct_gather_quant_out(
        hidden,
        expert_ids,
        sorted_ids,
        gather,
        scatter,
        tasks,
        task_count,
        output,
        scales,
        topk=4,
        expert_count=128,
    )

    identity = torch.arange(routes, dtype=torch.int32, device="cuda")
    expected_output = torch.empty_like(output)
    expected_scales = torch.empty_like(scales)
    iq2r_route_gather_quant_out(
        hidden,
        identity,
        expected_output,
        expected_scales,
        topk=4,
    )
    torch.testing.assert_close(sorted_ids, expert_ids, rtol=0, atol=0)
    torch.testing.assert_close(gather, identity, rtol=0, atol=0)
    torch.testing.assert_close(scatter, identity, rtol=0, atol=0)
    assert int(task_count.item()) == routes
    assert tasks[:, 0].cpu().tolist() == list(range(routes))
    assert tasks[:, 1].cpu().tolist() == [1] * routes
    torch.testing.assert_close(tasks[:, 2], expert_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        output.view(torch.uint8), expected_output.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_low_m_topk_route_quant_matches_aiter(tokens, router_dtype, with_bias):
    hidden, _, _ = _inputs(tokens, seed=0x2A00 + tokens)
    generator = torch.Generator(device="cuda").manual_seed(0x7060 + tokens)
    router_logits = torch.randn(
        (tokens, 128), generator=generator, dtype=torch.float32, device="cuda"
    ).to(router_dtype)
    # Exercise AITER's deterministic lower-expert-ID tie break.
    router_logits[:, 3] = router_logits[:, 7]
    router_bias = (
        torch.randn((128,), generator=generator, dtype=torch.bfloat16, device="cuda")
        if with_bias
        else None
    )
    expected_logits = (
        router_logits if router_bias is None else router_logits + router_bias
    )

    expected_weights = torch.empty((tokens, 4), dtype=torch.float32, device="cuda")
    expected_ids = torch.empty((tokens, 4), dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty_like(expected_ids)
    topk_softmax(
        expected_weights,
        expected_ids,
        token_expert_indices,
        expected_logits,
        True,
    )

    workspace = IQ2RMoeWorkspace.allocate(tokens, 4, device="cuda")
    actual_weights = workspace.topk_weights[:tokens]
    actual_ids = workspace.topk_ids[:tokens]
    routes = tokens * 4
    iq2r_route_topk_direct_gather_quant_out(
        hidden,
        router_logits,
        actual_weights,
        actual_ids,
        workspace.sorted_expert_ids[:routes],
        workspace.gather_indices[:routes],
        workspace.scatter_indices[:routes],
        workspace.tasks,
        workspace.task_count,
        workspace.route_input_fp8[:routes],
        workspace.route_input_scales[:routes],
        renormalize=True,
        router_bias=router_bias,
    )

    identity = torch.arange(routes, dtype=torch.int32, device="cuda")
    expected_fp8 = torch.empty_like(workspace.route_input_fp8[:routes])
    expected_scales = torch.empty_like(workspace.route_input_scales[:routes])
    iq2r_route_gather_quant_out(
        hidden,
        identity,
        expected_fp8,
        expected_scales,
        topk=4,
    )
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        workspace.sorted_expert_ids[:routes], expected_ids.reshape(-1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        workspace.gather_indices[:routes], identity, rtol=0, atol=0
    )
    torch.testing.assert_close(
        workspace.scatter_indices[:routes], identity, rtol=0, atol=0
    )
    assert int(workspace.task_count.item()) == routes
    torch.testing.assert_close(
        workspace.route_input_fp8[:routes].view(torch.uint8),
        expected_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        workspace.route_input_scales[:routes], expected_scales, rtol=0, atol=0
    )


@pytest.mark.parametrize("tokens", [8, 16])
@pytest.mark.parametrize("router_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_sorted_topk_route_quant_matches_aiter(tokens, router_dtype, with_bias):
    hidden, _, _ = _inputs(tokens, seed=0x2B00 + tokens)
    generator = torch.Generator(device="cuda").manual_seed(0x7160 + tokens)
    router_logits = torch.randn(
        (tokens, 128), generator=generator, dtype=torch.float32, device="cuda"
    ).to(router_dtype)
    router_logits[:, 13] = router_logits[:, 29]
    router_bias = (
        torch.randn((128,), generator=generator, dtype=torch.bfloat16, device="cuda")
        if with_bias
        else None
    )
    expected_logits = (
        router_logits if router_bias is None else router_logits + router_bias
    )

    expected_weights = torch.empty((tokens, 4), dtype=torch.float32, device="cuda")
    expected_ids = torch.empty((tokens, 4), dtype=torch.int32, device="cuda")
    token_expert_indices = torch.empty_like(expected_ids)
    topk_softmax(
        expected_weights,
        expected_ids,
        token_expert_indices,
        expected_logits,
        True,
    )

    workspace = IQ2RMoeWorkspace.allocate(tokens, 4, device="cuda")
    actual_weights = workspace.topk_weights[:tokens]
    actual_ids = workspace.topk_ids[:tokens]
    routes = tokens * 4
    task_capacity = iq2r_task_capacity(routes, 128, workspace.task_rows)
    iq2r_route_topk_sort_gather_quant_out(
        hidden,
        router_logits,
        actual_weights,
        actual_ids,
        workspace.sorted_expert_ids[:routes],
        workspace.gather_indices[:routes],
        workspace.scatter_indices[:routes],
        workspace.tasks[:task_capacity],
        workspace.task_count,
        workspace.route_input_fp8[:routes],
        workspace.route_input_scales[:routes],
        task_rows=workspace.task_rows,
        renormalize=True,
        router_bias=router_bias,
    )

    flat_ids = expected_ids.reshape(-1)
    expected_gather = torch.argsort(flat_ids, stable=True).to(torch.int32)
    expected_sorted_ids = flat_ids[expected_gather.long()]
    expected_scatter = torch.empty_like(expected_gather)
    expected_scatter[expected_gather.long()] = torch.arange(
        routes, dtype=torch.int32, device="cuda"
    )
    expected_fp8 = torch.empty_like(workspace.route_input_fp8[:routes])
    expected_scales = torch.empty_like(workspace.route_input_scales[:routes])
    iq2r_route_gather_quant_out(
        hidden,
        expected_gather,
        expected_fp8,
        expected_scales,
        topk=4,
    )

    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        workspace.sorted_expert_ids[:routes], expected_sorted_ids, rtol=0, atol=0
    )
    torch.testing.assert_close(
        workspace.gather_indices[:routes], expected_gather, rtol=0, atol=0
    )
    torch.testing.assert_close(
        workspace.scatter_indices[:routes], expected_scatter, rtol=0, atol=0
    )
    torch.testing.assert_close(
        workspace.route_input_fp8[:routes].view(torch.uint8),
        expected_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        workspace.route_input_scales[:routes], expected_scales, rtol=0, atol=0
    )

    counts = torch.bincount(flat_ids.cpu(), minlength=128).tolist()
    expected_tasks = []
    sorted_begin = 0
    for expert, expert_rows in enumerate(counts):
        for local in range(0, expert_rows, workspace.task_rows):
            expected_tasks.append(
                [
                    sorted_begin + local,
                    min(workspace.task_rows, expert_rows - local),
                    expert,
                ]
            )
        sorted_begin += expert_rows
    count = int(workspace.task_count.item())
    assert count == len(expected_tasks)
    assert workspace.tasks[:count].cpu().tolist() == expected_tasks


def test_fused_gather_quant_matches_canonical_split_path():
    hidden, _, _ = _inputs(3, seed=0xA8)
    gather = torch.tensor(
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.int32,
        device="cuda",
    )
    routes = gather.numel()
    gathered = torch.empty((routes, 2880), dtype=torch.bfloat16, device="cuda")
    actual = torch.empty((routes, 2880), dtype=torch.float8_e4m3fn, device="cuda")
    actual_scales = torch.empty((routes, 90), dtype=torch.uint8, device="cuda")

    iq2r_route_gather_indexed_out(hidden, gather, gathered, topk=4)
    expected, expected_scales = _quantize_mxfp8_reference(gathered)
    iq2r_route_gather_quant_out(hidden, gather, actual, actual_scales, topk=4)

    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(actual_scales, expected_scales)


def test_fused_gather_quant_accepts_padded_row_stride():
    hidden, _, _ = _inputs(3, seed=0xA8)
    padded = torch.empty((3, 3072), dtype=torch.bfloat16, device="cuda")
    padded[:, :2880].copy_(hidden)
    hidden_view = padded[:, :2880]
    assert hidden_view.stride() == (3072, 1)
    gather = torch.tensor(
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7],
        dtype=torch.int32,
        device="cuda",
    )
    expected = torch.empty(
        (gather.numel(), 2880), dtype=torch.float8_e4m3fn, device="cuda"
    )
    expected_scales = torch.empty(
        (gather.numel(), 90), dtype=torch.uint8, device="cuda"
    )
    actual = torch.empty_like(expected)
    actual_scales = torch.empty_like(expected_scales)

    iq2r_route_gather_quant_out(
        hidden.contiguous(), gather, expected, expected_scales, topk=4
    )
    iq2r_route_gather_quant_out(hidden_view, gather, actual, actual_scales, topk=4)

    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(actual_scales, expected_scales)


@pytest.mark.parametrize("family", ["group32", "parallel8"])
def test_fused_swiglu_quant_matches_canonical_split_path(monkeypatch, family):
    monkeypatch.setenv("IQ2R_SWIGLU_QUANT_FAMILY", family)
    generator = torch.Generator(device="cuda").manual_seed(0x51A6)
    gate_up = (torch.randn((7, 5760), generator=generator, device="cuda") * 4.0).to(
        torch.bfloat16
    )
    activated = torch.empty((7, 2880), dtype=torch.bfloat16, device="cuda")
    actual_activated = torch.empty_like(activated)
    actual = torch.empty((7, 2880), dtype=torch.float8_e4m3fn, device="cuda")
    actual_scales = torch.empty((7, 90), dtype=torch.uint8, device="cuda")

    iq2r_swiglu_out(gate_up, activated)
    expected, expected_scales = _quantize_mxfp8_reference(activated)
    iq2r_swiglu_quant_out(
        gate_up,
        actual,
        actual_scales,
        activated=actual_activated,
    )

    torch.testing.assert_close(actual_activated, activated, rtol=0, atol=0)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8))
    torch.testing.assert_close(actual_scales, expected_scales)


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_full_moe_matches_materialized_weight_oracle(first_expert, tokens):
    hidden, topk_weights, topk_ids = _inputs(tokens)
    workspace = IQ2RMoeWorkspace.allocate(
        tokens, 4, device="cuda", max_experts=1, task_rows=64
    )
    output = torch.empty_like(hidden)
    _run(first_expert, hidden, topk_weights, topk_ids, output, workspace)

    routes = tokens * 4
    gate_weight = iq2r_materialize_device(
        first_expert.gate_up_data,
        first_expert.gate_up_auxiliary,
        first_expert.gate_up_metadata,
    )
    down_weight = iq2r_materialize_device(
        first_expert.down_data,
        first_expert.down_auxiliary,
        first_expert.down_metadata,
    )
    a1 = _dequant_mxfp8(
        workspace.route_input_fp8[:routes], workspace.route_input_scales[:routes]
    )
    gate_up = (a1 @ gate_weight.T + first_expert.gate_up_bias[0].float()).to(
        torch.bfloat16
    )
    gate = torch.clamp(gate_up[:, 0::2].float(), max=7.0)
    up = torch.clamp(gate_up[:, 1::2].float(), min=-7.0, max=7.0)
    activated = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(torch.bfloat16)
    a2 = _dequant_mxfp8(
        workspace.intermediate_fp8[:routes],
        workspace.intermediate_scales[:routes],
    )
    # The fused kernel rounds the activation to BF16 before quantizing. Its
    # dedicated split-path parity test above isolates interleaving, clipping,
    # and E8M0 details; here a loose dequantization check guards end-to-end use.
    torch.testing.assert_close(a2, activated.float(), rtol=0.2, atol=0.25)
    route_output = (a2 @ down_weight.T + first_expert.down_bias[0].float()).to(
        torch.bfloat16
    )
    expected = (
        (route_output.reshape(tokens, 4, 2880).float() * topk_weights[..., None])
        .sum(dim=1)
        .to(torch.bfloat16)
    )
    relative_rmse = (
        output.float() - expected.float()
    ).square().mean().sqrt() / expected.float().square().mean().sqrt()
    assert relative_rmse.item() < 0.003
    assert torch.isfinite(output).all()


def test_repeated_graph_replay_uses_stable_buffers(first_expert):
    tokens = 2
    hidden, topk_weights, topk_ids = _inputs(tokens, seed=0xC0DE)
    workspace = IQ2RMoeWorkspace.allocate(
        4, 4, device="cuda", max_experts=1, task_rows=64
    )
    output = torch.empty_like(hidden)
    _run(first_expert, hidden, topk_weights, topk_ids, output, workspace)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(first_expert, hidden, topk_weights, topk_ids, output, workspace)

    for iteration in range(3):
        replacement, replacement_weights, _ = _inputs(tokens, seed=0xD000 + iteration)
        hidden.copy_(replacement)
        topk_weights.copy_(replacement_weights)
        graph.replay()
        torch.cuda.synchronize()
        captured = output.clone()

        eager_workspace = IQ2RMoeWorkspace.allocate(
            tokens, 4, device="cuda", max_experts=1, task_rows=64
        )
        expected = torch.empty_like(hidden)
        _run(
            first_expert,
            hidden,
            topk_weights,
            topk_ids,
            expected,
            eager_workspace,
        )
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
