# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pytest
import torch

from aiter.iq2r_checkpoint import load_iq2r_layer_checkpoint
from aiter.ops.iq2r import (
    iq2r_encode_device,
    iq2r_gemm,
    iq2r_materialize_device,
    iq2r_task_gemm,
)
from aiter.ops.iq2r_encoder import iq2r_encode_reference, iq2r_initial_codebook
from aiter.ops.iq2r_format import IQ2RMetadata
from aiter.ops.iq2r_reference import iq2r_materialize

_FIXTURE = Path("/models/openai/gpt-oss-120b-o0-e132-profile")


def _has_gfx950_fixture() -> bool:
    if not torch.cuda.is_available() or not (_FIXTURE / "config.json").is_file():
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


@pytest.mark.parametrize("k", [128, 2880])
def test_device_encoder_matches_reference(k):
    generator = torch.Generator(device="cuda").manual_seed(0x1022 + k)
    weight = (torch.randn((64, k), generator=generator, device="cuda") * 0.08).float()
    importance = torch.linspace(0.2, 2.0, k, device="cuda")
    codebook = iq2r_initial_codebook("cuda")
    expected_data, expected_auxiliary = iq2r_encode_reference(
        weight, importance, codebook
    )
    actual_data, actual_auxiliary = iq2r_encode_device(weight, importance, codebook)
    torch.testing.assert_close(actual_data, expected_data, rtol=0, atol=0)
    torch.testing.assert_close(actual_auxiliary, expected_auxiliary, rtol=0, atol=0)


@pytest.mark.parametrize("n", [64, 128])
def test_device_materializer_matches_independent_host_decoder(first_expert, n):
    metadata = IQ2RMetadata(n, 2880)
    data_cpu = first_expert.gate_up_data[:, : metadata.data_bytes].cpu().contiguous()
    auxiliary_cpu = (
        first_expert.gate_up_auxiliary[:, : metadata.auxiliary_bytes].cpu().contiguous()
    )
    expected = iq2r_materialize(data_cpu, auxiliary_cpu, metadata)[0]
    actual = iq2r_materialize_device(
        data_cpu.cuda(), auxiliary_cpu.cuda(), metadata
    ).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n,tile_n", [(64, 64), (128, 128)])
@pytest.mark.parametrize("m", [1, 4, 16, 17, 32])
def test_scaled_fp8_gemm_matches_materialized_weight(first_expert, n, tile_n, m):
    metadata = IQ2RMetadata(n, 2880)
    data = first_expert.gate_up_data[:, : metadata.data_bytes].contiguous()
    auxiliary = first_expert.gate_up_auxiliary[
        :, : metadata.auxiliary_bytes
    ].contiguous()
    bias = first_expert.gate_up_bias[:, :n].contiguous()
    weight = iq2r_materialize_device(data, auxiliary, metadata)

    generator = torch.Generator(device="cuda").manual_seed(0x1200 + n + m)
    activations = (
        torch.randn((m, metadata.logical_k), generator=generator, device="cuda") * 0.25
    ).to(torch.float8_e4m3fn)
    # E8M0 exponent 127 represents an exact scale of one.  Non-unity scale
    # behavior is covered separately so failures identify operand-layout bugs.
    scales = torch.full(
        (m, metadata.logical_k // 32), 127, dtype=torch.uint8, device="cuda"
    )
    actual = iq2r_gemm(
        activations,
        scales,
        data,
        auxiliary,
        metadata,
        tile_n=tile_n,
        bias=bias,
    ).float()
    expected = (
        (activations.float() @ weight.T + bias.float()).to(torch.bfloat16).float()
    )
    relative_rmse = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rmse.item() < 1e-3


def test_scaled_fp8_gemm_honors_independent_e8m0_blocks(first_expert):
    metadata = IQ2RMetadata(64, 2880)
    data = first_expert.gate_up_data[:, : metadata.data_bytes].contiguous()
    auxiliary = first_expert.gate_up_auxiliary[
        :, : metadata.auxiliary_bytes
    ].contiguous()
    weight = iq2r_materialize_device(data, auxiliary, metadata)
    generator = torch.Generator(device="cuda").manual_seed(0xE8A0)
    activations = torch.randn(
        (16, metadata.logical_k), generator=generator, device="cuda"
    ).to(torch.float8_e4m3fn)
    scale_pattern = torch.tensor([126, 127, 128, 129], dtype=torch.uint8, device="cuda")
    scales = scale_pattern[
        torch.arange(metadata.logical_k // 32, device="cuda") % 4
    ].repeat(16, 1)
    actual = iq2r_gemm(
        activations, scales, data, auxiliary, metadata, tile_n=64
    ).float()
    scale_values = torch.exp2(scales.float() - 127)
    dequantized = (
        activations.float().reshape(16, -1, 32) * scale_values[..., None]
    ).reshape(16, metadata.logical_k)
    expected = (dequantized @ weight.T).to(torch.bfloat16).float()
    relative_rmse = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rmse.item() < 1e-3


@pytest.mark.parametrize("n,tile_n", [(64, 64), (128, 128)])
def test_task_gemm_handles_expert_runs_and_partial_tiles(first_expert, n, tile_n):
    metadata = IQ2RMetadata(n, 2880)
    one_data = first_expert.gate_up_data[:, : metadata.data_bytes].contiguous()
    one_auxiliary = first_expert.gate_up_auxiliary[
        :, : metadata.auxiliary_bytes
    ].contiguous()
    data = one_data.repeat(2, 1)
    auxiliary = one_auxiliary.repeat(2, 1)
    bias = first_expert.gate_up_bias[:, :n].repeat(2, 1).contiguous()
    bias[1] += 0.25
    weight = iq2r_materialize_device(data, auxiliary, metadata)

    rows = 25
    generator = torch.Generator(device="cuda").manual_seed(0x7A50 + n)
    activations = torch.randn(
        (rows, metadata.logical_k), generator=generator, device="cuda"
    ).to(torch.float8_e4m3fn)
    scales = torch.full(
        (rows, metadata.logical_k // 32), 127, dtype=torch.uint8, device="cuda"
    )
    tasks = torch.tensor(
        [[0, 3, 0], [3, 5, 1], [8, 17, 0]], dtype=torch.int32, device="cuda"
    )
    task_count = torch.tensor([3], dtype=torch.int32, device="cuda")
    actual = iq2r_task_gemm(
        activations,
        scales,
        data,
        auxiliary,
        tasks,
        task_count,
        metadata,
        tile_n=tile_n,
        bias=bias,
    ).float()
    expected = torch.empty_like(actual)
    for start, height, expert in tasks.cpu().tolist():
        expected[start : start + height] = (
            (
                activations[start : start + height].float() @ weight.T
                + bias[expert].float()
            )
            .to(torch.bfloat16)
            .float()
        )
    relative_rmse = (
        actual - expected
    ).square().mean().sqrt() / expected.square().mean().sqrt()
    assert relative_rmse.item() < 1e-3
