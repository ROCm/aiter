# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness gate for the TP GEMM1 sorted-range descriptor producer."""

from __future__ import annotations

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.moe_tp_ag_gemm1 import (
    launch_tp_stage1_descriptor_pack,
    prepare_tp_stage1_all_ready_metadata,
    prepare_tp_stage1_device_metadata,
)
from aiter.ops.flydsl.utils import is_flydsl_available

_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)
_TILE_M = 32
_TOKENS_PER_SOURCE = 32
_EXPERTS = 8
_TOPK = 1
_SOURCE_EXPERT_STRIDE = 16


def _source_metadata(local_experts: tuple[int, ...]):
    sorted_stride = (
        _TOKENS_PER_SOURCE * _TOPK + _EXPERTS * _TILE_M - _TOPK
    )
    sorted_ids = torch.zeros(sorted_stride, dtype=torch.int32, device="cuda")
    sorted_weights = torch.zeros(sorted_stride, dtype=torch.float32, device="cuda")
    expert_ids = torch.full(
        (_SOURCE_EXPERT_STRIDE,), _EXPERTS, dtype=torch.int32, device="cuda"
    )
    if local_experts:
        expert_ids[: len(local_experts)] = torch.tensor(
            local_experts, dtype=torch.int32, device="cuda"
        )
    num_valid_ids = torch.tensor(
        [len(local_experts) * _TILE_M], dtype=torch.int32, device="cuda"
    )
    sorted_scale = torch.zeros(64, dtype=torch.uint8, device="cuda")
    return (
        sorted_ids,
        sorted_weights,
        expert_ids,
        num_valid_ids,
        sorted_scale,
    )


def _assert_descriptor_state(device_metadata, reference_metadata):
    torch.cuda.synchronize()
    work_blocks = reference_metadata.work_descriptors.numel()
    assert int(device_metadata.num_valid_ids[0].item()) == work_blocks * _TILE_M
    torch.testing.assert_close(
        device_metadata.work_descriptors[:work_blocks],
        reference_metadata.work_descriptors,
        atol=0,
        rtol=0,
    )

    expected_counts = torch.zeros(
        (_EXPERTS, 3), dtype=torch.int32, device="cuda"
    )
    expected_starts = torch.zeros_like(expected_counts)
    expected_counts[0, 0] = 3
    expected_counts[7, 0] = 2
    expected_starts[7, 0] = 3
    expected_counts[3, 2] = 12
    torch.testing.assert_close(
        device_metadata.expert_source_counts, expected_counts, atol=0, rtol=0
    )
    torch.testing.assert_close(
        device_metadata.expert_source_starts, expected_starts, atol=0, rtol=0
    )


@_SKIP_GFX950_FLYDSL
def test_tp_gemm1_sorted_range_descriptor_empty_hot_and_graph_replay():
    """Empty sources/experts and one hot range preserve exact work ordering."""
    source_metadata = (
        _source_metadata((0, 0, 0, 7, 7)),
        _source_metadata(()),
        _source_metadata((3,) * 12),
    )
    reference_metadata = prepare_tp_stage1_all_ready_metadata(
        source_metadata,
        tokens_per_source=_TOKENS_PER_SOURCE,
        experts=_EXPERTS,
        topk=_TOPK,
        tile_m=_TILE_M,
    )
    device_metadata = prepare_tp_stage1_device_metadata(
        source_metadata,
        tokens_per_source=_TOKENS_PER_SOURCE,
        experts=_EXPERTS,
        topk=_TOPK,
        tile_m=_TILE_M,
    )
    _assert_descriptor_state(device_metadata, reference_metadata)

    launch_tp_stage1_descriptor_pack(device_metadata, wait_source_ready=True)
    _assert_descriptor_state(device_metadata, reference_metadata)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_tp_stage1_descriptor_pack(device_metadata, wait_source_ready=True)
    for _ in range(32):
        graph.replay()
    _assert_descriptor_state(device_metadata, reference_metadata)

    epoch = int(device_metadata.source_current_epoch.item())
    assert epoch > 1
    assert int(device_metadata.source_ready_entry.item()) == epoch
    assert int(device_metadata.source_ready_errors.item()) == 0
    assert torch.all(device_metadata.source_ready == epoch)
    assert torch.all(device_metadata.source_payload_epoch == epoch)
    assert torch.all(device_metadata.source_observed_epoch == epoch)
