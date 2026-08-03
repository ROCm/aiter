# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""CPU correctness tests for the MoonEP gfx950 prototype contracts."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.moonep import MoonEPPlanConfig, build_reference_plan


def _config(*, rank: int, tokens: int, top_k: int, padding: int = 1):
    return MoonEPPlanConfig(
        rank=rank,
        world_size=2,
        num_tokens=tokens,
        top_k=top_k,
        num_experts=4,
        prefetch_slots=2,
        token_padding=padding,
    )


def test_reference_plan_balanced_routes_stay_home():
    tokens_per_expert = torch.tensor(
        [[2, 2, 0, 0], [0, 0, 2, 2]], dtype=torch.int32
    )
    plan = build_reference_plan(
        _config(rank=0, tokens=4, top_k=1),
        torch.tensor([[0], [0], [1], [1]], dtype=torch.int32),
        tokens_per_expert,
    )

    raw_dst, primary = plan.decode_dst(plan.dst)
    assert torch.equal(raw_dst // plan.config.num_dispatch_rows, torch.zeros_like(raw_dst))
    assert bool(primary.all())
    assert bool((plan.experts_to_copy == -1).all())
    assert torch.equal(plan.alloc.sum(dim=1), torch.full((2,), 4, dtype=torch.int32))


def test_reference_plan_moves_hot_expert_and_selects_prefetch_slot():
    tokens_per_expert = torch.tensor(
        [[4, 0, 0, 0], [4, 0, 0, 0]], dtype=torch.int32
    )
    plan = build_reference_plan(
        _config(rank=1, tokens=4, top_k=1),
        torch.zeros((4, 1), dtype=torch.int32),
        tokens_per_expert,
    )

    raw_dst, primary = plan.decode_dst(plan.dst)
    assert bool(primary.all())
    assert bool((raw_dst // plan.config.num_dispatch_rows == 1).all())
    assert int(plan.experts_to_copy[1, 0]) == 0
    assert int(plan.group_expert_ids[0]) == -1
    assert int(plan.group_expert_ids[4]) == 0
    assert int(plan.cu_seqlens[-1]) == 4
    assert torch.equal(plan.alloc[:, 0], torch.tensor([4, 4], dtype=torch.int32))


def test_reference_plan_negative_encodes_same_rank_topk_duplicates():
    tokens_per_expert = torch.tensor(
        [[2, 2, 0, 0], [0, 0, 2, 2]], dtype=torch.int32
    )
    plan = build_reference_plan(
        _config(rank=0, tokens=2, top_k=2),
        torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        tokens_per_expert,
    )

    raw_dst, primary = plan.decode_dst(plan.dst)
    assert bool(primary[:, 0].all())
    assert bool((~primary[:, 1]).all())
    assert bool((raw_dst // plan.config.num_dispatch_rows == 0).all())
    assert torch.equal(-plan.dst[:, 1] - 1, raw_dst[:, 1])


def test_reference_plan_padding_records_zero_fill_rows():
    tokens_per_expert = torch.tensor(
        [[1, 3, 0, 0], [0, 0, 1, 3]], dtype=torch.int32
    )
    plan = build_reference_plan(
        _config(rank=0, tokens=4, top_k=1, padding=4),
        torch.tensor([[0], [1], [1], [1]], dtype=torch.int32),
        tokens_per_expert,
    )

    # expert0 has one real row padded to four; expert1 already has four minus
    # one? It has three real rows, so both groups carry padding.
    assert torch.equal(
        plan.zero_fill_ranges[:2],
        torch.tensor([[1, 3], [7, 1]], dtype=torch.int32),
    )
    assert int(plan.cu_seqlens[1]) == 8


def test_reference_plan_rejects_inconsistent_local_histogram():
    tokens_per_expert = torch.tensor(
        [[2, 2, 0, 0], [0, 0, 2, 2]], dtype=torch.int32
    )
    try:
        build_reference_plan(
            _config(rank=0, tokens=4, top_k=1),
            torch.tensor([[0], [0], [0], [1]], dtype=torch.int32),
            tokens_per_expert,
        )
    except ValueError as exc:
        assert "histogram" in str(exc)
    else:
        raise AssertionError("expected an inconsistent histogram failure")
