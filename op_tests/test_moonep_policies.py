# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Policy-level tests for MoRI physical-id planning."""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.moonep import (
    MoonEPDecodePolicy,
    MoonEPPlanConfig,
    MoonEPPrefillPlanner,
    MoonEPPrefillPolicy,
    build_prefill_reference_plan,
)


def _overflow_routing():
    # R=2, E/R=4, B=1.  All routes hit rank-0's four experts, so the balanced
    # allocation borrows two experts on rank 1.  Only one can occupy its single
    # prefetch slot; the other must return to rank 0.
    per_rank = torch.tensor([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=torch.int32)
    all_topk = [per_rank.clone(), per_rank.clone()]
    histogram = torch.stack(
        [torch.bincount(x.reshape(-1).long(), minlength=8) for x in all_topk]
    ).to(torch.int32)
    return all_topk, histogram


def test_prefill_plan_emits_mori_physical_ids_and_rolls_back_overflow():
    all_topk, histogram = _overflow_routing()
    cfg = MoonEPPlanConfig(
        rank=0,
        world_size=2,
        num_tokens=4,
        top_k=2,
        num_experts=8,
        prefetch_slots=1,
    )
    plan = build_prefill_reference_plan(cfg, all_topk[0], histogram)

    assert plan.num_experts_per_rank == 5
    assert plan.experts_to_copy.tolist() == [[-1], [1]]
    assert plan.rank_route_counts.tolist() == [12, 4]
    assert plan.residual_imbalance.tolist() == [4, -4]
    assert plan.planned_topk_ids.tolist() == [
        [0, 9],
        [2, 3],
        [0, 9],
        [2, 3],
    ]

    # Every non-owner allocation has a selected prefetch slot, and every
    # emitted id decodes into that same slot map.
    epr = cfg.experts_per_rank
    width = cfg.virtual_experts_per_rank
    for dest in range(cfg.world_size):
        for expert in range(cfg.num_experts):
            count = int(plan.alloc[dest, expert])
            if count and expert // epr != dest:
                assert expert in plan.experts_to_copy[dest].tolist()

    for logical, physical in zip(
        all_topk[0].reshape(-1).tolist(),
        plan.planned_topk_ids.reshape(-1).tolist(),
    ):
        dest, slot = divmod(physical, width)
        assert slot == int(plan.expert_to_slot[dest, logical])


def test_prefill_plan_is_deterministic_on_every_source_rank():
    all_topk, histogram = _overflow_routing()
    expected_alloc = None
    expected_copy = None
    observed = torch.zeros(2, 8, dtype=torch.int32)
    for rank in range(2):
        cfg = MoonEPPlanConfig(
            rank=rank,
            world_size=2,
            num_tokens=4,
            top_k=2,
            num_experts=8,
            prefetch_slots=1,
        )
        plan = build_prefill_reference_plan(cfg, all_topk[rank], histogram)
        if expected_alloc is None:
            expected_alloc = plan.alloc
            expected_copy = plan.experts_to_copy
        else:
            assert torch.equal(plan.alloc, expected_alloc)
            assert torch.equal(plan.experts_to_copy, expected_copy)

        width = cfg.virtual_experts_per_rank
        for logical, physical in zip(
            all_topk[rank].reshape(-1).tolist(),
            plan.planned_topk_ids.reshape(-1).tolist(),
        ):
            dest = physical // width
            observed[dest, logical] += 1

    assert torch.equal(observed, expected_alloc)


def test_prefill_policy_runs_histogram_exchange_before_planning():
    all_topk, histogram = _overflow_routing()
    cfg = MoonEPPlanConfig(
        rank=0,
        world_size=2,
        num_tokens=4,
        top_k=2,
        num_experts=8,
        prefetch_slots=1,
    )

    class Exchange:
        def __init__(self):
            self.local = None

        def publish(self, local):
            self.local = local.clone()
            return histogram

    class Planner:
        def build(self, topk, all_histograms):
            assert torch.equal(all_histograms, histogram)
            return build_prefill_reference_plan(cfg, topk, all_histograms)

    exchange = Exchange()
    policy = MoonEPPrefillPolicy(
        cfg, "cpu", histogram_exchange=exchange, planner=Planner()
    )
    plan = policy.plan(all_topk[0])

    assert torch.equal(exchange.local, histogram[0])
    assert plan.planned_topk_ids.tolist()[0] == [0, 9]


def test_decode_policy_keeps_owner_ids_and_needs_no_prefetch_slots():
    logical = torch.tensor([[0, 7], [3, 4]], dtype=torch.int64)
    policy = MoonEPDecodePolicy(world_size=2, num_experts=8)
    plan = policy.plan(logical)

    assert plan.num_experts_per_rank == 4
    assert plan.planned_topk_ids.dtype == torch.int32
    assert torch.equal(plan.planned_topk_ids, logical.to(torch.int32))


def test_gpu_prefill_planner_matches_reference():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    all_topk, histogram = _overflow_routing()
    cfg = MoonEPPlanConfig(
        rank=0,
        world_size=2,
        num_tokens=4,
        top_k=2,
        num_experts=8,
        prefetch_slots=1,
    )
    device = torch.device("cuda")
    topk = all_topk[0].to(device)
    tpe = histogram.to(device)
    reference = build_prefill_reference_plan(cfg, topk, tpe)
    actual = MoonEPPrefillPlanner(cfg, device).build(topk, tpe).clone()

    for field in (
        "planned_topk_ids",
        "experts_to_copy",
        "alloc",
        "expert_to_slot",
        "rank_route_counts",
        "residual_imbalance",
    ):
        assert torch.equal(getattr(actual, field), getattr(reference, field))
