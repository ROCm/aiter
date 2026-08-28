# SPDX-License-Identifier: MIT
"""Static manifest checks for the scoreboard/direct-atomic architecture."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    DIRECT_TILE_STAGE1_CONTRACT,
    DIRECT_TILE_STAGE2_CONTRACT,
    ROUTE_STORE_STAGE2_CONTRACT,
    _validate_direct_tile_debug_snapshot,
    _validate_direct_tile_operator,
)


def _operator(
    stage1_updates=None,
    stage2_updates=None,
    *,
    node_accumulation_mode="direct_atomic",
):
    stage1 = deepcopy(DIRECT_TILE_STAGE1_CONTRACT)
    stage2 = deepcopy(
        ROUTE_STORE_STAGE2_CONTRACT
        if node_accumulation_mode == "route_store"
        else DIRECT_TILE_STAGE2_CONTRACT
    )
    stage1.update(stage1_updates or {})
    if node_accumulation_mode == "route_store":
        stage2.update(
            {
                "node_reduce_blocks": 16,
                "node_reduce_vec_bytes": 8,
                "node_reduce_schedule": "token",
                "node_reduce_load_schedule": "interleaved",
            }
        )
    stage2.update(stage2_updates or {})
    stage2_launcher = SimpleNamespace(
        architecture_contract=stage2,
        node_accumulation_mode=node_accumulation_mode,
    )
    if node_accumulation_mode == "route_store":
        stage2_launcher.node_reduce_blocks = 16
        stage2_launcher.node_reduce_vec_bytes = 8
        stage2_launcher.node_reduce_schedule = "token"
        stage2_launcher.node_reduce_load_schedule = "interleaved"
    return SimpleNamespace(
        _stage1=SimpleNamespace(architecture_contract=stage1),
        _stage2=stage2_launcher,
    )


def test_direct_tile_manifest_accepts_exact_architecture():
    _validate_direct_tile_operator(_operator())


def test_route_store_manifest_accepts_in_kernel_register_reduce():
    _validate_direct_tile_operator(
        _operator(node_accumulation_mode="route_store")
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"epilogue": "direct_lsa_atomic_source_aligned_node_accumulator"},
        {"node_accumulation_mode": "direct_atomic"},
        {"node_reduce_blocks": 32},
        {"node_reduce_vec_bytes": 4},
        {"node_reduce_schedule": "tile"},
        {"node_reduce_load_schedule": "load_first"},
        {"uses_external_reduce_kernel": True},
    ],
)
def test_route_store_manifest_rejects_mismatched_launcher_contract(updates):
    with pytest.raises(AssertionError, match="Stage2 architecture contract"):
        _validate_direct_tile_operator(
            _operator(
                stage2_updates=updates,
                node_accumulation_mode="route_store",
            )
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"receive_comm_roles": 7},
        {"cross_node_comm_roles": 2},
        {"intra_node_comm_roles": 8},
        {"eos_tail": False},
        {"uses_rank_inbox": True},
        {"uses_group_sort": True},
    ],
)
def test_stage1_rejects_wrong_role_eos_or_inbox_contract(updates):
    with pytest.raises(AssertionError, match="Stage1 architecture contract"):
        _validate_direct_tile_operator(_operator(stage1_updates=updates))


def test_stage1_requires_distinct_allocation_and_arrival_counters():
    with pytest.raises(AssertionError, match="distinct counters"):
        _validate_direct_tile_operator(
            _operator(stage1_updates={"arrival_counter": "alloc_count"})
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"epilogue": "rank_partial_then_node_scan"},
        {"node_accumulator_dtype": "fp32"},
        {"node_ready_granularity": "tile"},
        {"uses_rank_partial": True},
        {"uses_node_scan": True},
        {"uses_external_reduce_kernel": True},
    ],
)
def test_stage2_rejects_rank_partial_or_external_reduce(updates):
    with pytest.raises(AssertionError, match="Stage2 architecture contract"):
        _validate_direct_tile_operator(_operator(stage2_updates=updates))


def _completed_snapshot():
    allocated = [32] * 63 + [16, 16] + [0] * 3
    active = len(allocated) - 3
    return {
        "comm_role_eos": [17] * 8,
        "alloc_count": allocated,
        "tile_arrived": allocated.copy(),
        "tile_ready": [17] * active + [0] * 3,
        "tail_tile": [0] * 63 + [1, 1] + [0] * 3,
        "tail_sealed": [0] * 63 + [17, 17] + [0] * 3,
        "node_atomic_expected": [8] * 128,
        "node_atomic_done": [8] * 128,
        "node_atomic_ready": [17] * 128,
        "node_ready_granularity": "token",
        "node_token_done_mismatch": 0,
        "protocol_error_count": [0],
    }


def test_completed_debug_snapshot_checks_eight_eos_dual_counts_and_tails():
    _validate_direct_tile_debug_snapshot(_completed_snapshot())


def test_completed_token_ready_snapshot_checks_second_level_counter():
    snapshot = _completed_snapshot()
    _validate_direct_tile_debug_snapshot(snapshot)


def test_token_ready_snapshot_rejects_incomplete_second_level_counter():
    snapshot = _completed_snapshot()
    snapshot["node_ready_granularity"] = "token"
    snapshot["node_token_done_mismatch"] = 1
    with pytest.raises(AssertionError, match="token tile-count"):
        _validate_direct_tile_debug_snapshot(snapshot)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("comm_role_eos", [17] * 7), "exactly 8 communication roles"),
        (("tile_arrived", [31] + [32] * 62 + [16, 16] + [0] * 3), "tile_arrived"),
        (("tail_sealed", [0] * 68), "EOS-sealed"),
        (("node_atomic_done", [7] + [8] * 127), "before all contributions"),
        (("protocol_error_count", [1]), "protocol_error_count"),
        (("node_expected_uniform_mismatch", 1), "expected count differs"),
        (("node_expected_done_mismatch", 1), "contributor count"),
        (("node_not_ready", 1), "readiness"),
    ],
)
def test_debug_snapshot_rejects_protocol_invariant_failures(mutation, match):
    snapshot = _completed_snapshot()
    name, value = mutation
    snapshot[name] = value
    with pytest.raises(AssertionError, match=match):
        _validate_direct_tile_debug_snapshot(snapshot)
