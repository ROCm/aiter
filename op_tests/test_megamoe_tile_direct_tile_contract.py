# SPDX-License-Identifier: MIT
"""Static manifest checks for the scoreboard/direct-atomic architecture."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from op_tests.multigpu_tests.bench_megamoe_tile_ep16_two_kernel import (
    DIRECT_TILE_STAGE1_CONTRACT,
    DIRECT_TILE_STAGE2_CONTRACT,
    _validate_direct_tile_debug_snapshot,
    _validate_direct_tile_operator,
)


def _operator(stage1_updates=None, stage2_updates=None):
    stage1 = deepcopy(DIRECT_TILE_STAGE1_CONTRACT)
    stage2 = deepcopy(DIRECT_TILE_STAGE2_CONTRACT)
    stage1.update(stage1_updates or {})
    stage2.update(stage2_updates or {})
    return SimpleNamespace(
        _stage1=SimpleNamespace(architecture_contract=stage1),
        _stage2=SimpleNamespace(architecture_contract=stage2),
    )


def test_direct_tile_manifest_accepts_exact_architecture():
    _validate_direct_tile_operator(_operator())


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
        {"node_accumulator_dtype": "bf16"},
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
        "protocol_error_count": [0],
    }


def test_completed_debug_snapshot_checks_eight_eos_dual_counts_and_tails():
    _validate_direct_tile_debug_snapshot(_completed_snapshot())


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("comm_role_eos", [17] * 7), "exactly 8 communication roles"),
        (("tile_arrived", [31] + [32] * 62 + [16, 16] + [0] * 3), "tile_arrived"),
        (("tail_sealed", [0] * 68), "EOS-sealed"),
        (("node_atomic_done", [7] + [8] * 127), "before all atomics"),
        (("protocol_error_count", [1]), "protocol_error_count"),
    ],
)
def test_debug_snapshot_rejects_protocol_invariant_failures(mutation, match):
    snapshot = _completed_snapshot()
    name, value = mutation
    snapshot[name] = value
    with pytest.raises(AssertionError, match=match):
        _validate_direct_tile_debug_snapshot(snapshot)
