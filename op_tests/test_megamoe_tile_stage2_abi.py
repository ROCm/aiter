# SPDX-License-Identifier: MIT

import pytest
import torch

from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import (
    Stage2ArenaLayout,
    Stage2NodePartialWire,
    validate_public_stage2_contract,
)


def test_k3_node_partial_wire_geometry():
    wire = Stage2NodePartialWire(hidden=7168, records_per_group=4)
    assert wire.record_bytes == 14_336
    assert wire.group_bytes == 57_344
    assert wire.group_count(128) == 32


def test_stage2_arena_regions_are_aligned_and_non_overlapping():
    layout = Stage2ArenaLayout.create()
    assert layout.hidden_tiles == 28
    assert layout.return_groups == 32
    last_end = 0
    for region in layout.regions:
        assert region.offset % region.alignment == 0
        assert region.offset >= last_end
        last_end = region.end
    assert layout.total_bytes % 4096 == 0
    assert layout.total_bytes >= last_end
    assert layout.region("node_accumulator").shape == (2, 2, 128, 7168)
    assert layout.region("node_expected").shape == (2, 2, 128, 28)
    assert layout.region("node_done").shape == (2, 2, 128, 28)
    assert layout.region("node_tile_ready").shape == (2, 2, 128, 28)
    assert layout.region("remote_node_tx").shape == (2, 128, 7168)
    assert layout.region("node_dest_rank_mask").shape == (2, 2, 128)
    assert layout.region("remote_partial_rx").shape == (2, 128, 7168)


def test_public_stage2_contract_contains_no_source_metadata():
    weights = torch.empty((128, 16), dtype=torch.float32)
    ids = torch.empty((128, 16), dtype=torch.int32)
    output = torch.empty((128, 7168), dtype=torch.bfloat16)
    assert validate_public_stage2_contract(weights, ids, output) == 128
    with pytest.raises(ValueError):
        validate_public_stage2_contract(weights, ids, output[:, :3584])
