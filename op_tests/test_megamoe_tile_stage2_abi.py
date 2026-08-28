# SPDX-License-Identifier: MIT

import pytest
import torch

from aiter.ops.flydsl.kernels.megamoe_tile.stage2_abi import (
    STAGE2_TIMELINE_FIELDS,
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
    assert not layout.include_route_slots
    assert not layout.include_rank_partials
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
    assert layout.region("node_token_done").shape == (2, 2, 128, 16)
    assert layout.region("node_token_ready").shape == (2, 2, 128)
    assert layout.region("timeline").shape == (2, len(STAGE2_TIMELINE_FIELDS))
    assert layout.region("timeline_gmm_worker_done").shape == (2, 256)
    assert layout.region("remote_node_tx").shape == (2, 128, 7168)
    assert layout.region("node_dest_rank_mask").shape == (2, 2, 128)
    assert layout.region("remote_partial_rx").shape == (2, 128, 7168)


def test_stage2_route_store_regions_are_optional_aligned_and_non_overlapping():
    default_layout = Stage2ArenaLayout.create()
    explicit_default_layout = Stage2ArenaLayout.create(include_route_slots=False)
    assert default_layout == explicit_default_layout
    with pytest.raises(KeyError):
        default_layout.region("route_slots")
    with pytest.raises(KeyError):
        default_layout.region("node_partial_done")
    with pytest.raises(KeyError):
        default_layout.region("node_partial_ready")

    layout = Stage2ArenaLayout.create(include_route_slots=True)
    assert layout.include_route_slots
    assert not layout.include_rank_partials

    route_slots = layout.region("route_slots")
    assert route_slots.shape == (2, 2, 128, 28, 16, 256)
    assert route_slots.dtype == torch.bfloat16
    assert route_slots.alignment == 256
    assert route_slots.nbytes == 112 * 1024 * 1024

    node_partial_done = layout.region("node_partial_done")
    assert node_partial_done.shape == (2, 2, 128, 16)
    assert node_partial_done.dtype == torch.int32
    assert node_partial_done.alignment == 64
    assert node_partial_done.nbytes == 32 * 1024

    node_partial_ready = layout.region("node_partial_ready")
    assert node_partial_ready.shape == (2, 2, 128)
    assert node_partial_ready.dtype == torch.int64
    assert node_partial_ready.alignment == 64
    assert node_partial_ready.nbytes == 4096

    # Optional storage is append-only: enabling it cannot perturb any of the
    # direct-atomic ABI offsets consumed by existing Stage-1/Stage-2 kernels.
    assert layout.regions[:-3] == default_layout.regions
    assert route_slots.offset >= default_layout.regions[-1].end
    assert node_partial_done.offset >= route_slots.end
    assert node_partial_ready.offset >= node_partial_done.end

    last_end = 0
    for region in layout.regions:
        assert region.offset % region.alignment == 0
        assert region.offset >= last_end
        last_end = region.end
    assert layout.total_bytes % 4096 == 0
    assert layout.total_bytes >= last_end


def test_stage2_rank_local_regions_are_optional_aligned_and_non_overlapping():
    default_layout = Stage2ArenaLayout.create()
    layout = Stage2ArenaLayout.create(include_rank_partials=True)

    assert not layout.include_route_slots
    assert layout.include_rank_partials
    assert layout.regions[:-12] == default_layout.regions

    rank_accumulator = layout.region("rank_accumulator")
    assert rank_accumulator.shape == (2, 16, 128, 7168)
    assert rank_accumulator.dtype == torch.bfloat16
    assert rank_accumulator.alignment == 256
    assert rank_accumulator.nbytes == 56 * 1024 * 1024

    rank_token_pending = layout.region("rank_token_pending")
    assert rank_token_pending.shape == (2, 16, 128, 16)
    assert rank_token_pending.dtype == torch.int32
    assert rank_token_pending.alignment == 64
    assert rank_token_pending.nbytes == 256 * 1024

    rank_token_ready = layout.region("rank_token_ready")
    assert rank_token_ready.shape == (2, 16, 128)
    assert rank_token_ready.dtype == torch.int64
    assert rank_token_ready.alignment == 64
    assert rank_token_ready.nbytes == 32 * 1024

    assert layout.region("rank_return_tx_slot").shape == (2, 128)
    assert layout.region("rank_return_rx_slot").shape == (2, 128)
    assert layout.region("rank_return_count").shape == (2, 16)
    assert layout.region("rank_reduce_queue").shape == (2, 256)
    assert layout.region("rank_reduce_queue_ready").shape == (2, 256)
    assert layout.region("rank_reduce_queue_tail").shape == (2, 16)

    assert layout.region("node_partial_done").shape == (2, 2, 128, 16)
    assert layout.region("node_partial_ready").shape == (2, 2, 128)
    rank_reduce_queue_head = layout.region("rank_reduce_queue_head")
    assert rank_reduce_queue_head.shape == (2, 16)
    assert rank_reduce_queue_head.dtype == torch.int32
    assert rank_reduce_queue_head.alignment == 64
    assert rank_reduce_queue_head.nbytes == 128
    assert rank_reduce_queue_head.offset >= layout.region(
        "node_partial_ready"
    ).end
    assert layout.regions[-1] == rank_reduce_queue_head

    last_end = 0
    for region in layout.regions:
        assert region.offset % region.alignment == 0
        assert region.offset >= last_end
        last_end = region.end
    assert layout.total_bytes % 4096 == 0
    assert layout.total_bytes >= last_end


def test_stage2_optional_modes_share_node_partial_completion_regions():
    layout = Stage2ArenaLayout.create(
        include_route_slots=True,
        include_rank_partials=True,
    )
    names = [region.name for region in layout.regions]

    assert names.count("node_partial_done") == 1
    assert names.count("node_partial_ready") == 1
    assert names.index("route_slots") < names.index("rank_accumulator")
    assert names.index("rank_reduce_queue_tail") < names.index("node_partial_done")
    assert names.index("node_partial_ready") < names.index(
        "rank_reduce_queue_head"
    )


def test_public_stage2_contract_contains_no_source_metadata():
    weights = torch.empty((128, 16), dtype=torch.float32)
    ids = torch.empty((128, 16), dtype=torch.int32)
    output = torch.empty((128, 7168), dtype=torch.bfloat16)
    assert validate_public_stage2_contract(weights, ids, output) == 128
    with pytest.raises(ValueError):
        validate_public_stage2_contract(weights, ids, output[:, :3584])
