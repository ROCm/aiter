# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest


@pytest.mark.parametrize("record_bytes", (2048, 4096))
def test_dispatch_fanout_broadcast_contract_uses_arena_stride(record_bytes: int):
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import build_dispatch_fanout_lsa
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    layout = HierCcoArenaLayout.create(
        max_m_tiles=8,
        max_source_tokens=2048,
        max_fanout_records=2048,
        fanout_record_bytes=record_bytes,
    )
    module = build_dispatch_fanout_lsa(layout, node_ranks=8)

    # The original arbitrary-plan launcher remains available. The broadcast
    # launcher derives its copy stride from the arena rather than a fixed ABI.
    assert module.launch is not None
    assert module.launch_broadcast is not None
    assert module.capacity == 2048
    assert module.record_bytes == record_bytes
    assert module.launch_broadcast.record_bytes == record_bytes
    assert f"_r{record_bytes}" in module.launch_broadcast.kernel_name
    assert module.launch_broadcast.sides == 2
    assert module.launch_broadcast.grid_blocks_per_record == 16
    assert module.launch_broadcast.requires_disjoint_slot_ranges is True

    module.validate_broadcast_plan(
        record_count=128,
        local_slot_base=0,
        remote_slot_base=1024,
    )
    module.validate_broadcast_plan(
        record_count=128,
        local_slot_base=7 * 128,
        remote_slot_base=1024 + 7 * 128,
    )
    assert module.broadcast_grid_blocks(128) == 2 * 8 * 128 == 2048


@pytest.mark.parametrize(
    ("record_count", "local_base", "remote_base"),
    (
        (0, 0, 1024),
        (128, -1, 1024),
        (128, 0, 2000),
        (128, 64, 127),
    ),
)
def test_dispatch_fanout_broadcast_contract_rejects_invalid_slot_ranges(
    record_count: int,
    local_base: int,
    remote_base: int,
):
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import build_dispatch_fanout_lsa
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    layout = HierCcoArenaLayout.create(
        max_m_tiles=1,
        max_source_tokens=2048,
        max_fanout_records=2048,
        fanout_record_bytes=4096,
    )
    module = build_dispatch_fanout_lsa(layout, node_ranks=8)
    with pytest.raises(ValueError):
        module.validate_broadcast_plan(record_count, local_base, remote_base)
