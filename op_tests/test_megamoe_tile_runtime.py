# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.kernels.megamoe_tile.runtime import (
    EpochPhase,
    HierCcoArenaLayout,
    HierEpoch,
    LayeredHierPipeline,
)


def _layout():
    return HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=4,
        chunk_bytes=64 * 1024,
        max_m_tiles=97,
        max_source_tokens=16 * 128,
    )


def test_hier_cco_arena_regions_are_aligned_and_disjoint():
    layout = _layout()
    previous = 0
    names = set()
    for region in layout.regions:
        assert region.name not in names
        names.add(region.name)
        assert region.offset % region.alignment == 0
        assert region.offset >= previous
        previous = region.end
    assert layout.total_bytes % 4096 == 0
    assert previous <= layout.total_bytes
    # Ready generations are ordinary registered-window data, one per QP/slot.
    assert layout.region("dispatch_ready").shape == (8, 4)
    assert layout.region("dispatch_ready").dtype == torch.int64
    assert layout.region("partial_eos").shape == (2,)
    assert layout.region("partial_eos").dtype == torch.int64
    assert layout.region("h1_queue_header").shape == (2, 4)
    assert layout.region("h1_ready_queue").shape == (2, 97 * 4)


def test_hier_cco_arena_views_and_epoch_pointers():
    layout = _layout()
    arena = layout.allocate_local()
    expected = layout.view(arena, "h1_input_expected", parity=1)
    ready = layout.view(arena, "h1_input_ready", parity=1)
    assert expected.shape == ready.shape == (97,)
    expected[3] = 11
    assert layout.view(arena, "h1_input_expected")[1, 3].item() == 11

    base = arena.data_ptr()
    ptrs = layout.epoch_pointers(base, 7)
    assert ptrs.parity == 1
    assert ptrs.rank_partial_epoch_ready == base + layout.offset(
        "partial_eos", parity=1
    )
    assert ptrs.plan_ready == base + layout.offset("plan_ready") + 8
    assert ptrs.h1_input_ready == base + layout.offset("h1_input_ready", parity=1)
    assert ptrs.h2_output_ready == base + layout.offset("h2_output_ready", parity=1)
    assert ptrs.h1_queue_header == base + layout.offset(
        "h1_queue_header", parity=1
    )
    assert layout.ring_chunk_offset("dispatch_tx", 7) == (
        layout.region("dispatch_tx").offset + 7 * 64 * 1024
    )
    with pytest.raises(ValueError):
        layout.ring_chunk_offset("dispatch_tx", 8)

    even_ptrs = layout.epoch_pointers(base, 8)
    assert even_ptrs.parity == 0
    assert even_ptrs.rank_partial_epoch_ready == base + layout.offset(
        "partial_eos", parity=0
    )
    assert (
        ptrs.rank_partial_epoch_ready - even_ptrs.rank_partial_epoch_ready
        == torch.empty((), dtype=torch.int64).element_size()
    )


def test_hier_epoch_and_layered_pipeline_reject_out_of_order_reuse():
    layout = _layout()
    epoch = HierEpoch(layout)
    epoch.begin(1)
    assert epoch.parity == 1 and epoch.ring_slot == 1
    with pytest.raises(RuntimeError):
        epoch.begin(2)
    epoch.advance(EpochPhase.PLAN, EpochPhase.DISPATCH)
    epoch.advance(EpochPhase.DISPATCH, EpochPhase.H1)
    epoch.advance(EpochPhase.H1, EpochPhase.H2)
    epoch.advance(EpochPhase.H2, EpochPhase.RETURN)
    epoch.complete()
    assert epoch.last_completed == 1
    with pytest.raises(ValueError):
        epoch.begin(1)

    pipeline = LayeredHierPipeline(layout, arena_base=0x100000)
    ptrs = pipeline.begin(9)
    assert ptrs.generation == 9
    pipeline.dispatch_submitted()
    pipeline.h1_submitted()
    pipeline.h2_submitted()
    pipeline.return_submitted()
    pipeline.finish()
    assert pipeline.epoch.phase == EpochPhase.COMPLETE
