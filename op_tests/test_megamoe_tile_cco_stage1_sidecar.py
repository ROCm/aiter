# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def _layout():
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    return HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=4,
        chunk_bytes=64 * 1024,
        max_m_tiles=8,
        max_source_tokens=64,
        max_h1_n_blocks=3,
    )


class _FakeModule:
    team = "world"

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("launch_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def test_stage1_sidecar_host_lifecycle_and_bounds():
    from aiter.ops.flydsl.kernels.megamoe_tile.cco.stage1_sidecar import CcoStage1Sidecar

    layout = _layout()
    fake = _FakeModule()
    sidecar = CcoStage1Sidecar(layout, fake)
    ptrs1 = layout.epoch_pointers(0x100000, 1)
    ptrs2 = layout.epoch_pointers(0x100000, 2)

    assert layout.region("dispatch_request").shape == (8, 4)
    assert layout.region("h1_queue_header").shape == (2, 4)
    assert layout.region("h1_ready_queue").shape == (2, 8 * 3)

    sidecar.post_dispatch(1, 2, 3, 1, 1, 1, stream=4)
    with pytest.raises(RuntimeError, match="credited/reclaimed"):
        sidecar.post_dispatch(1, 2, 3, 1, 1, 9, stream=4)
    sidecar.reclaim_dispatch(1, 3, 1, 1, stream=4)
    sidecar.post_dispatch(1, 2, 3, 1, 1, 9, stream=4)

    sidecar.publish_plan_expected(
        1, ptrs1, 4, expected_per_tile=1, stream=4
    )
    sidecar.mark_chunk_ready(3, 1, 1, ptrs1, 2, 1, stream=4)

    sidecar.publish_plan_expected(
        2, ptrs2, 2, expected_per_tile=1, stream=4
    )
    sidecar.enqueue_prepacked_tiles(
        3,
        2,
        2,
        ptrs2,
        total_work=6,
        first_flat_tile=0,
        tile_count=3,
        final_batch=False,
        stream=4,
    )
    sidecar.enqueue_prepacked_tiles(
        3,
        3,
        2,
        ptrs2,
        total_work=6,
        first_flat_tile=3,
        tile_count=3,
        final_batch=True,
        stream=4,
    )
    assert sidecar._receive_tail == 6
    assert sidecar._receive_done is True

    ptrs3 = layout.epoch_pointers(0x100000, 3)
    sidecar.publish_plan_expected(
        3, ptrs3, 8, expected_per_tile=1, stream=4
    )
    with pytest.raises(ValueError, match="ready-queue capacity"):
        sidecar.enqueue_prepacked_tiles(
            3,
            4,
            3,
            ptrs3,
            total_work=25,
            first_flat_tile=0,
            tile_count=1,
            final_batch=False,
            stream=4,
        )


def test_stage1_sidecar_ready_publishers_compile_and_run():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")

    from aiter.ops.flydsl.kernels.megamoe_tile.cco import TEAM_WORLD, build_stage1_sidecar_module

    layout = _layout()
    module = build_stage1_sidecar_module(
        layout,
        batch_per_qp=1,
        segment_bytes=2048,
        team=TEAM_WORLD,
    )
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    arena = layout.allocate_local(device=dev)
    ptrs = layout.epoch_pointers(arena.data_ptr(), 1)
    slot = 1

    layout.view(arena, "dispatch_ready")[slot].fill_(1)
    module.launch_publish_plan_expected(
        1,
        ptrs.plan_ready,
        ptrs.h1_input_expected,
        2,
        1,
        stream=stream,
    )
    module.launch_mark_chunk_ready(
        arena.data_ptr(),
        slot,
        1,
        ptrs.h1_input_ready,
        1,
        1,
        1,
        stream=stream,
    )
    module.launch_enqueue_prepacked(
        arena.data_ptr(),
        slot,
        1,
        ptrs.h1_queue_header,
        ptrs.h1_ready_queue,
        6,
        0,
        6,
        0,
        1,
        stream=stream,
    )
    torch.cuda.synchronize(dev)

    expected = layout.view(arena, "h1_input_expected", parity=1)
    ready = layout.view(arena, "h1_input_ready", parity=1)
    header = layout.view(arena, "h1_queue_header", parity=1)
    queue = layout.view(arena, "h1_ready_queue", parity=1)
    assert expected[:2].tolist() == [1, 1]
    assert ready[:2].tolist() == [0, 1]
    assert header.tolist() == [1, 6, 6, 1]
    assert queue[:6].tolist() == list(range(6))
