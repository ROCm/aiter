# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def _layout(num_qp=4):
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout

    return HierCcoArenaLayout.create(
        ring_depth=8,
        num_qp=num_qp,
        chunk_bytes=64 * 1024,
        max_m_tiles=8,
        max_source_tokens=64,
        max_h1_n_blocks=3,
    )


class _FakeModule:
    team = "world"
    batch_per_qp = 2

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("launch_"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record


def test_stage2_sidecar_host_lifecycle_and_record_geometry():
    from aiter.ops.flydsl.kernels.megamoe_tile.cco.stage2_sidecar import CcoStage2ReturnSidecar

    layout = _layout()
    sidecar = CcoStage2ReturnSidecar(layout, _FakeModule())
    ptrs = layout.epoch_pointers(0x100000, 1)

    assert layout.region("partial_request").shape == (8, 4)
    sidecar.post_partial_return(1, 2, 3, 1, 0, 1, stream=4)
    with pytest.raises(RuntimeError, match="credited/reclaimed"):
        sidecar.post_partial_return(1, 2, 3, 1, 0, 2, stream=4)
    sidecar.publish_received_partials(3, 0, 1, ptrs, 5, 8, stream=4)
    with pytest.raises(ValueError, match="bounded return batch"):
        sidecar.publish_received_partials(3, 0, 1, ptrs, 0, 9, stream=4)
    sidecar.reclaim_partial(1, 3, 0, 1, stream=4)
    assert sidecar.outstanding_slots == ()


@pytest.mark.parametrize(
    "num_qp,batch",
    [(1, 8), (2, 4), (4, 2)],
)
def test_stage2_default_record_bounded_geometries(num_qp, batch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")
    from aiter.ops.flydsl.kernels.megamoe_tile.cco import build_stage2_sidecar_module

    module = build_stage2_sidecar_module(
        _layout(num_qp), batch_per_qp=batch, team="world"
    )
    assert module.payload_bytes == 8 * 7424
    assert module.payload_bytes <= 64 * 1024


def test_stage2_received_publisher_compiles_and_runs():
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")
    from aiter.ops.flydsl.kernels.megamoe_tile.cco import TEAM_WORLD, build_stage2_sidecar_module

    layout = _layout()
    module = build_stage2_sidecar_module(layout, team=TEAM_WORLD)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    arena = layout.allocate_local(device=dev)
    ptrs = layout.epoch_pointers(arena.data_ptr(), 1)
    slot = 2
    layout.view(arena, "partial_ready")[slot].fill_(1)

    module.launch_publish_received(
        arena.data_ptr(),
        slot,
        1,
        ptrs.node_partial_ready,
        3,
        8,
        stream=stream,
    )
    torch.cuda.synchronize(dev)
    ready = layout.view(arena, "node_partial_ready", parity=1)
    assert torch.count_nonzero(ready[:3]).item() == 0
    assert ready[3:11].tolist() == [1] * 8
    assert torch.count_nonzero(ready[11:]).item() == 0
