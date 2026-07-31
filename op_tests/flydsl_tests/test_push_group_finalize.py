# SPDX-License-Identifier: MIT
"""Task 3: push-group finalize kernel produces GEMM1 tile metadata from counts."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs gfx1250"
)


def _ref(running, tile_m, cap, rank, epr):
    tile_row_base, expert_ids = [], []
    for le, count in enumerate(running):
        num_tiles = math.ceil(count / tile_m) if count > 0 else 0
        for t in range(num_tiles):
            tile_row_base.append(le * cap + t * tile_m)
            expert_ids.append(rank * epr + le)
    num_valid = len(tile_row_base) * tile_m
    return tile_row_base, expert_ids, num_valid


def test_finalize_metadata_matches_ref():
    from aiter.ops.flydsl.kernels.push_group_finalize_gfx1250 import (
        launch_push_group_finalize,
    )

    dev = torch.device("cuda")
    tile_m, cap, epr, rank = 64, 256, 4, 0
    running_list = [70, 0, 130, 64]
    running = torch.tensor(running_list, dtype=torch.int32, device=dev)

    max_tiles = epr * (cap // tile_m)
    tile_row_base = torch.full((max_tiles,), -1, dtype=torch.int32, device=dev)
    expert_ids = torch.full((max_tiles,), -1, dtype=torch.int32, device=dev)
    num_valid = torch.zeros(1, dtype=torch.int32, device=dev)

    launch_push_group_finalize(
        pg_running_ptr=running.data_ptr(),
        tile_row_base_ptr=tile_row_base.data_ptr(),
        expert_ids_ptr=expert_ids.data_ptr(),
        num_valid_ptr=num_valid.data_ptr(),
        num_local_experts=epr,
        cap=cap,
        tile_m=tile_m,
        rank=rank,
        experts_per_rank=epr,
    )
    torch.cuda.synchronize()

    ref_trb, ref_eid, ref_nv = _ref(running_list, tile_m, cap, rank, epr)
    nt = len(ref_trb)
    assert int(num_valid.item()) == ref_nv, (int(num_valid.item()), ref_nv)
    assert tile_row_base[:nt].cpu().tolist() == ref_trb
    assert expert_ids[:nt].cpu().tolist() == ref_eid
