# SPDX-License-Identifier: MIT
"""Task 3: push-group finalize kernel produces GEMM1 tile metadata from counts."""
import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs gfx1250"
)


def _ref(running, tile_m, cap, rank, epr):
    """Non-compacted fixed layout: expert ``le``'s tile ``t`` -> meta index
    ``le*tiles_per_expert + t``; every other slot keeps the host default
    (tile_row_base=-1, expert_ids=-1 sentinel, tile_valid=0). The GEMM launches a
    fixed E*tiles_per_expert grid and skips defaults, so no compaction is done."""
    tiles_per_expert = (cap + tile_m - 1) // tile_m
    max_tiles = epr * tiles_per_expert
    tile_row_base = [-1] * max_tiles
    expert_ids = [-1] * max_tiles
    tile_valid = [0] * max_tiles
    n_active = 0
    for le, count in enumerate(running):
        count = min(count, cap)
        num_tiles = math.ceil(count / tile_m) if count > 0 else 0
        for t in range(num_tiles):
            meta = le * tiles_per_expert + t
            tile_row_base[meta] = le * cap + t * tile_m
            expert_ids[meta] = rank * epr + le
            tile_valid[meta] = min(tile_m, count - t * tile_m)
        n_active += num_tiles
    num_valid = n_active * tile_m
    return tile_row_base, expert_ids, tile_valid, num_valid


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
    tile_valid = torch.zeros((max_tiles,), dtype=torch.int32, device=dev)
    num_valid = torch.zeros(1, dtype=torch.int32, device=dev)

    launch_push_group_finalize(
        pg_running_ptr=running.data_ptr(),
        tile_row_base_ptr=tile_row_base.data_ptr(),
        expert_ids_ptr=expert_ids.data_ptr(),
        num_valid_ptr=num_valid.data_ptr(),
        tile_valid_ptr=tile_valid.data_ptr(),
        num_local_experts=epr,
        cap=cap,
        tile_m=tile_m,
        rank=rank,
        experts_per_rank=epr,
    )
    torch.cuda.synchronize()

    ref_trb, ref_eid, ref_tv, ref_nv = _ref(running_list, tile_m, cap, rank, epr)
    assert int(num_valid.item()) == ref_nv, (int(num_valid.item()), ref_nv)
    assert tile_row_base.cpu().tolist() == ref_trb
    assert expert_ids.cpu().tolist() == ref_eid
    assert tile_valid.cpu().tolist() == ref_tv


def test_finalize_compact_valid_rows():
    """The optional compact path emits the exact set of OCCUPIED fixed-slot rows
    (le*cap + slot for slot < min(count, cap)) plus their total, for the
    route-indexed preshuffle. Order across experts is arbitrary (per-expert atomic
    claim), so compare as a set over the first ``num_routes`` entries."""
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
    tile_valid = torch.zeros((max_tiles,), dtype=torch.int32, device=dev)
    num_valid = torch.zeros(1, dtype=torch.int32, device=dev)

    max_land = 512
    valid_rows = torch.full((max_land,), -1, dtype=torch.int32, device=dev)
    valid_routes = torch.zeros(1, dtype=torch.int32, device=dev)

    launch_push_group_finalize(
        pg_running_ptr=running.data_ptr(),
        tile_row_base_ptr=tile_row_base.data_ptr(),
        expert_ids_ptr=expert_ids.data_ptr(),
        num_valid_ptr=num_valid.data_ptr(),
        tile_valid_ptr=tile_valid.data_ptr(),
        num_local_experts=epr,
        cap=cap,
        tile_m=tile_m,
        rank=rank,
        experts_per_rank=epr,
        valid_rows_ptr=valid_rows.data_ptr(),
        valid_routes_ptr=valid_routes.data_ptr(),
    )
    torch.cuda.synchronize()

    expected_rows = set()
    for le, count in enumerate(running_list):
        for slot in range(min(count, cap)):
            expected_rows.add(le * cap + slot)
    n = int(valid_routes.item())
    assert n == len(expected_rows), (n, len(expected_rows))
    got = set(valid_rows[:n].cpu().tolist())
    assert got == expected_rows, (sorted(got)[:8], sorted(expected_rows)[:8])
    # tail beyond num_routes is untouched (never read by the preshuffle).
    assert valid_rows[n:].cpu().eq(-1).all().item()

    # In compact mode the tile metadata is also densely packed into
    # [0, total_tiles) (arbitrary expert order via atomic claim), tail = sentinel.
    # num_valid still == total_tiles*tile_m (num_valid atomic doubles as the tile
    # cursor). Compare (tile_row_base, expert_id, tile_valid) triples as a set.
    total_tiles = int(num_valid.item()) // tile_m
    expected_tiles = set()
    for le, count in enumerate(running_list):
        count = min(count, cap)
        nt = (count + tile_m - 1) // tile_m
        for t in range(nt):
            expected_tiles.add(
                (le * cap + t * tile_m, rank * epr + le, min(tile_m, count - t * tile_m))
            )
    assert total_tiles == len(expected_tiles), (total_tiles, len(expected_tiles))
    got_tiles = set(
        zip(
            tile_row_base[:total_tiles].cpu().tolist(),
            expert_ids[:total_tiles].cpu().tolist(),
            tile_valid[:total_tiles].cpu().tolist(),
        )
    )
    assert got_tiles == expected_tiles, (
        sorted(got_tiles)[:4],
        sorted(expected_tiles)[:4],
    )
    # tail tiles keep host defaults (sentinel -> gemm skips them).
    assert tile_row_base[total_tiles:].cpu().eq(-1).all().item()
    assert tile_valid[total_tiles:].cpu().eq(0).all().item()
