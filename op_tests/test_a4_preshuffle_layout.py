# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks for the gfx1250 A4 TDM preshuffle contract."""

import pytest
import torch

from aiter.ops.flydsl.grouped_moe_gfx1250 import _preshuffle_a4_payload


@pytest.mark.parametrize(
    ("tile_m", "tile_k"),
    ((16, 128), (32, 256), (64, 512), (128, 128)),
)
def test_a4_preshuffle_layout(tile_m: int, tile_k: int) -> None:
    """Verify (..., m_rept, k_rept, 4, 16, 16) physical ordering."""
    experts = 2
    rows = tile_m * 2
    k_bytes = tile_k
    payload = torch.arange(experts * rows * k_bytes, dtype=torch.int64)
    payload = payload.remainder(251).to(torch.uint8).reshape(experts, rows, k_bytes)

    actual = _preshuffle_a4_payload(payload, tile_m=tile_m, tile_k=tile_k)
    m_tiles = rows // tile_m
    k_tiles = k_bytes // (tile_k // 2)
    m_rept = tile_m // 16
    k_rept = tile_k // 128
    expected = (
        payload.reshape(
            experts, m_tiles, m_rept, 16, k_tiles, k_rept, 4, 16
        )
        .permute(0, 1, 4, 2, 5, 6, 3, 7)
        .contiguous()
        .reshape_as(payload)
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tile_m", (16, 32, 64, 128))
@pytest.mark.parametrize("tile_k", (128, 256, 512))
def test_a4_preshuffle_ds_read_b128_has_no_bank_conflict(
    tile_m: int, tile_k: int
) -> None:
    """Check each gfx1250 b128 arbitration group uses every bank once."""
    m_rept = tile_m // 16
    k_rept = tile_k // 128
    tile_bytes = tile_m * tile_k // 2

    for ksl in range(k_rept):
        for mr in range(m_rept):
            for second_half in (0, 2 * 16 * 16):
                addresses = []
                for lane in range(32):
                    lane16 = lane % 16
                    kgrp = lane // 16
                    offset = (
                        (((mr * k_rept + ksl) * 4 + kgrp) * 16 + lane16)
                        * 16
                        + second_half
                    )
                    assert offset + 16 <= tile_bytes
                    addresses.append(offset)

                # gfx1250 has 64 four-byte LDS banks. Each 16-lane
                # ds_read_b128 arbitration group should cover all 64 banks
                # exactly once; lanes 16-31 form the second independent group.
                for group_start in range(0, 32, 16):
                    banks = []
                    for offset in addresses[group_start : group_start + 16]:
                        first_bank = (offset // 4) % 64
                        banks.extend((first_bank + word) % 64 for word in range(4))
                    assert sorted(banks) == list(range(64))


@pytest.mark.parametrize("tile_m", (16, 32, 64, 128))
@pytest.mark.parametrize("tile_k", (128, 256, 512))
def test_a4_preshuffle_m16_rounded_oob_is_a_contiguous_prefix(
    tile_m: int, tile_k: int
) -> None:
    """Rounded row OOB must retain every K repeat of each valid M16 group."""
    m_rept = tile_m // 16
    k_rept = tile_k // 128
    row_bytes = tile_k // 2
    cell_bytes = 4 * 16 * 16

    for valid_rows in (0, 1, 15, 16, 17, tile_m - 1, tile_m):
        rounded_rows = min(tile_m, (valid_rows + 15) // 16 * 16)
        prefix_bytes = rounded_rows * row_bytes
        valid_m_rept = rounded_rows // 16
        for mr in range(m_rept):
            for kr in range(k_rept):
                cell_offset = (mr * k_rept + kr) * cell_bytes
                assert (cell_offset < prefix_bytes) == (mr < valid_m_rept)


@pytest.mark.parametrize(
    ("tile_m", "tile_k"), ((15, 128), (16, 64), (32, 192))
)
def test_a4_preshuffle_rejects_unsupported_tiles(tile_m: int, tile_k: int) -> None:
    payload = torch.zeros((1, 32, 256), dtype=torch.uint8)
    with pytest.raises(ValueError, match="A4 preshuffle requires"):
        _preshuffle_a4_payload(payload, tile_m=tile_m, tile_k=tile_k)
