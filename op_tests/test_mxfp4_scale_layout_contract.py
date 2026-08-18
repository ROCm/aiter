# SPDX-License-Identifier: MIT
"""Regression checks for the A4W4 preshuffled scale-buffer ABI."""

import pytest


_EP16_BM = 32
_EP16_MAX_ROUTE_TILES = (16 * 128 + _EP16_BM - 1) // _EP16_BM + 56


def _reference_mfma_scale_offsets(groups_per_row):
    """Invert the consumer's ``[ku, k_lane, n_lane][ikxdl, im_a]`` layout."""

    assert groups_per_row % 8 == 0
    k_chunks = groups_per_row // 8
    chunk_dwords = k_chunks * 64
    offsets = {}
    for dword in range(chunk_dwords):
        coord = dword
        n_lane = coord % 16
        coord //= 16
        k_lane = coord % 4
        coord //= 4
        ku = coord % k_chunks
        coord //= k_chunks
        assert coord == 0
        for ikxdl in range(2):
            for im_a in range(2):
                row = im_a * 16 + n_lane
                group = ku * 8 + ikxdl * 4 + k_lane
                offsets[row, group] = dword * 4 + ikxdl * 2 + im_a
    return offsets


def _dispatch_input_scale_offset(grouped_row, group, groups_per_row):
    """BM32 direct-dispatch address consumed by gfx950 ``mxfp4_gemm1``."""

    physical = grouped_row // _EP16_BM
    row_in_tile = grouped_row % _EP16_BM
    im_a = row_in_tile // 16
    n_lane = row_in_tile % 16
    ku = group // 8
    ikxdl = (group % 8) // 4
    k_lane = group % 4
    chunk_dwords = (groups_per_row // 8) * 64
    dword = physical * chunk_dwords + ku * 64 + k_lane * 16 + n_lane
    return dword * 4 + ikxdl * 2 + im_a


def _h1_output_scale_writer_offset(grouped_row, scale_index, groups_per_row):
    """Address emitted by the GMM1 SiLU/A4 quant epilogue."""

    physical = grouped_row // _EP16_BM
    row_in_tile = grouped_row % _EP16_BM
    n_block = scale_index // 4
    wave_group = scale_index % 4
    ku = n_block // 2
    ikxdl = n_block % 2
    sub = row_in_tile // 16
    m_lane = row_in_tile % 16
    chunk_dwords = (groups_per_row // 8) * 64
    dword = physical * chunk_dwords + ku * 64 + wave_group * 16 + m_lane
    return dword * 4 + ikxdl * 2 + sub


def _stage2_input_scale_reader_offset(grouped_row, scale_index, groups_per_row):
    """Address selected by MegaMoE GMM2's scale_view plus MFMA opselA."""

    physical = grouped_row // _EP16_BM
    row_in_tile = grouped_row % _EP16_BM
    lane_div_16 = scale_index % 4
    lane_mod_16 = row_in_tile % 16
    chunk_kt = scale_index // 8
    ikxdl = (scale_index % 8) // 4
    im_a = row_in_tile // 16
    chunk_dwords = (groups_per_row // 8) * 64
    dword = (
        physical * chunk_dwords
        + chunk_kt * 64
        + lane_div_16 * 16
        + lane_mod_16
    )
    opsel_a = ikxdl * 2 + im_a
    return dword * 4 + opsel_a


def test_scale_k_group_count_rounds_up_to_host_preshuffle_extent():
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.kernels.megamoe_tile.gemm_common import (
        MXFP4_SCALE_LAYOUT_TAG,
        kas_c_k1_for,
        kas_per_chunk_dw_for,
        kbs_c_k1_for,
        kbs_stride_n0_dw_for,
    )

    assert MXFP4_SCALE_LAYOUT_TAG == "sc2"
    assert kas_c_k1_for(256) == 1
    assert kbs_c_k1_for(256) == 1
    # K=384 has twelve logical 1x32 scales.  The host shuffle stores sixteen,
    # so it needs two 256-K chunks rather than truncating to one.
    assert kas_c_k1_for(384) == 2
    assert kbs_c_k1_for(384) == 2
    assert kas_per_chunk_dw_for(384) == 128
    assert kbs_stride_n0_dw_for(384) == 128


@pytest.mark.parametrize("k", [32, 128, 256, 384, 512, 768, 3584])
def test_scale_k_group_extent_matches_eight_group_padding(k):
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl.kernels.megamoe_tile.gemm_common import (
        kas_c_k1_for,
        kbs_c_k1_for,
    )

    expected_chunks = ((k // 32) + 7) // 8
    assert kas_c_k1_for(k) == expected_chunks
    assert kbs_c_k1_for(k) == expected_chunks


def test_ep16_bm32_k7168_input_scale_offsets_match_mfma_layout():
    groups_per_row = 7168 // 32
    chunk_bytes = _EP16_BM * groups_per_row
    reference = _reference_mfma_scale_offsets(groups_per_row)
    actual = {
        (row, group): _dispatch_input_scale_offset(row, group, groups_per_row)
        for row in range(_EP16_BM)
        for group in range(groups_per_row)
    }

    assert actual == reference
    assert len(actual) == chunk_bytes
    assert set(actual.values()) == set(range(chunk_bytes))

    last_physical = _EP16_MAX_ROUTE_TILES - 1
    last_base = last_physical * chunk_bytes
    for (row, group), local_offset in reference.items():
        grouped_row = last_physical * _EP16_BM + row
        assert (
            _dispatch_input_scale_offset(grouped_row, group, groups_per_row)
            == last_base + local_offset
        )
    assert (
        _dispatch_input_scale_offset(
            _EP16_MAX_ROUTE_TILES * _EP16_BM - 1,
            groups_per_row - 1,
            groups_per_row,
        )
        == _EP16_MAX_ROUTE_TILES * chunk_bytes - 1
    )


def test_ep16_h1_scale_writer_roundtrips_stage2_reader_and_fits_arena():
    groups_per_row = 3072 // 32
    rows = _EP16_MAX_ROUTE_TILES * _EP16_BM
    plane_bytes = rows * groups_per_row
    payload = bytearray(plane_bytes)
    visited = bytearray(plane_bytes)

    for row in range(rows):
        for scale_index in range(groups_per_row):
            writer_offset = _h1_output_scale_writer_offset(
                row, scale_index, groups_per_row
            )
            reader_offset = _stage2_input_scale_reader_offset(
                row, scale_index, groups_per_row
            )
            assert writer_offset == reader_offset
            assert 0 <= writer_offset < plane_bytes
            assert visited[writer_offset] == 0
            visited[writer_offset] = 1
            payload[writer_offset] = (row * 131 + scale_index * 17) & 0xFF

    assert visited.count(1) == plane_bytes
    assert (
        _h1_output_scale_writer_offset(rows - 1, groups_per_row - 1, groups_per_row)
        == plane_bytes - 1
    )

    for row in range(rows):
        for scale_index in range(groups_per_row):
            reader_offset = _stage2_input_scale_reader_offset(
                row, scale_index, groups_per_row
            )
            assert payload[reader_offset] == (
                row * 131 + scale_index * 17
            ) & 0xFF
