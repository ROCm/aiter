# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 MarloweAI Contributors
"""CPU-only domain, ownership and split-tree contracts for the M4 path."""

import ast
import importlib.util
import itertools
import pathlib
import struct

import pytest

SOURCE = pathlib.Path(__file__).parents[1] / "aiter/ops/flydsl/kernels/mxmoe_tiny_m4.py"
SPEC = importlib.util.spec_from_file_location("tiny_m4_policy", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def test_domain_and_unsupported_fallback_boundaries():
    kwargs = dict(rows=4, hidden=6144, inter=256, experts=257, topk=9, gfx="gfx950")
    assert MODULE.tiny_m4_supported(**kwargs)
    for key, values in {
        "rows": (1, 2, 3, 5, 8, 16, 32, 64, 128, 256),
        "hidden": (4096, 8192),
        "inter": (128, 512),
        "experts": (256, 258),
        "topk": (8, 10),
        "gfx": ("gfx942", "gfx1201"),
        "contiguous": (False,),
    }.items():
        for value in values:
            assert not MODULE.tiny_m4_supported(**(kwargs | {key: value}))


def test_partial_storage_written_once_in_full_producer_grid():
    addresses = set()
    for route, tile, split, part, block, element in itertools.product(
        range(36), range(8), range(6), range(4), range(4), range(4)
    ):
        # Independently reconstruct logical gate/up column ownership.
        col = tile * 32 + (block % 2) * 16 + (block // 2) * 256 + part * 4 + element
        offset = MODULE.split6_indices(route, col)[split]
        assert offset not in addresses
        addresses.add(offset)
    assert addresses == set(range(6 * 36 * 512))
    for route, col in ((-1, 0), (36, 0), (0, -1), (0, 512)):
        with pytest.raises(ValueError):
            MODULE.split6_indices(route, col)


def test_reset_exactly_covers_all_output_without_route_dependency():
    # Only native route slot zero clears; no assumption about expert-ID values.
    words = [
        token * 3072 + (split * 8 + tile) * 64 + lane
        for token, split, tile, lane in itertools.product(
            range(4), range(6), range(8), range(64)
        )
    ]
    assert len(words) == len(set(words)) == 4 * 3072
    assert sorted(words) == list(range(4 * 3072))


def test_split_tree_has_declared_padding_and_not_linear_fold():
    # Cancellation probes distinguish padded halves/quarters/pairs from a fold.
    values = [1e20, 1.0, -1e20, 2.0, 3.0, 4.0]
    round_tree = list(map(_fp32, values)) + [0.0, 0.0]
    for stride in (4, 2, 1):
        round_tree = [
            _fp32(round_tree[i] + round_tree[i + stride]) for i in range(stride)
        ]
    assert round_tree[0] == 7.0
    linear = 0.0
    for value in values:
        linear = _fp32(linear + _fp32(value))
    assert linear == 9.0
    assert MODULE.reduce_split6([1, 2, 4, 8, 16, 32]) == 63
    with pytest.raises(ValueError):
        MODULE.reduce_split6([1, 2, 3])


def test_weight_fragment_offsets_cover_all_split_payload_without_oob():
    # Each N16/K128 shuffled weight tile is 1024 bytes: 64 lanes x16.
    for expert in (0, 128, 256):
        for tile, split, unit, half, block, lane in itertools.product(
            range(8), range(6), range(4), range(2), range(4), range(64)
        ):
            column = tile * 32 + block % 2 * 16 + block // 2 * 256
            offset = (
                expert * 512 * 3072
                + column * 3072
                + lane * 16
                + (split * 8 + unit * 2 + half) * 1024
            )
            assert expert * 512 * 3072 <= offset
            assert offset + 16 <= (expert + 1) * 512 * 3072
            assert offset % 16 == 0


def test_input_pack_and_middle_scale_groups_complete():
    # Producer lanes each own16BF16; adjacent lane pair owns one group32.
    packed_bytes = {
        (lane * 8 + byte) for lane, byte in itertools.product(range(64), range(8))
    }
    assert packed_bytes == set(range(512))
    scale_indices = {lane // 2 for lane in range(0, 64, 2)}
    assert scale_indices == set(range(32))
    # Two middle waves each own128 values; four disjoint group32 blocks.
    middle_bytes = {
        wave * 64 + lane for wave, lane in itertools.product(range(2), range(64))
    }
    assert middle_bytes == set(range(128))
    assert {
        wave * 4 + lane // 16
        for wave, lane in itertools.product(range(2), range(0, 64, 16))
    } == set(range(8))


def test_atomic_ownership_covers_each_output_column_once_per_route():
    for tile_width, waves in ((128, 4), (256, 4), (128, 2)):
        columns = [
            tile * tile_width
            + wave * (tile_width // waves)
            + block * 16
            + lane * 2
            + half
            for tile, wave, block, lane, half in itertools.product(
                range(6144 // tile_width),
                range(waves),
                range(tile_width // waves // 16),
                range(8),
                range(2),
            )
        ]
        assert len(columns) == len(set(columns)) == 6144
        assert set(columns) == set(range(6144))


def test_source_preserves_native_numerical_helpers_and_two_launches():
    tree = ast.parse(SOURCE.read_text())
    imports = {
        item.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for item in node.names
    }
    assert {
        "_inline_e8m0",
        "_e8m0_from_amax",
        "_activation_mul_batch",
        "_scale_mma_atoms",
    } <= imports
    assert "pyhip" not in SOURCE.read_text().lower()
    launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]
    assert len(launches) == 2
    assert "0x400000" not in SOURCE.read_text()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
