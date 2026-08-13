"""Pure-data validation for the measured gfx942 dense GDN route table."""

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "aiter"
    / "ops"
    / "_gdn_dense_gfx942_routes.py"
)
_SPEC = spec_from_file_location("_gdn_dense_gfx942_routes_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
routes = module_from_spec(_SPEC)
_SPEC.loader.exec_module(routes)


def _unpack_record(record: int) -> tuple[tuple[int, int, int, bool], str]:
    path = routes._PATHS[record & 0b11]
    key = record >> 2
    state_on = bool(key & 0b1)
    key >>= 1
    h = key & ((1 << 10) - 1)
    key >>= 10
    t = (key & ((1 << 11) - 1)) * 64
    b = key >> 11
    return (b, t, h, state_on), path


def test_route_metadata_and_counts_are_exact_and_read_only():
    assert routes.DENSE_GFX942_RECORD_COUNT == 477
    assert dict(routes.DENSE_GFX942_WINNER_COUNTS) == {
        "cf": 196,
        "cs": 25,
        "ws": 182,
        "wf": 74,
    }
    assert routes.DENSE_GFX942_SOURCE_SHA256 == (
        "3e333710940d3c6d9feec6f8792fd500be5d0ac28423dc165ef80622cd712d89"
    )
    assert routes.DENSE_GFX942_CLOSEOUT_DATE == "2026-07-31"
    assert routes.DENSE_GFX942_PLATFORM == "AMD Instinct MI308X / gfx942"
    assert routes.DENSE_GFX942_COMPUTE_UNITS == 80
    assert routes.DENSE_GFX942_MEASUREMENT_JIT_SHA256.startswith("7fbdffd")
    assert routes.DENSE_GFX942_FINAL_JIT_SHA256.startswith("70fff550")

    with pytest.raises(TypeError):
        routes.DENSE_GFX942_WINNER_COUNTS["cf"] = 0
    with pytest.raises(TypeError):
        routes.DENSE_GFX942_PROVENANCE["sha256"] = "changed"


def test_all_packed_keys_are_unique_aligned_and_lookup_exactly():
    records = list(routes._PACKED_ROUTES)
    assert len(records) == routes.DENSE_GFX942_RECORD_COUNT
    assert records == sorted(records)

    unpacked = [_unpack_record(record) for record in records]
    keys = [key for key, _ in unpacked]
    assert len(keys) == len(set(keys)) == 477
    assert all(t % 64 == 0 for (_, t, _, _) in keys)
    assert Counter(path for _, path in unpacked) == Counter(
        routes.DENSE_GFX942_WINNER_COUNTS
    )
    assert all(routes.lookup_dense_gfx942_path(*key) == path for key, path in unpacked)


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((1, 64, 10, False), "cf"),
        ((1, 64, 1, False), "cs"),
        ((1, 128, 11, False), "ws"),
        ((1, 192, 128, False), "wf"),
    ],
)
def test_representative_winner(shape, expected):
    assert routes.lookup_dense_gfx942_path(*shape) == expected


def test_same_t_and_bh_can_have_different_winners():
    # Both shapes have T=128 and B*H=8, so B and H must remain separate keys.
    assert routes.lookup_dense_gfx942_path(1, 128, 8, False) == "cs"
    assert routes.lookup_dense_gfx942_path(2, 128, 4, False) == "ws"


@pytest.mark.parametrize(
    "shape",
    [
        (0, 64, 1, False),
        (64, 64, 1, False),
        (1, 0, 1, False),
        (1, 65, 1, False),
        (1, 131072, 1, False),
        (1, 64, 0, False),
        (1, 64, 1024, False),
        (1.0, 64, 1, False),
        (1, 64.0, 1, False),
        (1, 64, 1.0, False),
        (1, 64, 1, 0),
    ],
)
def test_invalid_or_out_of_range_fields_are_rejected(shape):
    assert routes.lookup_dense_gfx942_path(*shape) is None


def test_valid_but_unmeasured_shape_is_not_extrapolated():
    assert routes.lookup_dense_gfx942_path(63, 64, 1023, True) is None
