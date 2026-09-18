# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-rule tests for the adaptive decode Top-K kernel.

These pin the config the kernel is built from, so a shape cannot start taking a
different kernel without a test saying so. They need no device.
"""

import pytest

from aiter.ops import topk
from aiter.ops.flydsl.kernels.topk import topk_per_row_decode_adaptive as km

# A cooperating unordered shape, which is the only build the early stop reaches.
_REACHES = {"rows": 1, "seq": 131_072, "k": 2048, "cu_count": 80}


def _cfg(**over):
    args = {**_REACHES, **over}
    return km.decode_adaptive_config(
        args["rows"],
        args["seq"],
        args["k"],
        ordered=args.get("ordered", False),
        cu_count=args["cu_count"],
    )


def test_early_stop_is_on_by_default():
    assert km.EARLY_STOP_DEFAULT is True
    assert km.early_stop_default() is True
    assert _cfg()["kw"].get("early_stop") is True


@pytest.mark.parametrize("name", [km.EARLY_STOP_ENV, "FLYDSL_TOPK_COMPACT_ES"])
@pytest.mark.parametrize("asked,expected", [("0", False), ("1", True), ("", False)])
def test_early_stop_env_overrides_both_names(monkeypatch, name, asked, expected):
    monkeypatch.setenv(name, asked)
    assert km.early_stop_default() is expected


def test_early_stop_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv(km.EARLY_STOP_ENV, "0")
    cfg = km.decode_adaptive_config(
        _REACHES["rows"],
        _REACHES["seq"],
        _REACHES["k"],
        ordered=False,
        early_stop=True,
        cu_count=_REACHES["cu_count"],
    )
    assert cfg["kw"].get("early_stop") is True


@pytest.mark.parametrize(
    "over,why",
    [
        ({"ordered": True}, "the last pass is what produces the order"),
        ({"rows": 128}, "the candidate buffer replaces the row walk"),
        ({"seq": 4096}, "an all-short build never runs that pass"),
    ],
)
def test_early_stop_is_absent_where_the_build_cannot_reach_it(over, why):
    cfg = _cfg(**over)
    assert "early_stop" not in cfg["kw"], why


def test_the_three_drops_are_the_ones_the_shapes_above_exercise():
    """Guards the test above: each shape must really hit its own reason."""
    assert _cfg(ordered=True)["kw"]["ordered"] is True
    assert _cfg(rows=128)["compact"] is True
    assert _cfg(seq=4096)["kw"]["tier_mode"] == "short"


# --- the gate ---------------------------------------------------------------
#
# The decode backend used to be chosen twice, by `topk.py` from a band table and
# again by the FlyDSL host from the arch, which could contradict each other and
# run the adaptive kernel on a shape only the chunked table admitted. These pin
# that there is one decision.

_BAND = ((8_192, 20_000, 2, 128),)


@pytest.fixture
def one_band(monkeypatch):
    """One known band, replacing whatever the shipped table carries, so the test
    reads the rule rather than the numbers. The arch is real because a miss has
    to fall through to the chunked gate, which only exists for a real one."""
    monkeypatch.setitem(
        topk._ADAPTIVE_BANDS, ("gfx942", 80), {True: {1024: _BAND}, False: {}}
    )
    topk._decode_backend.cache_clear()
    yield
    topk._decode_backend.cache_clear()


@pytest.mark.parametrize(
    "width,rows,k,stable,indices_only,expected",
    [
        (8_192, 8, 1024, True, True, topk.BACKEND_ADAPTIVE),
        (20_000, 128, 1024, True, True, topk.BACKEND_ADAPTIVE),
        (8_192, 1, 1024, True, True, topk.BACKEND_CHUNKED),  # below min_rows
        (8_192, 256, 1024, True, True, topk.BACKEND_UPSTREAM),  # above max_rows
        (32_768, 8, 1024, True, True, topk.BACKEND_UPSTREAM),  # outside the width band
        (8_192, 8, 2048, True, True, topk.BACKEND_CHUNKED),  # k has no band
        # The other emit has no band here, and gfx942's chunked gate does not
        # take a narrow unordered buffer either, so it is the HIP kernel.
        (8_192, 8, 1024, False, True, topk.BACKEND_UPSTREAM),
        (8_192, 8, 1024, True, False, topk.BACKEND_CHUNKED),  # wants values
    ],
)
def test_the_band_decides_and_a_miss_falls_through(
    one_band, width, rows, k, stable, indices_only, expected
):
    got = topk._decode_backend("gfx942", 80, stable, width, rows, k, indices_only)
    assert got == expected


def test_an_unmeasured_cu_count_is_not_guessed_at(one_band):
    """The same arch at another CU count keeps the upstream path, because the
    grid the bands were measured against is built from that count.

    The count is taken as one the table does not hold, so that measuring a new
    card is a table edit and does not also reach in here.
    """
    measured = {cu for arch, cu in topk._ADAPTIVE_BANDS if arch == "gfx942"}
    unmeasured = next(cu for cu in range(1, 1024) if cu not in measured)
    assert (
        topk._decode_backend("gfx942", unmeasured, True, 8_192, 8, 1024, True)
        != topk.BACKEND_ADAPTIVE
    )


def test_the_shipped_bands_are_well_formed():
    """A malformed band fails open, routing an unmeasured shape to the kernel,
    so the shape of the table is pinned rather than left to review."""
    for device, per_emit in topk._ADAPTIVE_BANDS_BY_K_GROUP.items():
        arch, cu_count = device
        assert isinstance(arch, str) and cu_count > 0
        assert set(per_emit) == {True, False}, device
        for stable, per_group in per_emit.items():
            for ks, bands in per_group.items():
                seen_to = 0
                for min_w, max_w, min_r, max_r in bands:
                    assert min_w <= max_w, (device, stable, ks)
                    assert 0 < min_r <= max_r, (device, stable, ks)
                    # Ascending and disjoint, so one width cannot match two
                    # bands and make the answer depend on the tuple order.
                    assert min_w > seen_to, (device, stable, ks, min_w)
                    seen_to = max_w


def test_every_k_in_a_group_gets_that_groups_bands():
    """The lookup table is expanded from k groups at import; this is the
    expansion, which a typo in a group tuple would otherwise silently drop."""
    for device, per_emit in topk._ADAPTIVE_BANDS_BY_K_GROUP.items():
        for stable, per_group in per_emit.items():
            for ks, bands in per_group.items():
                for k in ks:
                    assert topk._ADAPTIVE_BANDS[device][stable][k] == bands


def test_the_flydsl_host_takes_the_gates_answer_rather_than_its_own():
    """The signature is the contract: a backend handed in is not re-derived."""
    import inspect

    from aiter.ops.flydsl.topk.topk_per_row import flydsl_top_k_per_row_decode

    assert "backend" in inspect.signature(flydsl_top_k_per_row_decode).parameters
