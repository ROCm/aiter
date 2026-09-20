# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the adaptive decode Top-K kernel.

Most of these pin the config the kernel is built from, so a shape cannot start
taking a different kernel without a test saying so; those need no device. The
last section needs one: it checks on the card that the kernel the table names is
the kernel that launches, which no host-side reading of the table can show.
"""

import pytest
import torch

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


# --- on the card ------------------------------------------------------------
#
# Everything above reads the table. None of it can show that the kernel the
# table names is the kernel that runs, because that is decided at the launch
# site, so this section observes the launch by patching it and checks what it
# wrote. Skipped without a device, and a no-op on a card the table does not
# carry -- there is no band there, so there is no claim to check.

# Rows the bands are expressed in. Band edges are checked from both sides, so
# the neighbours of min_rows and max_rows have to be on this axis too.
_ROWS_AXIS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# A cell is allocated in full. The widest band times the tallest row count is
# half a billion float32, which is not what an op test should ask a shared card
# for, so the corner is left to the measurement harness and everything under
# the cap is checked here.
_MAX_ELEMENTS = 64 << 20


def _band_edge_cells(arch, cu_count):
    """Both sides of every band edge in the shipped table, read off the table.

    A hand-written cell list goes stale the moment the table is re-fitted, and
    a stale list passes by testing nothing.
    """
    out, seen = [], []

    def add(width, rows, k, stable):
        key = (width, rows, k, stable)
        if rows in _ROWS_AXIS and key not in seen and rows * width <= _MAX_ELEMENTS:
            seen.append(key)
            out.append(key)

    for stable, per_group in topk._ADAPTIVE_BANDS_BY_K_GROUP[(arch, cu_count)].items():
        for ks, bands in per_group.items():
            k = ks[0]
            for min_w, max_w, min_r, max_r in bands:
                for w in (min_w, max_w):
                    add(w, min_r, k, stable)  # inside, at the bottom edge
                    add(w, max_r, k, stable)  # inside, at the top edge
                    i = _ROWS_AXIS.index(min_r)
                    if i:
                        add(w, _ROWS_AXIS[i - 1], k, stable)  # just below
                    j = _ROWS_AXIS.index(max_r)
                    if j + 1 < len(_ROWS_AXIS):
                        add(w, _ROWS_AXIS[j + 1], k, stable)  # just above
    return out


def _this_card():
    if not torch.cuda.is_available():
        pytest.skip("needs a device: this section observes a launch")
    props = torch.cuda.get_device_properties(0)
    return props.gcnArchName.split(":")[0], props.multi_processor_count


def test_the_kernel_the_table_names_is_the_one_that_launches(monkeypatch):
    """Per band-edge cell: the gate agrees with the table, the launch agrees
    with the gate, and the indices select the right values."""
    import aiter
    from aiter.ops.flydsl.topk import topk_per_row as host

    arch, cu_count = _this_card()
    if (arch, cu_count) not in topk._ADAPTIVE_BANDS_BY_K_GROUP:
        pytest.skip(
            f"{arch} at {cu_count} CU is not in the table, so it claims nothing"
        )

    launched = []

    def spy(name, attr):
        real = getattr(host, attr)

        def wrapper(*args, **kwargs):
            launched.append(name)
            return real(*args, **kwargs)

        monkeypatch.setattr(host, attr, wrapper)

    spy(topk.BACKEND_ADAPTIVE, "_run_adaptive")
    spy(topk.BACKEND_CHUNKED, "_run_compiled")
    spy("port", "flydsl_radix_topk_one_block")

    # The gate answers "not ours" and never names the fallback kernel, so what
    # a declined cell launches is an arch question. On the one-block arches
    # upstream runs the FlyDSL port, which is spied above and reports by name;
    # elsewhere it runs the HIP kernel, which is not on this module and so is
    # observed by nothing having been spied.
    declined_runs = (
        "port" if arch in host._FLYDSL_TOPK_ONE_BLOCK_ARCHES else topk.BACKEND_UPSTREAM
    )

    cells = _band_edge_cells(arch, cu_count)
    assert cells, "the table carries this card, so it has band edges to check"

    torch.manual_seed(0)
    bad = []
    for width, rows, k, stable in cells:
        want = topk._decode_backend(arch, cu_count, stable, width, rows, k, True)
        # float32: the FlyDSL decode path takes no other dtype.
        logits = torch.randn(rows, width, dtype=torch.float32, device="cuda")
        seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
        out = torch.empty(rows, k, dtype=torch.int32, device="cuda")

        launched.clear()
        got = topk.decode_backend_for_call(
            logits,
            1,
            seq_lens,
            out,
            rows,
            logits.stride(0),
            logits.stride(1),
            k,
            stable,
            None,
        )
        aiter.top_k_per_row_decode(
            logits,
            1,
            seq_lens,
            out,
            rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
            stable=stable,
        )
        # `_run_adaptive` and the port both go on to call `_run_compiled`, so
        # the first name recorded is the one that was dispatched to.
        ran = launched[0] if launched else topk.BACKEND_UPSTREAM

        # Compare selected values, not indices: a row of this many float32
        # samples carries ties, and either index of a tied pair is a correct
        # answer, so an index comparison reports a right kernel as wrong.
        want_values = torch.sort(torch.topk(logits, k, dim=1).values, dim=1).values
        got_values = torch.sort(logits.gather(1, out.long()), dim=1).values

        expect_ran = declined_runs if got == topk.BACKEND_UPSTREAM else got
        if got != want or ran != expect_ran or not torch.equal(got_values, want_values):
            bad.append(
                f"width={width} rows={rows} k={k} stable={stable}: "
                f"gate said {got!r} (table says {want!r}), "
                f"{ran!r} launched (expected {expect_ran!r}), "
                f"values {'match' if torch.equal(got_values, want_values) else 'WRONG'}"
            )

        del logits, seq_lens, out, want_values, got_values
        torch.cuda.empty_cache()

    assert not bad, f"{len(bad)}/{len(cells)} cells:\n" + "\n".join(bad)
