# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the adaptive decode Top-K kernel.

The host checks pin the config the kernel is built from and the table that routes
to it, and need no device. The sweep runs every band edge of this card through the
gate, checks that the kernel the table names is the kernel that launches, and times
the bounded call against the unbounded one, which never takes the adaptive kernel.
"""

import argparse
import contextlib
import itertools
import os
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import topk
from aiter.ops.flydsl.kernels.topk import topk_per_row_decode_adaptive as km
from aiter.ops.flydsl.topk import topk_per_row as host
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


def _raises(exc, fn, match):
    try:
        fn()
    except exc as e:
        assert match in str(e), e
        return
    raise AssertionError(f"expected {exc.__name__} matching {match!r}")


# --- the early stop ---------------------------------------------------------

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


def check_early_stop_is_on_by_default():
    assert km.EARLY_STOP_DEFAULT is True
    assert km.early_stop_default() is True
    assert _cfg()["kw"].get("early_stop") is True
    return 1


def check_early_stop_env_overrides_both_names():
    cases = list(
        itertools.product(
            (km.EARLY_STOP_ENV, "FLYDSL_TOPK_COMPACT_ES"),
            (("0", False), ("1", True), ("", False)),
        )
    )
    for name, (asked, expected) in cases:
        with mock.patch.dict(os.environ, {name: asked}):
            assert km.early_stop_default() is expected, (name, asked)
    return len(cases)


def check_early_stop_explicit_argument_beats_the_environment():
    with mock.patch.dict(os.environ, {km.EARLY_STOP_ENV: "0"}):
        cfg = km.decode_adaptive_config(
            _REACHES["rows"],
            _REACHES["seq"],
            _REACHES["k"],
            ordered=False,
            early_stop=True,
            cu_count=_REACHES["cu_count"],
        )
    assert cfg["kw"].get("early_stop") is True
    return 1


def check_early_stop_is_absent_where_the_build_cannot_reach_it():
    cases = [
        ({"ordered": True}, "the last pass is what produces the order"),
        ({"rows": 128}, "the candidate buffer replaces the row walk"),
        ({"seq": 4096}, "an all-short build never runs that pass"),
    ]
    for over, why in cases:
        assert "early_stop" not in _cfg(**over)["kw"], why
    return len(cases)


def check_the_three_drops_are_the_ones_the_shapes_above_exercise():
    """Guards the check above: each shape must really hit its own reason."""
    assert _cfg(ordered=True)["kw"]["ordered"] is True
    assert _cfg(rows=128)["compact"] is True
    assert _cfg(seq=4096)["kw"]["tier_mode"] == "short"
    return 1


# --- the gate ---------------------------------------------------------------
#
# The decode backend used to be chosen twice, by `topk.py` from a band table and
# again by the FlyDSL host from the arch, which could contradict each other and
# run the adaptive kernel on a shape only the chunked table admitted. These pin
# that there is one decision.

_BAND = ((8_192, 20_000, 2, 128),)


@contextlib.contextmanager
def _one_band():
    """One known band, replacing whatever the shipped table carries, so a check
    reads the rule rather than the numbers. The arch is real because a miss has
    to fall through to the chunked gate, which only exists for a real one."""
    band = {("gfx942", 80): {True: {1024: _BAND}, False: {}}}
    with mock.patch.dict(topk._ADAPTIVE_BANDS, band):
        topk._decode_backend.cache_clear()
        try:
            yield
        finally:
            topk._decode_backend.cache_clear()


def check_the_band_decides_and_a_miss_falls_through():
    cases = [
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
    ]
    with _one_band():
        for width, rows, k, stable, indices_only, expected in cases:
            got = topk._decode_backend(
                "gfx942", 80, stable, width, rows, k, indices_only, width
            )
            assert got == expected, (width, rows, k, stable, indices_only, got)
    return len(cases)


def check_an_unmeasured_cu_count_is_not_guessed_at():
    """The same arch at another CU count keeps the upstream path, because the
    grid the bands were measured against is built from that count.

    The count is taken as one the table does not hold, so that measuring a new
    card is a table edit and does not also reach in here.
    """
    with _one_band():
        measured = {cu for arch, cu in topk._ADAPTIVE_BANDS if arch == "gfx942"}
        unmeasured = next(cu for cu in range(1, 1024) if cu not in measured)
        got = topk._decode_backend(
            "gfx942", unmeasured, True, 8_192, 8, 1024, True, 8_192
        )
        assert got != topk.BACKEND_ADAPTIVE
    return 1


def check_the_shipped_bands_are_well_formed():
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
    return 1


def check_every_k_in_a_group_gets_that_groups_bands():
    """The lookup table is expanded from k groups at import; this is the
    expansion, which a typo in a group tuple would otherwise silently drop."""
    for device, per_emit in topk._ADAPTIVE_BANDS_BY_K_GROUP.items():
        for stable, per_group in per_emit.items():
            for ks, bands in per_group.items():
                for k in ks:
                    assert topk._ADAPTIVE_BANDS[device][stable][k] == bands
    return 1


# --- the CU count -----------------------------------------------------------


@contextlib.contextmanager
def _card_of_304_cu():
    """A 304 CU device, with the cached count cleared on both sides."""

    class _Props:
        multi_processor_count = 304

    with mock.patch.object(torch.cuda, "get_device_properties", lambda _: _Props()):
        topk._decode_cu_count.cache_clear()
        try:
            yield
        finally:
            topk._decode_cu_count.cache_clear()


def check_cu_num_can_lower_the_count_but_never_raise_it():
    cases = [(None, 304), ("0", 304), ("80", 80), ("304", 304), ("512", 304)]
    for cu_num, expected in cases:
        with _card_of_304_cu(), mock.patch.dict(os.environ):
            if cu_num is None:
                os.environ.pop("CU_NUM", None)
            else:
                os.environ["CU_NUM"] = cu_num
            assert topk._decode_cu_count(0) == expected, cu_num
    return len(cases)


def check_the_launcher_sizes_its_grid_from_the_gates_count():
    """A second reader of the CU count could disagree with the gate's, and the
    band it admitted would then describe a grid that was never built."""

    class _Stop(Exception):
        pass

    seen = []

    def config(*args, cu_count, **kwargs):
        seen.append(cu_count)
        raise _Stop("config reached")

    with mock.patch.object(topk, "_decode_cu_count", lambda _: 123), mock.patch.object(
        host._adaptive, "decode_adaptive_config", config
    ):
        _raises(
            _Stop,
            lambda: host._run_adaptive(
                torch.empty(1, 8), 1, None, None, 1, 8, 8, 1, 8, False, None
            ),
            "config reached",
        )
    assert seen == [123]
    return 1


# --- max_row_len ------------------------------------------------------------
#
# A decode buffer is sized to the model's maximum context and a call fills
# whatever part of it the requests need. Without a bound the host has to
# configure for the whole buffer, which the PR description prices.


def check_the_bound_is_the_shorter_of_what_is_held_and_what_is_promised():
    cases = [
        (1 << 20, 8_192, 8_192),  # said less than it holds
        (8_192, 1 << 20, 8_192),  # said more: the buffer still bounds it
        (8_192, 8_192, 8_192),
    ]
    for width, bound, expected in cases:
        assert topk.decode_adaptive_width(width, bound) == expected, (width, bound)
    return len(cases)


def check_a_bound_no_row_could_have_is_refused():
    """Zero is the plausible mistake -- an empty batch, a length not yet filled
    in -- and it would otherwise configure for a row of nothing. `None` is
    refused rather than read as the buffer: the gate declines before it gets
    here, so a `None` arriving is a caller that skipped the gate."""
    cases = [0, -1, None]
    for bad in cases:
        _raises(
            ValueError,
            lambda bad=bad: topk.decode_adaptive_width(1 << 20, bad),
            "max_row_len",
        )
    return len(cases)


def check_the_gate_reads_the_bound_not_the_buffer():
    """The band is 8192..20000 wide. A 1M buffer is outside it and a 1M buffer
    carrying 8192 is inside it, and those are the same tensor."""
    args = ("gfx942", 80, True, 1 << 20, 8, 1024, True)
    with _one_band():
        assert topk._decode_backend(*args, 1 << 20) != topk.BACKEND_ADAPTIVE
        assert topk._decode_backend(*args, 8_192) == topk.BACKEND_ADAPTIVE
    return 1


def check_no_bound_declines_the_adaptive_kernel_and_only_that_one():
    """What `None` costs, and what it must not cost.

    It gives up the adaptive kernel, because the host cannot size that launch
    without a length. It must leave every other decode kernel where it is: the
    chunked bands are read at the physical width with or without a bound, and
    routing that shape to the one-block kernel instead is a measured regression.
    """
    adaptive = ("gfx942", 80, True, 8_192, 8, 1024, True)
    # A shape no adaptive band claims, so the bound cannot be what decides it.
    chunked = ("gfx942", 80, True, 8_192, 1, 1024, True)
    with _one_band():
        assert topk._decode_backend(*adaptive, 8_192) == topk.BACKEND_ADAPTIVE
        assert topk._decode_backend(*adaptive, None) == topk.BACKEND_CHUNKED
        assert topk._decode_backend(*chunked, 8_192) == topk.BACKEND_CHUNKED
        assert topk._decode_backend(*chunked, None) == topk.BACKEND_CHUNKED
    return 1


def check_a_bound_does_not_move_the_chunked_bands():
    """The chunked kernel reads the whole buffer, so its bands were fitted on
    the physical width and a bound must not re-ask them at the live length:
    doing so admitted the chunked kernel where the one-block kernel was faster.
    """
    padded = ("gfx942", 80, True, 1 << 20, 64, 2048, True)
    with _one_band():
        assert topk._decode_backend(*padded, None) == topk.BACKEND_UPSTREAM
        assert topk._decode_backend(*padded, 4_096) == topk.BACKEND_UPSTREAM
    return 1


def check_the_bound_is_what_moves_the_config_off_the_padded_launch():
    """The mechanism behind the measurement, checked where it is decided.

    A buffer wide enough can never take the single-workgroup tier, so a short
    row inside one is launched as if it were long. That is a compile-time
    choice, which is also why the bound has to be a guarantee: the tier is
    picked before any row is read.
    """
    padded = km.decode_adaptive_config(rows=8, seq=1 << 20, k=1024, cu_count=80)
    bounded = km.decode_adaptive_config(rows=8, seq=8_192, k=1024, cu_count=80)

    assert padded["kw"]["tier_mode"] == "auto"
    assert bounded["kw"]["tier_mode"] == "short"
    assert km.needs_workspace_zero(
        1 << 20, 1024, padded["kw"]["tiered_short_max"], tier_mode="auto"
    )
    assert not km.needs_workspace_zero(
        8_192, 1024, bounded["kw"]["tiered_short_max"], tier_mode="short"
    )
    return 1


HOST_CHECKS = [
    check_early_stop_is_on_by_default,
    check_early_stop_env_overrides_both_names,
    check_early_stop_explicit_argument_beats_the_environment,
    check_early_stop_is_absent_where_the_build_cannot_reach_it,
    check_the_three_drops_are_the_ones_the_shapes_above_exercise,
    check_the_band_decides_and_a_miss_falls_through,
    check_an_unmeasured_cu_count_is_not_guessed_at,
    check_the_shipped_bands_are_well_formed,
    check_every_k_in_a_group_gets_that_groups_bands,
    check_cu_num_can_lower_the_count_but_never_raise_it,
    check_the_launcher_sizes_its_grid_from_the_gates_count,
    check_the_bound_is_the_shorter_of_what_is_held_and_what_is_promised,
    check_a_bound_no_row_could_have_is_refused,
    check_the_gate_reads_the_bound_not_the_buffer,
    check_no_bound_declines_the_adaptive_kernel_and_only_that_one,
    check_a_bound_does_not_move_the_chunked_bands,
    check_the_bound_is_what_moves_the_config_off_the_padded_launch,
]


# --- on the card ------------------------------------------------------------
#
# Everything above reads the table. None of it can show that the kernel the
# table names is the kernel that runs, because that is decided at the launch
# site, so the sweep observes the launch by patching it and checks what it wrote.

# Rows the bands are expressed in. Band edges are checked from both sides, so
# the neighbours of min_rows and max_rows have to be on this axis too.
_ROWS_AXIS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

# A cell is allocated in full. The widest band times the tallest row count is
# half a billion float32, which is not what an op test should ask a shared card
# for, so the corner is left to the measurement harness and everything under
# the cap is checked here.
_MAX_ELEMENTS = 64 << 20

# How the logits a call receives sit in memory. `ragged` is a buffer four times
# the bound with each row's own length below it, the first row exactly on it.
_STRIDE0 = {
    "padded": lambda w: w + 3,
    "overlapping": lambda w: w // 2,
    "broadcast": lambda w: 0,
}
_LAYOUTS = ["contiguous", "padded", "overlapping", "broadcast", "ragged"]
_DATA = ["planted", "random", "tied"]


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


def _planted(lens, width, k):
    """Logits whose top k per row sit at known, strided positions inside that
    row's own live length, so an off-by-one at either end shows up as a wrong
    value. Every entry is distinct, so there is nothing for a tie to hide."""
    rows = len(lens)
    x = -torch.arange(width, dtype=torch.float32).repeat(rows, 1)
    for r, n in enumerate(lens):
        stride = max(n // k, 1)
        pos = torch.arange(k) * stride + (r % stride)
        x[r, pos] = 1000.0 + torch.arange(k, dtype=torch.float32)
    return x


def _tied(rows, width):
    """Logits on a coarse grid, so every row ties across its k-th value."""
    x = (torch.randn(rows, width) * 8).round() / 8
    return x + 0.0  # the kernel ranks -0.0 below +0.0; leave only one zero


def _values(data, lens, width, k):
    if data == "planted":
        return _planted(lens, width, k)
    if data == "tied":
        return _tied(len(lens), width)
    return torch.randn(len(lens), width, dtype=torch.float32)


def _inputs(rows, live, k, data, layout):
    """The logits view and `seq_lens` a decode call in this layout receives.

    A strided view is cloned out of its source so that its storage ends exactly
    at its last element, and a read past a row has nowhere legitimate to land.
    """
    lens = [live] * rows
    if layout == "contiguous":
        logits = _values(data, lens, live, k)
    elif layout == "ragged":
        lens = [live] + torch.randint(k, live + 1, (rows - 1,)).tolist()
        logits = _values(data, lens, 4 * live, k)
    else:
        stride0 = _STRIDE0[layout](live)
        n = (rows - 1) * stride0 + live
        src = (
            _values(data, lens, stride0, k)
            if layout == "padded"
            else _values(data, [n], n, k)
        )
        logits = src.reshape(-1)[:n].clone().as_strided((rows, live), (stride0, 1))
    return logits, torch.tensor(lens, dtype=torch.int32)


def _stable_indices(logits, k):
    """The stable contract: ties go to the smaller index, output ascending."""
    kth = torch.topk(logits, k, dim=1).values[:, -1:]
    above = logits > kth
    at = logits == kth
    room = k - above.sum(dim=1, keepdim=True)
    keep = above | (at & (at.cumsum(dim=1, dtype=torch.int32) <= room))
    return keep.nonzero()[:, 1].view(-1, k).to(torch.int32)


def _err(masked, out, k, stable, msg):
    """Mismatch ratio against the reference. The stable contract names one index
    per tie, so a stable call compares indices; otherwise either index of a tied
    pair is correct, so it compares the values selected."""
    if stable:
        want, got = _stable_indices(masked, k), out
    else:
        want = torch.sort(torch.topk(masked, k, dim=1).values, dim=1).values
        got = torch.sort(masked.gather(1, out.long()), dim=1).values
    return checkAllclose(
        want.to(dtypes.fp32), got.to(dtypes.fp32), rtol=0, atol=0, msg=msg
    )


@contextlib.contextmanager
def _launches():
    """Record, in call order, which decode kernel entry points a call reaches.

    `_run_adaptive` and the port both go on to call `_run_compiled`, so the
    first name recorded is the one that was dispatched to.
    """
    seen = []

    def spy(name, attr):
        real = getattr(host, attr)

        def wrapper(*args, **kwargs):
            seen.append(name)
            return real(*args, **kwargs)

        return mock.patch.object(host, attr, wrapper)

    with spy(topk.BACKEND_ADAPTIVE, "_run_adaptive"), spy(
        topk.BACKEND_CHUNKED, "_run_compiled"
    ), spy("port", "flydsl_radix_topk_one_block"):
        yield seen


def _declined_runs(arch):
    """What a declined call launches, which is an arch question: the one-block
    arches run the FlyDSL port, spied by name; elsewhere the HIP kernel, which is
    not on the host module and so is observed by nothing having been spied."""
    if arch in host._FLYDSL_TOPK_ONE_BLOCK_ARCHES:
        return "port"
    return topk.BACKEND_UPSTREAM


@benchmark()
def test_topk_decode(rows, live, k, stable, data, layout):
    arch = get_gfx()
    cu_count = topk._decode_cu_count(torch.cuda.current_device())
    logits, seq_lens = _inputs(rows, live, k, data, layout)
    width = logits.shape[1]
    stride0, stride1 = logits.stride()
    # The reference sees each row's live part only, so a read past it selects a
    # value the reference does not have.
    live_cols = torch.arange(width)[None, :] < seq_lens[:, None]
    masked = torch.where(live_cols, logits, float("-inf"))

    # Top-K does no arithmetic, so TB/s is the only roofline metric.
    nbytes = (int(seq_lens.sum()) + rows * k) * logits.element_size()
    ret = {"gfx": arch, "cu": cu_count}
    routed = True
    # `bounded` is how a model calls it; `unbounded` omits max_row_len, which
    # declines the adaptive kernel and so is what runs without this change.
    for name, bound in {"bounded": live, "unbounded": None}.items():
        out = torch.empty(rows, k, dtype=torch.int32)

        def call(out=out, bound=bound):
            aiter.top_k_per_row_decode(
                logits,
                1,
                seq_lens,
                out,
                rows,
                stride0,
                stride1,
                k=k,
                stable=stable,
                max_row_len=bound,
            )

        adaptive_width = (
            None if bound is None else topk.decode_adaptive_width(width, bound)
        )
        table = topk._decode_backend(
            arch, cu_count, stable, width, rows, k, True, adaptive_width
        )
        gate = topk.decode_backend_for_call(
            logits,
            1,
            seq_lens,
            out,
            rows,
            stride0,
            stride1,
            k,
            stable,
            None,
            max_row_len=bound,
        )
        with _launches() as seen:
            call()
        ran = seen[0] if seen else topk.BACKEND_UPSTREAM
        expect_ran = _declined_runs(arch) if gate == topk.BACKEND_UPSTREAM else gate
        routed &= gate == table and ran == expect_ran

        err = _err(masked, out, k, stable, msg=f"{name}: ")
        _, us = run_perftest(call)
        ret[f"{name} kernel"] = ran
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    ret["routed"] = routed
    return ret


def main():
    # Whole-op arch gate goes here: @benchmark always returns the call-args dict.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("adaptive decode Top-K unsupported on %s", get_gfx())
        return

    arch = get_gfx()
    cu_count = topk._decode_cu_count(torch.cuda.current_device())
    carried = (arch, cu_count) in topk._ADAPTIVE_BANDS_BY_K_GROUP
    edges = sorted(
        {(r, w, k) for w, r, k, _ in _band_edge_cells(arch, cu_count)}
        if carried
        else ()
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=edges,
        help="""rows,live,k per shape. Default: both sides of every band edge
    this card's table carries, none if it carries no bands.
    e.g.: -s 8,65536,2048""",
    )
    parser.add_argument(
        "--stable",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
        help="""emit mode. e.g.: --stable true""",
    )
    parser.add_argument(
        "--data",
        type=str,
        nargs="*",
        default=_DATA,
        choices=_DATA,
        help="""logits values. e.g.: --data tied""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        default=_LAYOUTS,
        choices=_LAYOUTS,
        help="""how the logits sit in memory. e.g.: -l contiguous ragged""",
    )
    args = parser.parse_args()

    host_rows = []
    for check in HOST_CHECKS:
        try:
            host_rows.append({"check": check.__name__, "cases": check(), "ok": True})
        except AssertionError as e:
            aiter.logger.error("%s failed: %r", check.__name__, e)
            host_rows.append({"check": check.__name__, "cases": 0, "ok": False})
    host_df = pd.DataFrame(host_rows)
    aiter.logger.info(
        "adaptive decode Top-K host checks (markdown):\n%s",
        host_df.to_markdown(index=False),
    )

    if not args.shape:
        aiter.logger.warning(
            "%s at %d CU carries no adaptive bands; no band edges to sweep",
            arch,
            cu_count,
        )

    torch.manual_seed(0)
    df = []
    for (rows, live, k), stable, data, layout in itertools.product(
        args.shape, args.stable, args.data, args.layout
    ):
        # A tie only names one index under the stable contract.
        if data == "tied" and not stable:
            continue
        if rows * live * (4 if layout == "ragged" else 1) > _MAX_ELEMENTS:
            continue
        df.append(test_topk_decode(rows, live, k, stable, data, layout))
        torch.cuda.empty_cache()
    df = pd.DataFrame(df)
    if not df.empty:
        aiter.logger.info(
            "adaptive decode Top-K summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    assert host_df["ok"].all(), "host checks failed:\n" + host_df.to_markdown()
    if not df.empty:
        bad = df[(df["bounded err"] != 0) | (df["unbounded err"] != 0) | ~df["routed"]]
        assert bad.empty, f"{len(bad)} of {len(df)} cells bad:\n" + bad.to_markdown()


if __name__ == "__main__":
    main()
