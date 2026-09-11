# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structural checks on the FlyDSL all-reduce dispatch tables.

No GPU required.

Run with ``pytest op_tests/flydsl_tests/test_flydsl_allreduce_policy.py``.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from aiter.ops.flydsl import allreduce_policy as P
from aiter.ops.flydsl.kernels.one_shot_allreduce import (
    SUPPORTED_ATOMS,
    oneshot_ladder,
)
from aiter.ops.flydsl.kernels.quick_allreduce_int4 import MESH_ST_LADDER, SUPER_TILES
from aiter.ops.flydsl.kernels.quick_allreduce_int4_ring import (
    RING_SUPER_TILES,
    ring_st_ladder,
)

WORLDS = P.SUPPORTED_WORLDS
CELLS = [(link, ws) for link in P.LINKS for ws in WORLDS]


@pytest.mark.parametrize("cell", CELLS)
def test_every_cell_present(cell):
    """A missing (link, world) is a KeyError on the critical path, not a
    fallback -- the communicator resolves the policy before it builds anything."""
    assert cell in P.FAMILY_POLICY


@pytest.mark.parametrize("cell", CELLS)
def test_thresholds_partition_by_size(cell):
    """The three families must tile the size axis in order, with no gap and no
    overlap. ``FamilyPolicy.__post_init__`` enforces it; this pins that the
    shipped values actually satisfy it rather than that the check exists."""
    p = P.FAMILY_POLICY[cell]
    assert 0 < p.oneshot_max <= p.oneshot_max_exact <= p.mesh_max <= p.max_bytes
    assert p.min_bytes <= p.oneshot_max


def test_exact_window_is_never_narrower():
    """Preferring the bit-exact schedule can only widen its window. A row where
    the exact boundary sat *below* the speed one would mean the default policy
    is both slower and less accurate, which is not a trade anyone chose."""
    for cell in CELLS:
        p = P.FAMILY_POLICY[cell]
        assert p.oneshot_max_exact >= p.oneshot_max, cell


def test_oneshot_ceiling_shrinks_with_world_size():
    """Wire volume is ``(N-1)*S`` against a two-shot's ``2(N-1)/N*S``, a ratio
    of ``N/2``. The ceiling must therefore fall as the world grows -- this is
    the property the single 192 KiB constant could not express, and the reason
    the table is keyed on world size at all."""
    for link in P.LINKS:
        ceilings = [P.FAMILY_POLICY[(link, ws)].oneshot_max for ws in sorted(WORLDS)]
        if link == "xgmi":
            # Placeholder rows are flat by construction; assert that, so this
            # test starts failing the moment they are replaced by a real fit
            # that does not obey the trend.
            assert len(set(ceilings)) == 1
        else:
            assert ceilings == sorted(ceilings, reverse=True), ceilings


def test_xgmi_never_selects_the_ring():
    """The xGMI rows are an unmeasured placeholder. Conservative there means the
    mesh -- the documented default on a meshed fabric -- and never the ring,
    which trades fanout for the per-destination locality a PCIe host wants."""
    for ws in WORLDS:
        p = P.resolve("xgmi", ws, mode="fast")
        assert "ring" not in P.families_reachable(p)
        assert P.pick_family(1 << 30, p) == "mesh"


@pytest.mark.parametrize("ws", WORLDS)
def test_pick_family_is_monotone(ws):
    """Family choice must never go backwards as the payload grows.

    Three families only ever coexist in ``"fast"`` mode -- ``"exact"`` is
    single-family by design (see ``test_exact_mode_is_oneshot_only``), so that
    is the mode this checks the three-way monotonicity in.
    """
    p = P.resolve("pcie", ws, mode="fast")
    order = {"oneshot": 0, "mesh": 1, "ring": 2}
    seen = [order[P.pick_family(n, p)] for n in (1 << k for k in range(4, 31))]
    assert seen == sorted(seen)
    # And all three are actually reachable on PCIe, or a family is dead code.
    assert set(P.families_reachable(p)) == {"oneshot", "mesh", "ring"}


@pytest.mark.parametrize("ws", WORLDS)
def test_ladders_are_well_formed(ws):
    """Every ladder starts at 0, ascends, and names values its kernel accepts.

    A ladder that does not start at 0 leaves the smallest payloads with no rung;
    one that is not ascending makes ``_pick_st``/``_pick_cfg`` -- which take the
    *last* rung at or below the payload -- select something arbitrary.
    """
    for name, rungs, valid_st in (
        ("mesh", MESH_ST_LADDER[ws], SUPER_TILES),
        ("ring", ring_st_ladder(ws), RING_SUPER_TILES),
    ):
        assert rungs, name
        assert rungs[0][0] == 0, name
        assert [r[0] for r in rungs] == sorted(r[0] for r in rungs), name
        for _floor, st, cap in rungs:
            assert st in valid_st, (name, st)
            assert cap >= 1, (name, cap)

    one = oneshot_ladder(ws)
    assert one and one[0][0] == 0
    assert [r[0] for r in one] == sorted(r[0] for r in one)
    for _floor, atoms, cap, fanout in one:
        assert atoms in SUPPORTED_ATOMS
        assert cap >= 1
        assert fanout in ("peer", "atom")


@pytest.mark.parametrize("ws", WORLDS)
def test_ladder_rungs_fall_inside_their_dispatch_window(ws):
    """A rung above its family's window is an engine and an IPC inbox built for
    payloads that can never arrive.

    This is exactly the bug the windowed refit removed: fitted over the whole
    sweep, the TP8 one-shot ladder wanted a second rung at 192 KiB, four times
    above anything that schedule is dispatched at.
    """
    # "fast" mode: exact mode has no mesh/ring window at all (mesh_max ==
    # oneshot_max there by design), which would make every mesh rung above the
    # smallest look like it fails this check for the wrong reason.
    p = P.resolve("pcie", ws, mode="fast")
    for _floor, *_ in oneshot_ladder(ws)[1:]:
        assert _floor < p.oneshot_max, ("oneshot", ws, _floor)
    for floor, *_ in MESH_ST_LADDER[ws][1:]:
        assert floor < p.mesh_max, ("mesh", ws, floor)
    # Ring rungs are offsets into an unbounded window, so only the ordering
    # above constrains them.


def _env(**kw):
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items()}, clear=False)


def test_accuracy_mode_env():
    with _env(AITER_FLY_AR_ACCURACY="fast"):
        assert P.accuracy_mode() == "fast"
    with _env(AITER_FLY_AR_ACCURACY="EXACT"):
        assert P.accuracy_mode() == "exact"
    # An unrecognised value warns and falls back rather than raising: a typo in
    # an env var must not take a model down.
    with _env(AITER_FLY_AR_ACCURACY="nonsense"):
        assert P.accuracy_mode() == P.DEFAULT_ACCURACY


def test_exact_mode_is_oneshot_only():
    """``"exact"`` is not just a wider one-shot boundary -- it is a different
    policy shape. Above ``oneshot_max_exact`` there is no mesh/ring window at
    all: ``mesh_max`` and ``max_bytes`` collapse onto ``oneshot_max``, so
    ``should_fly_all_reduce`` declines any larger payload instead of routing
    it to a quantized schedule. A caller who never touches
    ``AITER_FLY_AR_ACCURACY`` gets bit-exact FlyDSL or no FlyDSL, never
    quantized FlyDSL.
    """
    for link in P.LINKS:
        for ws in WORLDS:
            exact = P.resolve(link, ws, mode="exact")
            assert exact.mesh_max == exact.oneshot_max
            assert exact.max_bytes == exact.oneshot_max
            assert P.families_reachable(exact) == ("oneshot",)
            # oneshot_max itself is still the *wider* (oneshot_max_exact)
            # boundary, same as before this policy shape existed.
            fast = P.resolve(link, ws, mode="fast")
            assert exact.oneshot_max >= fast.oneshot_max


def test_fast_mode_still_prefers_exactness_where_free():
    """``"fast"`` keeps the original two-boundary shape: one-shot up to
    ``oneshot_max``, mesh/ring beyond it -- unaffected by ``"exact"`` existing
    as a separate, stricter policy."""
    for ws in WORLDS:
        p = P.resolve("pcie", ws, mode="fast")
        assert p.mesh_max > p.oneshot_max
        assert p.max_bytes > p.mesh_max


def test_enable_flag_is_opt_in_only():
    with _env(AITER_FLY_AR="1"):
        assert P.enabled() is True
    with _env(AITER_FLY_AR="0"):
        assert P.enabled() is False
    with _env(AITER_FLY_AR=""):
        assert P.enabled() is False
    with _env(AITER_FLY_AR="true"):
        assert P.enabled() is False


def test_byte_overrides():
    # ONESHOT_MAX_VAR applies in both modes -- exact mode's default here is
    # oneshot_max_exact (48 KiB for TP8/pcie), so leaving mode unset still
    # exercises the override against a real baseline.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert P.resolve("pcie", 8).oneshot_max == 65536
    # MESH_MAX_VAR only has an effect in "fast" mode -- see
    # test_exact_mode_ignores_mesh_max_override for the exact-mode case.
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="1048576"):
        assert P.resolve("pcie", 4, mode="fast").mesh_max == 1048576
    # -1 is the house sentinel for "unset, use the table".
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="-1"):
        assert P.resolve("pcie", 8).oneshot_max == (48 << 10)
    # Garbage warns and is ignored.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="lots"):
        assert P.resolve("pcie", 8).oneshot_max == (48 << 10)


def test_exact_mode_ignores_mesh_max_override():
    """Honouring ``AITER_FLY_AR_MESH_MAX_BYTES`` in exact mode would reopen the
    mesh/ring window ``"exact"`` exists to close, so it is ignored there (with
    a warning) rather than applied."""
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="1048576"):
        p = P.resolve("pcie", 4, mode="exact")
        assert p.mesh_max == p.oneshot_max
        assert p.mesh_max != 1048576


def test_override_cannot_invert_the_partition():
    """Pushing the one-shot ceiling above the mesh floor means "give me the
    one-shot up to here", not "crash" -- the mesh window closes instead.

    "fast" mode: exact mode already has no independent mesh window to invert
    (mesh_max tracks oneshot_max unconditionally there), so this property is
    only meaningful where the two boundaries are otherwise independent.
    """
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES=str(64 << 20)):
        p = P.resolve("pcie", 4, mode="fast")
        assert p.mesh_max >= p.oneshot_max
        assert P.pick_family(1 << 20, p) == "oneshot"


def test_resolve_rejects_unknown_keys():
    with pytest.raises(ValueError):
        P.resolve("infiniband", 4)
    with pytest.raises(ValueError):
        P.resolve("pcie", 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
