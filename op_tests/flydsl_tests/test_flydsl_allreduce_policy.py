# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structural checks on the FlyDSL all-reduce dispatch tables.

No GPU required.

Run with ``pytest op_tests/flydsl_tests/test_flydsl_allreduce_policy.py``.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from aiter.dist.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    is_weak_contiguous,
)
from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce
from aiter.ops.flydsl import allreduce_policy as P
from aiter.ops.flydsl.allreduce_policy import (
    mesh_st_ladder,
    oneshot_ladder,
    ring_st_ladder,
)
from aiter.ops.flydsl.kernels.comm.allreduce_shared import (
    SUPPORTED_BLOCKS,
    AllReduceAlgorithm,
    FamilyPolicy,
    Window,
)
from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import SUPPORTED_ATOMS
from aiter.ops.flydsl.kernels.comm.quick_allreduce_int4 import SUPER_TILES
from aiter.ops.flydsl.kernels.comm.quick_allreduce_int4_ring import RING_SUPER_TILES
from aiter.ops.flydsl.one_shot_allreduce import resolve_oneshot_max_bytes

WORLDS = P.SUPPORTED_WORLDS
ARCHES = P.SUPPORTED_ARCHS
# Every platform that exists, as ``(link, world_size, arch)``.
CELLS = [
    (link, ws, arch)
    for arch in ARCHES
    for link in P.LINKS
    for ws in WORLDS
    if (arch, link) not in P.NOT_SUPPORTED_PLATFORMS
]
ONESHOT = AllReduceAlgorithm.OneShot
MESH = AllReduceAlgorithm.TwoShotMesh
RING = AllReduceAlgorithm.TwoShotRing


def _row(link, ws, arch):
    return P.family_policy(arch=arch, link=link, world_size=ws)


def test_every_platform_has_one_policy():
    """One row per existing ``(arch, link, world_size)``, and none for a
    platform that does not exist."""
    keys = [(r.arch, r.link, r.world_size) for r in P.FAMILY_POLICIES]
    assert len(keys) == len(set(keys))
    assert sorted(keys) == sorted((a, link, ws) for link, ws, a in CELLS)


def test_no_such_platform_raises():
    for (arch, link), reason in P.NOT_SUPPORTED_PLATFORMS.items():
        with pytest.raises(ValueError, match=reason):
            P.family_policy(arch=arch, link=link, world_size=4)


def _policy(**windows):
    return FamilyPolicy(arch="gfx950", link="pcie", world_size=4, **windows)


def test_window_bounds():
    w = Window(16, 64)
    assert not w.contains(0) and w.contains(16) and w.contains(48)
    assert not w.contains(64)
    assert Window(16, None).contains(1 << 40)
    assert Window(32, 32).is_empty and not Window(32, 32).contains(32)
    with pytest.raises(ValueError):
        Window(64, 16)
    with pytest.raises(ValueError):
        Window(-16, None)


def test_windows_must_not_overlap():
    with pytest.raises(ValueError, match="overlaps"):
        _policy(oneshot=Window(0, 256), mesh=Window(128, None), ring=None)
    with pytest.raises(ValueError, match="overlaps"):
        # An unbounded window must be the last one.
        _policy(oneshot=None, mesh=Window(0, None), ring=Window(1 << 20, None))
    # Gaps and disabled algorithms are fine.
    p = _policy(oneshot=Window(0, 64), mesh=Window(128, None), ring=Window(5, 5))
    assert p.algorithms(quantized=True) == (ONESHOT, MESH)
    assert p.window(RING, quantized=True) is None
    assert p.select(96, quantized=True) is None


def test_exact_oneshot_must_extend_into_the_quantized_windows():
    # The extension lies in the mesh window: fine.
    p = _policy(
        oneshot=Window(0, 128),
        mesh=Window(128, None),
        ring=None,
        exact_oneshot=Window(0, 1024),
    )
    assert p.select(512, quantized=True) == MESH
    assert p.select(512, quantized=False) == ONESHOT
    assert p.select(2048, quantized=False) is None
    assert p.window(MESH, quantized=False) is None
    # The extension reaches a gap that CustomAllreduce would serve with the
    # one-shot while select() says cross_device_reduce: rejected.
    with pytest.raises(ValueError, match="do not cover"):
        _policy(
            oneshot=Window(0, 128),
            mesh=Window(512, None),
            ring=None,
            exact_oneshot=Window(0, 1024),
        )


@pytest.mark.parametrize("cell", CELLS)
def test_quantized_algorithms_cover_the_axis_in_order(cell):
    """With quantization on, the enabled windows read one-shot, mesh, ring in
    payload order."""
    order = _row(*cell).algorithms(quantized=True)
    assert list(order) == [a for a in (ONESHOT, MESH, RING) if a in order]


@pytest.mark.parametrize("cell", CELLS)
def test_dispatch_is_monotone(cell):
    """The algorithm never goes backwards as the payload grows.

    ``fallback`` (``cross_device_reduce``) sits between the one-shot and the
    mesh: where it beats both FlyDSL families, the payload falls through to
    it, and only larger payloads reach the mesh.
    """
    row = _row(*cell)
    order = {ONESHOT: 0, None: 1, MESH: 2, RING: 3}
    for quantized in (False, True):
        first = min(
            (
                w.min
                for a in AllReduceAlgorithm
                if (w := row.window(a, quantized=quantized)) is not None
            ),
            default=0,
        )
        sizes = [1 << k for k in range(4, 33) if 1 << k >= first]
        seen = [order[row.select(n, quantized=quantized)] for n in sizes]
        assert seen == sorted(seen), quantized


@pytest.mark.parametrize("cell", CELLS)
def test_exact_mode_runs_only_the_one_shot(cell):
    row = _row(*cell)
    assert row.algorithms(quantized=False) in ((), (ONESHOT,))


def test_every_platform_has_one_ladder_per_algorithm():
    keys = sorted((a, link, ws) for link, ws, a in CELLS)
    for table in (P.ONESHOT_LADDER, P.MESH_ST_LADDER, P.RING_ST_LADDER):
        assert sorted(table) == keys


@pytest.mark.parametrize("cell", CELLS)
def test_ladders_are_well_formed(cell):
    """Every ladder starts at 0, ascends, and names values its kernel accepts.

    A ladder that does not start at 0 leaves the smallest payloads with no
    rung; one that is not ascending makes ``select_rung``, which takes the
    *last* rung at or below the payload, select something arbitrary.
    """
    link, ws, arch = cell
    kw = {"world_size": ws, "link": link, "arch": arch}
    for name, rungs, valid_st in (
        ("mesh", mesh_st_ladder(**kw), SUPER_TILES),
        ("ring", ring_st_ladder(**kw), RING_SUPER_TILES),
    ):
        assert rungs, name
        assert rungs[0].min_bytes == 0, name
        floors = [r.min_bytes for r in rungs]
        assert floors == sorted(floors), name
        for r in rungs:
            assert r.super_tile in valid_st, (name, r)
            assert r.grid_cap >= 1, (name, r)
            assert r.block in SUPPORTED_BLOCKS, (name, r)

    one = oneshot_ladder(**kw)
    assert one and one[0].min_bytes == 0
    floors = [r.min_bytes for r in one]
    assert floors == sorted(floors)
    for r in one:
        assert r.atoms in SUPPORTED_ATOMS, r
        assert r.grid_cap >= 1, r
        assert r.fanout in ("peer", "atom"), r
        assert r.block in SUPPORTED_BLOCKS, r


def test_ladder_lookup_rejects_unknown_keys():
    """No fallback rungs: an unknown platform raises."""
    for ladder in (oneshot_ladder, mesh_st_ladder, ring_st_ladder):
        with pytest.raises(P.NoPolicyError):
            ladder(world_size=4, link="pcie", arch="gfx942")
        with pytest.raises(ValueError):
            ladder(world_size=4, link="nosuchlink", arch="gfx950")
        with pytest.raises(P.NoPolicyError):
            ladder(world_size=4, link="xgmi", arch="gfx1250")


@pytest.mark.parametrize("cell", CELLS)
def test_oneshot_max_bytes_is_keyed_on_the_fabric(cell):
    """The default ceiling is where the fallback overtakes the one-shot, which
    is a property of the arch and the fabric.
    """
    link, ws, arch = cell
    one = _row(*cell).window(ONESHOT, quantized=False)
    got = resolve_oneshot_max_bytes(world_size=ws, link=link, arch=arch)
    assert got == one.max - 1


def test_oneshot_max_bytes_follows_the_overrides():
    """The standalone default ceiling is the one CustomAllreduce serves."""
    kw = {"world_size": 4, "link": "pcie", "arch": "gfx950"}
    with _env(
        AITER_FLY_AR_ONESHOT_MAX_BYTES="65536", AITER_FLY_AR_MESH_MIN_BYTES="65536"
    ):
        assert resolve_oneshot_max_bytes(**kw) == 65536 - 1
    # A maximum of 0 disables the one-shot.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="0"):
        assert resolve_oneshot_max_bytes(**kw) == 0
    # Invalid overrides raise, as they do in the dispatchers.
    with (
        _env(AITER_FLY_AR_ONESHOT_MAX_BYTES=str(1 << 20)),
        pytest.raises(ValueError, match="overlaps"),
    ):
        resolve_oneshot_max_bytes(**kw)


@pytest.mark.parametrize("cell", CELLS)
def test_ladder_rungs_fall_inside_their_dispatch_window(cell):
    """A rung above its algorithm's window is an engine and an IPC inbox built
    for payloads that can never arrive.

    This is the bug the windowed refit removed: fitted over the whole sweep,
    the TP8 one-shot ladder wanted a second rung at 192 KiB, four times above
    anything that schedule is dispatched at.
    """
    link, ws, _arch = cell
    row = _row(*cell)
    one = row.window(ONESHOT, quantized=False)
    kw = {"world_size": ws, "link": link, "arch": _arch}
    for r in oneshot_ladder(**kw)[1:]:
        assert one.contains(r.min_bytes), ("oneshot", cell, r)
    mesh = row.window(MESH, quantized=True)
    if mesh is not None and mesh.max is not None:
        for r in mesh_st_ladder(**kw)[1:]:
            assert r.min_bytes < mesh.max, ("mesh", cell, r)
    # Ring rungs are offsets into an unbounded window, so only the ordering
    # above constrains them.


def _env(**kw):
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items()}, clear=False)


def _resolve(link="pcie", ws=4, arch="gfx950"):
    return P.resolve_family_policy(world_size=ws, arch=arch, link=link)


@pytest.mark.parametrize("ws", WORLDS)
def test_gfx950_xgmi_never_rings(ws):
    """On gfx950/xGMI the ring is disabled, and the payloads between the
    one-shot and the mesh windows go to ``cross_device_reduce``."""
    row = _row("xgmi", ws, "gfx950")
    assert row.window(RING, quantized=True) is None
    one = row.window(ONESHOT, quantized=True)
    mesh = row.window(MESH, quantized=True)
    for k in range(4, 33):
        got = row.select(1 << k, quantized=True)
        assert got != RING
        if one.max <= 1 << k < mesh.min:
            assert got is None


def test_enable_flag_is_opt_in_only():
    with _env(AITER_FLY_AR="1"):
        assert P.enabled() is True
    with _env(AITER_FLY_AR="0"):
        assert P.enabled() is False
    with _env(AITER_FLY_AR=""):
        assert P.enabled() is False
    with _env(AITER_FLY_AR="true"):
        assert P.enabled() is False


def test_no_overrides_is_the_table():
    assert _resolve() == _row("pcie", 4, "gfx950")


def test_window_overrides():
    table = _row("pcie", 4, "gfx950")
    # Move the one-shot/mesh boundary: it takes both variables.
    with _env(
        AITER_FLY_AR_ONESHOT_MAX_BYTES="65536", AITER_FLY_AR_MESH_MIN_BYTES="65536"
    ):
        p = _resolve()
        assert p.window(ONESHOT, quantized=True) == Window(0, 65536)
        # The ONESHOT variables set the exact-mode window too.
        assert p.window(ONESHOT, quantized=False) == Window(0, 65536)
        assert p.window(MESH, quantized=True) == Window(65536, table.mesh.max)
    # No upper limit, and a maximum of 0 disables the ring.
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="inf", AITER_FLY_AR_RING_MAX_BYTES="0"):
        p = _resolve()
        assert p.window(MESH, quantized=True).max is None
        assert p.window(RING, quantized=True) is None
    # gfx950/PCIe TP8 has a ring: an unbounded mesh would overlap it.
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="None", AITER_FLY_AR_RING_MAX_BYTES="0"):
        p = _resolve(ws=8)
        assert p.window(MESH, quantized=True).max is None


def test_override_enables_a_disabled_algorithm():
    """gfx950/xGMI has no ring. Overriding its window creates one, with the
    other bound unbounded."""
    with _env(
        AITER_FLY_AR_MESH_MAX_BYTES=str(512 << 20),
        AITER_FLY_AR_RING_MIN_BYTES=str(512 << 20),
    ):
        p = _resolve("xgmi", 4)
        assert p.window(RING, quantized=True) == Window(512 << 20, None)
        assert p.select(1 << 30, quantized=True) == RING


def test_equal_bounds_disable():
    with _env(
        AITER_FLY_AR_RING_MIN_BYTES=str(1 << 30),
        AITER_FLY_AR_RING_MAX_BYTES=str(1 << 30),
    ):
        assert _resolve(ws=8).window(RING, quantized=True) is None


def test_ignored_overrides():
    table = _resolve()
    for value in ("-1", "", "lots", "-7"):
        with _env(AITER_FLY_AR_MESH_MIN_BYTES=value):
            assert _resolve() == table, value
    # A minimum cannot be unbounded.
    with _env(AITER_FLY_AR_MESH_MIN_BYTES="inf"):
        assert _resolve() == table


def test_overlapping_overrides_raise():
    # Raising the one-shot ceiling without moving the mesh floor.
    with (
        _env(AITER_FLY_AR_ONESHOT_MAX_BYTES=str(1 << 20)),
        pytest.raises(ValueError, match="overlaps"),
    ):
        _resolve()
    # A mesh floor above the mesh ceiling.
    with _env(AITER_FLY_AR_MESH_MIN_BYTES=str(1 << 30)), pytest.raises(ValueError):
        _resolve()


def test_resolve_rejects_unknown_keys():
    with pytest.raises(ValueError):
        _resolve(link="infiniband")
    with pytest.raises(ValueError):
        _resolve(ws=3)
    with pytest.raises(P.NoPolicyError):
        _resolve(link="pcie", arch="gfx942")


def _full_storage_transpose(nbytes: int) -> torch.Tensor:
    """A bf16 view that is weakly but not strictly contiguous.

    The transpose of a whole contiguous allocation covers its storage exactly,
    so ``is_weak_contiguous`` accepts it while ``Tensor.is_contiguous()`` does
    not -- the gap the FlyDSL selectors have to close.
    """
    cols = 64
    rows = nbytes // (cols * 2)
    t = torch.zeros(rows, cols, dtype=torch.bfloat16).t()
    assert not t.is_contiguous() and is_weak_contiguous(t)
    return t


def test_selectors_reject_non_contiguous():
    """Both FlyDSL selectors decline a weakly-contiguous view.

    ``CustomAllreduce`` admits weakly-contiguous tensors, but both FlyDSL
    engines require strict contiguity and raise otherwise. Once dispatch has
    picked FlyDSL the exception replaces the fallback, so the selectors must
    decline these tensors themselves. Each selector is run against a stand-in
    ``self`` (the fields it reads), so no process group or GPU is needed. The
    contiguous copy of the same payload is accepted, which shows the refusal
    comes from contiguity rather than from size, dtype or alignment.
    """
    policy = _row("pcie", 4, "gfx950")
    qr = SimpleNamespace(
        _fly_policy=policy, _fly_engines={MESH: object(), RING: object()}
    )
    t = _full_storage_transpose(policy.window(MESH, quantized=True).min + (64 << 10))
    assert QuickAllReduce._should_fly(qr, t.contiguous())
    assert not QuickAllReduce._should_fly(qr, t)

    one = policy.window(ONESHOT, quantized=False)
    ca = SimpleNamespace(
        _fly_oneshot=object(), _fly_window=one, device=torch.device("cpu")
    )
    t = _full_storage_transpose(max(one.min, 1 << 12))
    assert CustomAllreduce._should_fly_oneshot(ca, t.contiguous(), True, False)
    assert not CustomAllreduce._should_fly_oneshot(ca, t, True, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
