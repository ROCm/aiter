# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Structural checks on the FlyDSL all-reduce dispatch tables.

No GPU required.

Run with ``pytest op_tests/flydsl_tests/test_flydsl_allreduce_policy.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from aiter.dist.device_communicators.custom_all_reduce import (
    CustomAllreduce,
    is_weak_contiguous,
)
from aiter.dist.device_communicators.flydsl_backends import (
    FlyDSLAllReduceRMSNorm,
)
from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce
from aiter.ops.flydsl import allreduce_policy as P
from aiter.ops.flydsl.allreduce_policy import (
    RING_SUPER_TILES,
    SUPER_TILES,
    SUPPORTED_ATOMS,
    SUPPORTED_BLOCKS,
    SUPPORTED_SPLITS,
    AllReduceAlgorithm,
    FamilyPolicy,
    Window,
    fused_mesh_st_ladder,
    fused_oneshot_ladder,
    fused_ring_st_ladder,
    mesh_st_ladder,
    oneshot_ladder,
    ring_st_ladder,
)
from aiter.ops.flydsl.one_shot_allreduce import (
    resolve_fused_oneshot_max_bytes,
    resolve_oneshot_max_bytes,
)

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


def test_mesh_and_ring_windows_must_not_overlap():
    with pytest.raises(ValueError, match="overlaps"):
        _policy(oneshot=None, mesh=Window(0, 256), ring=Window(128, None))
    with pytest.raises(ValueError, match="overlaps"):
        # An unbounded window must be the last one.
        _policy(oneshot=None, mesh=Window(0, None), ring=Window(1 << 20, None))
    # Gaps and disabled algorithms are fine.
    p = _policy(oneshot=Window(0, 64), mesh=Window(128, None), ring=Window(5, 5))
    assert p.algorithms(quantized=True) == (ONESHOT, MESH)
    assert p.window(RING, quantized=True) is None
    assert p.select(96, quantized=True) is None


def test_unbounded_one_shot_window():
    """A one-shot that wins at every size, in both modes, is a valid policy."""
    row = _policy(oneshot=Window(0, None), mesh=None, ring=None)
    assert row.select(1 << 30, quantized=True) == ONESHOT
    assert row.select(1 << 30, quantized=False) == ONESHOT


def test_mesh_and_ring_take_precedence_over_the_one_shot():
    """The one-shot window may overlap the quantized windows. With quantization
    on, ``QuickAllReduce`` serves the mesh and the ring first, and ``select``
    follows it. With quantization off, the one-shot serves its whole window."""
    p = _policy(oneshot=Window(0, 1024), mesh=Window(128, 512), ring=Window(512, None))
    assert p.select(64, quantized=True) == ONESHOT
    assert p.select(256, quantized=True) == MESH
    assert p.select(768, quantized=True) == RING
    assert p.select(256, quantized=False) == ONESHOT
    assert p.select(768, quantized=False) == ONESHOT
    assert p.select(2048, quantized=False) is None
    assert p.window(MESH, quantized=False) is None
    assert p.window(ONESHOT, quantized=True) == p.window(ONESHOT, quantized=False)


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
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert resolve_oneshot_max_bytes(**kw) == 65536 - 1
    # A maximum of 0 disables the one-shot.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="0"):
        assert resolve_oneshot_max_bytes(**kw) == 0
    # The mesh variables do not move the one-shot window.
    with _env(AITER_FLY_AR_MESH_MIN_BYTES="65536"):
        assert (
            resolve_oneshot_max_bytes(**kw) == _row("pcie", 4, "gfx950").oneshot.max - 1
        )
    # Invalid overrides raise, as they do in the dispatchers.
    with (
        _env(AITER_FLY_AR_MESH_MAX_BYTES="inf"),
        pytest.raises(ValueError, match="overlaps"),
    ):
        resolve_oneshot_max_bytes(**kw)


@pytest.mark.parametrize("cell", CELLS)
def test_ladder_rungs_fall_inside_their_dispatch_window(cell):
    """A rung above its algorithm's window is an engine and an IPC inbox built
    for payloads that can never arrive."""
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
    # The ONESHOT variables move the window CustomAllreduce serves, in both
    # modes, and leave the mesh and the ring alone.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        p = _resolve()
        for quantized in (False, True):
            assert p.window(ONESHOT, quantized=quantized) == Window(0, 65536)
        assert p.window(MESH, quantized=True) == table.mesh
        assert p.window(RING, quantized=True) == table.ring
        # Between the one-shot and the mesh, cross_device_reduce.
        assert p.select(96 << 10, quantized=True) is None
    # A maximum of 0 disables only the one-shot.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="0"):
        p = _resolve()
        assert p.window(ONESHOT, quantized=False) is None
        assert p.window(MESH, quantized=True) == table.mesh
    # Lowering the mesh floor alone moves the boundary with quantization on.
    with _env(AITER_FLY_AR_MESH_MIN_BYTES="65536"):
        p = _resolve()
        assert p.select(96 << 10, quantized=True) == MESH
        assert p.select(96 << 10, quantized=False) == ONESHOT
    # A minimum alone keeps the table maximum.
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="1024"):
        assert _resolve().oneshot == Window(1024, table.oneshot.max)
    # No upper limit, and a maximum of 0 disables the ring.
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="inf", AITER_FLY_AR_RING_MAX_BYTES="0"):
        p = _resolve()
        assert p.window(MESH, quantized=True).max is None
        assert p.window(RING, quantized=True) is None
    # gfx950/PCIe TP8 has a ring: an unbounded mesh would overlap it.
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="None", AITER_FLY_AR_RING_MAX_BYTES="0"):
        p = _resolve(ws=8)
        assert p.window(MESH, quantized=True).max is None


def test_link_variable_backs_up_an_unreadable_probe():
    # The KFD probe comes first; the variable is only the backup. When the probe
    # is unreadable the variable decides, and is stripped, lowercased and
    # validated against LINKS.
    probe = "aiter.ops.flydsl.allreduce_host_shared.probe_xgmi_peer_links"
    P._probe_link.cache_clear()
    with mock.patch(probe, return_value=None):
        with _env(AITER_FLY_AR_LINK=" PCIe "):
            assert P.detect_link() == "pcie"
        with _env(AITER_FLY_AR_LINK="xgmi"):
            assert P.detect_link() == "xgmi"
            assert P.resolve_link(None) == "xgmi"
        with _env(AITER_FLY_AR_LINK="nvlink"), pytest.raises(ValueError, match="LINK"):
            P.detect_link()
    P._probe_link.cache_clear()


def test_unreadable_topology_needs_the_link_variable():
    env = {k: v for k, v in os.environ.items() if k != "AITER_FLY_AR_LINK"}
    probe = "aiter.ops.flydsl.allreduce_host_shared.probe_xgmi_peer_links"
    for found, expected in ((True, "xgmi"), (False, "pcie")):
        P._probe_link.cache_clear()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch(probe, return_value=found),
        ):
            assert P.detect_link() == expected
    P._probe_link.cache_clear()
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch(probe, return_value=None),
    ):
        with pytest.raises(ValueError, match="AITER_FLY_AR_LINK"):
            P.detect_link()
        # The variable replaces the probe, so an unreadable topology is fine.
        with _env(AITER_FLY_AR_LINK="xgmi"):
            assert P.detect_link() == "xgmi"
    P._probe_link.cache_clear()


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
    # Raising the mesh ceiling without moving the ring floor.
    with (
        _env(AITER_FLY_AR_MESH_MAX_BYTES=str(64 << 20)),
        pytest.raises(ValueError, match="overlaps"),
    ):
        _resolve()
    # Raising the one-shot ceiling into the mesh window is fine.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES=str(64 << 20)):
        assert _resolve().select(32 << 20, quantized=True) == RING
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


@pytest.mark.parametrize("ws", WORLDS)
def test_default_codecs(ws):
    """The ring's reduce-scatter lap is INT6 from ``RS_INT6_MIN_WORLD``; the
    mesh, which has no INT6 path, stays INT4 on both laps."""
    rs = "int6" if ws >= P.RS_INT6_MIN_WORLD else "int4"
    assert P.default_codecs("ring", ws) == (rs, "int4")
    assert P.default_codecs("mesh", ws) == ("int4", "int4")
    for algorithm, (rs_codecs, ag_codecs) in (
        ("mesh", (P.MESH_CODECS, P.MESH_CODECS)),
        ("ring", (P.RS_CODECS, P.AG_CODECS)),
    ):
        got_rs, got_ag = P.default_codecs(algorithm, ws)
        assert got_rs in rs_codecs and got_ag in ag_codecs


def test_default_inbox_memory():
    """Uncached on xGMI and at TP2, fine-grained on PCIe above TP2."""
    for ws in WORLDS:
        assert P.default_inbox_memory(ws, "xgmi") == "uncached"
        assert P.default_inbox_memory(ws, "pcie") == (
            "uncached" if ws == 2 else "finegrained"
        )
    assert set(P.INBOX_MEMORY_MODES) - {"auto"} == {"uncached", "finegrained"}


@pytest.mark.parametrize("cell", CELLS)
@pytest.mark.parametrize("fused", (False, True))
def test_quantized_floor_is_where_quantization_starts(cell, fused):
    """A mesh or ring host refuses what the policy would not quantize, and
    accepts everything it would."""
    link, ws, arch = cell
    row = (P.fused_family_policy if fused else P.family_policy)(
        arch=arch, link=link, world_size=ws
    )
    floor = P.quantized_floor(world_size=ws, link=link, arch=arch, fused=fused)
    quantized = [
        w for alg in (MESH, RING) if (w := row.window(alg, quantized=True)) is not None
    ]
    if not quantized:
        assert floor == 0
        return
    assert row.select(floor, quantized=True) in (MESH, RING)
    if floor > 0:
        assert row.select(floor - 1, quantized=True) not in (MESH, RING)


def test_quantized_floor_follows_the_overrides():
    kw = {"world_size": 4, "link": "pcie", "arch": "gfx950"}
    fused = P.quantized_floor(fused=True, **kw)
    with _env(AITER_FLY_AR_MESH_MIN_BYTES=str(256 << 10)):
        assert P.quantized_floor(**kw) == 256 << 10
        # The plain variable leaves the fused floor alone.
        assert P.quantized_floor(fused=True, **kw) == fused
    with _env(AITER_FLY_AR_FUSED_MESH_MIN_BYTES=str(512 << 10)):
        assert P.quantized_floor(fused=True, **kw) == 512 << 10
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="0", AITER_FLY_AR_RING_MAX_BYTES="0"):
        assert P.quantized_floor(**kw) == 0


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


# Fused all-reduce + RMSNorm
# ==========================
#
# The fused kernels have tables of their own, read by the same ``FamilyPolicy``
# and ``select_rung`` machinery. ``AITER_QUICK_REDUCE_QUANTIZATION=INT4`` opens
# the quantized windows, and the overrides carry a ``FUSED`` infix.


def _fused_row(link, ws, arch):
    return P.fused_family_policy(arch=arch, link=link, world_size=ws)


def _resolve_fused(link="pcie", ws=4, arch="gfx950"):
    return P.resolve_fused_policy(world_size=ws, arch=arch, link=link)


def test_every_platform_has_one_fused_policy():
    keys = sorted((r.arch, r.link, r.world_size) for r in P.FUSED_FAMILY_POLICIES)
    assert keys == sorted((a, link, ws) for link, ws, a in CELLS)


def test_every_platform_has_one_fused_ladder_per_algorithm():
    keys = sorted((a, link, ws) for link, ws, a in CELLS)
    for table in (
        P.FUSED_ONESHOT_LADDER,
        P.FUSED_MESH_ST_LADDER,
        P.FUSED_RING_ST_LADDER,
    ):
        assert sorted(table) == keys


@pytest.mark.parametrize("cell", CELLS)
def test_fused_dispatch_is_monotone(cell):
    row = _fused_row(*cell)
    order = {ONESHOT: 0, None: 1, MESH: 2, RING: 3}
    for quantized in (False, True):
        sizes = [1 << k for k in range(4, 33)]
        seen = [order[row.select(n, quantized=quantized)] for n in sizes]
        # Past the last window everything falls back, so drop the tail.
        while seen and seen[-1] == order[None]:
            seen.pop()
        assert seen == sorted(seen), quantized


@pytest.mark.parametrize("cell", CELLS)
def test_fused_unquantized_runs_only_the_one_shot(cell):
    assert _fused_row(*cell).algorithms(quantized=False) == (ONESHOT,)


@pytest.mark.parametrize("cell", CELLS)
def test_fused_quantized_reaches_past_the_one_shot(cell):
    """Quantization pays above the one-shot on every platform, so there is
    always a quantized window, and it is unbounded above."""
    row = _fused_row(*cell)
    algs = row.algorithms(quantized=True)
    assert algs[0] == ONESHOT and len(algs) >= 2, algs
    assert row.window(algs[-1], quantized=True).max is None


@pytest.mark.parametrize("cell", CELLS)
def test_fused_ladders_are_well_formed(cell):
    """The fused ladders obey the rules of the plain ones, with no ``block``:
    a fused build sizes its workgroup to the token row."""
    link, ws, arch = cell
    kw = {"world_size": ws, "link": link, "arch": arch}
    for name, rungs, valid_st in (
        ("mesh", fused_mesh_st_ladder(**kw), SUPER_TILES),
        ("ring", fused_ring_st_ladder(**kw), RING_SUPER_TILES),
    ):
        assert rungs and rungs[0].min_bytes == 0, name
        floors = [r.min_bytes for r in rungs]
        assert floors == sorted(floors), name
        for r in rungs:
            assert r.super_tile in valid_st, (name, r)
            assert r.grid_cap >= 1, (name, r)

    one = fused_oneshot_ladder(**kw)
    assert one and one[0].min_bytes == 0
    floors = [r.min_bytes for r in one]
    assert floors == sorted(floors)
    for r in one:
        assert r.atoms in SUPPORTED_ATOMS, r
        assert r.grid_cap >= 1, r
        assert r.fanout in ("peer", "atom"), r
        assert r.split in SUPPORTED_SPLITS, r


def test_fused_row_atoms_divide_the_chunk():
    """Every ``FUSED_QR_ROW_ATOMS`` entry divides a rank's reduce-scatter
    chunk, ``8 // world_size`` atoms, and names a platform that exists."""
    platforms = {(a, link, ws) for link, ws, a in CELLS}
    for (arch, link, ws, algorithm), atoms in P.FUSED_QR_ROW_ATOMS.items():
        assert (arch, link, ws) in platforms, (arch, link, ws)
        assert algorithm in ("mesh", "ring"), algorithm
        assert atoms >= 1 and (8 // ws) % atoms == 0, (arch, link, ws, atoms)


def test_fused_ladder_lookup_rejects_unknown_keys():
    for ladder in (fused_oneshot_ladder, fused_mesh_st_ladder, fused_ring_st_ladder):
        with pytest.raises(P.NoPolicyError):
            ladder(world_size=4, link="pcie", arch="gfx942")
        with pytest.raises(ValueError):
            ladder(world_size=4, link="nosuchlink", arch="gfx950")


@pytest.mark.parametrize("cell", CELLS)
def test_fused_oneshot_max_bytes_is_the_window_max(cell):
    link, ws, arch = cell
    want = _fused_row(*cell).window(ONESHOT, quantized=False).max - 1
    got = resolve_fused_oneshot_max_bytes(world_size=ws, link=link, arch=arch)
    assert got == want


def test_fused_overrides_are_separate_from_the_plain_ones():
    plain = _resolve()
    fused = _resolve_fused()
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert _resolve_fused() == fused
    with _env(
        AITER_FLY_AR_FUSED_ONESHOT_MAX_BYTES="65536",
        AITER_FLY_AR_FUSED_MESH_MIN_BYTES="65536",
    ):
        got = _resolve_fused()
        assert got.window(ONESHOT, quantized=True) == Window(0, 65536)
        assert got.window(ONESHOT, quantized=False) == Window(0, 65536)
        assert got.window(MESH, quantized=True).min == 65536
        assert _resolve() == plain


class _FakeFusedEngine:
    """The geometry methods the fused predicate asks an engine."""

    def supports_hidden(self, hidden):
        return True

    def pads_hidden(self, hidden):
        return hidden


def _fused_backend(quantized, link="pcie", ws=4, arch="gfx950"):
    """A ``FlyDSLAllReduceRMSNorm`` with fake engines and every width ready, so
    no process group or GPU is needed. The families are the ones its
    constructor builds."""
    policy = _fused_row(link, ws, arch)
    b = object.__new__(FlyDSLAllReduceRMSNorm)
    b.disabled = False
    b.device = torch.device("cpu")
    b.policy = policy
    b._quantized = quantized
    b._engines = {
        alg.value: _FakeFusedEngine()
        for alg in policy.algorithms(quantized=quantized)
        if quantized != (alg == ONESHOT)
    }
    b._ready = {(fam, h) for fam in b._engines for h in (64, 4096)}
    b._warned_capture = False
    return b


def test_fused_backends_split_the_families():
    """The exact backend (``CustomAllreduce``) runs only the one-shot; the
    quantized one (``QuickAllReduce``) only the mesh and the ring, and declines
    the one-shot's payloads so the exact backend gets them."""
    policy = _fused_row("pcie", 4, "gfx950")
    exact, quant = _fused_backend(False), _fused_backend(True)
    assert set(exact._engines) == {ONESHOT.value}
    assert ONESHOT.value not in quant._engines and quant._engines
    small = policy.window(ONESHOT, quantized=False).min + 4096
    big = policy.window(MESH, quantized=True).min + (64 << 10)
    assert exact.family_for(small) == ONESHOT.value
    assert quant.family_for(small) is None
    assert quant.family_for(big) == policy.select(big, quantized=True).value
    assert exact.family_for(big) == (
        ONESHOT.value if policy.window(ONESHOT, quantized=False).contains(big) else None
    )


def test_fused_selector_rejects_what_the_engines_reject():
    """The fused predicate declines a weakly-contiguous view, a misaligned
    tensor, a non-bf16 one and a wrong-width weight: once dispatch picks
    FlyDSL, the engine's exception replaces the fallback."""
    b = _fused_backend(False)
    hidden = 4096
    inp = torch.zeros(4, hidden, dtype=torch.bfloat16)
    res = torch.zeros_like(inp)
    w = torch.ones(hidden, dtype=torch.bfloat16)
    assert b.should_fly_fused_ar_rms(inp, res, w)

    weak = torch.zeros(hidden, 4, dtype=torch.bfloat16).t()
    assert is_weak_contiguous(weak) and not weak.is_contiguous()
    assert not b.should_fly_fused_ar_rms(weak, res, w)
    assert not b.should_fly_fused_ar_rms(inp, weak, w)

    flat = torch.zeros(4 * hidden + 1, dtype=torch.bfloat16)
    shifted = flat[1:].view(4, hidden)
    assert shifted.data_ptr() % 16
    assert not b.should_fly_fused_ar_rms(shifted, res, w)
    assert not b.should_fly_fused_ar_rms(inp.half(), res, w)
    assert not b.should_fly_fused_ar_rms(inp, res, w[:-64])


def test_fused_hiddens_env():
    with _env(AITER_FLY_AR_FUSED_HIDDENS=""):
        assert P.fused_hiddens((4096, 0, 4096)) == (4096,)
    with _env(AITER_FLY_AR_FUSED_HIDDENS="7168, 2048,x,-1"):
        assert P.fused_hiddens((4096,)) == (2048, 4096, 7168)


def test_resolve_fused_rejects_unknown_keys():
    with pytest.raises(ValueError):
        _resolve_fused(link="infiniband")
    with pytest.raises(ValueError):
        _resolve_fused(ws=3)
    with pytest.raises(P.NoPolicyError):
        _resolve_fused(link="pcie", arch="gfx942")
