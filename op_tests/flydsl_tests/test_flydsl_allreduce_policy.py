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
from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
    SUPPORTED_ATOMS,
    SUPPORTED_BLOCKS,
    oneshot_ladder,
)
from aiter.ops.flydsl.kernels.comm.quick_allreduce_codec import (
    SUPPORTED_BLOCKS as TWO_STAGE_BLOCKS,
)
from aiter.ops.flydsl.kernels.comm.quick_allreduce_int4 import (
    SUPER_TILES,
    mesh_st_ladder,
)
from aiter.ops.flydsl.kernels.comm.quick_allreduce_int4_ring import (
    RING_SUPER_TILES,
    ring_st_ladder,
)
from aiter.ops.flydsl.one_shot_allreduce import max_payload_bytes

WORLDS = P.SUPPORTED_WORLDS
# Every arch with policy rows of its own, and one without (``FAMILY_POLICY``).
ARCHES = ("gfx942", "gfx950")
CELLS = [(link, ws, arch) for arch in ARCHES for link in P.LINKS for ws in WORLDS]


@pytest.mark.parametrize("cell", CELLS)
def test_resolved_slots_partition_by_size(cell):
    """Each slot's own view must be internally ordered.

    The two views do not have to meet -- where ``cross_device_reduce`` beats
    both FlyDSL families they may leave a gap -- but neither may be inverted.
    """
    one = P.resolve_oneshot(*cell)
    quant = P.resolve_quant(*cell)
    assert 0 < one.max_bytes
    assert one.min_bytes <= one.max_bytes
    assert 0 < quant.floor <= quant.mesh_max <= quant.max_bytes


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("link", P.LINKS)
@pytest.mark.parametrize("ws", WORLDS)
def test_dispatch_is_monotone(ws, link, arch):
    """The path choice must never go backwards as the payload grows.

    Composed across both slots, in the order ``CudaCommunicator.all_reduce``
    consults them, so this is the ordering a payload actually experiences.
    Below ``min_bytes`` the payload falls back by design (the Aiter kernel is
    faster there); above it the order must hold.
    """
    # ``fallback`` sits between the one-shot and the mesh: where
    # ``cross_device_reduce`` beats both FlyDSL families, the payload falls
    # through to it, and only larger payloads reach the mesh.
    order = {"oneshot": 0, "fallback": 1, "mesh": 2, "ring": 3}
    floor = P.resolve_oneshot(link, ws, arch).min_bytes
    sizes = [1 << k for k in range(4, 31)]
    below = [_slot_of(link, ws, n, arch) for n in sizes if n < floor]
    assert set(below) <= {"fallback"}
    seen = [order[_slot_of(link, ws, n, arch)] for n in sizes if n >= floor]
    assert seen == sorted(seen)


@pytest.mark.parametrize("ws", WORLDS)
def test_ladders_are_well_formed(ws):
    """Every ladder starts at 0, ascends, and names values its kernel accepts.

    A ladder that does not start at 0 leaves the smallest payloads with no rung;
    one that is not ascending makes ``_pick_st``/``_pick_cfg`` -- which take the
    *last* rung at or below the payload -- select something arbitrary.
    """
    for link in P.LINKS:
        for name, rungs, valid_st in (
            ("mesh", mesh_st_ladder(ws, link), SUPER_TILES),
            ("ring", ring_st_ladder(ws, link), RING_SUPER_TILES),
        ):
            assert rungs, (name, link)
            assert rungs[0].min_bytes == 0, (name, link)
            floors = [r.min_bytes for r in rungs]
            assert floors == sorted(floors), (name, link)
            for r in rungs:
                assert r.super_tile in valid_st, (name, link, r)
                assert r.grid_cap >= 1, (name, link, r)
                assert r.block in TWO_STAGE_BLOCKS, (name, link, r)

    for link in P.LINKS:
        one = oneshot_ladder(ws, link)
        assert one and one[0].min_bytes == 0, link
        floors = [r.min_bytes for r in one]
        assert floors == sorted(floors), link
        for r in one:
            assert r.atoms in SUPPORTED_ATOMS, (link, r)
            assert r.grid_cap >= 1, (link, r)
            assert r.fanout in ("peer", "atom"), (link, r)
            assert r.block in SUPPORTED_BLOCKS, (link, r)


@pytest.mark.parametrize("ws", WORLDS)
def test_oneshot_ladder_unknown_fabric_falls_back(ws):
    """An unknown fabric gets a single conservative rung rather than silently
    borrowing another fabric's table."""
    assert len(oneshot_ladder(ws, "nosuchlink")) == 1


@pytest.mark.parametrize("cell", CELLS)
def test_max_payload_bytes_is_keyed_on_the_fabric(cell):
    """The default ceiling is where the fallback overtakes the one-shot, which
    is a property of the arch and the fabric.
    """
    link, ws, arch = cell
    assert (
        max_payload_bytes(ws, link, arch)
        == P.family_policy(link, ws, arch).oneshot_max_exact
    )


def test_arch_rows_override_only_their_arch():
    """An arch row replaces the default row on that arch alone."""
    for (arch, link, ws), row in P.ARCH_FAMILY_POLICY.items():
        assert P.family_policy(link, ws, arch) == row
        assert P.family_policy(link, ws, "gfx942") == P.FAMILY_POLICY[(link, ws)]


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("ws", WORLDS)
def test_ladder_rungs_fall_inside_their_dispatch_window(ws, arch):
    """A rung above its family's window is an engine and an IPC inbox built for
    payloads that can never arrive.

    This is exactly the bug the windowed refit removed: fitted over the whole
    sweep, the TP8 one-shot ladder wanted a second rung at 192 KiB, four times
    above anything that schedule is dispatched at.

    The one-shot window is the wider of the two ceilings.
    """
    for link in P.LINKS:
        quant = P.resolve_quant(link, ws, arch)
        # The one-shot never serves above its own ceiling: with the quant slot
        # open, the payloads between that ceiling and the quant floor fall
        # through to ``cross_device_reduce``.
        one_hi = P.resolve_oneshot(link, ws, arch).max_bytes
        for r in oneshot_ladder(ws, link)[1:]:
            assert r.min_bytes < one_hi, ("oneshot", link, ws, r)
        for r in mesh_st_ladder(ws, link)[1:]:
            assert r.min_bytes < quant.mesh_max, ("mesh", link, ws, r)
    # Ring rungs are offsets into an unbounded window, so only the ordering
    # above constrains them.


def _env(**kw):
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items()}, clear=False)


def _slot_of(link: str, ws: int, nbytes: int, arch: str) -> str:
    """Which path *nbytes* reaches, composed in real dispatch order.

    ``CudaCommunicator.all_reduce`` tries the quick-reduce slot, then the
    custom-all-reduce slot, then RCCL. Assumes the quantized slot is open
    (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``).
    """
    quant = P.resolve_quant(link, ws, arch)
    if quant.floor < nbytes <= quant.max_bytes:
        return P.pick_quant_family(nbytes, quant)
    one = P.resolve_oneshot(link, ws, arch)
    if one.min_bytes <= nbytes <= one.max_bytes:
        return "oneshot"
    return "fallback"


@pytest.mark.parametrize("cell", CELLS)
def test_slots_share_one_boundary(cell):
    """The quick-reduce floor is the table's ``oneshot_max``, and one override
    moves it and the one-shot ceiling together.

    One number read from two sides. If they ever drift apart, payloads either
    get double-claimed (the floor drops below the crossover) or fall into a hole
    neither family serves.
    """
    link, ws, arch = cell
    assert (
        P.resolve_quant(link, ws, arch).floor
        == P.family_policy(link, ws, arch).oneshot_max
    )
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert P.resolve_quant(link, ws, arch).floor == 65536
        # The override moves both readings, or the boundary splits in two.
        assert P.resolve_oneshot(link, ws, arch).max_bytes == 65536


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
    table_one = P.family_policy("pcie", 8, "gfx942").oneshot_max_exact
    table_floor = P.family_policy("pcie", 8, "gfx942").oneshot_max
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert P.resolve_oneshot("pcie", 8, "gfx942").max_bytes == 65536
        assert P.resolve_quant("pcie", 8, "gfx942").floor == 65536
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="1048576"):
        assert P.resolve_quant("pcie", 4, "gfx942").mesh_max == 1048576
    # -1 is the house sentinel for "unset, use the table".
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="-1"):
        assert P.resolve_oneshot("pcie", 8, "gfx942").max_bytes == table_one
        assert P.resolve_quant("pcie", 8, "gfx942").floor == table_floor
    # Garbage warns and is ignored.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="lots"):
        assert P.resolve_oneshot("pcie", 8, "gfx942").max_bytes == table_one
        assert P.resolve_quant("pcie", 8, "gfx942").floor == table_floor


def test_oneshot_min_override():
    """``AITER_FLY_AR_ONESHOT_MIN_BYTES`` moves only the one-shot's small-payload
    floor; the ceiling and the quant floor are untouched."""
    table_min = P.family_policy("xgmi", 8, "gfx942").min_bytes
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="65536"):
        assert P.resolve_oneshot("xgmi", 8, "gfx942").min_bytes == 65536
        # ceiling and quant slot are independent of the min override
        assert (
            P.resolve_oneshot("xgmi", 8, "gfx942").max_bytes
            == P.family_policy("xgmi", 8, "gfx942").oneshot_max_exact
        )
        assert (
            P.resolve_quant("xgmi", 8, "gfx942").floor
            == P.family_policy("xgmi", 8, "gfx942").oneshot_max
        )
    # 0 is a valid override: accept every size down to the custom-AR floor.
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="0"):
        assert P.resolve_oneshot("xgmi", 8, "gfx942").min_bytes == 0
    # -1 is the house sentinel for "unset, use the table".
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="-1"):
        assert P.resolve_oneshot("xgmi", 8, "gfx942").min_bytes == table_min
    # Garbage warns and is ignored.
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="lots"):
        assert P.resolve_oneshot("xgmi", 8, "gfx942").min_bytes == table_min


def test_override_cannot_invert_the_partition():
    """Pushing the one-shot ceiling past the mesh window means "give me the
    one-shot up to here", not "crash" -- the mesh window closes instead.

    Run on any cell whose table reaches the ring, with the ceiling placed just
    past that cell's own mesh window, so no tuned value is assumed.
    """
    cells = [
        c
        for c in CELLS
        if P.resolve_quant(*c).mesh_max + 16 < P.resolve_quant(*c).max_bytes
    ]
    if not cells:
        pytest.skip("no (link, world) in the table reaches the ring")
    link, ws, arch = cells[0]
    mesh_max = P.resolve_quant(link, ws, arch).mesh_max
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES=str(mesh_max + 16)):
        p = P.resolve_quant(link, ws, arch)
        assert p.mesh_max >= p.floor
        assert P.quant_families_reachable(p) == ("ring",)
        # A payload under the raised ceiling now reaches the one-shot, because
        # the quant slot's floor moved with it.
        assert _slot_of(link, ws, mesh_max, arch) == "oneshot"


def test_resolvers_reject_unknown_keys():
    for resolve in (P.resolve_oneshot, P.resolve_quant):
        with pytest.raises(ValueError):
            resolve("infiniband", 4)
        with pytest.raises(ValueError):
            resolve("pcie", 3)


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
    quant = P.resolve_quant("pcie", 4, "gfx942")
    qr = SimpleNamespace(
        _fly_policy=quant, _fly_engines={"mesh": object(), "ring": object()}
    )
    t = _full_storage_transpose(quant.floor + (64 << 10))
    assert QuickAllReduce._should_fly(qr, t.contiguous())
    assert not QuickAllReduce._should_fly(qr, t)

    one = P.resolve_oneshot("pcie", 4, "gfx942")
    ca = SimpleNamespace(
        _fly_oneshot=object(), _fly_policy=one, device=torch.device("cpu")
    )
    t = _full_storage_transpose(max(one.min_bytes, 1 << 12))
    assert CustomAllreduce._should_fly_oneshot(ca, t.contiguous(), True, False)
    assert not CustomAllreduce._should_fly_oneshot(ca, t, True, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
