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
from aiter.ops.flydsl.kernels.one_shot_allreduce import (
    SUPPORTED_ATOMS,
    SUPPORTED_BLOCKS,
    oneshot_ladder,
)
from aiter.ops.flydsl.kernels.quick_allreduce_codec import (
    SUPPORTED_BLOCKS as TWO_STAGE_BLOCKS,
)
from aiter.ops.flydsl.kernels.quick_allreduce_mesh import (
    SUPER_TILES,
    mesh_st_ladder,
)
from aiter.ops.flydsl.kernels.quick_allreduce_ring import (
    RING_SUPER_TILES,
    ring_st_ladder,
)
from aiter.ops.flydsl.one_shot_allreduce import max_payload_bytes

WORLDS = P.SUPPORTED_WORLDS
CELLS = [(link, ws) for link in P.LINKS for ws in WORLDS]


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


@pytest.mark.parametrize("ws", WORLDS)
def test_dispatch_is_monotone(ws):
    """The path choice must never go backwards as the payload grows.

    Composed across both slots, in the order ``CudaCommunicator.all_reduce``
    consults them, so this is the ordering a payload actually experiences.
    Below ``min_bytes`` the payload falls back by design (the Aiter kernel is
    faster there); above it the order must hold.
    """
    order = {"oneshot": 0, "mesh": 1, "ring": 2, "fallback": 3}
    floor = P.resolve_oneshot("pcie", ws).min_bytes
    sizes = [1 << k for k in range(4, 31)]
    below = [_slot_of("pcie", ws, n) for n in sizes if n < floor]
    assert set(below) <= {"fallback"}
    seen = [order[_slot_of("pcie", ws, n)] for n in sizes if n >= floor]
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
            assert rungs[0][0] == 0, (name, link)
            assert [r[0] for r in rungs] == sorted(r[0] for r in rungs), (name, link)
            for _floor, st, cap, block, skip_self in rungs:
                assert st in valid_st, (name, link, st)
                assert cap >= 1, (name, link, cap)
                assert block in TWO_STAGE_BLOCKS, (name, link, block)
                assert isinstance(skip_self, bool), (name, link, skip_self)
                # The ring never writes its own inbox, so it has no self round
                # trip to skip; the host rejects the combination.
                assert not (name == "ring" and skip_self), (link, ws)

    for link in P.LINKS:
        one = oneshot_ladder(ws, link)
        assert one and one[0][0] == 0, link
        assert [r[0] for r in one] == sorted(r[0] for r in one), link
        for _floor, atoms, cap, fanout, block, skip_self in one:
            assert atoms in SUPPORTED_ATOMS, (link, atoms)
            assert cap >= 1, (link, cap)
            assert fanout in ("peer", "atom"), (link, fanout)
            assert block in SUPPORTED_BLOCKS, (link, block)
            assert isinstance(skip_self, bool), (link, skip_self)


@pytest.mark.parametrize("ws", WORLDS)
def test_oneshot_ladder_unknown_fabric_falls_back(ws):
    """An unknown fabric gets a single conservative rung rather than silently
    borrowing another fabric's table."""
    assert len(oneshot_ladder(ws, "nosuchlink")) == 1


def test_max_payload_bytes_is_keyed_on_the_fabric():
    """The default ceiling is where the fallback overtakes the one-shot, which
    is a property of the fabric.
    """
    for ws in WORLDS:
        for link in P.LINKS:
            assert (
                max_payload_bytes(ws, link)
                == P.FAMILY_POLICY[(link, ws)].oneshot_max_exact
            )


@pytest.mark.parametrize("ws", WORLDS)
def test_ladder_rungs_fall_inside_their_dispatch_window(ws):
    """A rung above its family's window is an engine and an IPC inbox built for
    payloads that can never arrive.

    This is exactly the bug the windowed refit removed: fitted over the whole
    sweep, the TP8 one-shot ladder wanted a second rung at 192 KiB, four times
    above anything that schedule is dispatched at.

    The one-shot window is the wider of the two ceilings.
    """
    for link in P.LINKS:
        quant = P.resolve_quant(link, ws)
        # The one-shot's window is the wider of the two ceilings: it serves up
        # to its own ceiling when the quant slot is closed, and up to the quant
        # floor when that slot is open.
        one_hi = max(P.resolve_oneshot(link, ws).max_bytes, quant.floor)
        for _floor, *_ in oneshot_ladder(ws, link)[1:]:
            assert _floor < one_hi, ("oneshot", link, ws, _floor)
        for floor, *_ in mesh_st_ladder(ws, link)[1:]:
            assert floor < quant.mesh_max, ("mesh", link, ws, floor)
    # Ring rungs are offsets into an unbounded window, so only the ordering
    # above constrains them.


def _env(**kw):
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items()}, clear=False)


def _slot_of(link: str, ws: int, nbytes: int) -> str:
    """Which path *nbytes* reaches, composed in real dispatch order.

    ``CudaCommunicator.all_reduce`` tries the quick-reduce slot, then the
    custom-all-reduce slot, then RCCL. Assumes the quantized slot is open
    (``AITER_QUICK_REDUCE_QUANTIZATION=INT4``).
    """
    quant = P.resolve_quant(link, ws)
    if quant.floor < nbytes <= quant.max_bytes:
        return P.pick_quant_family(nbytes, quant)
    one = P.resolve_oneshot(link, ws)
    if one.min_bytes <= nbytes <= one.max_bytes:
        return "oneshot"
    return "fallback"


@pytest.mark.parametrize("cell", CELLS)
def test_slots_share_one_boundary(cell):
    """The quick-reduce floor *is* the one-shot/mesh crossover.

    One number read from two sides. If they ever drift apart, payloads either
    get double-claimed (the floor drops below the crossover) or fall into a hole
    neither family serves.
    """
    link, ws = cell
    assert P.resolve_quant(link, ws).floor == P.FAMILY_POLICY[cell].oneshot_max
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert P.resolve_quant(link, ws).floor == 65536
        # The override moves both readings, or the boundary splits in two.
        assert P.resolve_oneshot(link, ws).max_bytes == 65536


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
    table_one = P.FAMILY_POLICY[("pcie", 8)].oneshot_max_exact
    table_floor = P.FAMILY_POLICY[("pcie", 8)].oneshot_max
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="65536"):
        assert P.resolve_oneshot("pcie", 8).max_bytes == 65536
        assert P.resolve_quant("pcie", 8).floor == 65536
    with _env(AITER_FLY_AR_MESH_MAX_BYTES="1048576"):
        assert P.resolve_quant("pcie", 4).mesh_max == 1048576
    # -1 is the house sentinel for "unset, use the table".
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="-1"):
        assert P.resolve_oneshot("pcie", 8).max_bytes == table_one
        assert P.resolve_quant("pcie", 8).floor == table_floor
    # Garbage warns and is ignored.
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES="lots"):
        assert P.resolve_oneshot("pcie", 8).max_bytes == table_one
        assert P.resolve_quant("pcie", 8).floor == table_floor


def test_oneshot_min_override():
    """``AITER_FLY_AR_ONESHOT_MIN_BYTES`` moves only the one-shot's small-payload
    floor; the ceiling and the quant floor are untouched."""
    table_min = P.FAMILY_POLICY[("xgmi", 8)].min_bytes
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="65536"):
        assert P.resolve_oneshot("xgmi", 8).min_bytes == 65536
        # ceiling and quant slot are independent of the min override
        assert (
            P.resolve_oneshot("xgmi", 8).max_bytes
            == P.FAMILY_POLICY[("xgmi", 8)].oneshot_max_exact
        )
        assert P.resolve_quant("xgmi", 8).floor == P.FAMILY_POLICY[("xgmi", 8)].oneshot_max
    # 0 is a valid override: accept every size down to the custom-AR floor.
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="0"):
        assert P.resolve_oneshot("xgmi", 8).min_bytes == 0
    # -1 is the house sentinel for "unset, use the table".
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="-1"):
        assert P.resolve_oneshot("xgmi", 8).min_bytes == table_min
    # Garbage warns and is ignored.
    with _env(AITER_FLY_AR_ONESHOT_MIN_BYTES="lots"):
        assert P.resolve_oneshot("xgmi", 8).min_bytes == table_min


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
    link, ws = cells[0]
    mesh_max = P.resolve_quant(link, ws).mesh_max
    with _env(AITER_FLY_AR_ONESHOT_MAX_BYTES=str(mesh_max + 16)):
        p = P.resolve_quant(link, ws)
        assert p.mesh_max >= p.floor
        assert P.quant_families_reachable(p) == ("ring",)
        # A payload under the raised ceiling now reaches the one-shot, because
        # the quant slot's floor moved with it.
        assert _slot_of(link, ws, mesh_max) == "oneshot"


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
    quant = P.resolve_quant("pcie", 4)
    qr = SimpleNamespace(
        _fly_policy=quant, _fly_engines={"mesh": object(), "ring": object()}
    )
    t = _full_storage_transpose(quant.floor + (64 << 10))
    assert QuickAllReduce._should_fly(qr, t.contiguous())
    assert not QuickAllReduce._should_fly(qr, t)

    one = P.resolve_oneshot("pcie", 4)
    ca = SimpleNamespace(
        _fly_oneshot=object(), _fly_policy=one, device=torch.device("cpu")
    )
    t = _full_storage_transpose(max(one.min_bytes, 1 << 12))
    assert CustomAllreduce._should_fly_oneshot(ca, t.contiguous(), True, False)
    assert not CustomAllreduce._should_fly_oneshot(ca, t, True, False)


# ---------------------------------------------------------------------------
# Fused all-reduce+RMSNorm sub-API.
#
# The fused families all live in one aiter slot (QuickAllReduce), so unlike the
# plain path they keep a single unified view -- FusedPolicy -- with its own
# resolver, family picker, reachability and env overrides. accuracy_mode adds a
# second axis the plain path does not have: exact (default) leaves only the
# one-shot reachable, fast opens the quantized mesh/ring beyond it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", CELLS)
def test_fused_every_cell_has_a_policy(cell):
    assert cell in P.FUSED_FAMILY_POLICY


def test_accuracy_mode_env():
    assert P.DEFAULT_ACCURACY == "exact"
    with _env(AITER_FLY_AR_ACCURACY="exact"):
        assert P.accuracy_mode() == "exact"
    with _env(AITER_FLY_AR_ACCURACY="fast"):
        assert P.accuracy_mode() == "fast"
    # Unset and garbage both fall back to the exact default rather than
    # silently enabling the quantized families.
    with mock.patch.dict(os.environ, {}, clear=True):
        assert P.accuracy_mode() == P.DEFAULT_ACCURACY
    with _env(AITER_FLY_AR_ACCURACY="lossy"):
        assert P.accuracy_mode() == P.DEFAULT_ACCURACY


@pytest.mark.parametrize("cell", CELLS)
def test_fused_exact_mode_is_oneshot_only(cell):
    """Exact mode never reaches a quantized family, on any cell.

    Dropping this would silently quantize the fused output by default -- an
    accuracy regression for a tensor that feeds the next layer. The one-shot
    serves up to its widened ``oneshot_max_exact`` ceiling and above that the
    path declines.
    """
    link, ws = cell
    p = P.resolve_fused(link, ws, mode="exact")
    assert p.mesh_max == 0
    assert p.ring_max == 0
    assert P.fused_families_reachable(p) == ("oneshot",)
    assert p.oneshot_max == P.FUSED_FAMILY_POLICY[cell].oneshot_max_exact


@pytest.mark.parametrize("cell", CELLS)
def test_fused_fast_mode_reaches_past_the_oneshot(cell):
    """Fast mode opens at least one quantized family beyond the one-shot."""
    link, ws = cell
    p = P.resolve_fused(link, ws, mode="fast")
    families = P.fused_families_reachable(p)
    assert families[0] == "oneshot"
    assert len(families) >= 2


def test_fused_pcie_tp2_empty_mesh_reaches_the_ring():
    """The pcie/2 fast cell has an empty mesh window, so it skips straight to
    the ring.

    Its ``mesh_max`` equals its ``oneshot_max``: the mesh interval
    ``(oneshot_max, mesh_max]`` is empty, so no payload is ever dispatched to
    mesh and the ring picks up everything above the one-shot.
    """
    p = P.resolve_fused("pcie", 2, mode="fast")
    assert p.mesh_max == p.oneshot_max
    assert P.fused_families_reachable(p) == ("oneshot", "ring")
    assert P.pick_fused_family(p.oneshot_max, p) == "oneshot"
    assert P.pick_fused_family(p.oneshot_max + 1, p) == "ring"


@pytest.mark.parametrize("cell", CELLS)
def test_fused_pick_family_is_monotone(cell):
    """The fused family choice never goes backwards as the payload grows, and
    every family it picks is one the policy says is reachable."""
    link, ws = cell
    p = P.resolve_fused(link, ws, mode="fast")
    reachable = P.fused_families_reachable(p)
    order = {"oneshot": 0, "mesh": 1, "ring": 2}
    seen = []
    for n in (1 << k for k in range(4, 31)):
        fam = P.pick_fused_family(n, p)
        assert fam in reachable, (cell, n, fam, reachable)
        seen.append(order[fam])
    assert seen == sorted(seen)


def test_fused_family_windows_partition_by_size():
    """On a cell with all three families, the boundaries hand off cleanly.

    pcie/4 fast reaches one-shot, mesh and ring, so it exercises both
    crossovers at once.
    """
    p = P.resolve_fused("pcie", 4, mode="fast")
    assert P.fused_families_reachable(p) == ("oneshot", "mesh", "ring")
    assert P.pick_fused_family(p.oneshot_max, p) == "oneshot"
    assert P.pick_fused_family(p.oneshot_max + 1, p) == "mesh"
    assert P.pick_fused_family(p.mesh_max, p) == "mesh"
    assert P.pick_fused_family(p.mesh_max + 1, p) == "ring"


def test_fused_max_bytes_is_none_when_a_family_is_unbounded():
    """An open-ended family means no integer ceiling; exact mode has one."""
    assert P.resolve_fused("pcie", 4, mode="fast").max_bytes is None
    assert P.resolve_fused("xgmi", 2, mode="fast").max_bytes is None
    assert isinstance(P.resolve_fused("pcie", 4, mode="exact").max_bytes, int)


def test_fused_byte_overrides():
    table_one = P.FUSED_FAMILY_POLICY[("pcie", 8)].oneshot_max_exact
    with _env(AITER_FLY_AR_FUSED_ONESHOT_MAX_BYTES="65536"):
        assert P.resolve_fused("pcie", 8, mode="exact").oneshot_max == 65536
    with _env(AITER_FLY_AR_FUSED_MESH_MAX_BYTES="1048576"):
        # The mesh window is floored at the one-shot ceiling.
        p = P.resolve_fused("pcie", 4, mode="fast")
        assert p.mesh_max == max(1048576, p.oneshot_max)
    # -1 is the house sentinel for "unset, use the table".
    with _env(AITER_FLY_AR_FUSED_ONESHOT_MAX_BYTES="-1"):
        assert P.resolve_fused("pcie", 8, mode="exact").oneshot_max == table_one
    # Garbage warns and is ignored.
    with _env(AITER_FLY_AR_FUSED_ONESHOT_MAX_BYTES="lots"):
        assert P.resolve_fused("pcie", 8, mode="exact").oneshot_max == table_one


def test_fused_exact_mode_ignores_the_mesh_override():
    """The mesh override is meaningless in exact mode, which has no mesh
    window, so it is dropped rather than reopening one."""
    with _env(AITER_FLY_AR_FUSED_MESH_MAX_BYTES="1048576"):
        p = P.resolve_fused("pcie", 4, mode="exact")
        assert p.mesh_max == 0
        assert P.fused_families_reachable(p) == ("oneshot",)


def test_fused_min_bytes_override():
    with _env(AITER_FLY_AR_FUSED_MIN_BYTES="12345"):
        assert P.resolve_fused("pcie", 2, mode="fast").min_bytes == 12345
        assert P.resolve_fused("xgmi", 4, mode="exact").min_bytes == 12345


def test_resolve_fused_rejects_unknown_keys():
    with pytest.raises(ValueError):
        P.resolve_fused("infiniband", 4)
    with pytest.raises(ValueError):
        P.resolve_fused("pcie", 3)
    with pytest.raises(ValueError):
        P.resolve_fused("pcie", 4, mode="lossy")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
